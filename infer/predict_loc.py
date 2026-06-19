#!/usr/bin/env python3
"""TRACE inference — MODE A (localization only): classification + forgery bounding boxes.

Runs the TRACE model (in-domain LN-tuned DINOv2-L + end-to-end LoRA + Token Reference
Attention head) on a folder of images and writes ONE DDL-X submission JSON per
image, in the official 3-field schema:

  {
    "Classification result": "fake" | "real",
    "Bounding boxes": [[x0,y0,x1,y1], ...] | "None",   # int 1..1000 (/1000 coords)
    "Visible forgery traces": ""                         # filled by the caption stage (mode B)
  }

- classification: softmax P(fake) >= cls_thr (config.model.cls_thr, val-calibrated).
- boxes: connected-component decode of the fakeness map (config.decode), /1000 INT, clamped 1..1000.
- real, or fake with no surviving CC box -> "Bounding boxes": "None".

TRACE resizes images by STRETCH to a square, and /1000 normalized coords are stretch-invariant,
so predicted boxes are already in the original image's /1000 frame (no un-pad needed).

For a complete (caption-filled) submission, run the caption stage next (infer/caption_qwen_vllm.py)
then package (submit/package_submit.py).

Example:
  uv run python infer/predict_loc.py --images /path/to/test/image --out-dir out/json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import trace_config as C
from models.trace import TRACEHead
from train.train_trace import TRACEBackbone, parse_blocks
from core.data_metrics import (IMAGENET_MEAN, IMAGENET_STD, INPUT,
                                       jpeg_recompress, cc_boxes_1000)

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def preprocess(path: Path, jpeg_q: int = 90) -> torch.Tensor:
    """load -> (JPEG de-bias q90) -> stretch-resize INPUT -> ImageNet normalize. Matches DDLXDataset(jpeg='test')."""
    im = Image.open(path).convert("RGB")
    if jpeg_q:
        im = jpeg_recompress(im, jpeg_q)
    im = im.resize((INPUT, INPUT), Image.Resampling.BILINEAR)
    pix = (torch.from_numpy(np.asarray(im, np.float32) / 255.).permute(2, 0, 1)
           - IMAGENET_MEAN) / IMAGENET_STD
    return pix


def build_model(dev):
    m = C.MODEL
    vfm = TRACEBackbone(m["backbone"], dev, backbone_ckpt=str(C.CKPT_BACKBONE), lora_rank=m["lora_rank"],
                     lora_targets=tuple(m["lora_targets"].split(",")),
                     lora_blocks=parse_blocks(m["lora_blocks"]), lora_alpha=m["lora_alpha"])
    ck = torch.load(C.CKPT_TRACE, map_location="cpu", weights_only=False)
    vfm.load_state_dict(ck["lora"], strict=False)
    vfm.eval()
    head = TRACEHead(dim=1024, scales=(m["grid"],), out_size=m["out_size"], cls_dim=1024,
                     decoder_type=m["decoder_type"]).to(dev)
    head.decoder.dynamic_upsample = True
    head.load_state_dict(ck["head"])
    head.eval()
    return vfm, head


def to_int_boxes(boxes_1000):
    """/1000 float boxes -> list of [4 int] clamped to 1..1000 with positive area; drop degenerate."""
    out = []
    for b in boxes_1000:
        x0, y0, x1, y1 = (int(max(1, min(1000, round(v)))) for v in b)
        if x0 < x1 and y0 < y1:
            out.append([x0, y0, x1, y1])
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path, help="folder of input images (recursed)")
    ap.add_argument("--out-dir", required=True, type=Path, help="output dir for per-image JSONs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cls-thr", type=float, default=None, help="override config.model.cls_thr")
    ap.add_argument("--cc-thr", type=float, default=None, help="override config.decode.cc_thr")
    ap.add_argument("--cc-min-px", type=int, default=None, help="override config.decode.cc_min_px")
    ap.add_argument("--no-jpeg", action="store_true", help="disable the q90 JPEG de-bias re-encode")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    S = C.MODEL["out_size"]
    cls_thr = a.cls_thr if a.cls_thr is not None else C.MODEL["cls_thr"]
    thr = a.cc_thr if a.cc_thr is not None else C.DECODE["cc_thr"]
    min_px = a.cc_min_px if a.cc_min_px is not None else C.DECODE["cc_min_px"]

    vfm, head = build_model(dev)
    jpeg_q = 0 if a.no_jpeg else 90
    a.out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in a.images.rglob("*") if p.suffix.lower() in IMG_EXT)
    if a.limit:
        paths = paths[:a.limit]
    if not paths:
        raise SystemExit(f"no images under {a.images}")
    print(f"[predict-loc] images={len(paths)} cls_thr={cls_thr} CC(thr={thr},min_px={min_px})", flush=True)

    n_fake = n_real = n_box = n_fake_nobox = 0
    for start in range(0, len(paths), a.batch_size):
        chunk = paths[start:start + a.batch_size]
        pix = torch.stack([preprocess(p, jpeg_q) for p in chunk]).to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zt, zv, cls = vfm(pix)
            out = head(zt, zv, cls)
        fake_prob = out["image_logit"].float().softmax(1)[:, 1].cpu().numpy()
        prob = torch.sigmoid(out["mask_logit"].squeeze(1).float()).cpu().numpy()
        for i, p in enumerate(chunk):
            if fake_prob[i] >= cls_thr:
                boxes = to_int_boxes(cc_boxes_1000(prob[i], thr, min_px, S))
                payload = {"Classification result": "fake",
                           "Bounding boxes": boxes if boxes else "None",
                           "Visible forgery traces": ""}
                n_fake += 1
                if boxes:
                    n_box += len(boxes)
                else:
                    n_fake_nobox += 1
            else:
                payload = {"Classification result": "real",
                           "Bounding boxes": "None",
                           "Visible forgery traces": ""}
                n_real += 1
            (a.out_dir / f"{p.stem}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                                      encoding="utf-8")
        print(f"[predict-loc] {min(start + a.batch_size, len(paths))}/{len(paths)} "
              f"fake={n_fake} real={n_real}", flush=True)

    summary = {"total": len(paths), "fake": n_fake, "real": n_real,
               "total_boxes": n_box, "fake_no_box": n_fake_nobox,
               "cls_thr": cls_thr, "cc_thr": thr, "cc_min_px": min_px}
    (a.out_dir.parent / "predict_loc_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[predict-loc] DONE {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
