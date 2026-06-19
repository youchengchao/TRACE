#!/usr/bin/env python3
"""Shared library for TRACE: the dataset loader, the frozen backbone, evaluation metrics, and
box decoding. Imported by both the training and inference code.

Pipeline pieces defined here:
  - load_csv / DDLXDataset : read the train/val/test split file and load + preprocess images
    (load -> optional JPEG re-encode to remove a format shortcut -> resize -> ImageNet-normalize).
  - FrozenBackbone         : a frozen DINOv2 image encoder (from timm) with an optional adapted
    checkpoint loaded into it. Used by this file's own training entry point.
  - cc_boxes_1000          : decode a forgery-probability map into boxes via connected components
    (coordinates in the benchmark's 0-1000 range).

Localization metrics (computed on fake images with ground-truth boxes):
  region_IoU           : overlap of the union of predicted boxes vs. the union of ground-truth
                         boxes, rasterized on a 1000x1000 grid (the benchmark's primary metric).
  instance_matched_IoU : one-to-one box matching (Hungarian assignment), normalized by the larger
                         of the predicted / ground-truth box counts (penalizes wrong box counts).
  best_match_IoU       : for each ground-truth box, the IoU of the best-overlapping prediction,
                         averaged over ground-truth boxes.
"""
from __future__ import annotations

import argparse, csv, io, json, math, random, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label as cc_label
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

CODE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CODE_ROOT))
from dataset.ddlx_annotations import parse_ddlx_boxes
from models.trace import TRACEHead
from trace_config import SPLIT_CSV   # machine-specific paths live in TRACE/config.yaml

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
TIMM_NAME = {"v2": "vit_large_patch14_reg4_dinov2.lvd142m", "v3": "vit_large_patch16_dinov3"}
INPUT = 448


def load_csv(split):
    out = []
    for r in csv.DictReader(open(SPLIT_CSV, newline="")):
        if r["split"] == split:
            out.append(SimpleNamespace(image_id=r["image_id"], image_path=r["image_path"],
                                       json_path=r["json_path"], label=int(r["label"])))
    return out


def boxes1000(jp):
    return parse_ddlx_boxes(Path(jp))[1] if jp else []


def jpeg_recompress(im: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO(); im.convert("RGB").save(buf, format="JPEG", quality=q); buf.seek(0)
    return Image.open(buf).convert("RGB")


class DDLXDataset(Dataset):
    """load -> (JPEG de-bias) -> resize 448 -> normalize. jpeg='train' random q[40,95], 'test' q90, ''=off."""
    def __init__(self, records, jpeg=""):
        self.recs = records; self.jpeg = jpeg

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        rec = self.recs[i]
        im = Image.open(rec.image_path).convert("RGB")
        if self.jpeg == "train":
            im = jpeg_recompress(im, random.randint(40, 95))
        elif self.jpeg == "test":
            im = jpeg_recompress(im, 90)
        im = im.resize((INPUT, INPUT), Image.Resampling.BILINEAR)
        pix = (torch.from_numpy(np.asarray(im, np.float32) / 255.).permute(2, 0, 1)
               - IMAGENET_MEAN) / IMAGENET_STD
        return pix, rec.image_id


def collate(b):
    return torch.stack([x[0] for x in b]), [x[1] for x in b]


class FrozenBackbone(nn.Module):
    """frozen raw DINOv2/v3; taps block11 (L12 -> z_type) & block17 (L18 -> z_value); CLS."""
    def __init__(self, kind, dev, backbone_ckpt=None):
        super().__init__()
        self.backbone = timm.create_model(TIMM_NAME[kind], pretrained=True, num_classes=0,
                                          dynamic_img_size=True).to(dev).eval()
        if backbone_ckpt:                          # adapted (LayerNorm-tuned) backbone checkpoint
            bb = torch.load(backbone_ckpt, map_location="cpu", weights_only=False)["backbone"]
            info = self.backbone.load_state_dict({k: v.to(dev) for k, v in bb.items()}, strict=False)
            print(f"[FrozenBackbone] loaded adapted backbone (missing={len(info.missing_keys)} "
                  f"unexpected={len(info.unexpected_keys)})", flush=True)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.prefix = int(getattr(self.backbone, "num_prefix_tokens", 5) or 5)
        self.cap = {}
        for li in (11, 17):
            self.backbone.blocks[li].register_forward_hook(
                lambda _m, _i, out, b=li: self.cap.__setitem__(b, out))

    @torch.no_grad()
    def forward(self, x):
        feats = self.backbone.forward_features(x)
        cls = feats[:, 0]
        b = x.shape[0]
        def grid(t):
            tok = t[:, self.prefix:, :]; s = int(round(tok.shape[1] ** 0.5))
            return tok.transpose(1, 2).reshape(b, -1, s, s)
        return grid(self.cap[11]), grid(self.cap[17]), cls   # z_type, z_value, cls


def raster_gt(boxes, S):
    m = np.zeros((S, S), np.float32)
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v / 1000 * S)) for v in b]
        m[y1:y2, x1:x2] = 1.0
    return m


def cc_boxes_1000(prob, thr, min_px, S):
    lab, n = cc_label(prob >= thr); out = []
    for k in range(1, n + 1):
        ys, xs = np.where(lab == k)
        if len(ys) < min_px: continue
        out.append([xs.min()/S*1000, ys.min()/S*1000, (xs.max()+1)/S*1000, (ys.max()+1)/S*1000])
    return out


def iou_box(a, b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1]); ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.


def union_raster(boxes, G):
    m=np.zeros((G,G),bool); s=G/1000.0
    for x1,y1,x2,y2 in boxes:
        xa,ya=max(0,int(round(x1*s))),max(0,int(round(y1*s)))
        xb,yb=min(G,int(round(x2*s))),min(G,int(round(y2*s)))
        if xb>xa and yb>ya: m[ya:yb,xa:xb]=True
    return m


def region_iou(pb, gt, G=256):       # monitor uses G=256 for speed; final test at G=1000
    if not gt: return None
    if not pb: return 0.0
    p,g=union_raster(pb,G),union_raster(gt,G); u=(p|g).sum()
    return float((p&g).sum())/u if u else 0.0


def instance_matched_iou(pb, gt):
    if not gt: return None
    if not pb: return 0.0
    M=np.zeros((len(pb),len(gt)))
    for i,p in enumerate(pb):
        for j,g in enumerate(gt): M[i,j]=iou_box(p,g)
    ri,ci=linear_sum_assignment(-M)
    return float(M[ri,ci].sum())/max(len(pb),len(gt))


def best_match_iou_list(pb, gt):     # per-GT best, micro over GT boxes
    return [max((iou_box(p,g) for p in pb), default=0.0) for g in gt]


def dice_bce(logit, gt):
    bce = F.binary_cross_entropy_with_logits(logit, gt)
    p = torch.sigmoid(logit); inter = (p*gt).sum((-1,-2))
    dice = 1 - (2*inter+1)/(p.sum((-1,-2))+gt.sum((-1,-2))+1)
    return bce + dice.mean()


@torch.no_grad()
def evaluate(vfm, head, vl, meta, dev, S, thr, min_px, limit=0):
    head.eval()
    region, matched, best, ys, ss = [], [], [], [], []
    for bi, (pix, ids) in enumerate(vl):
        if limit and bi >= limit: break
        pix = pix.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zt, zv, cls = vfm(pix)
            out = head(zt, zv, cls)
        prob = torch.sigmoid(out["mask_logit"].squeeze(1).float()).cpu().numpy()
        fp = out["image_logit"].float().softmax(1)[:, 1].cpu().numpy()
        for i, iid in enumerate(ids):
            lab, jp = meta[iid]
            ys.append(lab); ss.append(float(fp[i]))
            if lab != 1: continue
            gt = boxes1000(jp)
            if not gt: continue
            pb = cc_boxes_1000(prob[i], thr, min_px, S)
            r = region_iou(pb, gt)
            if r is not None: region.append(r)
            matched.append(instance_matched_iou(pb, gt))
            best += best_match_iou_list(pb, gt)
    head.train()
    return {"region_IoU": float(np.mean(region)) if region else 0.0,
            "instance_matched_IoU": float(np.mean(matched)) if matched else 0.0,
            "best_match_IoU": float(np.mean(best)) if best else 0.0,
            "cls_auc": float(roc_auc_score(ys, ss)) if len(set(ys)) > 1 else 0.0,
            "n_fake": len(region)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["v2", "v3"], required=True)
    p.add_argument("--backbone-ckpt", type=str, default="")  # adapted backbone checkpoint
    p.add_argument("--decoder-type", default="gatedres", choices=["gatedres","res"])
    p.add_argument("--res-fe-channel", default="raw", choices=["raw","evidence"])
    p.add_argument("--out-size", type=int, choices=[112, 224, 448], required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--lambda-cls", type=float, default=1.0)
    p.add_argument("--eval-fakes", type=int, default=2000)
    p.add_argument("--eval-reals", type=int, default=2000)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--limit-steps", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260527); np.random.seed(20260527); random.seed(20260527)
    dev = torch.device("cuda"); torch.backends.cuda.matmul.allow_tf32 = True
    S = args.out_size
    grid = 32 if args.backbone == "v2" else 28
    min_px = max(8, int(round(125 * (S / 224) ** 2)))

    tr = load_csv("train"); va_full = load_csv("val")
    rng = random.Random(20260611)
    vf = [r for r in va_full if r.label == 1]; rng.shuffle(vf)
    vr = [r for r in va_full if r.label == 0]; rng.shuffle(vr)
    va = vf[: args.eval_fakes] + vr[: args.eval_reals]
    meta = {r.image_id: (r.label, r.json_path) for r in va_full + tr}
    gt_cache = {r.image_id: boxes1000(r.json_path) for r in tr}
    print(f"[raw_{args.backbone}_{S}] grid={grid} min_px={min_px} train={len(tr)} "
          f"eval_subset={len(va)}", flush=True)
    tl = DataLoader(DDLXDataset(tr, jpeg="train"), batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=collate, drop_last=True,
                    persistent_workers=True)
    vl = DataLoader(DDLXDataset(va, jpeg="test"), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate, persistent_workers=True)

    vfm = FrozenBackbone(args.backbone, dev, backbone_ckpt=args.backbone_ckpt or None)
    head = TRACEHead(dim=1024, scales=(grid,), out_size=S, cls_dim=1024,
                     decoder_type=args.decoder_type).to(dev)
    if args.decoder_type == "res":
        head.res_fe_channel = args.res_fe_channel   # 'raw' = senior mainline res_fe
    head.decoder.dynamic_upsample = True       # progressive from native grid, no hardcoded 32
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(tl)
    def lr_at(step):
        if step < args.warmup: return step / max(1, args.warmup)
        t = (step - args.warmup) / max(1, total - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    print(f"[raw_{args.backbone}_{S}] head {sum(q.numel() for q in head.parameters())/1e6:.2f}M "
          f"trainable; backbone frozen", flush=True)
    (args.output_dir / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        | {"grid": grid, "min_px": min_px, "timm": TIMM_NAME[args.backbone]}, indent=2))

    ZERO = np.zeros((S, S), np.float32)
    hist, best = [], {"region_IoU": -1.}
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); run = rm = rc = 0.; n = 0; head.train()
        for pix, ids in tl:
            pix = pix.to(dev, non_blocking=True)
            gts = [gt_cache.get(i, []) for i in ids]
            gm = torch.from_numpy(np.stack([raster_gt(g, S) if g else ZERO for g in gts])).to(dev)
            yl = torch.tensor([1 if g else 0 for g in gts], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zt, zv, cls = vfm(pix)
                out = head(zt, zv, cls)
                Lm = dice_bce(out["mask_logit"].squeeze(1).float(), gm)
                Lc = F.cross_entropy(out["image_logit"].float(), yl)
                loss = Lm + args.lambda_cls * Lc
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            run += float(loss.detach()); rm += float(Lm.detach()); rc += float(Lc.detach()); n += 1
            if n % args.log_every == 0:
                print(f"[raw_{args.backbone}_{S}]  ep{ep} step{n}/{len(tl)} loss={run/n:.4f} "
                      f"(Lm={rm/n:.4f} Lcls={rc/n:.4f}) lr={sched.get_last_lr()[0]:.2e}", flush=True)
            if args.limit_steps and n >= args.limit_steps: break
        md = evaluate(vfm, head, vl, meta, dev, S, args.thr, min_px, limit=args.limit_val)
        row = {"epoch": ep, "loss": run/max(n,1), "Lm": rm/max(n,1), "Lcls": rc/max(n,1),
               "sec": time.time()-t0, **md}
        hist.append(row); (args.output_dir/"history.json").write_text(json.dumps(hist, indent=2))
        print(f"[raw_{args.backbone}_{S}] ep{ep} ** region_IoU={md['region_IoU']:.4f} ** "
              f"(instance_matched_IoU={md['instance_matched_IoU']:.4f} "
              f"best_match_IoU={md['best_match_IoU']:.4f} cls_auc={md['cls_auc']:.4f}) "
              f"n_fake={md['n_fake']} sec={row['sec']:.0f}", flush=True)
        if md["region_IoU"] > best["region_IoU"]:
            best = {"epoch": ep, **md}
            torch.save({"head": head.state_dict(), "args": vars(args), "metrics": md},
                       args.output_dir/"head_best.pt")
            (args.output_dir/"best.json").write_text(json.dumps(best, indent=2))
    print(f"\n[raw_{args.backbone}_{S}] BEST ep={best['epoch']} region_IoU={best['region_IoU']:.4f} "
          f"instance_matched_IoU={best['instance_matched_IoU']:.4f} "
          f"best_match_IoU={best['best_match_IoU']:.4f}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
