#!/usr/bin/env python3
"""Train TRACE, our deepfake region-localization model, on the DDL-X dataset.

TRACE has three parts:
  1. Backbone: a DINOv2-Large Vision Transformer (frozen, pretrained image encoder). Its base
     weights are NEVER updated. We adapt it cheaply with LoRA (Low-Rank Adaptation): small trainable
     low-rank matrices injected into the attention query/key/value projections of transformer
     blocks 0-17 (rank 8). The LoRA matrices are initialized to zero, so the adapted backbone starts
     identical to the frozen one and is then fine-tuned by the localization loss.
  2. Decode head (class TRACEHead): a "Token Reference Attention" head that turns the backbone
     features into (a) a per-pixel forgery probability map and (b) an image-level real/fake logit.
  3. Loss: pixel-level mask loss (Dice + binary cross-entropy) for localization + cross-entropy for
     the real/fake classification. Backbone-LoRA and head are trained together, end to end.

Why LoRA only on blocks 0-17 (not the full network): the decode head reads intermediate features
from blocks 11 and 17, so adapting later blocks cannot change what the head sees. Restricting LoRA
to the query/key/value projections of blocks 0-17 adapts only "where attention looks" at a tiny
cost (~0.59M trainable parameters).

Reference point: the same head on a fully-frozen backbone reaches region-overlap IoU 0.7794;
this LoRA-adapted version is the improvement we keep.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

CODE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CODE_ROOT))
from models.trace import TRACEHead, inject_lora
from core.data_metrics import (DDLXDataset, collate, raster_gt, load_csv, boxes1000,
                                       dice_bce, evaluate, TIMM_NAME)


def parse_blocks(s):
    if "-" in s:
        a, b = s.split("-"); return range(int(a), int(b) + 1)
    return [int(x) for x in s.split(",") if x != ""]


class TRACEBackbone(nn.Module):
    """raw DINOv2/v3 + optional adapted-backbone checkpoint, base FROZEN, trainable LoRA injected.
    No @torch.no_grad on forward -> LoRA adapters receive gradients (base still frozen)."""
    def __init__(self, kind, dev, backbone_ckpt=None, lora_rank=8, lora_targets=("qkv",),
                 lora_blocks=range(0, 18), lora_alpha=16):
        super().__init__()
        self.backbone = timm.create_model(TIMM_NAME[kind], pretrained=True, num_classes=0,
                                          dynamic_img_size=True).to(dev)
        if backbone_ckpt:
            bb = torch.load(backbone_ckpt, map_location="cpu", weights_only=False)["backbone"]
            info = self.backbone.load_state_dict({k: v.to(dev) for k, v in bb.items()}, strict=False)
            print(f"[TRACEBackbone] loaded adapted backbone (missing={len(info.missing_keys)} "
                  f"unexpected={len(info.unexpected_keys)})", flush=True)
        for p in self.backbone.parameters():
            p.requires_grad_(False)                       # freeze base FIRST
        n = (inject_lora(self.backbone, lora_rank, alpha=lora_alpha,
                         targets=tuple(lora_targets), block_ids=lora_blocks)  # then add trainable LoRA
             if lora_rank > 0 else 0)                      # lora_rank<=0 -> fully frozen (eval frozen baseline)
        self.backbone = self.backbone.to(dev)
        ntrain = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"[TRACEBackbone] injected LoRA r={lora_rank} targets={lora_targets} "
              f"blocks={list(lora_blocks)} on {n} linears -> {ntrain/1e6:.2f}M trainable LoRA params",
              flush=True)
        self.prefix = int(getattr(self.backbone, "num_prefix_tokens", 5) or 5)
        self.cap = {}
        for li in (11, 17):
            self.backbone.blocks[li].register_forward_hook(
                lambda _m, _i, out, b=li: self.cap.__setitem__(b, out))

    def forward(self, x):                                  # NO no_grad: LoRA must get gradients
        feats = self.backbone.forward_features(x)
        cls = feats[:, 0]; b = x.shape[0]
        def grid(t):
            tok = t[:, self.prefix:, :]; s = int(round(tok.shape[1] ** 0.5))
            return tok.transpose(1, 2).reshape(b, -1, s, s)
        return grid(self.cap[11]), grid(self.cap[17]), cls


class SimpleDecoder(nn.Module):
    """SINGLE-head decoder: a tapped grid (B,dim,g,g) -> progressive conv upsample -> out mask logit.
    No ReferenceAttention / gate / graph -- just a plain decoder. The minimal localization head."""
    def __init__(self, dim, out_size=224, ch=128):
        super().__init__()
        self.inp = nn.Conv2d(dim, ch, 1)
        self.sizes = [s for s in (56, 112, 224, 448) if s <= out_size]
        if not self.sizes or self.sizes[-1] != out_size:
            self.sizes.append(out_size)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(16, ch), nn.GELU())
            for _ in self.sizes])
        self.head = nn.Conv2d(ch, 1, 1)

    def forward(self, grid):
        x = self.inp(grid)
        for s, blk in zip(self.sizes, self.blocks):
            x = blk(F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False))
        return self.head(x)                          # (B,1,out,out)


class SingleLocHead(nn.Module):
    """Ablation baseline (selected with --head single): LoRA-adapted backbone + a plain decoder
    head, WITHOUT the Token Reference Attention. Same call signature as TRACEHead for evaluation."""
    def __init__(self, dim=1024, out_size=224, cls_dim=1024):
        super().__init__()
        self.decoder = SimpleDecoder(dim, out_size=out_size)
        self.cls_head = nn.Sequential(nn.LayerNorm(cls_dim), nn.Linear(cls_dim, 2))

    def forward(self, zt, zv, cls):                  # zt (L12) ignored: single-tap L18 = zv
        return {"mask_logit": self.decoder(zv), "image_logit": self.cls_head(cls)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["v2", "v3"], default="v2")
    p.add_argument("--backbone-ckpt", type=str, default="")
    p.add_argument("--out-size", type=int, default=224)
    p.add_argument("--head", default="t1", choices=["t1", "single"])   # t1 = full TRACE head (default); single = plain-decoder ablation
    p.add_argument("--decoder-type", default="gatedres", choices=["gatedres", "res"])
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-targets", default="qkv")            # comma list: qkv,fc1,fc2
    p.add_argument("--lora-blocks", default="0-17")            # range a-b or comma list
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)          # learning rate for the decode head
    p.add_argument("--lr-lora", type=float, default=2e-5)     # learning rate for the LoRA parameters
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--lambda-cls", type=float, default=1.0)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--eval-fakes", type=int, default=2000)
    p.add_argument("--eval-reals", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--limit-steps", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260527); np.random.seed(20260527)
    dev = torch.device("cuda"); torch.backends.cuda.matmul.allow_tf32 = True
    S = args.out_size
    grid = 32 if args.backbone == "v2" else 28
    min_px = max(8, int(round(125 * (S / 224) ** 2)))

    tr = load_csv("train"); va_full = load_csv("val")
    import random as _r
    rng = _r.Random(20260611)
    vf = [r for r in va_full if r.label == 1]; rng.shuffle(vf)
    vr = [r for r in va_full if r.label == 0]; rng.shuffle(vr)
    va = vf[: args.eval_fakes] + vr[: args.eval_reals]
    meta = {r.image_id: (r.label, r.json_path) for r in va_full + tr}
    gt_cache = {r.image_id: boxes1000(r.json_path) for r in tr}
    print(f"[train-trace] backbone={args.backbone} grid={grid} S={S} train={len(tr)} eval={len(va)}", flush=True)
    tl = DataLoader(DDLXDataset(tr, jpeg="train"), batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=collate, drop_last=True, persistent_workers=True)
    vl = DataLoader(DDLXDataset(va, jpeg="test"), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate, persistent_workers=True)

    vfm = TRACEBackbone(args.backbone, dev, backbone_ckpt=args.backbone_ckpt or None, lora_rank=args.lora_rank,
                     lora_targets=args.lora_targets.split(","), lora_blocks=parse_blocks(args.lora_blocks),
                     lora_alpha=args.lora_alpha)
    if args.head == "single":
        head = SingleLocHead(dim=1024, out_size=S, cls_dim=1024).to(dev)
    else:
        head = TRACEHead(dim=1024, scales=(grid,), out_size=S, cls_dim=1024,
                         decoder_type=args.decoder_type).to(dev)
        head.decoder.dynamic_upsample = True
    lora_params = [p for p in vfm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": head.parameters(), "lr": args.lr},
                             {"params": lora_params, "lr": args.lr_lora}], weight_decay=1e-4)
    total = args.epochs * len(tl)
    def lr_at(step):
        if step < args.warmup: return step / max(1, args.warmup)
        t = (step - args.warmup) / max(1, total - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    print(f"[train-trace] head {sum(q.numel() for q in head.parameters())/1e6:.2f}M + "
          f"LoRA {sum(q.numel() for q in lora_params)/1e6:.2f}M trainable; base frozen", flush=True)
    (args.output_dir / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        | {"grid": grid, "min_px": min_px, "timm": TIMM_NAME[args.backbone],
           "baseline_frozen_backbone_iou": 0.7794}, indent=2))

    ZERO = np.zeros((S, S), np.float32)
    hist, best = [], {"region_IoU": -1.}
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); run = rm = rc = 0.; n = 0; head.train(); vfm.train()
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
                print(f"[train-trace] ep{ep} step{n}/{len(tl)} loss={run/n:.4f} "
                      f"(Lm={rm/n:.4f} Lcls={rc/n:.4f}) lr={sched.get_last_lr()[0]:.2e}", flush=True)
            if args.limit_steps and n >= args.limit_steps: break
        vfm.eval()
        md = evaluate(vfm, head, vl, meta, dev, S, args.thr, min_px, limit=args.limit_val)
        row = {"epoch": ep, "loss": run/max(n,1), "Lm": rm/max(n,1), "Lcls": rc/max(n,1),
               "sec": time.time()-t0, **md}
        hist.append(row); (args.output_dir/"history.json").write_text(json.dumps(hist, indent=2))
        print(f"[train-trace] epoch {ep} ** region_IoU={md['region_IoU']:.4f} ** "
              f"(matched={md['instance_matched_IoU']:.4f} best={md['best_match_IoU']:.4f} "
              f"cls_auc={md['cls_auc']:.4f}) sec={row['sec']:.0f} (frozen-backbone baseline=0.7794)", flush=True)
        if md["region_IoU"] > best["region_IoU"]:
            best = {"epoch": ep, **md}
            torch.save({"head": head.state_dict(),
                        "lora": {k: v.detach().cpu() for k, v in vfm.state_dict().items()
                                 if any(s in k for s in (".A.", ".B."))},
                        "args": vars(args), "metrics": md}, args.output_dir/"trace_best.pt")
            (args.output_dir/"best.json").write_text(json.dumps(best, indent=2))
    print(f"\n[train-trace] BEST epoch={best['epoch']} region_IoU={best['region_IoU']:.4f} "
          f"(frozen-backbone baseline 0.7794)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
