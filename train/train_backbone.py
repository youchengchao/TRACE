#!/usr/bin/env python3
"""Stage 1 of training: adapt the DINOv2-Large backbone, producing ckpt/backbone_best.pt.

This is the *first* of TRACE's two training stages (stage 2 is train/train_trace.py). Here we adapt
a frozen DINOv2-Large image encoder to the deepfake domain with a **classification-only** objective —
no localization yet. The backbone stays frozen except its LayerNorm affine parameters; a linear head
predicts real/fake from the CLS embedding. Output: {backbone, cls_head} saved as backbone_best.pt.

Objective = cross-entropy + two regularizers on the L2-normalized CLS embedding:
  - uniformity : spreads embeddings over the unit sphere (avoids collapse),
  - alignment  : pulls same-label embeddings together.
(These two terms follow the "uniformity/alignment" representation-learning objective.)

Two phases: (1) warm up the linear head with the backbone fully frozen, then (2) also unfreeze the
backbone's LayerNorm affine parameters. Early-stops on validation AUC.

Run:
  uv run python train/train_backbone.py --output-dir logs/backbone
The output logs/backbone/backbone_best.pt is the file used as ckpt/backbone_best.pt by train_trace.py.
"""
from __future__ import annotations

import argparse, csv, json, math, sys, time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True   # a few DDL-X jpgs are truncated; pad rather than crash

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trace_config import SPLIT_CSV   # train/val/test split CSV path (from config.yaml)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
INPUT = 448


def alignment(embeddings, labels, alpha: float = 2):
    """Pull same-label normalized embeddings together (mean pairwise distance^alpha within a class)."""
    n_samples = embeddings.size(0)
    if n_samples < 2:
        return torch.tensor(0.0, device=embeddings.device)
    labels_equal_mask = (labels[:, None] == labels[None, :]).triu(diagonal=1)
    positive_indices = torch.nonzero(labels_equal_mask, as_tuple=False)
    if positive_indices.numel() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    x = embeddings[positive_indices[:, 0]]
    y = embeddings[positive_indices[:, 1]]
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(x, t: float = 2, clip_value: float = 1e-6):
    """Spread embeddings over the unit sphere (log of the mean Gaussian-potential over all pairs)."""
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().clamp(min=clip_value).log()


class DDLXClsDataset(Dataset):
    """Read the split CSV -> (image @448, label). Classification-only: no mask/box loading."""
    def __init__(self, split, training):
        self.training = training
        self.recs = []
        for r in csv.DictReader(open(SPLIT_CSV)):
            if r["split"] != split:
                continue
            self.recs.append((Path(r["image_path"]), int(r["label"])))

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        ip, lab = self.recs[i]
        im = Image.open(ip).convert("RGB")
        if self.training and np.random.rand() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im448 = im.resize((INPUT, INPUT), Image.Resampling.BILINEAR)
        pix = (torch.from_numpy(np.asarray(im448, np.float32) / 255.).permute(2, 0, 1) - MEAN) / STD
        return pix, lab


def collate(b):
    return torch.stack([x[0] for x in b]), torch.tensor([x[1] for x in b])


class BackboneClassifier(nn.Module):
    """Frozen DINOv2-L/14 (LayerNorm affine trainable) + a linear real/fake head on the CLS embedding."""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m",
                                          pretrained=True, dynamic_img_size=True)
        self.dim = int(self.backbone.num_features)
        self.cls_head = nn.Linear(self.dim, 2)

    def forward(self, x):
        feats = self.backbone.forward_features(x)            # (B, N, C)
        cls = feats[:, 0]                                    # CLS embedding
        logits = self.cls_head(cls)
        l2 = F.normalize(cls, p=2, dim=1)
        return logits, l2


def set_stage(model, stage):
    """stage 1: only the head trains. stage 2: also the backbone's LayerNorm affine params."""
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    if stage == 2:
        for n, p in model.backbone.named_parameters():
            if (".norm1." in n) or (".norm2." in n) or n.endswith("norm.weight") or n.endswith("norm.bias"):
                p.requires_grad_(True)
    for p in model.cls_head.parameters():
        p.requires_grad_(True)


@torch.no_grad()
def evaluate(model, vl, dev, limit=0):
    model.eval()
    iy, isc = [], []
    for bi, (pix, lab) in enumerate(vl):
        if limit and bi >= limit:
            break
        pix = pix.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(pix)
        isc.extend(logits.float().softmax(1)[:, 1].cpu().numpy().tolist())
        iy.extend(lab.numpy().tolist())
    model.train()
    return {"image_auc": float(roc_auc_score(iy, isc)),
            "image_ap": float(average_precision_score(iy, isc)), "n": len(iy)}


def plot_curves(hist, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped: {e}", flush=True); return
    xs = list(range(1, len(hist) + 1))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(xs, [h["loss"] for h in hist], "b-o", label="train loss")
    ax1.set_xlabel("epoch (concat stages)"); ax1.set_ylabel("train loss", color="b")
    ax2 = ax1.twinx()
    ax2.plot(xs, [h["image_auc"] for h in hist], "r-s", label="val image_auc")
    ax2.plot(xs, [h["image_ap"] for h in hist], "g-^", label="val image_ap")
    ax2.set_ylabel("val metric", color="r")
    fig.legend(loc="upper center", ncol=3); fig.tight_layout()
    fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"[plot] -> {out_png}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--stage1-epochs", type=int, default=1)
    p.add_argument("--stage2-epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-head", type=float, default=4e-4)
    p.add_argument("--lr-ln", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--ce", type=float, default=1.0)
    p.add_argument("--uniformity", type=float, default=0.5)
    p.add_argument("--alignment", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--limit-steps", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)   # eval on first N val batches (0 = all)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260527); np.random.seed(20260527)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    tl = DataLoader(DDLXClsDataset("train", True), batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=collate, drop_last=True, persistent_workers=True)
    vl = DataLoader(DDLXClsDataset("val", False), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate, persistent_workers=True)
    model = BackboneClassifier().to(dev)
    (args.output_dir / "config.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        | {"backbone": "DINOv2-L/14 (lvd142m)", "input": INPUT, "split_csv": str(SPLIT_CSV),
           "task": "classification-only", "loss": "ce + uniformity + alignment"}, indent=2))
    print(f"[train-backbone] train={len(tl.dataset)} val={len(vl.dataset)} "
          f"ce={args.ce} unif={args.uniformity} align={args.alignment}", flush=True)

    def run_stage(stage, epochs, hist, best):
        set_stage(model, stage)
        if stage == 2:
            ln = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
            heads = list(model.cls_head.parameters())
            opt = torch.optim.AdamW([{"params": ln, "lr": args.lr_ln},
                                     {"params": heads, "lr": args.lr_head}], weight_decay=args.wd)
        else:
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=args.lr_head, weight_decay=args.wd)
        total = epochs * len(tl)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5*(1+math.cos(math.pi*min(s,total)/max(1,total))))
        for ep in range(1, epochs + 1):
            t0 = time.time(); run = rc = ra = 0.; n = 0
            for pix, lab in tl:
                pix = pix.to(dev, non_blocking=True); y = lab.to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, l2 = model(pix)
                    Lce = F.cross_entropy(logits.float(), y)
                    Lun = uniformity(l2.float())
                    Lal = alignment(l2.float(), y)
                    loss = args.ce*Lce + args.uniformity*Lun + args.alignment*Lal
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
                run += float(loss.detach()); rc += float(Lce.detach()); ra += float(Lal.detach()); n += 1
                if n % args.log_every == 0:
                    print(f"[s{stage}] ep{ep} step{n}/{len(tl)} loss={run/n:.4f} (ce={rc/n:.4f} "
                          f"align={ra/n:.4f}) lr={sched.get_last_lr()[-1]:.2e}", flush=True)
                if args.limit_steps and n >= args.limit_steps: break
            md = evaluate(model, vl, dev, limit=args.limit_val)
            row = {"stage": stage, "epoch": ep, "loss": run/max(n,1), "sec": time.time()-t0, **md}
            hist.append(row); (args.output_dir/"history.json").write_text(json.dumps(hist, indent=2))
            plot_curves(hist, args.output_dir/"curves.png")
            print(f"[s{stage}] ep{ep} ** img_auc={md['image_auc']:.4f} img_ap={md['image_ap']:.4f} **"
                  f" sec={row['sec']:.0f}", flush=True)
            score = md["image_auc"]
            if score > best["score"]:
                best.update(score=score, **row)
                torch.save({"backbone": model.backbone.state_dict(), "cls_head": model.cls_head.state_dict(),
                            "metrics": md, "args": vars(args)}, args.output_dir/"backbone_best.pt")
                (args.output_dir/"best.json").write_text(json.dumps(best, indent=2))
                best["no_improve"] = 0
            else:
                best["no_improve"] = best.get("no_improve", 0) + 1
                if stage == 2 and best["no_improve"] >= args.patience:
                    print(f"[s{stage}] early stop @ep{ep}", flush=True); return True
        return False

    hist, best = [], {"score": -1.}
    run_stage(1, args.stage1_epochs, hist, best)
    best["no_improve"] = 0
    run_stage(2, args.stage2_epochs, hist, best)
    print(f"\n[train-backbone] BEST img_auc={best.get('image_auc')} img_ap={best.get('image_ap')} "
          f"-> {args.output_dir}/backbone_best.pt", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
