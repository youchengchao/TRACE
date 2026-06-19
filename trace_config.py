"""Single source of truth for TRACE paths + hyper-parameters.

Loads config.yaml (next to this file) and exposes the constants the training /
inference code imports. Machine-specific absolute paths live ONLY in config.yaml
(or env overrides) — never hardcoded in the code modules.

TRACE: in-domain LN-tuned DINOv2-L backbone + end-to-end LoRA + a Token Reference
Attention decode head (TRACEHead, GatedRes decoder @224). NO instance head / no
graph; forgery boxes come from a connected-component decode of the fakeness map.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def _abs(p: str) -> Path:
    """Resolve relative paths (e.g. ckpt/...) against the package root; keep abs paths."""
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


# ── data ──────────────────────────────────────────────────────────────────────
SPLIT_CSV = os.environ.get("TRACE_SPLIT_CSV", CFG["data"]["split_csv"])
DDLX_ROOT = Path(os.environ.get("TRACE_DDLX_ROOT", CFG["data"]["ddlx_root"]))

# ── checkpoints (relative paths resolve to TRACE/) ───────────────────────────────
CKPT_BACKBONE = _abs(CFG["ckpt"]["backbone"])
CKPT_TRACE = _abs(CFG["ckpt"]["trace"])

# ── model / decode ──────────────────────────────────────────────────────────────
MODEL = CFG["model"]
DECODE = CFG["decode"]
CAPTION = CFG["caption"]


def resolve_qwen_path() -> str:
    """Return a local Qwen weight dir if present; otherwise return the HF model id."""
    for cand in [*CAPTION["local_candidates"], CAPTION["download_to"]]:
        p = _abs(cand)
        if p.exists() and any(p.glob("*.safetensors")):
            return str(p)
    return CAPTION["model_id"]


def qwen_download_dir() -> Path:
    """Directory where first-run Qwen downloads should be stored."""
    return _abs(CAPTION["download_to"])
