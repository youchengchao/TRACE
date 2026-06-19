#!/usr/bin/env python3
"""Idempotent environment fixes required to serve Qwen3-VL-8B with vLLM 0.11 + transformers 5.9.

Two known incompatibilities (both reproduced on this stack), with the fixes B14 used:
  1. vLLM's get_cached_tokenizer reads `tokenizer.all_special_tokens_extended`, which the
     transformers-5.9 Qwen2Tokenizer no longer exposes -> AttributeError at engine init.
     Fix: getattr fallback to `tokenizer.all_special_tokens`.
  2. Qwen3VLTextConfig lacks `tie_word_embeddings` -> vLLM crashes loading weights.
     Fix: set "tie_word_embeddings": false in the model config.json (root + text_config).

Run once before launching the server (serve_qwen_vllm.sh calls it automatically). Safe to re-run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import trace_config as C


def patch_vllm_tokenizer() -> bool:
    import vllm
    tk = Path(vllm.__file__).parent / "transformers_utils" / "tokenizer.py"
    src = tk.read_text()
    old = "tokenizer_all_special_tokens_extended = (\n        tokenizer.all_special_tokens_extended)"
    new = ("tokenizer_all_special_tokens_extended = (\n        getattr(tokenizer, "
           "\"all_special_tokens_extended\", tokenizer.all_special_tokens))")
    if "getattr(tokenizer, \"all_special_tokens_extended\"" in src:
        print(f"[patch] vllm tokenizer already patched: {tk}")
        return False
    if old not in src:
        print(f"[patch] WARN expected snippet not found in {tk} — vLLM version differs, check manually")
        return False
    tk.write_text(src.replace(old, new))
    print(f"[patch] patched vllm tokenizer getattr-fallback: {tk}")
    return True


def patch_model_config() -> bool:
    mp = Path(C.resolve_qwen_path())
    cfg = mp / "config.json"
    if not cfg.exists():
        print(f"[patch] model not local yet ({mp}); skip config patch (apply after download)")
        return False
    data = json.loads(cfg.read_text())
    changed = False
    if data.get("tie_word_embeddings") is not False:
        data["tie_word_embeddings"] = False; changed = True
    if isinstance(data.get("text_config"), dict) and data["text_config"].get("tie_word_embeddings") is not False:
        data["text_config"]["tie_word_embeddings"] = False; changed = True
    if changed:
        cfg.write_text(json.dumps(data, indent=2))
        print(f"[patch] set tie_word_embeddings=false in {cfg}")
    else:
        print(f"[patch] model config already has tie_word_embeddings=false: {cfg}")
    return changed


if __name__ == "__main__":
    patch_vllm_tokenizer()
    patch_model_config()
    print("[patch] done")
