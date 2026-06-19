#!/bin/bash
# Launch the vLLM OpenAI-compatible server for Qwen3-VL-8B (the caption backend).
# Applies the required env patches first, then serves. Leave it running; drive it with
# infer/caption_qwen_vllm.py. Ctrl-C to stop.
#
# Usage: bash infer/serve_qwen_vllm.sh [GPU_ID] [PORT]
set -e
GPU="${1:-0}"
PORT="${2:-8000}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_FROZEN=1   # uv run uses uv.lock exactly (never re-resolve to newer versions)

# resolve the Qwen weights (existing project cache, else the HF/modelscope id for auto-download)
MODEL="$(cd "$ROOT" && uv run python -c 'import trace_config as C; print(C.resolve_qwen_path())')"
MAXLEN="$(cd "$ROOT" && uv run python -c 'import trace_config as C; print(C.CAPTION["max_model_len"])')"
GMU="$(cd "$ROOT" && uv run python -c 'import trace_config as C; print(C.CAPTION["gpu_memory_utilization"])')"
MAXSEQ="$(cd "$ROOT" && uv run python -c 'import trace_config as C; print(C.CAPTION["max_num_seqs"])')"
echo "### Qwen model -> $MODEL (max_model_len=$MAXLEN gpu_mem_util=$GMU max_num_seqs=$MAXSEQ)"

echo "### applying vLLM/Qwen env patches"
(cd "$ROOT" && uv run python infer/patch_vllm_for_qwen.py)

echo "### starting vLLM OpenAI server on GPU $GPU port $PORT (max-model-len 4096, eager)"
cd "$ROOT"
# --enforce-eager: skip CUDA-graph capture (its warmup is the memory spike that OOMs on a shared
# GPU) and the slow torch.compile. Slightly slower decode, far more robust under memory contention.
CUDA_VISIBLE_DEVICES=$GPU exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --max-model-len "$MAXLEN" \
    --gpu-memory-utilization "$GMU" \
    --max-num-seqs "$MAXSEQ" \
    --enforce-eager \
    --trust-remote-code
