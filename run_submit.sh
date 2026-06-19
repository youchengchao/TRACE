#!/bin/bash
# TRACE end-to-end DDL-X submission builder: localization -> caption -> package zip.
#
# Usage (needs only an NVIDIA driver; no CUDA toolkit / conda env required):
#   bash run_submit.sh <IMAGE_DIR> <OUT_DIR> [GPU_ID] [MODE] [PORT]
#     MODE = full (default): localization + Qwen3-VL captions + zip  (valid submission)
#          = loc           : localization only (boxes, empty captions; NOT a valid submission)
#
# 'full' mode auto-starts a vLLM Qwen server, runs the caption client, then shuts the server down.
#
# Example:
#   bash run_submit.sh /path/to/DDL_X/test/images out/test_submit 0 full
set -e
IMG_DIR="${1:?need IMAGE_DIR}"
OUT="${2:?need OUT_DIR}"
GPU="${3:-0}"
MODE="${4:-full}"
PORT="${5:-8000}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JSON_DIR="$OUT/json"
export UV_FROZEN=1   # uv run uses uv.lock exactly (never re-resolve to newer versions)

echo "### [1/3] LOCALIZATION (TRACE) -> $JSON_DIR"
CUDA_VISIBLE_DEVICES=$GPU uv run python "$ROOT/infer/predict_loc.py" \
    --images "$IMG_DIR" --out-dir "$JSON_DIR"

if [ "$MODE" = "full" ]; then
  echo "### [2/3] CAPTIONS (Qwen3-VL-8B via vLLM)"
  echo "###   starting vLLM server (GPU $GPU, port $PORT) ..."
  # own process group (setsid) so we can reliably kill the whole server tree (uv -> python -> EngineCore)
  setsid bash "$ROOT/infer/serve_qwen_vllm.sh" "$GPU" "$PORT" > "$OUT/vllm_server.log" 2>&1 &
  SPGID=$!
  trap 'kill -- -$SPGID 2>/dev/null || true' EXIT
  echo "###   waiting for server /health (pgid $SPGID, up to ~20 min for first-time compile) ..."
  ready=0
  for i in $(seq 1 240); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then ready=1; echo "###   server ready"; break; fi
    sleep 5
  done
  if [ $ready -ne 1 ]; then echo "### server not ready in time — see $OUT/vllm_server.log"; exit 1; fi
  uv run python "$ROOT/infer/caption_qwen_vllm.py" \
      --images "$IMG_DIR" --json-dir "$JSON_DIR" --cache-dir "$OUT/caption_cache" \
      --base-url "http://localhost:$PORT/v1"
  kill -- -$SPGID 2>/dev/null || true; trap - EXIT
  CAP_FLAG="--require-nonempty-caption"
else
  echo "### [2/3] SKIP captions (MODE=loc) — result will NOT pass require-nonempty-caption"
  CAP_FLAG=""
fi

echo "### [3/3] PACKAGE + VALIDATE -> $OUT/submit.zip"
uv run python "$ROOT/submit/package_submit.py" \
    --json-dir "$JSON_DIR" --image-dir "$IMG_DIR" --output-dir "$OUT" \
    --allow-fake-no-box $CAP_FLAG

echo "### DONE -> $OUT/submit.zip"
