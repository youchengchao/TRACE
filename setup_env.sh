#!/bin/bash
# Build the TRACE environment.
#
#   ./setup_env.sh            reproduce (default): exact locked env from uv.lock
#                             (torch 2.8.0+cu128 — the verified build).
#   ./setup_env.sh auto       same pinned versions, but auto-match torch's CUDA tag to THIS host.
#
# Why two modes: uv.lock pins torch 2.8.0+cu128 for exact reproducibility. On a machine whose CUDA
# differs (e.g. an older driver where the +cu128 wheel won't run), 'auto' keeps the pinned torch
# VERSION (2.8.x is required by vLLM 0.11) and only swaps the +cuXXX build — uv reads the host CUDA
# via nvidia-smi and picks the matching wheel (cu126/cu128/cu129/...). Everything else stays exactly
# locked, so vLLM/transformers/fastapi are untouched.
set -e
cd "$(dirname "$0")"
MODE="${1:-reproduce}"

# keep in sync with uv.lock / pyproject (vLLM 0.11 requires torch 2.8.x)
TORCH_VER="2.8.0"
TV_VER="0.23.0"

echo "### [1/2] uv sync --frozen  (install the exact locked env)"
uv sync --frozen

if [ "$MODE" = "auto" ]; then
  echo "### [2/2] auto: matching torch's CUDA tag to this host (keeping torch==$TORCH_VER)"
  echo -n "###   host driver: "; nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "(no nvidia-smi -> CPU build)"
  uv pip install "torch==$TORCH_VER" "torchvision==$TV_VER" --torch-backend=auto --reinstall
else
  echo "### [2/2] reproduce: keeping the locked torch ($TORCH_VER+cu128)"
fi

uv run python -c "import torch; print('### torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
echo "### env ready"
