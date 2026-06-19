#!/bin/bash
# Download the large backbone weight (not stored in git) from Google Drive into ckpt/.
# trace_best.pt is already in the repo; only backbone_best.pt (~1.2 GB) is fetched here.
# Run AFTER setup_env.sh (uses gdown from the environment). Idempotent: skips if already present.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Google Drive file ID — from the file's "Share" link
#   https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
# Make sure it is shared as "Anyone with the link".
BACKBONE_ID="1YLUzBfv0OU-vYMYC9sgb2fqF2V0GAvw6"   # backbone_best.pt (~1.2 GB)

mkdir -p ckpt
if [ -f ckpt/backbone_best.pt ]; then
  echo "already present: ckpt/backbone_best.pt"
else
  echo "downloading backbone_best.pt (~1.2 GB) ..."
  uv run gdown "$BACKBONE_ID" -O ckpt/backbone_best.pt
fi
echo "done -> ckpt/backbone_best.pt"
