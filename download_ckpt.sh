#!/bin/bash
# Download the large backbone weight (not stored in git) from Google Drive into ckpt/.
# trace_best.pt is already in the repo; only backbone_best.pt (~1.2 GB) is fetched here.
# Run AFTER setup_env.sh (uses gdown from the environment). Idempotent: skips if already present.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BACKBONE_ID="1YLUzBfv0OU-vYMYC9sgb2fqF2V0GAvw6"
BACKBONE_URL="https://drive.google.com/file/d/${BACKBONE_ID}/view?usp=sharing"
BACKBONE_PATH="ckpt/backbone_best.pt"
TMP_PATH="${BACKBONE_PATH}.tmp"
MIN_BYTES=1000000000

mkdir -p ckpt

check_size() {
  local path="$1"
  local bytes
  bytes="$(wc -c < "$path")"
  if [ "$bytes" -lt "$MIN_BYTES" ]; then
    echo "ERROR: $path is only $bytes bytes; expected about 1.2 GB."
    echo "Remove the partial file and re-run this script, or download manually:"
    echo "$BACKBONE_URL"
    exit 1
  fi
}

if [ -f "$BACKBONE_PATH" ]; then
  check_size "$BACKBONE_PATH"
  echo "already present: $BACKBONE_PATH"
else
  echo "downloading backbone_best.pt (~1.2 GB) ..."
  rm -f "$TMP_PATH"
  uv run gdown "$BACKBONE_URL" -O "$TMP_PATH"
  check_size "$TMP_PATH"
  mv "$TMP_PATH" "$BACKBONE_PATH"
fi
echo "done -> $BACKBONE_PATH"
