# TRACE

Token Reference Attention for auditable deepfake region localization on DDL-X / IJCAI 2026 Track 3.

TRACE predicts:

- real/fake label
- manipulated-region boxes
- visible-forgery description

## Quick Start

```bash
bash setup_env.sh
bash download_ckpt.sh
bash run_submit.sh /path/to/test/images out/test_submit 0 full
```

Output:

```text
out/test_submit/submit.zip
```

Use `loc` instead of `full` to skip Qwen captions:

```bash
bash run_submit.sh /path/to/test/images out/test_submit 0 loc
```

## Contents

| Part | Files |
|---|---|
| GenD-style DINOv2 adaptation | `train/train_backbone.py` |
| TRACE training | `train/train_trace.py`, `models/trace.py` |
| Localization inference | `infer/predict_loc.py` |
| Qwen captioning | `infer/serve_qwen_vllm.sh`, `infer/caption_qwen_vllm.py` |
| Submission zip/validation | `submit/package_submit.py`, `submit/validate_submit.py` |

## Checkpoints

| Path | Git status |
|---|---|
| `ckpt/trace_best.pt` | committed, about 6 MB |
| `ckpt/backbone_best.pt` | downloaded by `download_ckpt.sh`, about 1.2 GB |
| `ckpt/Qwen3-VL-8B-Instruct/` | downloaded on first `full` run if missing |

Large downloaded weights are ignored by git.

## Environment

`setup_env.sh` installs the pinned `uv.lock` environment. Tested with Python 3.12, PyTorch 2.8
CUDA 12.8 wheels, vLLM 0.11.0, and transformers 5.9.0. Only an NVIDIA driver is required.

## Training

Set dataset paths in `config.yaml`, or override them:

```bash
export TRACE_SPLIT_CSV=/path/to/DDL_X/table1_train_val_test_splits.csv
export TRACE_DDLX_ROOT=/path/to/DDL_X/images
```

Train the two stages:

```bash
uv run python train/train_backbone.py --output-dir logs/backbone
cp logs/backbone/backbone_best.pt ckpt/backbone_best.pt

uv run python train/train_trace.py \
  --backbone v2 \
  --backbone-ckpt ckpt/backbone_best.pt \
  --head t1 \
  --decoder-type gatedres \
  --lora-rank 8 \
  --lora-blocks 0-17 \
  --lora-alpha 16 \
  --epochs 12 \
  --batch-size 16 \
  --num-workers 8 \
  --output-dir logs/trace
```

## Qwen Captions

`run_submit.sh ... full` manages the vLLM server automatically. To serve manually:

```bash
bash infer/serve_qwen_vllm.sh 0 8000
```

Qwen3-VL-8B is loaded from `caption.local_candidates`; if missing, it is downloaded to
`caption.download_to` in `config.yaml` (`ckpt/Qwen3-VL-8B-Instruct` by default).

## Results

Held-out DDL-X test split:

| Metric | TRACE | DeCLIP | Dolos |
|---|---:|---:|---:|
| Union region IoU | **0.8255** | 0.7858 | 0.6640 |
| Strict IoU | **0.7466** | 0.7063 | 0.5301 |
| Per-GT-box IoU | **0.7249** | 0.6886 | 0.5394 |
| Detection AUC | **1.000** | 0.8301 | 0.9806 |

## Layout

```text
config.yaml, trace_config.py       configuration
setup_env.sh, download_ckpt.sh     setup
run_submit.sh                      end-to-end submission
train/                             training scripts
models/                            TRACE model modules
infer/                             inference and captioning
submit/                            validation and packaging
ckpt/                              committed and downloaded checkpoints
```
