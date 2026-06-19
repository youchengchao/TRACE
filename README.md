# TRACE — Token Reference Attention for Auditable Deepfake Region Localization

DDL-X benchmark submission (IJCAI 2026 Track 3). For each image, output: **is it fake**, **where**
the manipulated regions are (boxes), and a **text description** of the forgery.

## TL;DR

- **Model** — frozen `DINOv2-Large` + **LoRA** + a **Token Reference Attention** head → per-pixel
  forgery map (→ boxes) + real/fake score. Captions from off-the-shelf **Qwen3-VL-8B** (via vLLM).
- **Performance** (held-out test) — union box IoU **0.826**, detection AUC **1.000**; beats DeCLIP
  (0.786) and Dolos (0.664). [Full table](#results).
- **Run** — `bash setup_env.sh` → `bash download_ckpt.sh` → `bash run_submit.sh <images> <out> <gpu> full` → `<out>/submit.zip`.
- **Needs** — one NVIDIA GPU with a recent driver (CUDA 12.x). No CUDA toolkit / conda env required —
  the pinned torch wheel ships its own CUDA runtime.

## Run it

```bash
bash setup_env.sh                                            # build the pinned environment (uv)
bash download_ckpt.sh                                        # fetch backbone_best.pt (~1.2 GB, not in git)
bash run_submit.sh /path/to/test/images out/test_submit 0 full   # localization + captions + zip
#   -> out/test_submit/submit.zip   (mode "loc" = boxes only, skip captions)
```

One JSON per image, in the benchmark's 3-field format (boxes are ints in 0–1000; real → `"None"`):

```json
{ "Classification result": "fake",
  "Bounding boxes": [[263, 362, 741, 817]],
  "Visible forgery traces": "A close-up portrait ... Summary: This image has been tampered with." }
```

<details><summary>Run the stages separately</summary>

```bash
uv run python infer/predict_loc.py --images <IMG> --out-dir out/json                 # boxes
uv run python infer/caption_qwen_vllm.py --images <IMG> --json-dir out/json --cache-dir out/cap  # captions
uv run python submit/package_submit.py --json-dir out/json --image-dir <IMG> --output-dir out \
    --allow-fake-no-box --require-nonempty-caption                                   # validate + zip
```
</details>

## Environment

`setup_env.sh` runs `uv sync --frozen`, installing the **exact pinned versions** from `uv.lock`
(~161 packages; torch 2.8.0+cu128 · vllm 0.11.0 · transformers 5.9.0). Tested on Python 3.12.
Only the **NVIDIA driver** is required — the torch wheel bundles the CUDA 12.8 runtime, so no separate
CUDA toolkit (`nvcc`) is needed. Different host CUDA? `bash setup_env.sh auto` keeps the pinned torch
version and only swaps its CUDA build to match your driver.

## How it works

- **Backbone**: `DINOv2-Large` (timm), **frozen**.
- **LoRA**: trainable low-rank adapters on the attention q/k/v of blocks 0–17 (rank 8, ~0.6M params,
  zero-init). Only LoRA + head are trained, end to end.
- **Token Reference Attention head** (`models/trace.py`): each patch is compared against a reference
  built from the other patches; the **deviation** (what the rest of the image can't explain) drives a
  per-pixel forgery map + a real/fake score.
- **Decode**: threshold the map → connected components → boxes. `fake` if score ≥ `cls_thr`.
- **Captions**: Qwen3-VL-8B describes each image, then lists anomalies inside the predicted boxes.

Decode defaults (`config.yaml`): threshold 0.9, min component 160 px, `cls_thr` 0.11.

## Results

Held-out **test split**. Localization on fake images with ground-truth boxes; detection on all images.
DeCLIP / Dolos are published methods run under the same protocol.

| metric (↑ better) | meaning | **TRACE** | DeCLIP | Dolos |
|---|---|---:|---:|---:|
| union region IoU | all predicted vs. all ground-truth boxes (merged) | **0.8255** | 0.7858 | 0.6640 |
| strict IoU | one-to-one box matching (Hungarian) | **0.7466** | 0.7063 | 0.5301 |
| per-GT-box IoU | best prediction per ground-truth box | **0.7249** | 0.6886 | 0.5394 |
| detection AUC | real-vs-fake over all images | **1.000** | 0.8301 | 0.9806 |

## Layout

```
config.yaml / trace_config.py   # config + loader      run_submit.sh / setup_env.sh   # entry scripts
ckpt/  backbone_best.pt (1.2G)  trace_best.pt (6M)
models/trace.py                 # TRACEHead + LoRA      core/data_metrics.py   # data, metrics, decode
dataset/ddlx_annotations.py     # parse DDL-X annotations
train/  train_backbone.py (1a)  train_trace.py (1b)    # (1) TRAINING — two stages
infer/  predict_loc.py  caption_qwen_vllm.py  serve_qwen_vllm.sh  patch_vllm_for_qwen.py   # (2) inference
submit/ package_submit.py  validate_submit.py           # (3) packaging
```

## Training (reproduce the checkpoints)

Two stages — backbone adaptation, then the localization head:

```bash
# (1a) adapt the DINOv2-Large backbone (classification-only) -> backbone_best.pt
uv run python train/train_backbone.py --output-dir logs/backbone
cp logs/backbone/backbone_best.pt ckpt/backbone_best.pt

# (1b) train the LoRA + Token Reference Attention head on top -> trace_best.pt
uv run python train/train_trace.py --backbone v2 --backbone-ckpt ckpt/backbone_best.pt \
    --head t1 --decoder-type gatedres --lora-rank 8 --lora-blocks 0-17 --lora-alpha 16 \
    --epochs 12 --batch-size 16 --num-workers 8 --output-dir logs/trace
```

Data paths come from `config.yaml → data` (or env `TRACE_SPLIT_CSV` / `TRACE_DDLX_ROOT`). Test split
is held out from training. (To only run inference, skip training and use the shipped checkpoints.)

## vLLM caption server

`run_submit.sh full` manages it for you; to run by hand: `bash infer/serve_qwen_vllm.sh <gpu> <port>`.
Qwen weights auto-resolve from `config.yaml → caption.local_candidates` (else downloaded on first
`full` run to `config.yaml → caption.download_to`, default `ckpt/Qwen3-VL-8B-Instruct`).
`patch_vllm_for_qwen.py` applies two idempotent compatibility fixes the pinned vLLM/transformers need.
Default `--enforce-eager` is robust on a shared GPU; on a dedicated GPU drop it and shard with the
client's `--num-shards` / `--shard-id` for speed.
