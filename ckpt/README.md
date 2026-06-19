# Checkpoints

| file | size | how to get it |
|---|---|---|
| `trace_best.pt` | 6 MB | already in the repo (committed) |
| `backbone_best.pt` | 1.2 GB | `bash download_ckpt.sh` — fetched from Google Drive (too large for git) |

`trace_best.pt` is the trained LoRA + decode head. `backbone_best.pt` is the adapted DINOv2-Large
backbone; it exceeds GitHub's file-size limit, so it is hosted on Google Drive and downloaded by
`download_ckpt.sh` (run it after `bash setup_env.sh`). If Google Drive is blocked on your network,
download it manually and place it here as `ckpt/backbone_best.pt`:

```text
https://drive.google.com/file/d/1YLUzBfv0OU-vYMYC9sgb2fqF2V0GAvw6/view?usp=sharing
```

Both `trace_best.pt` and `backbone_best.pt` are required to run inference.

> Note: the **Qwen3-VL-8B** caption model is not here and not on Google Drive — it is an off-the-shelf
> public model that vLLM downloads automatically from Hugging Face on first run.
