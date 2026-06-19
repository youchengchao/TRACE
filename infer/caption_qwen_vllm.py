#!/usr/bin/env python3
"""TRACE inference — MODE B add-on: forensic captions via Qwen3-VL-8B-Instruct (served by vLLM).

This is the API CLIENT. It talks to a vLLM OpenAI-compatible server (start it first with
infer/serve_qwen_vllm.sh) over HTTP, so the client itself needs no torch/vllm — just PIL + stdlib.

Consumes the per-image localization JSONs from predict_loc.py and fills "Visible forgery traces":
  pass 1 (all images) : neutral general description
  pass 2 (fake images): bulleted anomaly cues inside the predicted boxes
  assembly            : general (+ cues) + the static summary sentence

Caches general/<id>.txt and fake_cues/<id>.txt under --cache-dir for resume. Concurrency via a
thread pool; optional sharding (--num-shards/--shard-id) to drive several servers in parallel.

Example (after serve_qwen_vllm.sh is up on :8000):
  uv run python infer/caption_qwen_vllm.py --images <IMG_DIR> --json-dir out/json \
      --cache-dir out/caption_cache --base-url http://localhost:8000/v1
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import trace_config as C

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

SYSTEM_PROMPT = (
    "You are a forensic image analysis expert. Your job is to analyze the provided image "
    "for digital manipulation, forgery, editing, or AI generation."
)
PROMPT_GENERAL = (
    "Please provide a detailed, neutral general description of this image. "
    "Describe the objects, the scene, layout, colors, and background. "
    "Do NOT mention or look for digital manipulation, anomalies, or forgery. "
    "Output ONLY a single, comprehensive paragraph describing the visible content."
)
PROMPT_ANOMALIES = (
    "Prior general description of this image:\n{general_description}\n\n"
    "Prior model analysis details for this image:\n"
    "- Predicted Status: FAKE\n"
    "- Predicted Tampered Bounding Boxes: {bboxes}\n\n"
    "Inspect the image, focusing strictly on the regions inside the specified bounding boxes "
    "(which are in 1000x1000 normalized coordinates, [x_min, y_min, x_max, y_max]).\n"
    "Generate ONLY a bulleted list of 3-5 visual anomalies observed in those areas. Do not describe "
    "the general scene. Do not include markdown backticks or any introductory or concluding text.\n\n"
    "Required Output Format:\n"
    "- **[Anomaly Category 1]**: [Detailed description of the anomaly in the specified region]\n"
    "- **[Anomaly Category 2]**: [Description...]\n"
    "...\n"
    "- **[Anomaly Category N]**: [Description...]"
)
REAL_TAIL = ("No signs of manipulation are present. The lighting, edges, and texture are consistent "
             "and show no signs of digital modification.\n\nSummary: This image has not been tampered with.")
FAKE_SUMMARY = "Summary: This image has been tampered with."


def encode_image(path: Path, resize_to: int) -> str:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if max(w, h) > resize_to:
        s = resize_to / max(w, h)
        im = im.resize((int(w * s), int(h * s)), Image.Resampling.BILINEAR)
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_vlm(base_url, api_key, model, image_b64, user_prompt, max_tokens, retries=4, timeout=180):
    url = f"{base_url.rstrip('/')}/chat/completions"
    body = json.dumps({
        "model": model, "temperature": 0.0, "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": user_prompt}]}]}).encode()
    hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers=hdr, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read().decode())
            return out["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            last = e; time.sleep(2 * (k + 1))
    raise RuntimeError(f"API failed after {retries} retries: {last}")


def find_image(image_dir: Path, image_id: str):
    for ext in IMG_EXT:
        p = image_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    hits = list(image_dir.rglob(f"{image_id}.*"))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--json-dir", required=True, type=Path, help="predict_loc.py output (updated in place)")
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--model", default=None, help="served model name (default = resolved Qwen path)")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    cap = C.CAPTION
    model = a.model or C.resolve_qwen_path()
    resize_to, max_tokens = cap["resize_to"], cap["max_new_tokens"]
    gen_dir = a.cache_dir / "general"; gen_dir.mkdir(parents=True, exist_ok=True)
    cue_dir = a.cache_dir / "fake_cues"; cue_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(a.json_dir.glob("*.json"))
    json_paths = [p for i, p in enumerate(json_paths) if i % a.num_shards == a.shard_id]
    if a.limit:
        json_paths = json_paths[:a.limit]
    items = []
    for jp in json_paths:
        img = find_image(a.images, jp.stem)
        if img is None:
            print(f"[caption] WARN no image for {jp.stem}", flush=True); continue
        data = json.loads(jp.read_text())
        items.append({"id": jp.stem, "img": img, "json": jp, "data": data,
                      "fake": data.get("Classification result", "real").lower() == "fake"})
    print(f"[caption] shard {a.shard_id}/{a.num_shards}: {len(items)} items "
          f"({sum(x['fake'] for x in items)} fake) -> {a.base_url} model={model}", flush=True)

    def gen_one(it):
        out = gen_dir / f"{it['id']}.txt"
        if out.exists():
            return
        txt = call_vlm(a.base_url, a.api_key, model, encode_image(it["img"], resize_to),
                       PROMPT_GENERAL, max_tokens)
        out.write_text(txt, encoding="utf-8")

    def cue_one(it):
        out = cue_dir / f"{it['id']}.txt"
        if out.exists():
            return
        gen = (gen_dir / f"{it['id']}.txt").read_text(encoding="utf-8").strip()
        prompt = PROMPT_ANOMALIES.format(general_description=gen,
                                         bboxes=it["data"].get("Bounding boxes", "None"))
        txt = call_vlm(a.base_url, a.api_key, model, encode_image(it["img"], resize_to), prompt, max_tokens)
        out.write_text(txt, encoding="utf-8")

    def run_pool(fn, work, tag):
        todo = [it for it in work
                if not ((gen_dir if tag == "general" else cue_dir) / f"{it['id']}.txt").exists()]
        print(f"[caption] pass {tag}: {len(todo)} to generate", flush=True)
        t0, done = time.time(), 0
        with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            futs = {ex.submit(fn, it): it for it in todo}
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001
                    print(f"[caption] ERR {futs[f]['id']}: {e}", flush=True)
                done += 1
                if done % 50 == 0 or done == len(todo):
                    print(f"[caption] {tag} {done}/{len(todo)} ({done/max(time.time()-t0,1e-9):.2f} img/s)",
                          flush=True)

    run_pool(gen_one, items, "general")
    run_pool(cue_one, [it for it in items if it["fake"]], "fake_cues")

    n = 0
    for it in items:
        gen = (gen_dir / f"{it['id']}.txt").read_text(encoding="utf-8").strip()
        if it["fake"]:
            cues = (cue_dir / f"{it['id']}.txt").read_text(encoding="utf-8").strip()
            caption = f"{gen}\n\n{cues}\n\n{FAKE_SUMMARY}"
        else:
            caption = f"{gen}\n\n{REAL_TAIL}"
        it["data"]["Visible forgery traces"] = caption
        it["json"].write_text(json.dumps(it["data"], indent=2, ensure_ascii=False), encoding="utf-8")
        n += 1
    print(f"[caption] DONE assembled captions into {n} JSONs at {a.json_dir}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
