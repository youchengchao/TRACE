#!/usr/bin/env python3
"""Validate and zip DDL-X per-image JSON outputs."""
from __future__ import annotations

import argparse
import csv
import json
import time
import zipfile
from pathlib import Path
from typing import Any


REQUIRED_KEYS = ["Classification result", "Bounding boxes", "Visible forgery traces"]


def iter_image_ids(image_dir: Path) -> list[str]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p.stem for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


def valid_bbox_value(label: str, boxes: Any, allow_fake_no_box: bool) -> tuple[bool, int, str]:
    if label == "real":
        return (True, 0, "") if boxes in (None, "None") else (False, 0, "real sample has non-empty boxes")
    if label != "fake":
        return False, 0, "invalid classification label"
    if boxes in (None, "None", []):
        if allow_fake_no_box:
            return True, 0, ""
        return False, 0, "fake sample has no bbox list"
    if not isinstance(boxes, list) or not boxes:
        return False, 0, "fake sample has no bbox list"
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4:
            return False, 0, "bbox is not a length-4 list"
        if any(not isinstance(v, int) or v < 1 or v > 1000 for v in box):
            return False, 0, "bbox coordinate is not an int in 1..1000"
        if box[0] >= box[2] or box[1] >= box[3]:
            return False, 0, "bbox has non-positive area"
    return True, len(boxes), ""


def validate_json_dir(
    json_dir: Path,
    image_dir: Path,
    require_nonempty_caption: bool,
    max_examples: int,
    allow_fake_no_box: bool,
) -> tuple[dict[str, Any], list[dict[str, str]], list[str]]:
    expected_ids = iter_image_ids(image_dir)
    json_ids = sorted(p.stem for p in json_dir.glob("*.json"))
    expected_set = set(expected_ids)
    json_set = set(json_ids)
    missing = sorted(expected_set - json_set)
    extra = sorted(json_set - expected_set)
    invalid: list[dict[str, str]] = []
    fake_count = real_count = total_boxes = empty_caption_count = nonempty_caption_count = fake_no_box_count = 0

    for image_id in expected_ids:
        path = json_dir / f"{image_id}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("top-level JSON is not an object")
        except Exception as exc:  # noqa: BLE001
            invalid.append({"image_id": image_id, "reason": f"invalid JSON: {exc}"})
            continue
        missing_keys = [key for key in REQUIRED_KEYS if key not in data]
        if missing_keys:
            invalid.append({"image_id": image_id, "reason": f"missing keys: {','.join(missing_keys)}"})
            continue
        label = data.get("Classification result")
        fake_count += int(label == "fake")
        real_count += int(label == "real")
        boxes = data.get("Bounding boxes")
        ok, num_boxes, reason = valid_bbox_value(str(label), boxes, allow_fake_no_box)
        if not ok:
            invalid.append({"image_id": image_id, "reason": reason})
        if label == "fake" and boxes in (None, "None", []):
            fake_no_box_count += 1
        total_boxes += num_boxes
        caption = data.get("Visible forgery traces")
        if not isinstance(caption, str):
            invalid.append({"image_id": image_id, "reason": "caption field is not a string"})
        elif caption.strip():
            nonempty_caption_count += 1
        else:
            empty_caption_count += 1
            if require_nonempty_caption:
                invalid.append({"image_id": image_id, "reason": "empty caption"})

    invalid.extend({"image_id": image_id, "reason": "missing JSON"} for image_id in missing[:max_examples])
    invalid.extend({"image_id": image_id, "reason": "extra JSON"} for image_id in extra[:max_examples])
    summary = {
        "status": "valid" if not missing and not extra and not invalid else "invalid",
        "image_dir": str(image_dir),
        "json_dir": str(json_dir),
        "expected_images": len(expected_ids),
        "json_files": len(json_ids),
        "missing_json_count": len(missing),
        "extra_json_count": len(extra),
        "invalid_schema_count": len(invalid),
        "fake_jsons": fake_count,
        "real_jsons": real_count,
        "fake_no_box_jsons": fake_no_box_count,
        "total_boxes": total_boxes,
        "empty_caption_count": empty_caption_count,
        "nonempty_caption_count": nonempty_caption_count,
        "require_nonempty_caption": require_nonempty_caption,
        "allow_fake_no_box": allow_fake_no_box,
        "invalid_examples": invalid[:max_examples],
        "missing_examples": missing[:max_examples],
        "extra_examples": extra[:max_examples],
    }
    return summary, invalid, expected_ids


def write_invalid_cases(path: Path, invalid: list[dict[str, str]]) -> None:
    if not invalid:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "reason"])
        writer.writeheader()
        writer.writerows(invalid)


def zip_submit(json_dir: Path, image_ids: list[str], zip_path: Path, compresslevel: int) -> dict[str, Any]:
    t0 = time.time()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as zf:
        for image_id in image_ids:
            path = json_dir / f"{image_id}.json"
            zf.write(path, arcname=f"json/{path.name}")
    elapsed = time.time() - t0
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_entry = zf.testzip()
        entries = zf.namelist()
    return {
        "zip_path": str(zip_path),
        "num_entries": len(entries),
        "bad_entry": bad_entry,
        "size_bytes": zip_path.stat().st_size,
        "elapsed_seconds": round(elapsed, 3),
        "throughput_images_per_second": round(len(image_ids) / max(elapsed, 1e-9), 3),
        "compresslevel": compresslevel,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json-dir", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--zip-name", default="submit.zip")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--source-run-dir", default="")
    p.add_argument("--require-nonempty-caption", action="store_true")
    p.add_argument("--allow-fake-no-box", action="store_true")
    p.add_argument("--compresslevel", type=int, default=6)
    p.add_argument("--max-examples", type=int, default=20)
    args = p.parse_args()

    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation, invalid, image_ids = validate_json_dir(
        args.json_dir,
        args.image_dir,
        args.require_nonempty_caption,
        args.max_examples,
        args.allow_fake_no_box,
    )
    (args.output_dir / "validation_summary.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
    write_invalid_cases(args.output_dir / "invalid_cases.csv", invalid)
    if validation["status"] != "valid":
        raise SystemExit(json.dumps(validation, indent=2, ensure_ascii=False))
    zip_info = zip_submit(args.json_dir, image_ids, args.output_dir / args.zip_name, args.compresslevel)
    summary = {
        "status": "completed",
        "checkpoint": args.checkpoint,
        "source_run_dir": args.source_run_dir,
        "validation": validation,
        "zip": zip_info,
        "total_elapsed_seconds": round(time.time() - t0, 3),
    }
    (args.output_dir / "package_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
