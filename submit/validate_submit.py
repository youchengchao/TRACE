#!/usr/bin/env python3
"""Validate DDL-X submission JSON files and summarize output size."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_KEYS = {"Classification result", "Bounding boxes", "Visible forgery traces"}


def validate_file(path: Path) -> tuple[dict, list[dict]]:
    errors = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, [{"file": path.name, "kind": "json", "message": str(exc)}]

    keys = set(data)
    if keys != REQUIRED_KEYS:
        errors.append({"file": path.name, "kind": "keys", "value": sorted(keys)})

    label = data.get("Classification result")
    if label not in {"fake", "real"}:
        errors.append({"file": path.name, "kind": "label", "value": label})

    boxes = data.get("Bounding boxes")
    if boxes == "None":
        return {"label": label, "boxes": 0, "fake_nobox": label == "fake"}, errors
    if not isinstance(boxes, list):
        errors.append({"file": path.name, "kind": "box_container", "value": type(boxes).__name__})
        return {"label": label, "boxes": 0, "fake_nobox": False}, errors

    for box in boxes:
        if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, int) for v in box)):
            errors.append({"file": path.name, "kind": "box_schema", "value": box})
            break
        x1, y1, x2, y2 = box
        if not (1 <= x1 < x2 <= 1000 and 1 <= y1 < y2 <= 1000):
            errors.append({"file": path.name, "kind": "box_range", "value": box})
            break
    return {"label": label, "boxes": len(boxes), "fake_nobox": False}, errors


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("json_dir", type=Path)
    p.add_argument("--expected-count", type=int, default=0)
    p.add_argument("--max-errors", type=int, default=20)
    args = p.parse_args()

    files = sorted(args.json_dir.glob("*.json"))
    errors = []
    fake = real = fake_nobox = total_boxes = max_boxes = total_bytes = 0
    if args.expected_count and len(files) != args.expected_count:
        errors.append({"kind": "file_count", "expected": args.expected_count, "actual": len(files)})

    for path in files:
        total_bytes += path.stat().st_size
        stats, file_errors = validate_file(path)
        errors.extend(file_errors)
        if stats.get("label") == "fake":
            fake += 1
        elif stats.get("label") == "real":
            real += 1
        fake_nobox += int(bool(stats.get("fake_nobox")))
        n_boxes = int(stats.get("boxes", 0))
        total_boxes += n_boxes
        max_boxes = max(max_boxes, n_boxes)

    summary = {
        "json_dir": str(args.json_dir),
        "files": len(files),
        "fake": fake,
        "real": real,
        "fake_nobox": fake_nobox,
        "total_boxes": total_boxes,
        "max_boxes_image": max_boxes,
        "size_mb": total_bytes / (1024 * 1024),
        "errors": len(errors),
        "first_errors": errors[: args.max_errors],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
