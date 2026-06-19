"""Parse DDL-X bounding-box annotation files.

Each DDL-X annotation is a JSON file with a "Bounding boxes" field that is either the string
"None" (real image / no manipulated region) or a list of boxes [x_min, y_min, x_max, y_max] in
0-1000 normalized coordinates.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple


def parse_ddlx_boxes(json_path: Path) -> Tuple[Optional[None], List[List[float]]]:
    """Read a DDL-X annotation JSON and return (None, boxes), where boxes is a list of
    [x_min, y_min, x_max, y_max] in 0-1000 coordinates (empty list if there are none)."""
    data = json.loads(Path(json_path).read_text())
    raw = data.get("Bounding boxes", "None")
    if raw == "None" or not raw:
        return None, []
    return None, [list(map(float, b)) for b in raw]
