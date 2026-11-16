from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected object")
                    rows.append(row)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


def photo_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"Invalid photo_id: {value!r}")
    value = str(value)
    if not value.isascii() or not value.isdigit() or int(value) <= 0:
        raise ValueError(f"Expected positive numeric photo_id: {value!r}")
    return str(int(value))


def indexed(rows):
    result = {}
    for row in rows:
        key = photo_id(row["photo_id"])
        if key in result:
            raise ValueError(f"Duplicate photo_id: {key}")
        result[key] = dict(row, photo_id=key)
    return result


def boxes_valid(boxes):
    if not isinstance(boxes, list):
        raise ValueError("boxes must be a list")
    for box in boxes:
        if not isinstance(box, dict):
            raise ValueError("box must be an object")
        values = [box[k] for k in ("left", "top", "width", "height")]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError(f"Non-finite or non-numeric box: {box}")
        left, top, width, height = values
        if min(left, top) < 0 or min(width, height) <= 0 or left + width > 1.000001 or top + height > 1.000001:
            raise ValueError(f"Box outside normalized image or empty: {box}")
        if "score" in box and (isinstance(box["score"], bool) or not isinstance(box["score"], (int, float))
                               or not math.isfinite(box["score"]) or not 0 <= box["score"] <= 1):
            raise ValueError("Invalid confidence score")
    return boxes


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
