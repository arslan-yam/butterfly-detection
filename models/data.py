from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from urllib.request import Request, urlopen

from .io import boxes_valid, file_sha, indexed, read_jsonl, write_json, write_jsonl


def prepare(sources, output, stage, expected_count=None):
    rows = indexed([row for source in sources for row in read_jsonl(source)])
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} photos, found {len(rows)}")
    manifest, labels = [], []
    for key, row in rows.items():
        image = row.get("image")
        if not isinstance(image, str) or not image.strip():
            raise ValueError(f"{key}: missing image URL or local path")
        manifest_row = dict(photo_id=key, image=image)
        if row.get("group"):
            manifest_row["group"] = str(row["group"])
        manifest.append(manifest_row)
        if stage != "unlabeled":
            boxes = boxes_valid(row["boxes"])
            if row.get("status") not in ("butterflies", "no_butterflies"):
                raise ValueError(f"{key}: annotation not resolved: {row.get('status')}")
            if bool(boxes) != (row["status"] == "butterflies"):
                raise ValueError(f"{key}: status/boxes mismatch")
            if row.get("n_boxes", len(boxes)) != len(boxes):
                raise ValueError(f"{key}: n_boxes mismatch")
            labels.append(dict(photo_id=key, boxes=boxes))
    output = Path(output)
    # Never leave old labels next to a new, unlabeled manifest.
    if stage == "unlabeled" and (output / "annotations.jsonl").exists():
        raise ValueError("Use a new output directory for an unlabeled dataset")
    write_jsonl(output / "manifest.jsonl", manifest)
    if labels:
        write_jsonl(output / "annotations.jsonl", labels)
    write_json(output / "dataset.json", dict(
        stage=stage, photos=len(rows), boxes=sum(len(r["boxes"]) for r in labels),
        negative_photos=sum(not r["boxes"] for r in labels),
        sources=[dict(path=str(s), sha256=file_sha(s)) for s in sources],
        manifest_sha256=file_sha(output / "manifest.jsonl"),
        annotations_sha256=file_sha(output / "annotations.jsonl") if labels else None,
        note="Raw bbox before independent review; not a golden set" if stage == "raw" else "",
    ))


def open_image(path):
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        source.load()
        return ImageOps.exif_transpose(source).convert("RGB")


def download(manifest, output, retries=3):
    """Cache original bytes and record their hashes. Failed photos remain explicit."""
    rows = indexed(read_jsonl(manifest))
    output = Path(output)
    cache = output / "images"
    cache.mkdir(parents=True, exist_ok=True)
    old = indexed(read_jsonl(output / "images.jsonl")) if (output / "images.jsonl").exists() else {}
    records = []
    for key, row in rows.items():
        path = cache / (key + ".image")
        previous = old.get(key, {})
        record = dict(photo_id=key, image=row["image"], path=str(path.relative_to(output)), error=None)
        for attempt in range(retries):
            try:
                reusable = (path.exists() and previous.get("image") == row["image"]
                            and previous.get("sha256") == file_sha(path))
                if not reusable:
                    temporary = path.with_suffix(".part")
                    if row["image"].startswith("https://") or row["image"].startswith("http://"):
                        request = Request(row["image"], headers={"User-Agent": "butterfly-detection/1.0"})
                        with urlopen(request, timeout=60) as response, temporary.open("wb") as target:
                            shutil.copyfileobj(response, target)
                    else:
                        # Local paths are relative to the manifest, making exam datasets portable.
                        source = Path(row["image"])
                        if not source.is_absolute():
                            source = Path(manifest).parent / source
                        shutil.copyfile(source, temporary)
                    open_image(temporary)
                    temporary.replace(path)
                image = open_image(path)
                record.update(sha256=file_sha(path), width=image.width, height=image.height, error=None)
                break
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < retries:
                    time.sleep(min(2 ** attempt, 4))
        records.append(record)
        # Checkpoint every photo; interrupted downloads can be resumed safely.
        write_jsonl(output / "images.jsonl", records)
        print(f"[{len(records)}/{len(rows)}] {key}: {record['error'] or 'ok'}", flush=True)
    write_json(output / "download.json", dict(manifest_sha256=file_sha(manifest),
                                               failed=sum(bool(r["error"]) for r in records)))
    return records
