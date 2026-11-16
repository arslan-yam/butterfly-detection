from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .backends import HuggingFaceBackend
from .data import open_image
from .geometry import nms
from .io import boxes_valid, digest, file_sha, indexed, read_jsonl, write_json, write_jsonl


def validate_config(config):
    common = {"name", "backend", "model_id", "revision", "nms_iou", "seed", "hypothesis"}
    specific = {"dino": {"prompt", "threshold", "text_threshold"},
                "owlv2": {"queries", "threshold"},
                "qwen": {"prompt", "min_pixels", "max_pixels", "max_new_tokens", "load_in_4bit"}}
    backend = config.get("backend")
    if backend not in specific:
        raise ValueError(f"Unknown backend: {backend}")
    expected = common | specific[backend]
    if set(config) != expected:
        raise ValueError(f"Config keys missing={expected - set(config)}, unknown={set(config) - expected}")
    for key in ("threshold", "text_threshold", "nms_iou"):
        if key in config and not (key == "nms_iou" and config[key] is None):
            if isinstance(config[key], bool) or not isinstance(config[key], (int, float)) or not 0 <= config[key] <= 1:
                raise ValueError(f"{key} must be in [0,1]")
    for key in ("name", "model_id", "revision", "prompt", "hypothesis"):
        if key in config and (not isinstance(config[key], str) or not config[key].strip()):
            raise ValueError(f"{key} must be nonempty text")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise ValueError("seed must be integer")
    if backend == "owlv2" and (not isinstance(config["queries"], list) or not config["queries"]
                                or any(not isinstance(q, str) or not q.strip() for q in config["queries"])):
        raise ValueError("queries must be a nonempty list of text")
    if backend == "qwen":
        for key in ("min_pixels", "max_pixels", "max_new_tokens"):
            if type(config[key]) is not int or config[key] <= 0:
                raise ValueError(f"{key} must be positive integer")
        if config["min_pixels"] > config["max_pixels"] or type(config["load_in_4bit"]) is not bool:
            raise ValueError("Invalid Qwen pixel range or quantization flag")
    return config


def environment():
    packages = {}
    for name in ("torch", "torchvision", "transformers", "accelerate", "huggingface-hub", "Pillow", "bitsandbytes"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    build_path = Path("/app/build-info.json")
    container = json.loads(build_path.read_text(encoding="utf-8")) if build_path.exists() else None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=packages,
                git_commit=commit, git_dirty=dirty, container=container,
                image_id=os.environ.get("BUTTERFLY_IMAGE_ID"),
                physical_gpu=os.environ.get("BUTTERFLY_PHYSICAL_GPU"))


def resolve_revision(model_id, revision):
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    from huggingface_hub import HfApi
    return HfApi().model_info(model_id, revision=revision).sha


def lock_configs(configs, output):
    """Resolve all branches once, including a shared revision for paired experiments."""
    output = Path(output)
    originals = {Path(p).name: validate_config(json.loads(Path(p).read_text(encoding="utf-8"))) for p in configs}
    if len(originals) != len(configs):
        raise ValueError("Config basenames must be unique")
    lock_path = output / "lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock["source_digest"] != digest(originals):
            raise ValueError("Locked source configs differ; use a new lock directory")
        for name, sha in lock["files"].items():
            if file_sha(output / name) != sha:
                raise ValueError(f"Locked config changed: {name}")
        return
    resolved, files = {}, {}
    for name, config in originals.items():
        key = (config["model_id"], config["revision"])
        if key not in resolved:
            resolved[key] = resolve_revision(*key)
        write_json(output / name, dict(config, revision=resolved[key]))
        files[name] = file_sha(output / name)
    write_json(lock_path, dict(source_digest=digest(originals), files=files))


def run(manifest, images, config_path, output, device="auto", dtype="auto", resume=False, limit=None):
    import torch

    config = validate_config(json.loads(Path(config_path).read_text(encoding="utf-8")))
    rows = indexed(read_jsonl(manifest))
    image_rows = indexed(read_jsonl(images))
    if rows.keys() != image_rows.keys():
        raise ValueError("Image index and manifest must have identical photo IDs")
    for key, row in rows.items():
        if row["image"] != image_rows[key]["image"]:
            raise ValueError(f"{key}: cached source differs from manifest")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = dict(list(rows.items())[:limit])
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype == "auto":
        dtype = "bfloat16" if device == "cuda" and config["backend"] == "qwen" and torch.cuda.is_bf16_supported() else "float32"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path, predictions_path = output / "run.json", output / "predictions.jsonl"
    requested = dict(config=config, manifest_sha256=file_sha(manifest), images_sha256=file_sha(images),
                     device=device, dtype=dtype, limit=limit, environment=environment(),
                     implementation_sha256=digest({p.name: file_sha(p) for p in Path(__file__).parent.glob("*.py")}))
    # Environment is recorded, but git dirtiness is not a reason to discard a checkpoint.
    identity = {k: v for k, v in requested.items() if k != "environment"}
    identity["packages"] = requested["environment"]["packages"]
    identity["image_id"] = requested["environment"]["image_id"]
    fingerprint = digest(identity)
    if metadata_path.exists():
        if not resume:
            raise ValueError("Run exists; use --resume or a new output directory")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["fingerprint"] != fingerprint:
            raise ValueError("Config, input bytes or runtime differ; use a new run directory")
        revision = metadata["resolved_revision"]
    else:
        if predictions_path.exists():
            raise ValueError("Predictions exist without run metadata; use a new directory")
        revision = resolve_revision(config["model_id"], config["revision"])
        metadata = dict(requested, fingerprint=fingerprint, resolved_revision=revision,
                        created_at=datetime.now(timezone.utc).isoformat(), preprocessing="EXIF transpose, RGB; slow image processor at pinned revision; clip boxes to image, discard boxes entirely in padding; score-sorted class-agnostic NMS",
                        qwen_parse="JSON list; optional json fence; remove trailing commas outside strings; require box in prompted xyxy format, accept bbox_format=xyxy or exact duplicate of box; ignore supplementary metadata; validate coordinates without correction")
        write_json(metadata_path, metadata)
    previous = indexed(read_jsonl(predictions_path)) if predictions_path.exists() else {}
    if not previous.keys() <= rows.keys():
        raise ValueError("Predictions contain unexpected photo IDs")
    for row in previous.values():
        if not row.get("error"):
            boxes_valid(row["boxes"])
    results = {key: row for key, row in previous.items() if not row.get("error")}
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.benchmark = False
    # Verify all cached bytes, including images skipped during resume.
    for key in rows:
        info = image_rows[key]
        if not info.get("error"):
            if file_sha(Path(images).parent / info["path"]) != info["sha256"]:
                raise ValueError(f"{key}: cached image changed; rerun download")
    backend = None
    if len(results) < len(rows):
        start = time.perf_counter()
        backend = HuggingFaceBackend(config, device, dtype, revision)
        backend.synchronize()
        metadata["last_model_load_seconds"] = time.perf_counter() - start
        metadata["hardware"] = dict(device=device, cuda=torch.version.cuda,
                                    gpu=torch.cuda.get_device_name(0) if device == "cuda" else None)
        write_json(metadata_path, metadata)
    for key in rows:
        if key in results:
            continue
        info = image_rows[key]
        started = time.perf_counter()
        row = dict(photo_id=key, boxes=None, error=None)
        try:
            if info.get("error"):
                raise ValueError(f"Image unavailable: {info['error']}")
            image = open_image(Path(images).parent / info["path"])
            boxes, raw = backend.predict(image)
            row["boxes"] = boxes_valid(nms(boxes, config["nms_iou"]))
            if raw is not None:
                row["raw_text"] = raw
        except Exception as exc:
            row.update(error=f"{type(exc).__name__}: {exc}", boxes=None)
            if hasattr(exc, "raw_text"):
                row["raw_text"] = exc.raw_text
            with (output / "errors.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(photo_id=key, at=datetime.now(timezone.utc).isoformat(),
                                             error=row["error"], traceback=traceback.format_exc(),
                                             raw_text=row.get("raw_text")), ensure_ascii=False) + "\n")
            if device == "cuda":
                torch.cuda.empty_cache()
        backend.synchronize()
        row["elapsed_seconds"] = time.perf_counter() - started
        results[key] = row
        write_jsonl(predictions_path, [results[k] for k in rows if k in results])
        print(f"[{len(results)}/{len(rows)}] {key}: {row['error'] or str(len(row['boxes'])) + ' boxes'}", flush=True)
    metadata.update(finished_at=datetime.now(timezone.utc).isoformat(), photos=len(rows),
                    failed=sum(bool(r["error"]) for r in results.values()),
                    predictions_sha256=file_sha(predictions_path))
    write_json(metadata_path, metadata)
    return list(results.values())
