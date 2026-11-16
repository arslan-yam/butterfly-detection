"""Re-evaluate saved predictions against reviewed labels without running models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import prepare
from .evaluation import evaluate, leaderboard
from .io import atomic_text, file_sha, indexed, read_jsonl, write_json


CONFIGS = ("dino", "dino_threshold035", "owlv2", "owlv2_threshold020", "qwen")


def rescore(sources, previous, output):
    previous, output = Path(previous), Path(output)
    if previous.resolve() == output.resolve():
        raise ValueError("Keep original predictions and new metrics in separate directories")
    reference_path = previous / "manifest.jsonl"
    reference = indexed(read_jsonl(reference_path))
    reviewed = indexed([row for source in sources for row in read_jsonl(source)])
    if reference.keys() != reviewed.keys():
        raise ValueError("Reviewed annotations must contain exactly the original photo IDs")
    for key in reference:
        if reference[key]["image"] != reviewed[key].get("image"):
            raise ValueError(f"{key}: image source changed; cannot reuse saved predictions")
    # Check all inputs before writing a new dataset.
    predictions = {}
    for config in CONFIGS:
        path = previous / config / "predictions.jsonl"
        rows = indexed(read_jsonl(path))
        if rows.keys() != reference.keys() or any(row.get("error") for row in rows.values()):
            raise ValueError(f"{config}: need complete predictions without technical errors")
        metadata = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
        if metadata["manifest_sha256"] != file_sha(reference_path) or metadata["predictions_sha256"] != file_sha(path):
            raise ValueError(f"{config}: prediction provenance does not match files")
        predictions[config] = path
    prepare(sources, output, "golden", len(reference))
    # Annotation export order may change after review. Preserve the exact model
    # input manifest so the original run hashes stay verifiable.
    atomic_text(output / "manifest.jsonl", reference_path.read_text(encoding="utf-8"))
    metadata = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    metadata["manifest_sha256"] = file_sha(output / "manifest.jsonl")
    write_json(output / "dataset.json", metadata)
    for config, path in predictions.items():
        evaluate(output / "annotations.jsonl", path, output / "manifest.jsonl", output / config)
    leaderboard([output / config / "metrics.json" for config in CONFIGS], output)
    write_json(output / "rescore.json", dict(
        reference_manifest_sha256=file_sha(reference_path),
        predictions={config: file_sha(path) for config, path in predictions.items()},
        annotations_sha256=file_sha(output / "annotations.jsonl"),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--previous", required=True, help="Directory with manifest.jsonl and saved model runs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        rescore(**vars(args))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
