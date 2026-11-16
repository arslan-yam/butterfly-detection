from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

from .geometry import match_boxes
from .io import atomic_text, boxes_valid, digest, file_sha, indexed, read_jsonl, write_json, write_jsonl


def rates(tp, fp, fn):
    return dict(tp=tp, fp=fp, fn=fn, precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)


def aggregate(rows, threshold):
    key = f"iou_{threshold}"
    return rates(**{k: sum(row[key][k] for row in rows) for k in ("tp", "fp", "fn")})


def evaluate(annotations, predictions, manifest, output, allow_errors=False):
    golden, predicted, photos = [indexed(read_jsonl(path)) for path in (annotations, predictions, manifest)]
    if golden.keys() != photos.keys() or predicted.keys() != photos.keys():
        raise ValueError("Annotations, predictions and manifest must contain exactly the same IDs")
    dataset_path = Path(annotations).parent / "dataset.json"
    stage = "unknown"
    if dataset_path.exists():
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        if dataset["annotations_sha256"] != file_sha(annotations) or dataset["manifest_sha256"] != file_sha(manifest):
            raise ValueError("Dataset metadata does not match annotation/manifest bytes; rerun prepare")
        stage = dataset["stage"]
    run_path = Path(predictions).parent / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else None
    if run:
        if run["manifest_sha256"] != file_sha(manifest) or run.get("predictions_sha256") != file_sha(predictions):
            raise ValueError("Predictions/manifest differ from completed run metadata")
    errors = [key for key, row in predicted.items() if row.get("error")]
    if errors and not allow_errors:
        raise ValueError(f"{len(errors)} technical errors; rerun inference with --resume. Diagnostic metrics: --allow-errors")
    rows = []
    for key, gt in golden.items():
        boxes_valid(gt["boxes"])
        pred = predicted[key]
        # Diagnostic lower bound only: failed images retain all FN, never disappear.
        boxes = [] if pred.get("error") else boxes_valid(pred["boxes"])
        elapsed = pred.get("elapsed_seconds")
        if elapsed is not None and (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                                    or not math.isfinite(elapsed) or elapsed < 0):
            raise ValueError(f"{key}: invalid elapsed_seconds")
        groups = ["positive" if gt["boxes"] else "negative"]
        if gt["boxes"]:
            groups.append("multiple" if len(gt["boxes"]) > 1 else "single")
        if photos[key].get("group"):
            groups.append("source:" + str(photos[key]["group"]))
        row = dict(photo_id=key, groups=groups, n_golden=len(gt["boxes"]), n_predicted=len(boxes),
                   error=pred.get("error"), elapsed_seconds=elapsed)
        for threshold in (0.5, 0.75):
            row[f"iou_{threshold}"] = match_boxes(gt["boxes"], boxes, threshold)
        rows.append(row)
    negative = [r for r in rows if r["n_golden"] == 0]
    total_seconds = sum(r["elapsed_seconds"] or 0 for r in rows)
    report = dict(
        name=run["config"]["name"] if run else Path(predictions).parent.name,
        model_id=run["config"]["model_id"] if run else None,
        resolved_revision=run["resolved_revision"] if run else None,
        annotation_stage=stage, photos=len(rows), technical_errors=len(errors),
        eligible_for_selection=stage == "golden" and not errors and run is not None and run.get("limit") is None,
        dataset_fingerprint=digest(dict(manifest=file_sha(manifest), annotations=file_sha(annotations))),
        predictions_sha256=file_sha(predictions),
        iou_0_5=aggregate(rows, 0.5), iou_0_75=aggregate(rows, 0.75),
        negative_photos=len(negative), negative_fp_boxes=sum(r["n_predicted"] for r in negative),
        negative_photos_with_fp=sum(r["n_predicted"] > 0 for r in negative),
        total_seconds=total_seconds,
        mean_seconds=total_seconds / len(rows) if all(r["elapsed_seconds"] is not None for r in rows) else None,
        images_with_errors_iou_0_5=sum(bool(r["error"]) or r["iou_0.5"]["fp"] + r["iou_0.5"]["fn"] > 0 for r in rows),
        groups={group: {"photos": sum(group in r["groups"] for r in rows),
                        "iou_0_5": aggregate([r for r in rows if group in r["groups"]], 0.5),
                        "iou_0_75": aggregate([r for r in rows if group in r["groups"]], 0.75)}
                for group in sorted({g for r in rows for g in r["groups"]})},
        timing_note="Decode + preprocessing + forward/generate + postprocessing + device sync; excludes weights and downloads; first image is cold, no warmup. Resumed successes retain original time.",
    )
    write_json(Path(output) / "metrics.json", report)
    write_jsonl(Path(output) / "per_image.jsonl", rows)
    return report


def leaderboard(metrics, output):
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in metrics]
    if len({r["dataset_fingerprint"] for r in reports}) != 1:
        raise ValueError("Cannot compare metrics from different datasets/annotations")
    if len({r["name"] for r in reports}) != len(reports):
        raise ValueError("Run names must be unique")
    reports.sort(key=lambda r: (-r["iou_0_5"]["f1"], r["mean_seconds"] if r["mean_seconds"] is not None else math.inf))
    fields = ["name", "annotation_stage", "eligible_for_selection", "photos", "technical_errors",
              "f1_0.5", "precision_0.5", "recall_0.5", "f1_0.75", "tp", "fp", "fn",
              "negative_fp_boxes", "negative_photos_with_fp", "mean_seconds"]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for report in reports:
        row = {k: report[k] for k in fields if k in report}
        row.update({k: report["iou_0_5"][k] for k in ("tp", "fp", "fn")})
        row.update({f"{k}_0.5": report["iou_0_5"][k] for k in ("f1", "precision", "recall")})
        row["f1_0.75"] = report["iou_0_75"]["f1"]
        writer.writerow(row)
    output = Path(output)
    atomic_text(output / "leaderboard.csv", stream.getvalue())
    eligible = [r for r in reports if r["eligible_for_selection"]]
    required = {"IDEA-Research/grounding-dino-tiny", "google/owlv2-base-patch16-ensemble", "Qwen/Qwen3-VL-2B-Instruct"}
    complete = required <= {r["model_id"] for r in eligible}
    write_json(output / "selection.json", dict(
        status="ready" if complete else "provisional",
        best=eligible[0]["name"] if complete else None,
        reason="Highest F1@0.5; tie: lowest mean latency" if complete else "Need error-free full golden-set runs of all three required models",
        ranking=[r["name"] for r in reports], dataset_fingerprint=reports[0]["dataset_fingerprint"],
    ))


def export_contest(manifest, predictions, output, expected_count=400):
    expected, predicted = [indexed(read_jsonl(path)) for path in (manifest, predictions)]
    if len(expected) != expected_count or expected.keys() != predicted.keys():
        raise ValueError(f"Need exactly {expected_count} unique IDs matching exam manifest")
    result = []
    for key in expected:
        row = predicted[key]
        if row.get("error") is not None:
            raise ValueError(f"{key}: unresolved model error: {row['error']}")
        boxes = boxes_valid(row["boxes"])
        result.append(dict(photo_id=int(key), boxes=[{k: b[k] for k in ("left", "top", "width", "height")} for b in boxes], error=None))
    text = "".join(json.dumps(row, allow_nan=False) + "\n" for row in result)
    if len(text.encode()) > 5 * 1024 * 1024:
        raise ValueError("Contest JSONL exceeds 5 MiB")
    atomic_text(output, text)
