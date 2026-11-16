from __future__ import annotations

import json
import math

from .io import boxes_valid


def iou(a, b):
    w = max(0, min(a["left"] + a["width"], b["left"] + b["width"]) - max(a["left"], b["left"]))
    h = max(0, min(a["top"] + a["height"], b["top"] + b["height"]) - max(a["top"], b["top"]))
    intersection = w * h
    union = a["width"] * a["height"] + b["width"] * b["height"] - intersection
    return intersection / union if union else 0.0


def normalize_xyxy(coords, width, height, score=None):
    if len(coords) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in coords):
        raise ValueError(f"Invalid xyxy coordinates: {coords}")
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Reversed or empty xyxy box: {coords}")
    x1, x2 = [min(1.0, max(0.0, v / width)) for v in (x1, x2)]
    y1, y2 = [min(1.0, max(0.0, v / height)) for v in (y1, y2)]
    if x2 <= x1 or y2 <= y1:
        return None  # Predictions in processor padding have no image area.
    box = dict(left=x1, top=y1, width=x2 - x1, height=y2 - y1)
    if score is not None:
        box["score"] = float(score)
    return boxes_valid([box])[0]


def nms(boxes, threshold):
    if threshold is None:
        return boxes
    kept = []
    for box in sorted(boxes, key=lambda b: b.get("score", 1.0), reverse=True):
        if all(iou(box, previous) <= threshold for previous in kept):
            kept.append(box)
    return kept


def parse_qwen(text):
    """Accept a JSON fence and trailing commas; validate every box strictly."""
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[len("```json\n"):-3].strip()
    # Qwen sometimes puts a trailing comma before ] or }. Remove punctuation
    # outside quoted strings only; never extract guessed boxes from free text.
    cleaned, in_string, escaped = [], False, False
    for i, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == ",":
            rest = text[i + 1:].lstrip()
            previous = text[:i].rstrip()
            if rest.startswith(("]", "}")) and previous and previous[-1] not in "[{,:":
                continue
        cleaned.append(char)
    data = json.loads("".join(cleaned))
    if not isinstance(data, list):
        raise ValueError("Qwen must return a JSON list")
    result = []
    for item in data:
        if not isinstance(item, dict) or not {"bbox_format", "box"} <= item.keys():
            raise ValueError("Expected {bbox_format: xyxy, box: [x1,y1,x2,y2]}")
        coords = item["box"]
        # The prompt fixes box to xyxy. Sometimes Qwen copies the same four
        # coordinates into the format tag; this does not change the box field.
        tag = item["bbox_format"]
        if tag != "xyxy" and not (isinstance(tag, list) and tag == coords):
            raise ValueError("Expected bbox_format=xyxy or a duplicate of box coordinates")
        if not isinstance(coords, list) or len(coords) != 4:
            raise ValueError("Expected four coordinates")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1000 for v in coords):
            raise ValueError("Qwen coordinates must be in [0,1000]")
        result.append(normalize_xyxy(coords, 1000, 1000))
    return result


def match_boxes(golden, predicted, threshold):
    """Maximum-cardinality bipartite matching, as required by the contest."""
    edges = [[j for j, box in enumerate(predicted) if iou(gt, box) >= threshold] for gt in golden]
    owner = {}

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    tp = sum(augment(i, set()) for i in range(len(golden)))
    return dict(tp=tp, fp=len(predicted) - tp, fn=len(golden) - tp)
