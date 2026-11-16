import json


def is_empty(v):
    return v is None or v == "" or (isinstance(v, float) and v != v)


def clean(v):
    if is_empty(v):
        return ""
    return str(v).strip().strip('"')


def col(df, name):
    for c in df.columns:
        if c.split(":")[-1] == name:
            return df[c]
    raise KeyError(f"no column {name}")


def parse_boxes(raw):
    if is_empty(raw):
        return []
    raw = raw.replace("\\,", ",").strip()
    if not raw.startswith("["):
        raw = "[" + raw + "]"

    boxes = []
    for b in json.loads(raw):
        if b.get("shape") != "rectangle":
            continue
        left, top = max(b["left"], 0), max(b["top"], 0)
        right = min(b["left"] + b["width"], 1)
        bottom = min(b["top"] + b["height"], 1)
        if right > left and bottom > top:
            boxes.append({"left": round(left, 6), "top": round(top, 6),
                          "width": round(right - left, 6), "height": round(bottom - top, 6)})
    return boxes


def iou(a, b):
    x1 = max(a["left"], b["left"])
    y1 = max(a["top"], b["top"])
    x2 = min(a["left"] + a["width"], b["left"] + b["width"])
    y2 = min(a["top"] + a["height"], b["top"] + b["height"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0


def match(ref, pred, thr=0.5):
    good = [[iou(r, p) >= thr for p in pred] for r in ref]
    owner = [-1] * len(pred)

    def find_pair(i, seen):
        for j in range(len(pred)):
            if good[i][j] and not seen[j]:
                seen[j] = True
                if owner[j] == -1 or find_pair(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    tp = sum(find_pair(i, [False] * len(pred)) for i in range(len(ref)))
    ious = [iou(ref[owner[j]], pred[j]) for j in range(len(pred)) if owner[j] != -1]
    return tp, len(pred) - tp, len(ref) - tp, ious
