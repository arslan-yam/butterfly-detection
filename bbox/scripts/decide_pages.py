import argparse

import pandas as pd

from boxes import match, parse_boxes

ap = argparse.ArgumentParser()
ap.add_argument("--final", required=True)
ap.add_argument("--raw", required=True)
ap.add_argument("--out", default="page_decisions.csv")
ap.add_argument("--iou", type=float, default=0.5)
ap.add_argument("--page-share", type=float, default=0.75)
a = ap.parse_args()

final = pd.read_csv(a.final, sep="\t", dtype=str).set_index("photo_id")
raw = pd.read_json(a.raw, lines=True, dtype={"photo_id": str})
raw["pos"] = raw.groupby("assignment_id").cumcount() + 1

errors = []
for r in raw.itertuples():
    f = final.loc[r.photo_id]
    tp, fp, fn, _ = match(parse_boxes(f["boxes"]), r.boxes, a.iou)
    if r.status != f["status"] and f["status"] == "butterflies":
        errors.append(f"Фото {r.pos}: ответ «Бабочек нет», хотя на фото есть бабочка")
    elif r.status != f["status"]:
        errors.append(f"Фото {r.pos}: выделено не бабочка")
    elif fp or fn:
        errors.append(f"Фото {r.pos}: рамки неточные или не все бабочки выделены")
    else:
        errors.append("")
raw["error"] = errors
raw["ok"] = raw["error"] == ""

pages = raw.groupby(["assignment_id", "worker_id"], sort=False).agg(
    n=("ok", "size"),
    n_ok=("ok", "sum"),
    errors=("error", lambda s: "; ".join(e for e in s if e)),
).reset_index()
pages["accept"] = pages["n_ok"] >= a.page_share * pages["n"]
pages["comment"] = ""
bad = ~pages["accept"]
pages.loc[bad, "comment"] = (pages.loc[bad, "errors"]
                             + ". По инструкции ночные бабочки и моль тоже выделяются рамкой.")
pages.to_csv(a.out, index=False)

print(f"pages: {len(pages)}, accept: {pages['accept'].sum()}, reject: {bad.sum()}")
print(raw.groupby("worker_id")["ok"].agg(answers="size", errors=lambda s: (~s).sum()).to_string())
