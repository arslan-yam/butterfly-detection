import argparse
import json

import numpy as np
import pandas as pd

from boxes import clean, col, match, parse_boxes

STATUS_RU = {"butterflies": "«Есть бабочки»", "no_butterflies": "«Бабочек нет»", "load_error": "«Не загрузилось»"}
REJECT_TEXT = ("Перечитайте инструкцию: каждая бабочка — отдельная плотная рамка, ночные бабочки и моль "
               "тоже выделяются; ручейники, гусеницы и другие насекомые — нет.")

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--iou", type=float, default=0.7)
ap.add_argument("--page-share", type=float, default=0.75)
ap.add_argument("--threshold", type=int, default=75)
ap.add_argument("--min-pages", type=int, default=2)
a = ap.parse_args()

ref = {}
for r in pd.read_csv(a.ref, dtype=str).itertuples():
    ref[r.photo_id] = (r.status, json.loads(r.boxes) if isinstance(r.boxes, str) else [])

res = pd.read_csv(a.results, sep="\t", dtype=str)
res = res[col(res, "assignment_id").notna()]

rows = []
for (aid, worker), page in res.groupby([col(res, "assignment_id"), col(res, "worker_id")], sort=False):
    for pos, (_, r) in enumerate(page.iterrows(), start=1):
        ref_status, ref_boxes = ref[r["INPUT:photo_id"]]
        status = clean(r["OUTPUT:status"])
        tp, fp, fn, ious = match(ref_boxes, parse_boxes(r["OUTPUT:result"]), a.iou)

        error = ""
        if status != ref_status:
            error = f"фото {pos}: ответ {STATUS_RU.get(status, status)} вместо {STATUS_RU[ref_status]}"
        elif fp or fn:
            error = f"фото {pos}: рамки неточные или не все бабочки выделены"

        rows.append({"assignment_id": aid, "worker_id": worker, "photo_id": r["INPUT:photo_id"],
                     "correct": not error, "mean_iou": np.mean(ious) if ious else np.nan, "error": error})
photos = pd.DataFrame(rows)

pages = photos.groupby(["assignment_id", "worker_id"], sort=False).agg(
    n=("correct", "size"),
    n_correct=("correct", "sum"),
    mean_iou=("mean_iou", "mean"),
    errors=("error", lambda s: "; ".join(e for e in s if e)),
).reset_index()
pages["accept"] = pages["n_correct"] >= a.page_share * pages["n"]
pages["comment"] = ""
bad = ~pages["accept"]
pages.loc[bad, "comment"] = "Ошибки — " + pages.loc[bad, "errors"].str.slice(0, 400) + ". " + REJECT_TEXT
pages.to_csv("exam_decisions.csv", index=False)

workers = pages.groupby("worker_id").agg(
    pages=("accept", "size"),
    accepted=("accept", "sum"),
    photos_correct=("n_correct", "sum"),
    mean_iou=("mean_iou", "mean"),
).reset_index()
workers["photos_done"] = 4 * workers["pages"]
workers["skill"] = (100 * workers["photos_correct"] / workers["photos_done"]).round().astype(int)
workers["admit"] = (workers["pages"] >= a.min_pages) & (workers["skill"] >= a.threshold)
workers = workers.sort_values("skill", ascending=False)
workers.to_csv("exam_workers.csv", index=False)

admitted = workers[workers["admit"]]
admitted[["worker_id", "skill", "pages", "photos_correct"]].to_csv("exam_admitted.csv", index=False)

print(photos.groupby("photo_id")["correct"].mean().rename("share_correct").to_string())
print()
print(workers.to_string(index=False))
print(f"\nadmitted (>= {a.min_pages} pages, skill >= {a.threshold}): {len(admitted)}")
print(f"pages: {len(pages)}, accept: {pages['accept'].sum()}, reject: {bad.sum()}")
