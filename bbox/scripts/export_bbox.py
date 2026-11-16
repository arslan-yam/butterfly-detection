import argparse
import csv
import json

import pandas as pd

from boxes import clean, col, parse_boxes

ap = argparse.ArgumentParser()
ap.add_argument("--results", nargs="+", required=True)
a = ap.parse_args()

res = pd.concat([pd.read_csv(p, sep="\t", dtype=str) for p in a.results], ignore_index=True)
res = res[col(res, "assignment_id").notna()]

df = pd.DataFrame({
    "photo_id": res["INPUT:photo_id"],
    "image": res["INPUT:image"],
    "assignment_id": col(res, "assignment_id"),
    "worker_id": col(res, "worker_id"),
    "status": res["OUTPUT:status"].map(clean),
    "unsure": res.get("OUTPUT:unsure", pd.Series(index=res.index, dtype=str)).map(clean).str.lower() == "true",
    "comment": res.get("OUTPUT:comment", pd.Series(index=res.index, dtype=str)).map(clean),
    "boxes": res["OUTPUT:result"].map(parse_boxes),
})
df["n_boxes"] = df["boxes"].str.len()

mismatch = (df["status"] == "butterflies") != (df["n_boxes"] > 0)
if mismatch.any():
    print("warning: status does not match boxes in", mismatch.sum(), "answers")

df.to_json("bbox_raw.jsonl", orient="records", lines=True, force_ascii=False)

review = pd.DataFrame({
    "INPUT:image": df["image"],
    "INPUT:photo_id": df["photo_id"],
    "INPUT:boxes": df["boxes"].map(json.dumps),
    "INPUT:status": df["status"],
    "INPUT:bbox_assignment_id": df["assignment_id"],
    "INPUT:bbox_worker_id": df["worker_id"],
})
review.to_csv("review_tasks.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE)

flagged = df[df["unsure"] | (df["status"] == "load_error")]
flagged[["photo_id", "image", "worker_id", "status", "unsure", "comment"]].to_csv("bbox_flagged.csv", index=False)

print(f"answers: {len(df)}, photos: {df['photo_id'].nunique()}, workers: {df['worker_id'].nunique()}")
print(df["status"].value_counts().to_string())
print(f"boxes: {df['n_boxes'].sum()}, flagged: {len(flagged)}")
print("answers per worker:")
print(df["worker_id"].value_counts().to_string())
