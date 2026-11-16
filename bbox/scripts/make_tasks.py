import argparse
import os

import pandas as pd

S3 = "https://inaturalist-open-data.s3.amazonaws.com/photos/{}/original.{}"


def save_tasks(df, path):
    tasks = pd.DataFrame({"INPUT:image": df["image"], "INPUT:photo_id": df["photo_id"]})
    tasks.to_csv(path, sep="\t", index=False)
    print(path, len(tasks), "tasks")


ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["dataset", "exam"])
ap.add_argument("--src", required=True)
ap.add_argument("--out-dir", default=".")
ap.add_argument("--pilot", type=int, default=20)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

df = pd.read_csv(a.src, sep=None, engine="python", dtype=str)
df.columns = [c.split(":")[-1] for c in df.columns]
if "image" not in df.columns:
    df["image"] = [S3.format(p, ext) for p, ext in zip(df["photo_id"], df["ext"])]

assert df["photo_id"].is_unique, "duplicate photo_id"
assert df["image"].is_unique, "duplicate image links"
assert df["image"].str.startswith("https://").all(), "image links must be https"

if a.mode == "exam":
    save_tasks(df, os.path.join(a.out_dir, "tasks_exam.tsv"))
else:
    df = df.sample(frac=1, random_state=a.seed)
    save_tasks(df[:a.pilot], os.path.join(a.out_dir, "tasks_pilot.tsv"))
    save_tasks(df[a.pilot:], os.path.join(a.out_dir, "tasks_main.tsv"))
