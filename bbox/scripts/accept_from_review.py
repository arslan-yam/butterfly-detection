import argparse

import pandas as pd

REJECT_TEXT = ("Ревьюеры нашли ошибки в рамках на нескольких фото этой страницы "
               "(пропуски, лишние рамки или неточные границы).")

ap = argparse.ArgumentParser()
ap.add_argument("--verdicts", required=True)
ap.add_argument("--raw", required=True)
ap.add_argument("--page-share", type=float, default=0.75)
a = ap.parse_args()

verdicts = pd.read_csv(a.verdicts, dtype=str)
raw = pd.read_json(a.raw, lines=True, dtype={"photo_id": str})

verdicts["ok"] = verdicts["verdict"] == "ok"
pages = verdicts.groupby("bbox_assignment_id")["ok"].agg(n="size", n_ok="sum").reset_index()
pages["accept"] = pages["n_ok"] >= a.page_share * pages["n"]
pages["comment"] = pages["accept"].map({True: "", False: REJECT_TEXT})
pages.to_csv("bbox_page_decisions.csv", index=False)

images = raw[["photo_id", "image"]].drop_duplicates()
fix = verdicts[verdicts["verdict"] == "fix"].merge(images, on="photo_id").drop_duplicates("photo_id")
rework = pd.DataFrame({"INPUT:image": fix["image"], "INPUT:photo_id": fix["photo_id"]})
rework.to_csv("tasks_rework.tsv", sep="\t", index=False)

manual = verdicts[verdicts["verdict"] == "manual"]
manual.drop(columns="ok").to_csv("manual_review.csv", index=False)

print(f"pages: {len(pages)}, accept: {pages['accept'].sum()}, reject: {(~pages['accept']).sum()}")
print(f"rework: {len(rework)}, manual: {len(manual)}")
