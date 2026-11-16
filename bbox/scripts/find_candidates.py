import argparse
import time

import pandas as pd
import requests

API = "https://api.inaturalist.org/v1/observations"
S3 = "https://inaturalist-open-data.s3.amazonaws.com/photos/{}/original.{}"

CATEGORIES = {
    "butterfly": {"taxon_id": 47224, "term_id": 1, "term_value_id": 2},
    "moth": {"taxon_id": 47157, "without_taxon_id": 47224, "term_id": 1, "term_value_id": 2},
    "lookalike": {"taxon_id": "62164,48763"},
    "caterpillar": {"taxon_id": 47157, "term_id": 1, "term_value_id": 6},
    "pupa": {"taxon_id": 47157, "term_id": 1, "term_value_id": 4},
    "other_insect": {"taxon_id": "47201,47792,47822"},
    "plant": {"taxon_id": 47126},
}


def fetch(cat, params, n):
    rows, page = [], 1
    while len(rows) < n and page <= 5:
        q = {"photos": "true", "photo_license": "cc0", "per_page": 200,
             "page": page, "order_by": "votes", **params}
        res = requests.get(API, params=q, timeout=30).json()["results"]
        if not res:
            break
        for o in res:
            for p in o["photos"]:
                if p.get("license_code") != "cc0":
                    continue
                ext = p["url"].rsplit(".", 1)[-1].split("?")[0]
                rows.append({
                    "category": cat,
                    "photo_id": p["id"],
                    "image": S3.format(p["id"], ext),
                    "observation_url": f"https://www.inaturalist.org/observations/{o['id']}",
                    "taxon": (o.get("taxon") or {}).get("name"),
                    "attribution": p.get("attribution"),
                    "license": p["license_code"],
                })
                break
        page += 1
        time.sleep(1)
    return rows[:n]


ap = argparse.ArgumentParser()
ap.add_argument("--exclude")
ap.add_argument("--per-cat", type=int, default=40)
ap.add_argument("--out", default="candidates.csv")
a = ap.parse_args()

df = pd.DataFrame([r for c, p in CATEGORIES.items() for r in fetch(c, p, a.per_cat)])
df = df.drop_duplicates("photo_id")
if a.exclude:
    ex = pd.read_csv(a.exclude, sep=None, engine="python")
    ex.columns = [c.split(":")[-1] for c in ex.columns]
    used = set(ex["photo_id"].astype(int))
    df = df[~df["photo_id"].isin(used)]
df.to_csv(a.out, index=False)
print(df["category"].value_counts().to_string())
print("saved:", a.out)
