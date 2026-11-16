import argparse
import sys
from collections import Counter

import pandas as pd

from api import find_scheme, get_key, set_status
from boxes import col

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()

key = get_key()
res = pd.read_csv(a.results, sep="\t", dtype=str)
ids = col(res, "assignment_id").dropna().unique()
scheme, _ = find_scheme(key, f"/assignments/{ids[0]}")

print(f"key works, pages in file: {len(ids)}")
if not a.apply:
    sys.exit("dry run, nothing changed. add --apply to run")

stats = Counter()
for aid in ids:
    result = set_status(aid, "ACCEPTED", key, scheme)
    stats[result] += 1
    print(f"  accepted {aid}: {result}", flush=True)

print("\ntotal:", dict(stats))
