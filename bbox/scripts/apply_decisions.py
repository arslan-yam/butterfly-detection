import argparse
import sys
from collections import Counter

import pandas as pd

from api import find_scheme, get_key, set_status

ap = argparse.ArgumentParser()
ap.add_argument("--decisions", required=True)
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()

key = get_key()
dec = pd.read_csv(a.decisions, dtype={"assignment_id": str}, keep_default_na=False)
dec["accept"] = dec["accept"].astype(str) == "True"
scheme, _ = find_scheme(key, f"/assignments/{dec['assignment_id'].iloc[0]}")

print(f"key works, accept {dec['accept'].sum()} pages, reject {(~dec['accept']).sum()} pages")
if not a.apply:
    sys.exit("dry run, nothing changed. add --apply to run")

stats = Counter()
for r in dec.itertuples():
    if r.accept:
        result = set_status(r.assignment_id, "ACCEPTED", key, scheme)
        stats["accepted " + result] += 1
        print(f"  accepted {r.assignment_id}: {result}", flush=True)
    else:
        result = set_status(r.assignment_id, "REJECTED", key, scheme, r.comment)
        stats["rejected " + result] += 1
        print(f"  rejected {r.assignment_id}: {result}", flush=True)

print("\ntotal:", dict(stats))
