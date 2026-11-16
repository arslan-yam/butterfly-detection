import argparse
import sys
import urllib.parse
from collections import Counter

import pandas as pd

from api import call, find_scheme, get_key, set_status

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true")
ap.add_argument("--skill-name", default="bbox_exam")
ap.add_argument("--decisions", default="exam_decisions.csv")
ap.add_argument("--admitted", default="exam_admitted.csv")
a = ap.parse_args()

key = get_key()
dec = pd.read_csv(a.decisions)
admitted = pd.read_csv(a.admitted)
rejected = dec[~dec["accept"]]
accepted = dec[dec["accept"]]

scheme, resp = find_scheme(key, "/skills?" + urllib.parse.urlencode({"name": a.skill_name}))
skills = [s for s in resp["items"] if s["name"] == a.skill_name]
if len(skills) != 1:
    sys.exit(f"skill {a.skill_name} found {len(skills)} times")
skill_id = skills[0]["id"]

print(f"key works, skill {a.skill_name}: id {skill_id}")
print(f"plan: reject {len(rejected)} pages, accept {len(accepted)} pages, skill for {len(admitted)} workers")
if not a.apply:
    sys.exit("dry run, nothing changed. add --apply to run")

stats = Counter()
for r in rejected.itertuples():
    result = set_status(r.assignment_id, "REJECTED", key, scheme, r.comment)
    stats["rejected " + result] += 1
    print(f"  rejected {r.assignment_id}: {result}", flush=True)

for r in accepted.itertuples():
    result = set_status(r.assignment_id, "ACCEPTED", key, scheme)
    stats["accepted " + result] += 1
    print(f"  accepted {r.assignment_id}: {result}", flush=True)

for r in admitted.itertuples():
    body = {"skill_id": str(skill_id), "user_id": r.worker_id, "value": float(r.skill), "reason": "bbox exam"}
    code, resp = call("PUT", "/user-skills", key, scheme, body)
    if code in (200, 201):
        stats["skills"] += 1
    else:
        stats["skill errors"] += 1
        print(f"  skill error {r.worker_id}: {code} {resp}")

print("\ntotal:", dict(stats))
