#!/usr/bin/env bash
# Runs inside the image: bash models/docker/compare.sh [raw|golden] [sources...]
set -euo pipefail
stage=${1:-raw}
if [[ "$stage" != raw && "$stage" != golden ]]; then
  echo 'Stage must be raw or golden' >&2
  exit 2
fi
if (( $# > 0 )); then shift; fi
sources=("$@")
if (( ${#sources[@]} == 0 )); then
  if [[ "$stage" == golden ]]; then
    echo 'Provide the reviewed JSONL paths under /inputs explicitly' >&2
    exit 2
  fi
  sources=(/inputs/main/bbox_raw.jsonl /inputs/pilot/bbox_raw.jsonl)
fi
python -m models prepare --sources "${sources[@]}" --stage "$stage" \
  --expected-count 400 --output "/artifacts/data/$stage"
python -m models download --manifest "/artifacts/data/$stage/manifest.jsonl" \
  --output /artifacts/cache/team
bash models/run_comparison.sh "/artifacts/data/$stage" /artifacts/cache/team "/artifacts/runs/$stage"
