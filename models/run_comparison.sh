#!/usr/bin/env bash
# Run from repository root; one process/model at a time to release GPU memory.
set -euo pipefail
dataset=${1:?Usage: bash models/run_comparison.sh DATASET_DIR CACHE_DIR RUNS_DIR}
cache=${2:?Missing cache directory}
runs=${3:?Missing runs directory}
python -m models lock --configs models/configs/dino.json models/configs/dino_threshold035.json \
  models/configs/owlv2.json models/configs/owlv2_threshold020.json models/configs/qwen.json --output "$runs/configs"
for config in dino dino_threshold035 owlv2 owlv2_threshold020 qwen; do
  python -m models run --manifest "$dataset/manifest.jsonl" \
    --images "$cache/images.jsonl" --config "$runs/configs/$config.json" \
    --output "$runs/$config" --resume
  python -m models evaluate --manifest "$dataset/manifest.jsonl" \
    --annotations "$dataset/annotations.jsonl" \
    --predictions "$runs/$config/predictions.jsonl" --output "$runs/$config"
done
python -m models leaderboard --metrics \
  "$runs/dino/metrics.json" "$runs/dino_threshold035/metrics.json" \
  "$runs/owlv2/metrics.json" "$runs/owlv2_threshold020/metrics.json" "$runs/qwen/metrics.json" --output "$runs"
