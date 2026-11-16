#!/usr/bin/env bash
# Real GPU check on positive, negative and multiple-object examples. Not a benchmark.
set -euo pipefail
python - <<'PY'
import torch
from models.io import read_jsonl, write_jsonl
from models.data import prepare

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 1, "Exactly one GPU must be visible"
print("GPU:", torch.cuda.get_device_name(0), flush=True)
rows = read_jsonl('/inputs/main/bbox_raw.jsonl') + read_jsonl('/inputs/pilot/bbox_raw.jsonl')
selected = [next(r for r in rows if not r['boxes']),
            next(r for r in rows if len(r['boxes']) == 1),
            next(r for r in rows if len(r['boxes']) > 1)]
write_jsonl('/artifacts/data/smoke/source.jsonl', selected)
prepare(['/artifacts/data/smoke/source.jsonl'], '/artifacts/data/smoke', 'raw', 3)
PY
python -m models download --manifest /artifacts/data/smoke/manifest.jsonl --output /artifacts/cache/smoke
python -m models lock --configs models/configs/dino.json models/configs/dino_threshold035.json \
  models/configs/owlv2.json models/configs/owlv2_threshold020.json models/configs/qwen.json --output /artifacts/smoke/configs
for config in dino dino_threshold035 owlv2 owlv2_threshold020 qwen; do
  python -m models run --manifest /artifacts/data/smoke/manifest.jsonl \
    --images /artifacts/cache/smoke/images.jsonl --config "/artifacts/smoke/configs/$config.json" \
    --output "/artifacts/smoke/$config" --device cuda --resume
done
