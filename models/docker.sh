#!/usr/bin/env bash
# Usage: bash models/docker.sh python -m models ...
# Code comes from the image. Only input annotations, outputs and HF cache mount.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image=${BUTTERFLY_IMAGE:-butterfly-models:1.0}
artifacts=${BUTTERFLY_ARTIFACTS:-$repo/models/artifacts}
hf_cache=${BUTTERFLY_HF_CACHE:-$repo/models/cache/huggingface}
inputs=${BUTTERFLY_INPUTS:-$repo/bbox}
gpu=${GPU_DEVICE:-0}
if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
  echo 'GPU_DEVICE must be a single physical GPU index (default: 0)' >&2
  exit 2
fi
mkdir -p -- "$artifacts" "$hf_cache"
artifacts=$(cd -- "$artifacts" && pwd)
hf_cache=$(cd -- "$hf_cache" && pwd)
inputs=$(cd -- "$inputs" && pwd)
image_id=$(docker image inspect --format '{{.Id}}' "$image")
env_file=$(mktemp)
trap 'rm -f -- "$env_file"' EXIT
printf 'BUTTERFLY_IMAGE_ID=%s\nBUTTERFLY_PHYSICAL_GPU=%s\n' "$image_id" "$gpu" > "$env_file"
# GPU-aware Docker wrappers may derive GPU access from this variable.
# --gpus also makes the same script work with ordinary Docker installations.
export CUDA_VISIBLE_DEVICES="$gpu"
docker run --rm --init \
  --name "butterfly-${USER:-user}-$$" \
  --gpus "device=$gpu" --shm-size=2g \
  --user "$(id -u):$(id -g)" \
  --env-file "$env_file" \
  --mount "type=bind,src=$artifacts,dst=/artifacts" \
  --mount "type=bind,src=$hf_cache,dst=/hf-cache" \
  --mount "type=bind,src=$inputs,dst=/inputs,readonly" \
  "$image" "$@"
