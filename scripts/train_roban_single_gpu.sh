#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${UFO_CUDA_VISIBLE_DEVICES:-0}"
export UFO_ROBAN_MOTION_DIR="${UFO_ROBAN_MOTION_DIR:-$repo_dir/../BFM-Zero-ManagerOnly/bfm/data/named_roban_lafan_10s}"
export WANDB_ENTITY="xiechunyang1-hajimi"
export WANDB_PROJECT="${WANDB_PROJECT:-${BFM_WANDB_PROJECT:-hajimi}}"
export WANDB_GROUP="${WANDB_GROUP:-${BFM_WANDB_GROUP:-ufo_fb}}"

wandb_run_name="${WANDB_RUN_NAME:-${BFM_WANDB_NAME:-}}"
use_wandb="${UFO_USE_WANDB:-${BFM_USE_WANDB:-1}}"

if [[ ! -d "$UFO_ROBAN_MOTION_DIR" ]]; then
  echo "Roban motion directory does not exist: $UFO_ROBAN_MOTION_DIR" >&2
  exit 1
fi

cd "$repo_dir"
train_args=(
  --agent fb \
  --gpu-ids single \
  --data-manifest configs/data/roban_s22.yaml \
  --work-dir "${UFO_ROBAN_WORK_DIR:-runs/roban_s22_fb}" \
)

case "${use_wandb,,}" in
  1|true|yes|on)
    train_args+=(--use-wandb)
    ;;
esac

if [[ -n "$wandb_run_name" ]]; then
  train_args+=(--wandb-run-name "$wandb_run_name")
fi

train_args+=("$@")

exec ./run_train.sh "${train_args[@]}"
