#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"

cd "$repo_dir"

work_dir="${UFO_ROBAN_WORK_DIR:-runs/roban_s22_fb}"
mkdir -p "$work_dir"

log_path="${UFO_SERVER_LOG_PATH:-$work_dir/server_train.log}"
run_name_default="roban-fb-cuda${UFO_CUDA_VISIBLE_DEVICES:-0}-$(date +%Y%m%d-%H%M%S)"

export BFM_USE_WANDB="${BFM_USE_WANDB:-1}"
export BFM_WANDB_NAME="${BFM_WANDB_NAME:-$run_name_default}"
export BFM_WANDB_ENTITY="${BFM_WANDB_ENTITY:-xiechunyang1-hajimi}"
export BFM_WANDB_PROJECT="${BFM_WANDB_PROJECT:-hajimi}"
export BFM_WANDB_GROUP="${BFM_WANDB_GROUP:-ufo_fb}"

nohup "$script_dir/train_roban_single_gpu.sh" "$@" >>"$log_path" 2>&1 &
pid=$!

echo "pid=$pid"
echo "log=$log_path"
echo "wandb_entity=$BFM_WANDB_ENTITY"
echo "wandb_project=$BFM_WANDB_PROJECT"
echo "wandb_group=$BFM_WANDB_GROUP"
echo "wandb_name=$BFM_WANDB_NAME"
