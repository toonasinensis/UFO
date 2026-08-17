#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"

# -----------------------------------------------------------------------------
# Shared A/B training configuration (edit values here)
# -----------------------------------------------------------------------------
MOTION_DIR="/home/xiechunyang/wt_ws/wt_wbc/UFO/humanoidverse/data/roban/named_roban_lafan_10s"
DATA_MANIFEST="configs/data/roban_s22.yaml"
AGENT="fb"
NUM_ENVS_PER_GPU=1024
NUM_ENV_STEPS=192000000
BUFFER_SIZE_PER_GPU=3000000
CHECKPOINT_EVERY_STEPS=3200000
UPDATE_Z_EVERY_STEP=100
SEED=4728

OLD_GPUS="0,1"
OLD_ACTION_MAPPING="effort_kp"
OLD_WORK_DIR="/home/xiechunyang/wt_ws/wt_wbc/UFO/runs/roban_lafan10s_effort_kp_fb_2gpu"
OLD_TMP_DIR="/tmp/ufo_ab_effort_kp"
OLD_WANDB_RUN_NAME="roban-lafan10s-effort-kp-fb-2gpu"

NEW_GPUS="2,3"
NEW_ACTION_MAPPING="soft_limit_bias"
NEW_WORK_DIR="/home/xiechunyang/wt_ws/wt_wbc/UFO/runs/roban_lafan10s-soft-limit-bias-fb-2gpu"
NEW_TMP_DIR="/tmp/ufo_ab_soft_limit_bias"
NEW_WANDB_RUN_NAME="roban-lafan10s-soft-limit-bias-fb-2gpu"

WANDB_ENTITY_VALUE="xiechunyang1-hajimi"
WANDB_PROJECT_VALUE="hajimi"
WANDB_GROUP_VALUE="ufo_fb_action_mapping_ab"
# -----------------------------------------------------------------------------

if [[ ! -d "$MOTION_DIR" ]]; then
  echo "Motion directory does not exist: $MOTION_DIR" >&2
  exit 1
fi

mkdir -p "$OLD_WORK_DIR" "$NEW_WORK_DIR"
cd "$repo_dir"

launch_experiment() {
  local gpus="$1"
  local action_mapping="$2"
  local work_dir="$3"
  local tmp_dir="$4"
  local wandb_run_name="$5"
  local log_path="$work_dir/server_train.log"
  local cache_dir="$work_dir/cache"

  mkdir -p "$cache_dir" "$tmp_dir"
  nohup setsid env \
    UFO_ROBAN_MOTION_DIR="$MOTION_DIR" \
    UFO_CACHE_DIR="$cache_dir" \
    TMPDIR="$tmp_dir" \
    TEMP="$tmp_dir" \
    TMP="$tmp_dir" \
    CUDA_VISIBLE_DEVICES="$gpus" \
    WANDB_ENTITY="$WANDB_ENTITY_VALUE" \
    WANDB_PROJECT="$WANDB_PROJECT_VALUE" \
    WANDB_GROUP="$WANDB_GROUP_VALUE" \
    .venv/bin/python -m humanoidverse.train \
      --agent "$AGENT" \
      --gpu-ids all \
      --data-manifest "$DATA_MANIFEST" \
      --rebuild-motion-cache \
      --motion-cache-root "$cache_dir/motion_data" \
      --work-dir "$work_dir" \
      --action-mapping "$action_mapping" \
      --num-envs "$NUM_ENVS_PER_GPU" \
      --num-env-steps "$NUM_ENV_STEPS" \
      --buffer-size "$BUFFER_SIZE_PER_GPU" \
      --checkpoint-every-steps "$CHECKPOINT_EVERY_STEPS" \
      --update-z-every-step "$UPDATE_Z_EVERY_STEP" \
      --seed "$SEED" \
      --use-wandb \
      --wandb-run-name "$wandb_run_name" \
      </dev/null >>"$log_path" 2>&1 &

  echo "pid=$! gpus=$gpus mapping=$action_mapping log=$log_path wandb=$wandb_run_name"
}

launch_experiment "$OLD_GPUS" "$OLD_ACTION_MAPPING" "$OLD_WORK_DIR" "$OLD_TMP_DIR" "$OLD_WANDB_RUN_NAME"
launch_experiment "$NEW_GPUS" "$NEW_ACTION_MAPPING" "$NEW_WORK_DIR" "$NEW_TMP_DIR" "$NEW_WANDB_RUN_NAME"
