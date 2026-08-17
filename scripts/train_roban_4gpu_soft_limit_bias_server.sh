#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"

# -----------------------------------------------------------------------------
# Training configuration (edit values here)
# -----------------------------------------------------------------------------
MOTION_DIR="/home/xiechunyang/wt_ws/wt_wbc/UFO/data/selected_raw_my_data_10s"
DATA_MANIFEST="configs/data/roban_s22.yaml"
WORK_DIR="/home/xiechunyang/wt_ws/wt_wbc/UFO/runs/roban_soft_limit_bias_fb_4gpu"
CUDA_VISIBLE_DEVICES_VALUE="0,1,2,3"
GPU_IDS="all"
AGENT="fb"
ACTION_MAPPING="soft_limit_bias"
NUM_ENVS_PER_GPU=1024
NUM_ENV_STEPS=192000000
BUFFER_SIZE_PER_GPU=3000000
CHECKPOINT_EVERY_STEPS=3200000
UPDATE_Z_EVERY_STEP=100
SEED=4728
WANDB_ENTITY_VALUE="xiechunyang1-hajimi"
WANDB_PROJECT_VALUE="hajimi"
WANDB_GROUP_VALUE="ufo_fb"
WANDB_RUN_NAME="roban-soft-limit-bias-fb-4gpu"
SERVER_LOG="$WORK_DIR/server_train.log"
# -----------------------------------------------------------------------------

if [[ ! -d "$MOTION_DIR" ]]; then
  echo "Motion directory does not exist: $MOTION_DIR" >&2
  exit 1
fi

mkdir -p "$WORK_DIR"
cd "$repo_dir"

export UFO_ROBAN_MOTION_DIR="$MOTION_DIR"
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
export WANDB_ENTITY="$WANDB_ENTITY_VALUE"
export WANDB_PROJECT="$WANDB_PROJECT_VALUE"
export WANDB_GROUP="$WANDB_GROUP_VALUE"

nohup setsid .venv/bin/python -m humanoidverse.train \
  --agent "$AGENT" \
  --gpu-ids "$GPU_IDS" \
  --data-manifest "$DATA_MANIFEST" \
  --rebuild-motion-cache \
  --work-dir "$WORK_DIR" \
  --action-mapping "$ACTION_MAPPING" \
  --num-envs "$NUM_ENVS_PER_GPU" \
  --num-env-steps "$NUM_ENV_STEPS" \
  --buffer-size "$BUFFER_SIZE_PER_GPU" \
  --checkpoint-every-steps "$CHECKPOINT_EVERY_STEPS" \
  --update-z-every-step "$UPDATE_Z_EVERY_STEP" \
  --seed "$SEED" \
  --use-wandb \
  --wandb-run-name "$WANDB_RUN_NAME" \
  </dev/null >>"$SERVER_LOG" 2>&1 &

pid=$!
echo "pid=$pid"
echo "log=$SERVER_LOG"
echo "work_dir=$WORK_DIR"
echo "motion_dir=$MOTION_DIR"
echo "gpus=$CUDA_VISIBLE_DEVICES_VALUE"
echo "action_mapping=$ACTION_MAPPING"
echo "buffer_size_per_gpu=$BUFFER_SIZE_PER_GPU"
echo "wandb=$WANDB_ENTITY_VALUE/$WANDB_PROJECT_VALUE/$WANDB_RUN_NAME"
