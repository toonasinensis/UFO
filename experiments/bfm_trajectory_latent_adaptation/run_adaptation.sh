#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# =============================================================================
# 直接修改这里的参数。脚本不读取 UFO_* 等隐藏环境变量。
# =============================================================================

PYTHON_BIN=".venv/bin/python"
MODE="full"                     # smoke 或 full
RESUME_FROM_CHECKPOINT=true
RESUME_SOURCE_DIR="experiments/bfm_trajectory_latent_adaptation/outputs/dance1_fk50_0010_best_link_params_iter50"
MODEL_FOLDER="runs/新数据addlelay_onnx_153m_20260807"
ACTOR_ONNX="runs/新数据addlelay_onnx_153m_20260807/exported/FBcprAuxModel.onnx"
BACKWARD_ONNX="runs/新数据addlelay_onnx_153m_20260807/exported/backward_encoder.onnx"
MOTION="data/selected_raw_my_data_10s/dance1_subject2.roban_s22.fk50_0010.npz"
ROBOT_CONFIG="configs/robots/roban_s22.yaml"
DEVICE="cuda:0"
ONNX_PROVIDER="cuda"
SEED=0
DETERMINISM_TOLERANCE=0.02

# 从已完成的第 50 轮 checkpoint 继续到 50,000 轮。
PARTICLES=128
ITERATIONS=50000
BETA_ITERATION=0.85
BETA_HORIZON=0.90
INITIAL_SIGMA=0.20
MPPI_TEMPERATURE=0.25
TEMPORAL_SMOOTHING_WINDOW=5
FRAME_COUNT=500

# smoke 只优化前 49 个 action step（50 reference frames）。
SMOKE_PARTICLES=8
SMOKE_ITERATIONS=2
SMOKE_FRAME_COUNT=50
SMOKE_OUTPUT_DIR="experiments/bfm_trajectory_latent_adaptation/outputs/smoke"

# Sweep 最优 w60_s030：link-relative 权重 0.60，其余权重合计 0.40。
ROOT_POSITION_WEIGHT=0.085714
ROOT_ROTATION_WEIGHT=0.057143
BODY_POSITION_WEIGHT=0.60
BODY_ROTATION_WEIGHT=0.085714
JOINT_POSITION_WEIGHT=0.085714
BODY_LINEAR_VELOCITY_WEIGHT=0.042857
BODY_ANGULAR_VELOCITY_WEIGHT=0.042858

# Tracking error 的 Gaussian 尺度。
ROOT_POSITION_SIGMA=0.15
ROOT_ROTATION_SIGMA=0.35
BODY_POSITION_SIGMA=0.03
BODY_ROTATION_SIGMA=0.35
JOINT_POSITION_SIGMA=0.25
BODY_LINEAR_VELOCITY_SIGMA=0.60
BODY_ANGULAR_VELOCITY_SIGMA=1.50

SAVE_VIDEO=true
VIDEO_INTERVAL=50
VALIDATION_INTERVAL=0           # 长跑先关闭额外复验，只记录搜索曲线。
ENABLE_TENSORBOARD=true
HISTORY_FORMAT="jsonl"          # 长跑必须追加写，避免每轮重写全量 JSON。
SHOW_PROGRESS=false             # 关闭每帧 tqdm，只保留每轮摘要日志。
RENDER_SIZE=480
FPS=50
OUTPUT_DIR="experiments/bfm_trajectory_latent_adaptation/outputs/dance1_fk50_0010_w60_s030_iter50000"

if [[ "${MODE}" == "smoke" ]]; then
  PARTICLES="${SMOKE_PARTICLES}"
  ITERATIONS="${SMOKE_ITERATIONS}"
  FRAME_COUNT="${SMOKE_FRAME_COUNT}"
  RESUME_FROM_CHECKPOINT=false
  OUTPUT_DIR="${SMOKE_OUTPUT_DIR}"
elif [[ "${MODE}" != "full" ]]; then
  echo "MODE must be smoke or full, got: ${MODE}" >&2
  exit 2
fi

VIDEO_ARG="--no-save-video"
if [[ "${SAVE_VIDEO}" == "true" ]]; then VIDEO_ARG="--save-video"; fi
TENSORBOARD_ARG="--no-tensorboard"
if [[ "${ENABLE_TENSORBOARD}" == "true" ]]; then TENSORBOARD_ARG="--tensorboard"; fi
PROGRESS_ARG="--no-progress"
if [[ "${SHOW_PROGRESS}" == "true" ]]; then PROGRESS_ARG="--progress"; fi
RUN_LOCATION_ARGS=(--output-dir "${OUTPUT_DIR}")
if [[ "${RESUME_FROM_CHECKPOINT}" == "true" ]]; then
  if [[ ! -d "${OUTPUT_DIR}" ]]; then
    cp -a -- "${RESUME_SOURCE_DIR}" "${OUTPUT_DIR}"
    if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
      mv -- "${OUTPUT_DIR}/summary.json" "${OUTPUT_DIR}/summary_iter50.json"
    fi
    if [[ -f "${OUTPUT_DIR}/metrics_adapted.json" ]]; then
      mv -- "${OUTPUT_DIR}/metrics_adapted.json" "${OUTPUT_DIR}/metrics_adapted_iter50.json"
    fi
    if [[ -f "${OUTPUT_DIR}/tracking_timeseries.npz" ]]; then
      mv -- "${OUTPUT_DIR}/tracking_timeseries.npz" "${OUTPUT_DIR}/tracking_timeseries_iter50.npz"
    fi
  fi
  RUN_LOCATION_ARGS=(
    --resume-dir "${OUTPUT_DIR}"
    --resume-total-iterations "${ITERATIONS}"
    --resume-use-cli-config
  )
fi

exec "${PYTHON_BIN}" -m experiments.bfm_trajectory_latent_adaptation.adapt_trajectory \
  --model-folder "${MODEL_FOLDER}" \
  --actor-onnx "${ACTOR_ONNX}" \
  --backward-onnx "${BACKWARD_ONNX}" \
  --motion "${MOTION}" \
  --robot-config "${ROBOT_CONFIG}" \
  --device "${DEVICE}" \
  --onnx-provider "${ONNX_PROVIDER}" \
  --particles "${PARTICLES}" \
  --iterations "${ITERATIONS}" \
  --frame-count "${FRAME_COUNT}" \
  --sigma0 "${INITIAL_SIGMA}" \
  --beta-iteration "${BETA_ITERATION}" \
  --beta-horizon "${BETA_HORIZON}" \
  --temperature "${MPPI_TEMPERATURE}" \
  --smoothing-window "${TEMPORAL_SMOOTHING_WINDOW}" \
  --seed "${SEED}" \
  --determinism-tolerance "${DETERMINISM_TOLERANCE}" \
  --root-position-weight "${ROOT_POSITION_WEIGHT}" \
  --root-rotation-weight "${ROOT_ROTATION_WEIGHT}" \
  --body-position-weight "${BODY_POSITION_WEIGHT}" \
  --body-rotation-weight "${BODY_ROTATION_WEIGHT}" \
  --joint-position-weight "${JOINT_POSITION_WEIGHT}" \
  --body-linear-velocity-weight "${BODY_LINEAR_VELOCITY_WEIGHT}" \
  --body-angular-velocity-weight "${BODY_ANGULAR_VELOCITY_WEIGHT}" \
  --root-position-sigma "${ROOT_POSITION_SIGMA}" \
  --root-rotation-sigma "${ROOT_ROTATION_SIGMA}" \
  --body-position-sigma "${BODY_POSITION_SIGMA}" \
  --body-rotation-sigma "${BODY_ROTATION_SIGMA}" \
  --joint-position-sigma "${JOINT_POSITION_SIGMA}" \
  --body-linear-velocity-sigma "${BODY_LINEAR_VELOCITY_SIGMA}" \
  --body-angular-velocity-sigma "${BODY_ANGULAR_VELOCITY_SIGMA}" \
  "${RUN_LOCATION_ARGS[@]}" \
  "${VIDEO_ARG}" \
  --video-interval "${VIDEO_INTERVAL}" \
  --validation-interval "${VALIDATION_INTERVAL}" \
  "${TENSORBOARD_ARG}" \
  --history-format "${HISTORY_FORMAT}" \
  "${PROGRESS_ARG}" \
  --render-size "${RENDER_SIZE}" \
  --fps "${FPS}" \
  "$@"
