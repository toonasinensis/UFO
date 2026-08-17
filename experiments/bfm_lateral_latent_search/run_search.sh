#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# =============================================================================
# 直接修改这里的参数。脚本不从 UFO_* 环境变量读取隐藏配置。
# =============================================================================

# 文件与运行设备
# TASK、INITIAL_LATENT 和下方两个 TARGET_* 必须描述同一个动作。
PYTHON_BIN=".venv/bin/python"
TASK="move-ego-180-0.5"
MODEL_FOLDER="runs/新数据addlelay_onnx_153m_20260807"
ACTOR_ONNX="runs/新数据addlelay_onnx_153m_20260807/exported/FBcprAuxModel.onnx"
ROBOT_CONFIG="configs/robots/roban_s22.yaml"
DATA_PATH="cache/motion_data/roban_s22/roban_lafan_10s_inference_ufo.pkl"
INITIAL_LATENT="runs/新数据addlelay_onnx_153m_20260807/reward_inference/move-ego-180-0.5.npy"
DEVICE="cuda:0"
ONNX_PROVIDER="cuda"

# CEM 搜索规模：要设置迭代次数，直接修改 ITERATIONS。
POPULATION=256
ITERATIONS=10000
ELITE_FRACTION=0.125
# 对角 CEM 分布：初始每维标准差；每轮由 elite 样本重新估计，不再固定衰减。
INITIAL_SIGMA=0.20
# 协方差平滑系数：0 表示完全采用本轮 elite，越接近 1 更新越稳、越慢。
COVARIANCE_ALPHA=0.70
MIN_SIGMA=0.03
MAX_SIGMA=0.50
SEED=0

# 每个候选的仿真时间
ROLLOUT_SECONDS=5.0
WARMUP_SECONDS=1.0

# 平面速度 objective。机器人本体坐标：+X 前进，-X 后退，+Y 左移，-Y 右移。
TARGET_FORWARD_SPEED=-0.5
TARGET_LATERAL_SPEED=0.0
VELOCITY_SIGMA=0.20
YAW_RATE_SIGMA=0.75
# 脚掌水平项：复用训练 aux penalty_feet_ori；0.20 约等于单脚倾斜 11.5 度。
FOOT_FLATNESS_SIGMA=0.20
VELOCITY_WEIGHT=0.75
YAW_WEIGHT=0.15
FOOT_FLATNESS_WEIGHT=0.10
LATENT_REG=0.05
MIN_UPRIGHT=0.5
MIN_ROOT_HEIGHT=0.5
FALL_GRACE_SECONDS=0.2

# 输出与可视化
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="experiments/bfm_lateral_latent_search/outputs/${TASK}_cem_foot_flat_${RUN_TIMESTAMP}"
ROLLOUT_FRAMES=5000
SAVE_VIDEO=true
RENDER_SIZE=480
FPS=50
ENABLE_TENSORBOARD=true
TENSORBOARD_LOG_DIR="${OUTPUT_DIR}/tensorboard"

VIDEO_ARG="--no-save-video"
if [[ "${SAVE_VIDEO}" == "true" ]]; then
  VIDEO_ARG="--save-video"
fi

TENSORBOARD_ARG="--no-tensorboard"
if [[ "${ENABLE_TENSORBOARD}" == "true" ]]; then
  TENSORBOARD_ARG="--tensorboard"
fi

exec "${PYTHON_BIN}" -m experiments.bfm_lateral_latent_search.search_lateral_z \
  --task "${TASK}" \
  --model-folder "${MODEL_FOLDER}" \
  --actor-onnx "${ACTOR_ONNX}" \
  --onnx-provider "${ONNX_PROVIDER}" \
  --robot-config "${ROBOT_CONFIG}" \
  --data-path "${DATA_PATH}" \
  --initial-latent "${INITIAL_LATENT}" \
  --device "${DEVICE}" \
  --population "${POPULATION}" \
  --iterations "${ITERATIONS}" \
  --elite-fraction "${ELITE_FRACTION}" \
  --sigma "${INITIAL_SIGMA}" \
  --covariance-alpha "${COVARIANCE_ALPHA}" \
  --min-sigma "${MIN_SIGMA}" \
  --max-sigma "${MAX_SIGMA}" \
  --rollout-s "${ROLLOUT_SECONDS}" \
  --warmup-s "${WARMUP_SECONDS}" \
  --target-forward-speed "${TARGET_FORWARD_SPEED}" \
  --target-lateral-speed "${TARGET_LATERAL_SPEED}" \
  --velocity-sigma "${VELOCITY_SIGMA}" \
  --yaw-rate-sigma "${YAW_RATE_SIGMA}" \
  --foot-flatness-sigma "${FOOT_FLATNESS_SIGMA}" \
  --velocity-weight "${VELOCITY_WEIGHT}" \
  --yaw-weight "${YAW_WEIGHT}" \
  --foot-flatness-weight "${FOOT_FLATNESS_WEIGHT}" \
  --latent-reg "${LATENT_REG}" \
  --min-upright "${MIN_UPRIGHT}" \
  --min-root-height "${MIN_ROOT_HEIGHT}" \
  --fall-grace-s "${FALL_GRACE_SECONDS}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  --rollout-frames "${ROLLOUT_FRAMES}" \
  "${VIDEO_ARG}" \
  --render-size "${RENDER_SIZE}" \
  --fps "${FPS}" \
  "${TENSORBOARD_ARG}" \
  --tensorboard-log-dir "${TENSORBOARD_LOG_DIR}" \
  "$@"
