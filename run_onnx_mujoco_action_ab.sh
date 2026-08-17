#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ========================= Editable configuration =========================
PYTHON_BIN=".venv/bin/python"

# Select exactly one: effort_kp or soft_limit_bias.
MODEL_VARIANT="soft_limit_bias"
EFFORT_KP_MODEL_FOLDER="runs/roban_action_mapping_ab_g89628672_onnx/effort_kp"
SOFT_LIMIT_BIAS_MODEL_FOLDER="runs/roban_action_mapping_ab_g89628672_onnx/soft_limit_bias"

MOTION_FILE="/home/thl/wt_wbc/UFO/data/test/klcb_0601_16_15_001_Skeleton_0613.npz"
ROBOT_CONFIG="configs/robots/roban_s22.yaml"
LATENT_DEVICE="cpu"
LATENT_MODE="single_frame"
REBUILD_LATENT=true

SIM_FPS=200
DECIMATION=4
TORQUE_CLIP="fixed"
ONNX_PROVIDER="cpu"
ONNX_THREADS=0
INFERENCE_MODE="inline"
INITIAL_POSE="motion"
INITIAL_MOTION_FRAME=0
INITIAL_HOLD_SECONDS=1.0
MAX_STEPS=0
LOOP_PLAYBACK=true
HEADLESS=false
SHOW_REFERENCE=true
SAVE_DIAGNOSTICS=true
DIAGNOSTICS_ROOT="runs/roban_action_mapping_ab_g89628672_onnx/diagnostics"
# ========================================================================

# Optional shorthand keeps the editable default above while making A/B runs
# convenient: ./run_onnx_mujoco_action_ab.sh effort_kp
if [[ $# -gt 0 && "${1}" != -* ]]; then
    MODEL_VARIANT="${1}"
    shift
fi

case "${MODEL_VARIANT}" in
    effort_kp)
        MODEL_FOLDER="${EFFORT_KP_MODEL_FOLDER}"
        ;;
    soft_limit_bias)
        MODEL_FOLDER="${SOFT_LIMIT_BIAS_MODEL_FOLDER}"
        ;;
    *)
        echo "MODEL_VARIANT must be effort_kp or soft_limit_bias, got: ${MODEL_VARIANT}" >&2
        exit 2
        ;;
esac

for required_file in \
    "${PYTHON_BIN}" \
    "${MODEL_FOLDER}/config.json" \
    "${MODEL_FOLDER}/exported/FBcprAuxModel.onnx" \
    "${MODEL_FOLDER}/exported/FBcprAuxModel.meta.json" \
    "${MODEL_FOLDER}/exported/backward_encoder.onnx" \
    "${MOTION_FILE}" \
    "${ROBOT_CONFIG}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing required file: ${required_file}" >&2
        exit 1
    fi
done

if [[ "${REBUILD_LATENT}" == true ]]; then
    "${PYTHON_BIN}" -m humanoidverse.generate_onnx_latent \
        --model-folder "${MODEL_FOLDER}" \
        --motion "${MOTION_FILE}" \
        --robot-config "${ROBOT_CONFIG}" \
        --motion-id 0 \
        --device "${LATENT_DEVICE}" \
        --latent-mode "${LATENT_MODE}" \
        --rebuild-motion-cache
fi

sim_args=(
    --model-folder "${MODEL_FOLDER}"
    --motion "${MOTION_FILE}"
    --robot-config "${ROBOT_CONFIG}"
    --sim-fps "${SIM_FPS}"
    --decimation "${DECIMATION}"
    --torque-clip "${TORQUE_CLIP}"
    --onnx-provider "${ONNX_PROVIDER}"
    --onnx-threads "${ONNX_THREADS}"
    --inference-mode "${INFERENCE_MODE}"
    --initial-pose "${INITIAL_POSE}"
    --initial-motion-frame "${INITIAL_MOTION_FRAME}"
    --initial-hold-seconds "${INITIAL_HOLD_SECONDS}"
    --max-steps "${MAX_STEPS}"
    --diagnostics-root "${DIAGNOSTICS_ROOT}"
)

if [[ "${LOOP_PLAYBACK}" == true ]]; then
    sim_args+=(--loop)
else
    sim_args+=(--no-loop)
fi
if [[ "${HEADLESS}" == true ]]; then
    sim_args+=(--headless)
fi
if [[ "${SHOW_REFERENCE}" == false ]]; then
    sim_args+=(--no-reference)
fi
if [[ "${SAVE_DIAGNOSTICS}" == false ]]; then
    sim_args+=(--no-diagnostics)
fi

echo "[onnx-ab] variant=${MODEL_VARIANT} model=${MODEL_FOLDER} motion=${MOTION_FILE}"
exec "${PYTHON_BIN}" -m humanoidverse.onnx_mujoco_sim "${sim_args[@]}" "$@"
