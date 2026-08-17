#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${UFO_PYTHON:-.venv/bin/python}"
MODEL_FOLDER="${UFO_REWARD_MODEL_FOLDER:-runs/新数据addlelay_onnx_153m_20260807}"
BUFFER_PATH="${UFO_REWARD_BUFFER_PATH:-runs/xml有手base有imu_60m/checkpoint/buffers/train_rank_0}"
ROBOT_CONFIG="${UFO_REWARD_ROBOT_CONFIG:-configs/robots/roban_s22.yaml}"
ONNX_DEVICE="${UFO_REWARD_ONNX_DEVICE:-cpu}"
NUM_SAMPLES="${UFO_REWARD_NUM_SAMPLES:-400000}"
N_INFERENCES="${UFO_REWARD_N_INFERENCES:-1}"
MAX_WORKERS="${UFO_REWARD_MAX_WORKERS:-24}"
LATENT_FRAMES="${UFO_REWARD_LATENT_FRAMES:-5000}"
MOTION_FILE="${UFO_REWARD_MOTION:-/home/thl/Downloads/retargeter/aiming2_subject2.roban_s22.fk50.npz}"
ACTOR_PROVIDER="${UFO_REWARD_ACTOR_PROVIDER:-cpu}"
REUSE_LATENT="${UFO_REWARD_REUSE_LATENT:-true}"
TASK="${UFO_REWARD_TASK:-rotate-z--0.3-0.5}"

# Optional positional shorthand: ./run_reward_inference_onnx.sh <task>
if [[ $# -gt 0 && "${1}" != -* ]]; then
    TASK="${1}"
    shift
fi
LATENT_PATH="${UFO_REWARD_LATENT_PATH:-${MODEL_FOLDER}/reward_inference/${TASK}.npy}"

REUSE_ARGS=()
if [[ "${REUSE_LATENT}" == "true" || "${REUSE_LATENT}" == "1" ]]; then
    REUSE_ARGS+=(--reuse-existing)
fi

"${PYTHON_BIN}" -m humanoidverse.reward_inference_onnx \
    --model-folder "${MODEL_FOLDER}" \
    --buffer-path "${BUFFER_PATH}" \
    --robot-config "${ROBOT_CONFIG}" \
    --device "${ONNX_DEVICE}" \
    --num-samples "${NUM_SAMPLES}" \
    --n-inferences "${N_INFERENCES}" \
    --max-workers "${MAX_WORKERS}" \
    --latent-frames "${LATENT_FRAMES}" \
    --latent-npy-output "${LATENT_PATH}" \
    --tasks "${TASK}" \
    "${REUSE_ARGS[@]}" \
    "$@"

echo "[reward-onnx] Launching interactive MuJoCo viewer. Close the window or press Ctrl-C to stop."
echo "[reward-onnx] task=${TASK} latent=${LATENT_PATH}"
exec "${PYTHON_BIN}" -m humanoidverse.onnx_mujoco_sim \
    --model-folder "${MODEL_FOLDER}" \
    --motion "${MOTION_FILE}" \
    --latent "${LATENT_PATH}" \
    --robot-config "${ROBOT_CONFIG}" \
    --onnx-provider "${ACTOR_PROVIDER}" \
    --initial-pose default \
    --no-reference
