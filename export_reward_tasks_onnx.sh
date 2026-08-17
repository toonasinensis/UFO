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
REUSE_LATENT="${UFO_REWARD_REUSE_LATENT:-true}"
TASK_FILE="${1:-${UFO_REWARD_TASK_FILE:-${SCRIPT_DIR}/need}}"
OUTPUT_DIR="${UFO_REWARD_OUTPUT_DIR:-${MODEL_FOLDER}/reward_inference}"

if [[ ! -f "${TASK_FILE}" ]]; then
    echo "Task file does not exist: ${TASK_FILE}" >&2
    exit 1
fi

TASKS=()
while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
    task="${raw_line%%#*}"
    task="${task//$'\r'/}"
    if [[ -z "${task//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "${task}" =~ [[:space:]] ]]; then
        echo "Task names cannot contain whitespace: ${task}" >&2
        exit 1
    fi
    TASKS+=("${task}")
done < "${TASK_FILE}"

if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "No tasks found in ${TASK_FILE}" >&2
    exit 1
fi

REUSE_ARGS=()
if [[ "${REUSE_LATENT}" == "true" || "${REUSE_LATENT}" == "1" ]]; then
    REUSE_ARGS+=(--reuse-existing)
fi

echo "[reward-batch] task_file=${TASK_FILE}"
echo "[reward-batch] task_count=${#TASKS[@]}"
echo "[reward-batch] output_dir=${OUTPUT_DIR}"
printf '[reward-batch] task=%s\n' "${TASKS[@]}"

"${PYTHON_BIN}" -m humanoidverse.reward_inference_onnx \
    --model-folder "${MODEL_FOLDER}" \
    --buffer-path "${BUFFER_PATH}" \
    --robot-config "${ROBOT_CONFIG}" \
    --device "${ONNX_DEVICE}" \
    --num-samples "${NUM_SAMPLES}" \
    --n-inferences "${N_INFERENCES}" \
    --max-workers "${MAX_WORKERS}" \
    --latent-frames "${LATENT_FRAMES}" \
    --output-dir "${OUTPUT_DIR}" \
    --tasks "${TASKS[@]}" \
    "${REUSE_ARGS[@]}"

echo "[reward-batch] Completed. Task latents are in ${OUTPUT_DIR}/<task>.npy"
