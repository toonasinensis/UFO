#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${UFO_PYTHON:-.venv/bin/python}"
MODEL_FOLDER="${UFO_ONNX_MODEL_FOLDER:-runs/新数据addlelay_onnx}"
#/home/thl/wt_wbc/UFO/humanoidverse/data/roban/roban_npz/fallAndGetUp1_subject1.npz
#/home/thl/wt_wbc/soma-retargeter/external_data/lafan1/isaaclab_named/multipleActions1_subject1.isaaclab.named.npz
MOTION_FILE="${UFO_ONNX_MOTION:-/home/thl/wt_wbc/soma-retargeter/external_data/lafan1/isaaclab_named/aiming1_subject1.isaaclab.named.npz}"
ROBOT_CONFIG="${UFO_ONNX_ROBOT_CONFIG:-configs/robots/roban_s22.yaml}"
LATENT_DEVICE="${UFO_ONNX_LATENT_DEVICE:-cpu}"
REBUILD_LATENT="${UFO_ONNX_REBUILD_LATENT:-true}"

# Optional positional shorthand:
#   $1 = model folder
#   $2 = motion NPZ
# All remaining arguments are forwarded to humanoidverse.onnx_mujoco_sim.
if [[ $# -gt 0 && "${1}" != -* ]]; then
    MODEL_FOLDER="${1}"
    shift
fi
if [[ $# -gt 0 && "${1}" != -* ]]; then
    MOTION_FILE="${1}"
    shift
fi

if [[ "${REBUILD_LATENT}" == "true" || "${REBUILD_LATENT}" == "1" ]]; then
    echo "[onnx-run] Generating latent z for ${MOTION_FILE} on ${LATENT_DEVICE}..."
    "${PYTHON_BIN}" -m humanoidverse.generate_onnx_latent \
        --model-folder "${MODEL_FOLDER}" \
        --motion "${MOTION_FILE}" \
        --robot-config "${ROBOT_CONFIG}" \
        --motion-id 0 \
        --device "${LATENT_DEVICE}" \
        --rebuild-motion-cache
fi

exec "${PYTHON_BIN}" -m humanoidverse.onnx_mujoco_sim \
    --model-folder "${MODEL_FOLDER}" \
    --motion "${MOTION_FILE}" \
    "$@"
