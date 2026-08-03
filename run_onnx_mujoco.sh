#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${UFO_PYTHON:-.venv/bin/python}"
MODEL_FOLDER="${UFO_ONNX_MODEL_FOLDER:-runs/new_onxx}"
MOTION_FILE="${UFO_ONNX_MOTION:-humanoidverse/data/roban/named_roban_lafan/aiming1_subject1.npz}"

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

exec "${PYTHON_BIN}" -m humanoidverse.onnx_mujoco_sim \
    --model-folder "${MODEL_FOLDER}" \
    --motion "${MOTION_FILE}" \
    "$@"
