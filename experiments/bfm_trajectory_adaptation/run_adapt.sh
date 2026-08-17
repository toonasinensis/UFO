#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${UFO_PYTHON:-.venv/bin/python}"
MODEL_FOLDER="${UFO_ADAPT_MODEL_FOLDER:-runs/新数据addlelay_onnx_153m_20260807}"
MOTION="${UFO_ADAPT_MOTION:-/home/thl/Downloads/retargeter/dance1_subject2.roban_s22.fk50.npz}"
ROBOT_CONFIG="${UFO_ADAPT_ROBOT_CONFIG:-configs/robots/roban_s22.yaml}"

exec "${PYTHON_BIN}" -m experiments.bfm_trajectory_adaptation.adapt_trajectory \
  --model-folder "${MODEL_FOLDER}" \
  --motion "${MOTION}" \
  --robot-config "${ROBOT_CONFIG}" \
  "$@"
