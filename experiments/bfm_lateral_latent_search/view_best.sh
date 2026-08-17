#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${UFO_PYTHON:-.venv/bin/python}"
MODEL_FOLDER="${UFO_LATENT_SEARCH_MODEL_FOLDER:-runs/新数据addlelay_onnx_153m_20260807}"
ROBOT_CONFIG="${UFO_LATENT_SEARCH_ROBOT_CONFIG:-configs/robots/roban_s22.yaml}"
MOTION_FILE="${UFO_LATENT_SEARCH_MOTION:-/home/thl/Downloads/retargeter/aiming2_subject2.roban_s22.fk50.npz}"
OUTPUT_DIR="${1:-experiments/bfm_lateral_latent_search/outputs/move-ego-90-0.5_refined_onnx}"

exec "${PYTHON_BIN}" -m humanoidverse.onnx_mujoco_sim \
  --model-folder "${MODEL_FOLDER}" \
  --motion "${MOTION_FILE}" \
  --latent "/home/thl/wt_wbc/UFO/experiments/bfm_lateral_latent_search/outputs/move-ego-180-0.5_cem_foot_flat_20260812_153046/best_z_rollout.npy" \
  --robot-config "${ROBOT_CONFIG}" \
  --onnx-provider cpu \
  --initial-pose default \
  --no-reference
