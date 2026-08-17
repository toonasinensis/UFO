#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# =============================================================================
# Link-relative position 参数搜索。每一行依次为：
# name body_weight body_sigma root_pos root_rot body_rot joint_pos lin_vel ang_vel
# 所有七项 reward 权重在每一行中明确列出并合计为 1。
# =============================================================================

PYTHON_BIN=".venv/bin/python"
MODEL_FOLDER="runs/新数据addlelay_onnx_153m_20260807"
ACTOR_ONNX="runs/新数据addlelay_onnx_153m_20260807/exported/FBcprAuxModel.onnx"
BACKWARD_ONNX="runs/新数据addlelay_onnx_153m_20260807/exported/backward_encoder.onnx"
MOTION="data/selected_raw_my_data_10s/dance1_subject2.roban_s22.fk50_0010.npz"
ROBOT_CONFIG="configs/robots/roban_s22.yaml"
DEVICE="cuda:0"
ONNX_PROVIDER="cuda"
SEED=0

PARTICLES=64
ITERATIONS=40
FRAME_COUNT=500
INITIAL_SIGMA=0.20
BETA_ITERATION=0.85
BETA_HORIZON=0.90
MPPI_TEMPERATURE=0.25
TEMPORAL_SMOOTHING_WINDOW=5
DETERMINISM_TOLERANCE=0.02

ROOT_POSITION_SIGMA=0.15
ROOT_ROTATION_SIGMA=0.35
BODY_ROTATION_SIGMA=0.35
JOINT_POSITION_SIGMA=0.25
BODY_LINEAR_VELOCITY_SIGMA=0.60
BODY_ANGULAR_VELOCITY_SIGMA=1.50

OUTPUT_ROOT="experiments/bfm_trajectory_latent_adaptation/outputs/link_sweep_dance1_fk50_0010"
MAX_PARALLEL_TASKS=3

SWEEP_CONFIGS=(
  "w30_s100 0.30 0.10 0.15 0.10 0.15 0.15 0.075 0.075"
  "w30_s030 0.30 0.03 0.15 0.10 0.15 0.15 0.075 0.075"
  "w45_s050 0.45 0.05 0.117857 0.078571 0.117857 0.117857 0.058929 0.058929"
  "w45_s030 0.45 0.03 0.117857 0.078571 0.117857 0.117857 0.058929 0.058929"
  "w45_s020 0.45 0.02 0.117857 0.078571 0.117857 0.117857 0.058929 0.058929"
  "w60_s030 0.60 0.03 0.085714 0.057143 0.085714 0.085714 0.042857 0.042858"
)

mkdir -p "${OUTPUT_ROOT}"

run_one() {
  local config="$1"
  read -r name body_weight body_sigma root_pos root_rot body_rot joint_pos lin_vel ang_vel <<<"${config}"
  local output_dir="${OUTPUT_ROOT}/${name}"
  mkdir -p "${output_dir}"
  "${PYTHON_BIN}" -m experiments.bfm_trajectory_latent_adaptation.adapt_trajectory \
    --model-folder "${MODEL_FOLDER}" \
    --actor-onnx "${ACTOR_ONNX}" \
    --backward-onnx "${BACKWARD_ONNX}" \
    --motion "${MOTION}" \
    --robot-config "${ROBOT_CONFIG}" \
    --device "${DEVICE}" \
    --onnx-provider "${ONNX_PROVIDER}" \
    --seed "${SEED}" \
    --particles "${PARTICLES}" \
    --iterations "${ITERATIONS}" \
    --frame-count "${FRAME_COUNT}" \
    --sigma0 "${INITIAL_SIGMA}" \
    --beta-iteration "${BETA_ITERATION}" \
    --beta-horizon "${BETA_HORIZON}" \
    --temperature "${MPPI_TEMPERATURE}" \
    --smoothing-window "${TEMPORAL_SMOOTHING_WINDOW}" \
    --determinism-tolerance "${DETERMINISM_TOLERANCE}" \
    --root-position-weight "${root_pos}" \
    --root-rotation-weight "${root_rot}" \
    --body-position-weight "${body_weight}" \
    --body-rotation-weight "${body_rot}" \
    --joint-position-weight "${joint_pos}" \
    --body-linear-velocity-weight "${lin_vel}" \
    --body-angular-velocity-weight "${ang_vel}" \
    --root-position-sigma "${ROOT_POSITION_SIGMA}" \
    --root-rotation-sigma "${ROOT_ROTATION_SIGMA}" \
    --body-position-sigma "${body_sigma}" \
    --body-rotation-sigma "${BODY_ROTATION_SIGMA}" \
    --joint-position-sigma "${JOINT_POSITION_SIGMA}" \
    --body-linear-velocity-sigma "${BODY_LINEAR_VELOCITY_SIGMA}" \
    --body-angular-velocity-sigma "${BODY_ANGULAR_VELOCITY_SIGMA}" \
    --output-dir "${output_dir}" \
    --no-save-video \
    --tensorboard \
    >"${output_dir}/run.log" 2>&1
}

for config in "${SWEEP_CONFIGS[@]}"; do
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL_TASKS )); do
    wait -n
  done
  run_one "${config}" &
done
wait

"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

root = Path("experiments/bfm_trajectory_latent_adaptation/outputs/link_sweep_dance1_fk50_0010")
rows = []
for summary_path in sorted(root.glob("*/summary.json")):
    summary = json.loads(summary_path.read_text())
    baseline = summary["baseline_link_relative_position_error"]
    adapted = summary["adapted_link_relative_position_error"]
    rows.append({
        "run": summary_path.parent.name,
        "baseline_link_mm": 1000.0 * baseline,
        "adapted_link_mm": 1000.0 * adapted,
        "link_improvement_percent": 100.0 * (baseline - adapted) / baseline,
        "mpjpe_improvement_percent": summary["paired_mpjpe_improvement_percent"],
        "objective_improvement_percent": summary["paired_objective_improvement_percent"],
    })
rows.sort(key=lambda row: row["adapted_link_mm"])
(root / "ranking.json").write_text(json.dumps(rows, indent=2) + "\n")
for row in rows:
    print(row)
PY
