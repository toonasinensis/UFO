#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# =============================================================================
# Platform-triggered body-position-weight A/B. Edit this block before launch.
# The script never writes to MAIN_DIR: it snapshots its atomic checkpoint and
# creates three isolated branches with the same optimizer mean and CUDA RNG.
# =============================================================================
PYTHON_BIN=".venv/bin/python"
MAIN_DIR="experiments/bfm_trajectory_latent_adaptation/outputs/overnight_link_tuning_20260813/promoted_best_weight095"
FORK_ROOT="experiments/bfm_trajectory_latent_adaptation/outputs/overnight_link_tuning_20260813/refinements/fork_body_weight0975_1000_platform"
SOURCE_DIR="${FORK_ROOT}/source"
CONTROL_DIR="${FORK_ROOT}/body_weight095_control"
WEIGHT0975_DIR="${FORK_ROOT}/body_weight0975_trial"
WEIGHT1000_DIR="${FORK_ROOT}/body_weight1000_trial"

MODEL_FOLDER="runs/新数据addlelay_onnx_153m_20260807"
ACTOR_ONNX="runs/新数据addlelay_onnx_153m_20260807/exported/FBcprAuxModel.onnx"
BACKWARD_ONNX="runs/新数据addlelay_onnx_153m_20260807/exported/backward_encoder.onnx"
MOTION="data/selected_raw_my_data_10s/dance1_subject2.roban_s22.fk50_0010.npz"
ROBOT_CONFIG="configs/robots/roban_s22.yaml"
INITIAL_MEAN_Z="experiments/bfm_trajectory_latent_adaptation/outputs/overnight_link_tuning_20260813/initial_mean_z.npy"
DEVICE="cuda:0"
ONNX_PROVIDER="cuda"

# Keep the 50,000-step DIAL schedule; stop each branch after exactly 50 new steps.
PARTICLES=128
DIAL_TOTAL_ITERATIONS=50000
ADDITIONAL_STEPS=50
MIN_SOURCE_COMPLETED_STEPS=2150
FRAME_COUNT=500
SIGMA0=0.08
BETA_ITERATION=0.85
BETA_HORIZON=0.90
TEMPERATURE=0.10
SMOOTHING_WINDOW=5
SEED=0
DETERMINISM_TOLERANCE=0.02
MPPI_SCORE="objective"
MPPI_LINK_GUARD="baseline"
MPPI_LINK_BLEND_ALPHA=0.0

# All non-body weights retain their established proportions.
CONTROL_BODY_POSITION_WEIGHT=0.95
CONTROL_ROOT_POSITION_WEIGHT=0.010714250
CONTROL_ROOT_ROTATION_WEIGHT=0.007142875
CONTROL_BODY_ROTATION_WEIGHT=0.010714250
CONTROL_JOINT_POSITION_WEIGHT=0.010714250
CONTROL_BODY_LINEAR_VELOCITY_WEIGHT=0.005357125
CONTROL_BODY_ANGULAR_VELOCITY_WEIGHT=0.005357250

WEIGHT0975_BODY_POSITION_WEIGHT=0.975
WEIGHT0975_ROOT_POSITION_WEIGHT=0.005357125
WEIGHT0975_ROOT_ROTATION_WEIGHT=0.0035714375
WEIGHT0975_BODY_ROTATION_WEIGHT=0.005357125
WEIGHT0975_JOINT_POSITION_WEIGHT=0.005357125
WEIGHT0975_BODY_LINEAR_VELOCITY_WEIGHT=0.0026785625
WEIGHT0975_BODY_ANGULAR_VELOCITY_WEIGHT=0.002678625

WEIGHT1000_BODY_POSITION_WEIGHT=1.0
WEIGHT1000_ROOT_POSITION_WEIGHT=0.0
WEIGHT1000_ROOT_ROTATION_WEIGHT=0.0
WEIGHT1000_BODY_ROTATION_WEIGHT=0.0
WEIGHT1000_JOINT_POSITION_WEIGHT=0.0
WEIGHT1000_BODY_LINEAR_VELOCITY_WEIGHT=0.0
WEIGHT1000_BODY_ANGULAR_VELOCITY_WEIGHT=0.0

ROOT_POSITION_SIGMA=0.15
ROOT_ROTATION_SIGMA=0.35
BODY_POSITION_SIGMA=0.020
BODY_ROTATION_SIGMA=0.35
JOINT_POSITION_SIGMA=0.25
BODY_LINEAR_VELOCITY_SIGMA=0.60
BODY_ANGULAR_VELOCITY_SIGMA=1.50

# sequential is the cleaner comparison; parallel is faster but shares one GPU.
RUN_MODE="sequential"          # sequential or parallel
ENABLE_TENSORBOARD=true
SAVE_VIDEO=false
VIDEO_INTERVAL=0
VALIDATION_INTERVAL=0          # final validation still runs: 64 baseline + 64 candidate.
HISTORY_FORMAT="jsonl"
SHOW_PROGRESS=false
VIDEO_BEST_KIND="link"
RENDER_SIZE=480
FPS=50

# Simultaneous two-trial decision thresholds against the weight-0.95 control.
FAMILY_ALPHA=0.05
MIN_LINK_GAIN_MM=0.015
MAX_MPJPE_REGRESSION_UPPER_MM=5.0
MAX_ROOT_REGRESSION_UPPER_MM=5.0
MIN_FIXED_WEIGHT095_OBJECTIVE_DELTA=-0.001

for required in \
  "${MAIN_DIR}/checkpoint.pt" \
  "${MAIN_DIR}/run_config.json" \
  "${MODEL_FOLDER}/config.json" \
  "${ACTOR_ONNX}" \
  "${BACKWARD_ONNX}" \
  "${MOTION}" \
  "${ROBOT_CONFIG}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required file is missing: ${required}" >&2
    exit 2
  fi
done

if [[ "${RUN_MODE}" != "sequential" && "${RUN_MODE}" != "parallel" ]]; then
  echo "RUN_MODE must be sequential or parallel, got: ${RUN_MODE}" >&2
  exit 2
fi
if [[ -e "${FORK_ROOT}" ]]; then
  echo "Refusing to overwrite existing fork root: ${FORK_ROOT}" >&2
  exit 3
fi

# Validate explicit weights before creating any output.
"${PYTHON_BIN}" - \
  "${CONTROL_BODY_POSITION_WEIGHT}" "${CONTROL_ROOT_POSITION_WEIGHT}" \
  "${CONTROL_ROOT_ROTATION_WEIGHT}" "${CONTROL_BODY_ROTATION_WEIGHT}" \
  "${CONTROL_JOINT_POSITION_WEIGHT}" "${CONTROL_BODY_LINEAR_VELOCITY_WEIGHT}" \
  "${CONTROL_BODY_ANGULAR_VELOCITY_WEIGHT}" \
  "${WEIGHT0975_BODY_POSITION_WEIGHT}" "${WEIGHT0975_ROOT_POSITION_WEIGHT}" \
  "${WEIGHT0975_ROOT_ROTATION_WEIGHT}" "${WEIGHT0975_BODY_ROTATION_WEIGHT}" \
  "${WEIGHT0975_JOINT_POSITION_WEIGHT}" "${WEIGHT0975_BODY_LINEAR_VELOCITY_WEIGHT}" \
  "${WEIGHT0975_BODY_ANGULAR_VELOCITY_WEIGHT}" \
  "${WEIGHT1000_BODY_POSITION_WEIGHT}" "${WEIGHT1000_ROOT_POSITION_WEIGHT}" \
  "${WEIGHT1000_ROOT_ROTATION_WEIGHT}" "${WEIGHT1000_BODY_ROTATION_WEIGHT}" \
  "${WEIGHT1000_JOINT_POSITION_WEIGHT}" "${WEIGHT1000_BODY_LINEAR_VELOCITY_WEIGHT}" \
  "${WEIGHT1000_BODY_ANGULAR_VELOCITY_WEIGHT}" <<'PY'
import math
import sys

values = [float(value) for value in sys.argv[1:]]
for offset, name in ((0, "control"), (7, "weight0975"), (14, "weight1000")):
    total = sum(values[offset : offset + 7])
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise SystemExit(f"{name} weights sum to {total}, expected 1.0")
PY

# checkpoint.pt is written by adapt_trajectory through os.replace. cp therefore
# opens either the complete old inode or complete new inode, never a partial file.
mkdir -p "${SOURCE_DIR}"
cp --reflink=auto -- "${MAIN_DIR}/checkpoint.pt" "${SOURCE_DIR}/checkpoint.pt"
cp --reflink=auto -- "${MAIN_DIR}/run_config.json" "${SOURCE_DIR}/run_config.json"

SOURCE_COMPLETED_STEPS="$(
  "${PYTHON_BIN}" - \
    "${SOURCE_DIR}/checkpoint.pt" \
    "${SOURCE_DIR}/run_config.json" \
    "${MIN_SOURCE_COMPLETED_STEPS}" \
    "${DIAL_TOTAL_ITERATIONS}" \
    "${CONTROL_BODY_POSITION_WEIGHT}" <<'PY'
import json
import math
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
config = json.loads(open(sys.argv[2], encoding="utf-8").read())
minimum_step = int(sys.argv[3])
schedule_steps = int(sys.argv[4])
expected_weight = float(sys.argv[5])
completed = int(checkpoint["iteration"]) + 1
if completed < minimum_step:
    raise SystemExit(f"source is only at step {completed}, need >= {minimum_step}")
if "rng_state" not in checkpoint:
    raise SystemExit("source checkpoint has no rng_state")
if not bool(torch.isfinite(checkpoint["mean"]).all()):
    raise SystemExit("source optimizer mean contains NaN/Inf")
if int(config["iterations"]) != schedule_steps:
    raise SystemExit("source DIAL schedule does not match configured schedule")
if not math.isclose(float(config["body_position_weight"]), expected_weight):
    raise SystemExit("source is not the promoted weight-0.95 configuration")
print(completed)
PY
)"
TARGET_COMPLETED_STEPS=$(( SOURCE_COMPLETED_STEPS + ADDITIONAL_STEPS ))
if (( TARGET_COMPLETED_STEPS > DIAL_TOTAL_ITERATIONS )); then
  echo "Target ${TARGET_COMPLETED_STEPS} exceeds DIAL schedule ${DIAL_TOTAL_ITERATIONS}" >&2
  exit 4
fi

CHECKPOINT_SHA256="$(sha256sum "${SOURCE_DIR}/checkpoint.pt" | awk '{print $1}')"
"${PYTHON_BIN}" - \
  "${SOURCE_DIR}/snapshot_manifest.json" \
  "${SOURCE_COMPLETED_STEPS}" \
  "${TARGET_COMPLETED_STEPS}" \
  "${CHECKPOINT_SHA256}" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
value = {
    "created_at": datetime.now().astimezone().isoformat(),
    "checkpoint_iteration": int(sys.argv[2]) - 1,
    "completed_steps": int(sys.argv[2]),
    "target_completed_steps": int(sys.argv[3]),
    "checkpoint_sha256": sys.argv[4],
    "same_checkpoint_and_rng_for_all_branches": True,
}
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY

clone_source() {
  local destination="$1"
  mkdir -p "${destination}"
  cp --reflink=auto -- "${SOURCE_DIR}/checkpoint.pt" "${destination}/checkpoint.pt"
  cp --reflink=auto -- "${SOURCE_DIR}/run_config.json" "${destination}/run_config.json"
}

clone_source "${CONTROL_DIR}"
clone_source "${WEIGHT0975_DIR}"
clone_source "${WEIGHT1000_DIR}"
for branch_dir in "${CONTROL_DIR}" "${WEIGHT0975_DIR}" "${WEIGHT1000_DIR}"; do
  branch_sha256="$(sha256sum "${branch_dir}/checkpoint.pt" | awk '{print $1}')"
  if [[ "${branch_sha256}" != "${CHECKPOINT_SHA256}" ]]; then
    echo "Checkpoint hash mismatch after cloning ${branch_dir}" >&2
    exit 4
  fi
done

SAVE_VIDEO_ARG="--no-save-video"
if [[ "${SAVE_VIDEO}" == "true" ]]; then SAVE_VIDEO_ARG="--save-video"; fi
TENSORBOARD_ARG="--no-tensorboard"
if [[ "${ENABLE_TENSORBOARD}" == "true" ]]; then TENSORBOARD_ARG="--tensorboard"; fi
PROGRESS_ARG="--no-progress"
if [[ "${SHOW_PROGRESS}" == "true" ]]; then PROGRESS_ARG="--progress"; fi

run_branch() {
  local output_dir="$1"
  local reset_objective_best="$2"
  local body_position_weight="$3"
  local root_position_weight="$4"
  local root_rotation_weight="$5"
  local body_rotation_weight="$6"
  local joint_position_weight="$7"
  local body_linear_velocity_weight="$8"
  local body_angular_velocity_weight="$9"
  local reset_args=()
  if [[ "${reset_objective_best}" == "true" ]]; then
    reset_args=(--resume-reset-objective-best)
  fi

  "${PYTHON_BIN}" -m experiments.bfm_trajectory_latent_adaptation.adapt_trajectory \
    --model-folder "${MODEL_FOLDER}" \
    --actor-onnx "${ACTOR_ONNX}" \
    --backward-onnx "${BACKWARD_ONNX}" \
    --motion "${MOTION}" \
    --robot-config "${ROBOT_CONFIG}" \
    --initial-mean-z "${INITIAL_MEAN_Z}" \
    --device "${DEVICE}" \
    --onnx-provider "${ONNX_PROVIDER}" \
    --particles "${PARTICLES}" \
    --iterations "${DIAL_TOTAL_ITERATIONS}" \
    --frame-count "${FRAME_COUNT}" \
    --sigma0 "${SIGMA0}" \
    --beta-iteration "${BETA_ITERATION}" \
    --beta-horizon "${BETA_HORIZON}" \
    --temperature "${TEMPERATURE}" \
    --mppi-score "${MPPI_SCORE}" \
    --mppi-link-guard "${MPPI_LINK_GUARD}" \
    --mppi-link-blend-alpha "${MPPI_LINK_BLEND_ALPHA}" \
    --smoothing-window "${SMOOTHING_WINDOW}" \
    --seed "${SEED}" \
    --determinism-tolerance "${DETERMINISM_TOLERANCE}" \
    --root-position-weight "${root_position_weight}" \
    --root-rotation-weight "${root_rotation_weight}" \
    --body-position-weight "${body_position_weight}" \
    --body-rotation-weight "${body_rotation_weight}" \
    --joint-position-weight "${joint_position_weight}" \
    --body-linear-velocity-weight "${body_linear_velocity_weight}" \
    --body-angular-velocity-weight "${body_angular_velocity_weight}" \
    --root-position-sigma "${ROOT_POSITION_SIGMA}" \
    --root-rotation-sigma "${ROOT_ROTATION_SIGMA}" \
    --body-position-sigma "${BODY_POSITION_SIGMA}" \
    --body-rotation-sigma "${BODY_ROTATION_SIGMA}" \
    --joint-position-sigma "${JOINT_POSITION_SIGMA}" \
    --body-linear-velocity-sigma "${BODY_LINEAR_VELOCITY_SIGMA}" \
    --body-angular-velocity-sigma "${BODY_ANGULAR_VELOCITY_SIGMA}" \
    --resume-dir "${output_dir}" \
    --resume-total-iterations "${DIAL_TOTAL_ITERATIONS}" \
    --resume-use-cli-config \
    "${reset_args[@]}" \
    --stop-after-iteration "${TARGET_COMPLETED_STEPS}" \
    "${SAVE_VIDEO_ARG}" \
    --video-best-kind "${VIDEO_BEST_KIND}" \
    --video-interval "${VIDEO_INTERVAL}" \
    --validation-interval "${VALIDATION_INTERVAL}" \
    "${TENSORBOARD_ARG}" \
    --history-format "${HISTORY_FORMAT}" \
    "${PROGRESS_ARG}" \
    --render-size "${RENDER_SIZE}" \
    --fps "${FPS}" \
    >"${output_dir}/run.log" 2>&1
}

run_control() {
  run_branch "${CONTROL_DIR}" false \
    "${CONTROL_BODY_POSITION_WEIGHT}" \
    "${CONTROL_ROOT_POSITION_WEIGHT}" \
    "${CONTROL_ROOT_ROTATION_WEIGHT}" \
    "${CONTROL_BODY_ROTATION_WEIGHT}" \
    "${CONTROL_JOINT_POSITION_WEIGHT}" \
    "${CONTROL_BODY_LINEAR_VELOCITY_WEIGHT}" \
    "${CONTROL_BODY_ANGULAR_VELOCITY_WEIGHT}"
}

run_weight0975() {
  run_branch "${WEIGHT0975_DIR}" true \
    "${WEIGHT0975_BODY_POSITION_WEIGHT}" \
    "${WEIGHT0975_ROOT_POSITION_WEIGHT}" \
    "${WEIGHT0975_ROOT_ROTATION_WEIGHT}" \
    "${WEIGHT0975_BODY_ROTATION_WEIGHT}" \
    "${WEIGHT0975_JOINT_POSITION_WEIGHT}" \
    "${WEIGHT0975_BODY_LINEAR_VELOCITY_WEIGHT}" \
    "${WEIGHT0975_BODY_ANGULAR_VELOCITY_WEIGHT}"
}

run_weight1000() {
  run_branch "${WEIGHT1000_DIR}" true \
    "${WEIGHT1000_BODY_POSITION_WEIGHT}" \
    "${WEIGHT1000_ROOT_POSITION_WEIGHT}" \
    "${WEIGHT1000_ROOT_ROTATION_WEIGHT}" \
    "${WEIGHT1000_BODY_ROTATION_WEIGHT}" \
    "${WEIGHT1000_JOINT_POSITION_WEIGHT}" \
    "${WEIGHT1000_BODY_LINEAR_VELOCITY_WEIGHT}" \
    "${WEIGHT1000_BODY_ANGULAR_VELOCITY_WEIGHT}"
}

if [[ "${RUN_MODE}" == "parallel" ]]; then
  run_control & control_pid=$!
  run_weight0975 & weight0975_pid=$!
  run_weight1000 & weight1000_pid=$!
  status=0
  wait "${control_pid}" || status=1
  wait "${weight0975_pid}" || status=1
  wait "${weight1000_pid}" || status=1
  if (( status != 0 )); then
    echo "At least one A/B branch failed; inspect per-branch run.log files" >&2
    exit 5
  fi
else
  run_control
  run_weight0975
  run_weight1000
fi

# Compare independently validated link candidates under one fixed weight-0.95
# objective. Two trial-vs-control intervals use a Bonferroni family correction.
"${PYTHON_BIN}" - \
  "${CONTROL_DIR}" \
  "${WEIGHT0975_DIR}" \
  "${WEIGHT1000_DIR}" \
  "${FORK_ROOT}/comparison.json" \
  "${FAMILY_ALPHA}" \
  "${MIN_LINK_GAIN_MM}" \
  "${MAX_MPJPE_REGRESSION_UPPER_MM}" \
  "${MAX_ROOT_REGRESSION_UPPER_MM}" \
  "${MIN_FIXED_WEIGHT095_OBJECTIVE_DELTA}" \
  "${SOURCE_COMPLETED_STEPS}" \
  "${TARGET_COMPLETED_STEPS}" <<'PY'
import json
import math
import os
import sys
from pathlib import Path
from statistics import NormalDist

control_dir, trial0975_dir, trial1000_dir = map(Path, sys.argv[1:4])
output_path = Path(sys.argv[4])
family_alpha = float(sys.argv[5])
minimum_link_gain_mm = float(sys.argv[6])
maximum_mpjpe_upper_mm = float(sys.argv[7])
maximum_root_upper_mm = float(sys.argv[8])
minimum_objective_delta = float(sys.argv[9])
source_step = int(sys.argv[10])
target_step = int(sys.argv[11])

fixed_weights = {
    "root_position": 0.010714250,
    "root_rotation": 0.007142875,
    "body_position": 0.95,
    "body_rotation": 0.010714250,
    "joint_position": 0.010714250,
    "body_linear_velocity": 0.005357125,
    "body_angular_velocity": 0.005357250,
}

def load_result(path):
    value = json.loads((path / "metrics_link_best.json").read_text(encoding="utf-8"))
    counts = value["validation_counts"]
    if counts != {"baseline": 64, "adapted": 64}:
        raise SystemExit(f"{path} validation counts are {counts}, expected 64/64")
    return value

def fixed_objective(result):
    return sum(fixed_weights[name] * result["rewards"][name] for name in fixed_weights)

control = load_result(control_dir)
trials = {
    "body_weight0975_trial": load_result(trial0975_dir),
    "body_weight1000_trial": load_result(trial1000_dir),
}
n = 64
trial_count = len(trials)
z_score = NormalDist().inv_cdf(1.0 - family_alpha / (2.0 * trial_count))
control_fixed_objective = fixed_objective(control)

rows = []
for name, trial in trials.items():
    control_metrics = control["metrics"]
    trial_metrics = trial["metrics"]
    control_std = control["population_std"]
    trial_std = trial["population_std"]

    def difference_radius(metric):
        return z_score * math.sqrt(
            control_std[metric] ** 2 / n + trial_std[metric] ** 2 / n
        )

    link_gain_mm = 1000.0 * (
        control_metrics["mean_body_position_error"]
        - trial_metrics["mean_body_position_error"]
    )
    link_radius_mm = 1000.0 * difference_radius("mean_body_position_error")
    mpjpe_regression_mm = 1000.0 * (
        trial_metrics["mpjpe"] - control_metrics["mpjpe"]
    )
    mpjpe_upper_mm = mpjpe_regression_mm + 1000.0 * difference_radius("mpjpe")
    root_regression_mm = 1000.0 * (
        trial_metrics["mean_root_position_error"]
        - control_metrics["mean_root_position_error"]
    )
    root_upper_mm = root_regression_mm + 1000.0 * difference_radius(
        "mean_root_position_error"
    )
    objective_delta = fixed_objective(trial) - control_fixed_objective
    passed = (
        link_gain_mm >= max(minimum_link_gain_mm, link_radius_mm)
        and mpjpe_upper_mm <= maximum_mpjpe_upper_mm
        and root_upper_mm <= maximum_root_upper_mm
        and objective_delta >= minimum_objective_delta
    )
    rows.append(
        {
            "name": name,
            "validated_link_mm": 1000.0
            * trial_metrics["mean_body_position_error"],
            "validated_link_gain_mm": link_gain_mm,
            "simultaneous_link_radius_mm": link_radius_mm,
            "required_link_gain_mm": max(minimum_link_gain_mm, link_radius_mm),
            "mpjpe_mm": 1000.0 * trial_metrics["mpjpe"],
            "mpjpe_regression_mm": mpjpe_regression_mm,
            "mpjpe_regression_upper_mm": mpjpe_upper_mm,
            "root_position_mm": 1000.0
            * trial_metrics["mean_root_position_error"],
            "root_position_regression_mm": root_regression_mm,
            "root_position_regression_upper_mm": root_upper_mm,
            "fixed_weight095_objective": fixed_objective(trial),
            "fixed_weight095_objective_delta": objective_delta,
            "passed": passed,
        }
    )

passed = [row for row in rows if row["passed"]]
winner = "body_weight095_control"
if len(passed) == 1:
    winner = passed[0]["name"]
elif len(passed) == 2:
    first, second = passed
    result_by_name = {
        "body_weight0975_trial": trials["body_weight0975_trial"],
        "body_weight1000_trial": trials["body_weight1000_trial"],
    }
    pair_radius_mm = z_score * 1000.0 * math.sqrt(
        result_by_name[first["name"]]["population_std"][
            "mean_body_position_error"
        ] ** 2 / n
        + result_by_name[second["name"]]["population_std"][
            "mean_body_position_error"
        ] ** 2 / n
    )
    if abs(first["validated_link_mm"] - second["validated_link_mm"]) <= pair_radius_mm:
        winner = "body_weight0975_trial"
    else:
        winner = min(passed, key=lambda row: row["validated_link_mm"])["name"]

value = {
    "fork_completed_steps": source_step,
    "comparison_completed_steps": target_step,
    "same_checkpoint_and_rng": True,
    "validation_count_per_candidate": n,
    "simultaneous_family_alpha": family_alpha,
    "simultaneous_z_score": z_score,
    "control_validated_link_mm": 1000.0
    * control["metrics"]["mean_body_position_error"],
    "control_mpjpe_mm": 1000.0 * control["metrics"]["mpjpe"],
    "control_root_position_mm": 1000.0
    * control["metrics"]["mean_root_position_error"],
    "control_fixed_weight095_objective": control_fixed_objective,
    "thresholds": {
        "minimum_link_gain_mm": minimum_link_gain_mm,
        "maximum_mpjpe_regression_upper_mm": maximum_mpjpe_upper_mm,
        "maximum_root_regression_upper_mm": maximum_root_upper_mm,
        "minimum_fixed_weight095_objective_delta": minimum_objective_delta,
    },
    "trials": rows,
    "winner": winner,
}
temporary = output_path.with_name(f".{output_path.name}.tmp")
temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output_path)
print(json.dumps(value, indent=2))
PY
