#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Explicit experiment configuration (edit values here; no environment defaults)
# ==============================================================================
PROJECT_ROOT="/home/thl/wt_wbc/UFO"
PYTHON_BIN="/home/thl/wt_wbc/UFO/.venv/bin/python"

MODEL_FOLDER="/home/thl/wt_wbc/UFO/runs/roban_action_mapping_ab_g89628672_onnx/soft_limit_bias"
MOTION_FILE="/home/thl/wt_wbc/UFO/data/test/klcb_0601_16_15_001_Skeleton_0613.npz"
ROBOT_CONFIG="/home/thl/wt_wbc/UFO/configs/robots/roban_s22.yaml"

DEVICE="cuda:0"
ONNX_PROVIDER="cuda"
PARTICLES=64
ITERATIONS=25
START_FRAME=2950
FRAME_COUNT=101
SEED=40

SIGMA0=0.35
BETA_ITERATION=0.85
BETA_HORIZON=0.90
TEMPERATURE=0.70
SMOOTHING_WINDOW=5
VALIDATION_INTERVAL=25
DETERMINISM_TOLERANCE=0.02

JOINT_LIMIT_GUARD_FRACTION=0.995
JOINT_LIMIT_SAFE_FRACTION=0.88
TRACKING_REGRESSION_FRACTION=0.15

ROOT_POSITION_WEIGHT=0.04
ROOT_ROTATION_WEIGHT=0.03
BODY_POSITION_WEIGHT=0.09
BODY_ROTATION_WEIGHT=0.04
JOINT_POSITION_WEIGHT=0.04
BODY_LINEAR_VELOCITY_WEIGHT=0.03
BODY_ANGULAR_VELOCITY_WEIGHT=0.03
JOINT_LIMIT_WEIGHT=0.70

ROOT_POSITION_SIGMA=0.15
ROOT_ROTATION_SIGMA=0.35
BODY_POSITION_SIGMA=0.10
BODY_ROTATION_SIGMA=0.35
JOINT_POSITION_SIGMA=0.25
BODY_LINEAR_VELOCITY_SIGMA=0.60
BODY_ANGULAR_VELOCITY_SIGMA=1.50
JOINT_LIMIT_SIGMA=0.50

RUN_NAME="soft_limit_bias_limit_guard"
OUTPUT_ROOT="/home/thl/wt_wbc/UFO/experiments/bfm_trajectory_latent_adaptation/outputs"
ENABLE_PROGRESS=false
ENABLE_TENSORBOARD=false
RUN_MUJOCO_VALIDATION=true
MUJOCO_ONNX_PROVIDER="cuda"
MUJOCO_TORQUE_CLIP="fixed"
MUJOCO_SIM_FPS=200
MUJOCO_DECIMATION=4
MUJOCO_MAX_STEPS=0
MUJOCO_DIAGNOSTIC_FOLDER_NAME="motion_window__soft_limit_bias"
MAX_ADAPTED_HARD_VIOLATION_RAD=0.0
# ==============================================================================

usage() {
    printf '%s\n' \
        "Usage: $0 [--help]" \
        "" \
        "Edit the explicit configuration block at the top of this file, then run it." \
        "The script optimizes one motion window and performs baseline/adapted MuJoCo validation."
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if [[ $# -ne 0 ]]; then
    usage >&2
    exit 2
fi

for required_file in \
    "$PYTHON_BIN" \
    "$MOTION_FILE" \
    "$ROBOT_CONFIG" \
    "$MODEL_FOLDER/exported/FBcprAuxModel.onnx" \
    "$MODEL_FOLDER/exported/backward_encoder.onnx"; do
    if [[ ! -e "$required_file" ]]; then
        printf 'Missing required path: %s\n' "$required_file" >&2
        exit 1
    fi
done

if [[ "$ENABLE_PROGRESS" == true ]]; then
    PROGRESS_ARG="--progress"
else
    PROGRESS_ARG="--no-progress"
fi
if [[ "$ENABLE_TENSORBOARD" == true ]]; then
    TENSORBOARD_ARG="--tensorboard"
else
    TENSORBOARD_ARG="--no-tensorboard"
fi

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$OUTPUT_ROOT/${RUN_NAME}_${RUN_TIMESTAMP}"
SLICED_MOTION="$OUTPUT_DIR/mujoco_eval/motion_window.npz"
if [[ -e "$OUTPUT_DIR" ]]; then
    printf 'Refusing to reuse existing output directory: %s\n' "$OUTPUT_DIR" >&2
    exit 1
fi

cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m experiments.bfm_trajectory_latent_adaptation.adapt_trajectory \
    --model-folder "$MODEL_FOLDER" \
    --motion "$MOTION_FILE" \
    --robot-config "$ROBOT_CONFIG" \
    --device "$DEVICE" \
    --onnx-provider "$ONNX_PROVIDER" \
    --particles "$PARTICLES" \
    --iterations "$ITERATIONS" \
    --start-frame "$START_FRAME" \
    --frame-count "$FRAME_COUNT" \
    --seed "$SEED" \
    --sigma0 "$SIGMA0" \
    --beta-iteration "$BETA_ITERATION" \
    --beta-horizon "$BETA_HORIZON" \
    --temperature "$TEMPERATURE" \
    --smoothing-window "$SMOOTHING_WINDOW" \
    --validation-interval "$VALIDATION_INTERVAL" \
    --determinism-tolerance "$DETERMINISM_TOLERANCE" \
    --require-action-mapping soft_limit_bias \
    --require-limit-safe-best \
    --joint-limit-guard-fraction "$JOINT_LIMIT_GUARD_FRACTION" \
    --joint-limit-safe-fraction "$JOINT_LIMIT_SAFE_FRACTION" \
    --selection-tracking-regression-fraction "$TRACKING_REGRESSION_FRACTION" \
    --root-position-weight "$ROOT_POSITION_WEIGHT" \
    --root-rotation-weight "$ROOT_ROTATION_WEIGHT" \
    --body-position-weight "$BODY_POSITION_WEIGHT" \
    --body-rotation-weight "$BODY_ROTATION_WEIGHT" \
    --joint-position-weight "$JOINT_POSITION_WEIGHT" \
    --body-linear-velocity-weight "$BODY_LINEAR_VELOCITY_WEIGHT" \
    --body-angular-velocity-weight "$BODY_ANGULAR_VELOCITY_WEIGHT" \
    --joint-limit-weight "$JOINT_LIMIT_WEIGHT" \
    --root-position-sigma "$ROOT_POSITION_SIGMA" \
    --root-rotation-sigma "$ROOT_ROTATION_SIGMA" \
    --body-position-sigma "$BODY_POSITION_SIGMA" \
    --body-rotation-sigma "$BODY_ROTATION_SIGMA" \
    --joint-position-sigma "$JOINT_POSITION_SIGMA" \
    --body-linear-velocity-sigma "$BODY_LINEAR_VELOCITY_SIGMA" \
    --body-angular-velocity-sigma "$BODY_ANGULAR_VELOCITY_SIGMA" \
    --joint-limit-sigma "$JOINT_LIMIT_SIGMA" \
    --history-format jsonl \
    "$PROGRESS_ARG" \
    "$TENSORBOARD_ARG" \
    --output-dir "$OUTPUT_DIR"

if [[ "$RUN_MUJOCO_VALIDATION" == true ]]; then
    "$PYTHON_BIN" -m experiments.bfm_trajectory_latent_adaptation.slice_motion_window \
        --input "$MOTION_FILE" \
        --start-frame "$START_FRAME" \
        --frame-count "$FRAME_COUNT" \
        --output "$SLICED_MOTION"

    "$PYTHON_BIN" -m humanoidverse.onnx_mujoco_sim \
        --model-folder "$MODEL_FOLDER" \
        --motion "$SLICED_MOTION" \
        --latent "$OUTPUT_DIR/baseline_z.npy" \
        --robot-config "$ROBOT_CONFIG" \
        --onnx-provider "$MUJOCO_ONNX_PROVIDER" \
        --torque-clip "$MUJOCO_TORQUE_CLIP" \
        --sim-fps "$MUJOCO_SIM_FPS" \
        --decimation "$MUJOCO_DECIMATION" \
        --max-steps "$MUJOCO_MAX_STEPS" \
        --headless \
        --no-loop \
        --initial-pose motion \
        --initial-motion-frame 0 \
        --initial-hold-seconds 0 \
        --log-every 0 \
        --diagnostics-root "$OUTPUT_DIR/mujoco_eval/baseline"

    "$PYTHON_BIN" -m humanoidverse.onnx_mujoco_sim \
        --model-folder "$MODEL_FOLDER" \
        --motion "$SLICED_MOTION" \
        --latent "$OUTPUT_DIR/validated_objective_best_z.npy" \
        --robot-config "$ROBOT_CONFIG" \
        --onnx-provider "$MUJOCO_ONNX_PROVIDER" \
        --torque-clip "$MUJOCO_TORQUE_CLIP" \
        --sim-fps "$MUJOCO_SIM_FPS" \
        --decimation "$MUJOCO_DECIMATION" \
        --max-steps "$MUJOCO_MAX_STEPS" \
        --headless \
        --no-loop \
        --initial-pose motion \
        --initial-motion-frame 0 \
        --initial-hold-seconds 0 \
        --log-every 0 \
        --diagnostics-root "$OUTPUT_DIR/mujoco_eval/adapted"

    "$PYTHON_BIN" -m experiments.bfm_trajectory_latent_adaptation.audit_mujoco_limit_pair \
        --baseline-summary "$OUTPUT_DIR/mujoco_eval/baseline/$MUJOCO_DIAGNOSTIC_FOLDER_NAME/summary.json" \
        --adapted-summary "$OUTPUT_DIR/mujoco_eval/adapted/$MUJOCO_DIAGNOSTIC_FOLDER_NAME/summary.json" \
        --baseline-timeseries "$OUTPUT_DIR/mujoco_eval/baseline/$MUJOCO_DIAGNOSTIC_FOLDER_NAME/timeseries.npz" \
        --adapted-timeseries "$OUTPUT_DIR/mujoco_eval/adapted/$MUJOCO_DIAGNOSTIC_FOLDER_NAME/timeseries.npz" \
        --max-adapted-hard-violation-rad "$MAX_ADAPTED_HARD_VIOLATION_RAD" \
        --adapted-joint-limit-guard-fraction "$JOINT_LIMIT_GUARD_FRACTION" \
        --output "$OUTPUT_DIR/mujoco_eval/comparison.json"
fi

printf 'Completed output: %s\n' "$OUTPUT_DIR"
