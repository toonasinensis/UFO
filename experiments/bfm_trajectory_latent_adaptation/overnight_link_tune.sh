#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# =============================================================================
# 通宵 local link-relative position 调参。直接修改本配置块即可复跑。
# =============================================================================

PYTHON_BIN=".venv/bin/python"
STOP_AT="2026-08-14 08:00:00"
FINAL_PHASE_AT="2026-08-13 23:30:00"
FINAL_RUN_STOP_AT="2026-08-14 07:45:00"
MODEL_FOLDER="runs/新数据addlelay_onnx_153m_20260807"
ACTOR_ONNX="runs/新数据addlelay_onnx_153m_20260807/exported/FBcprAuxModel.onnx"
BACKWARD_ONNX="runs/新数据addlelay_onnx_153m_20260807/exported/backward_encoder.onnx"
MOTION="data/selected_raw_my_data_10s/dance1_subject2.roban_s22.fk50_0010.npz"
ROBOT_CONFIG="configs/robots/roban_s22.yaml"
OUTPUT_ROOT="experiments/bfm_trajectory_latent_adaptation/outputs/overnight_link_tuning_20260813"
INITIAL_MEAN_Z_SOURCE="experiments/bfm_trajectory_latent_adaptation/outputs/dance1_fk50_0010_w60_s030_iter50000/link_best_z.npy"
INITIAL_MEAN_Z="${OUTPUT_ROOT}/initial_mean_z.npy"
DEVICE="cuda:0"
ONNX_PROVIDER="cuda"

SCREEN_PARTICLES=64
SCREEN_ITERATIONS=50
FINAL_PARTICLES=128
FINAL_TOTAL_ITERATIONS=50000
VALIDATION_INTERVAL=0            # 每个短试结束时统一做 paired validation。
VIDEO_INTERVAL=50
FRAME_COUNT=500
SEED=0
REPLICATION_SEED=1
REPLICATION_COUNT=2
DETERMINISM_TOLERANCE=0.02
BETA_ITERATION=0.85
BETA_HORIZON=0.90

ROOT_POSITION_SIGMA=0.15
ROOT_ROTATION_SIGMA=0.35
BODY_ROTATION_SIGMA=0.35
JOINT_POSITION_SIGMA=0.25
BODY_LINEAR_VELOCITY_SIGMA=0.60
BODY_ANGULAR_VELOCITY_SIGMA=1.50

# name body_weight body_sigma sigma0 temperature smoothing_window seed
# 先筛 MPPI；前两组的实测表明 k5 优于 k1，且 T=0.25 后期 ESS 偏高，
# 因此保留一个 T=0.40/k1 对照，并把其余预算放到更有辨别力的 T=0.10–0.15。
MPPI_CONFIGS=(
  "w60_rs030_n040_t025_k1 0.60 0.030 0.04 0.25 1 0"
  "w60_rs030_n040_t025_k5 0.60 0.030 0.04 0.25 5 0"
  "w60_rs030_n040_t040_k1 0.60 0.030 0.04 0.40 1 0"
  "w60_rs030_n040_t015_k5 0.60 0.030 0.04 0.15 5 0"
  "w60_rs030_n040_t010_k5 0.60 0.030 0.04 0.10 5 0"
  "w60_rs030_n080_t025_k5 0.60 0.030 0.08 0.25 5 0"
  "w60_rs030_n080_t015_k5 0.60 0.030 0.08 0.15 5 0"
  "w60_rs030_n080_t010_k5 0.60 0.030 0.08 0.10 5 0"
  "w60_rs030_n120_t025_k5 0.60 0.030 0.12 0.25 5 0"
  "w60_rs030_n120_t015_k5 0.60 0.030 0.12 0.15 5 0"
)

# name body_weight body_sigma；运行时自动继承 MPPI 筛选冠军的超参数。
REWARD_CONFIGS=(
  "reward_w60_rs020 0.60 0.020"
  "reward_w60_rs025 0.60 0.025"
  "reward_w70_rs020 0.70 0.020"
  "reward_w70_rs025 0.70 0.025"
  "reward_w70_rs030 0.70 0.030"
  "reward_w80_rs020 0.80 0.020"
  "reward_w80_rs025 0.80 0.025"
  "reward_w80_rs030 0.80 0.030"
)

mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/scheduler.lock"
if ! flock -n 9; then
  echo "Another overnight scheduler already holds ${OUTPUT_ROOT}/scheduler.lock" >&2
  exit 3
fi

"${PYTHON_BIN}" - "${INITIAL_MEAN_Z_SOURCE}" "${INITIAL_MEAN_Z}" <<'PY'
import os, sys, tempfile
from pathlib import Path
import numpy as np

source = Path(sys.argv[1])
target = Path(sys.argv[2])
if not target.exists():
    values = np.load(source)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    with os.fdopen(fd, "wb") as stream:
        np.save(stream, values)
    os.replace(temporary, target)
PY

stop_epoch="$(date -d "${STOP_AT}" +%s)"
final_phase_epoch="$(date -d "${FINAL_PHASE_AT}" +%s)"
final_run_stop_epoch="$(date -d "${FINAL_RUN_STOP_AT}" +%s)"

write_status() {
  local phase="$1"
  local detail="$2"
  "${PYTHON_BIN}" - "${OUTPUT_ROOT}/status.json" "${phase}" "${detail}" <<'PY'
import json, os, sys, tempfile
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
value = {"updated_at": datetime.now().astimezone().isoformat(), "phase": sys.argv[2], "detail": sys.argv[3]}
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(value, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
os.replace(temporary, path)
PY
}

remaining_seconds() {
  local now
  now="$(date +%s)"
  echo $(( stop_epoch - now ))
}

final_run_remaining_seconds() {
  local now
  now="$(date +%s)"
  echo $(( final_run_stop_epoch - now ))
}

weights_for_body() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import sys
w = float(sys.argv[1])
r = (1.0 - w) / 0.40
values = [0.085714*r, 0.057143*r, 0.085714*r, 0.085714*r, 0.042857*r]
six = sum(values)
values.append(1.0 - w - six)
print(" ".join(f"{value:.9f}" for value in values))
PY
}

run_fresh_trial() {
  local config="$1"
  local name body_weight body_sigma sigma0 temperature smoothing trial_seed
  read -r name body_weight body_sigma sigma0 temperature smoothing trial_seed <<<"${config}"
  if [[ -z "${trial_seed}" ]]; then
    echo "Trial config must explicitly include a seed: ${config}" >&2
    return 12
  fi
  local output_dir="${OUTPUT_ROOT}/${name}"
  local summary_path="${output_dir}/summary.json"
  if [[ -f "${summary_path}" ]]; then
    write_status "screen" "skip completed ${name}"
    return
  fi
  if [[ -e "${output_dir}" ]]; then
    write_status "screen" "incomplete directory ${name}; leaving it for inspection"
    return
  fi
  if (( $(date +%s) >= final_phase_epoch )); then
    return 10
  fi
  mkdir -p "${output_dir}"
  local root_pos root_rot body_rot joint_pos lin_vel ang_vel
  read -r root_pos root_rot body_rot joint_pos lin_vel ang_vel <<<"$(weights_for_body "${body_weight}")"
  write_status "screen" "running ${name}"
  set +e
  "${PYTHON_BIN}" -m experiments.bfm_trajectory_latent_adaptation.adapt_trajectory \
    --model-folder "${MODEL_FOLDER}" \
    --actor-onnx "${ACTOR_ONNX}" \
    --backward-onnx "${BACKWARD_ONNX}" \
    --motion "${MOTION}" \
    --robot-config "${ROBOT_CONFIG}" \
    --initial-mean-z "${INITIAL_MEAN_Z}" \
    --output-dir "${output_dir}" \
    --device "${DEVICE}" \
    --onnx-provider "${ONNX_PROVIDER}" \
    --particles "${SCREEN_PARTICLES}" \
    --iterations "${SCREEN_ITERATIONS}" \
    --frame-count "${FRAME_COUNT}" \
    --sigma0 "${sigma0}" \
    --beta-iteration "${BETA_ITERATION}" \
    --beta-horizon "${BETA_HORIZON}" \
    --temperature "${temperature}" \
    --smoothing-window "${smoothing}" \
    --seed "${trial_seed}" \
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
    --no-save-video \
    --video-interval 0 \
    --video-best-kind link \
    --validation-interval "${VALIDATION_INTERVAL}" \
    --tensorboard \
    --history-format jsonl \
    --no-progress \
    >"${output_dir}/run.log" 2>&1
  local status=$?
  set -e
  if [[ ${status} -ne 0 ]]; then
    write_status "screen" "${name} failed with exit ${status}; continuing"
  fi
  return 0
}

rank_trials() {
  "${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import json, os, sys, tempfile
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/summary.json")):
    if path.parent.name == "final_best" or path.parent.name.startswith("."):
        continue
    summary = json.loads(path.read_text())
    link_metrics_path = path.parent / "metrics_link_best.json"
    if not link_metrics_path.exists():
        continue
    link_details = json.loads(link_metrics_path.read_text())
    adapted = link_details["metrics"]
    adapted_std = link_details["population_std"]
    paired_baseline = link_details["paired_baseline_metrics"]
    link = adapted.get("mean_body_position_error")
    mpjpe = adapted.get("mpjpe")
    objective = adapted.get("objective")
    if link is None:
        continue
    rows.append({
        "run": path.parent.name,
        "output_dir": str(path.parent.resolve()),
        "baseline_link_mm": 1000.0 * float(
            paired_baseline["mean_body_position_error"]
        ),
        "validated_link_mm": 1000.0 * float(link),
        "validated_link_std_mm": 1000.0 * float(adapted_std["mean_body_position_error"]),
        "baseline_mpjpe_mm": 1000.0 * float(paired_baseline["mpjpe"]),
        "validated_mpjpe_mm": 1000.0 * float(mpjpe),
        "baseline_objective": float(paired_baseline["objective"]),
        "validated_objective": float(objective),
        "eligible": (
            float(link) < float(paired_baseline["mean_body_position_error"])
            and float(mpjpe) < float(paired_baseline["mpjpe"])
            and float(objective) >= float(paired_baseline["objective"])
        ),
    })
rows.sort(key=lambda row: (not row["eligible"], row["validated_link_mm"]))
for filename, value in (("ranking.json", rows), ("latest_best.json", rows[0] if rows else None)):
    path = root / filename
    fd, temporary = tempfile.mkstemp(prefix=f".{filename}.", dir=root)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)
if rows:
    print(rows[0]["output_dir"])
PY
}

select_mppi_winner() {
  "${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import json, re, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = json.loads((root / "ranking.json").read_text())
for row in rows:
    if row["eligible"] and re.fullmatch(r"w60_rs030_n\d+_t\d+_k\d+", row["run"]):
        print(row["output_dir"])
        break
PY
}

prepare_replication_configs() {
  "${PYTHON_BIN}" - \
    "${OUTPUT_ROOT}" \
    "${REPLICATION_COUNT}" \
    "${REPLICATION_SEED}" <<'PY'
import json, os, sys, tempfile
from pathlib import Path

root = Path(sys.argv[1])
count = int(sys.argv[2])
seed = int(sys.argv[3])
plan_path = root / "replication_plan.json"
if plan_path.exists():
    base_runs = json.loads(plan_path.read_text())["base_runs"]
else:
    rows = json.loads((root / "ranking.json").read_text())
    base_runs = [
        row["run"] for row in rows
        if row["eligible"] and not row["run"].startswith("seed")
    ][:count]
    value = {"base_runs": base_runs, "replication_seed": seed}
    fd, temporary = tempfile.mkstemp(prefix=".replication_plan.", dir=root)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, plan_path)
for base_run in base_runs:
    config = json.loads((root / base_run / "run_config.json").read_text())
    print(
        f"seed{seed}__{base_run} "
        f"{config['body_position_weight']} {config['body_position_sigma']} "
        f"{config['sigma0']} {config['temperature']} "
        f"{config['smoothing_window']} {seed}"
    )
PY
}

select_robust_winner() {
  "${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import json, os, sys, tempfile
from pathlib import Path

root = Path(sys.argv[1])
selection_path = root / "selection.json"
if selection_path.exists():
    selection = json.loads(selection_path.read_text())
    print(selection["winner"]["output_dir"])
    raise SystemExit
ranking = json.loads((root / "ranking.json").read_text())
by_run = {row["run"]: row for row in ranking}
plan_path = root / "replication_plan.json"
groups = []
incomplete_groups = []
if plan_path.exists():
    plan = json.loads(plan_path.read_text())
    seed = int(plan["replication_seed"])
    for base_run in plan["base_runs"]:
        names = (base_run, f"seed{seed}__{base_run}")
        members = [by_run[name] for name in names if name in by_run]
        if len(members) == 2 and all(row["eligible"] for row in members):
            groups.append({
                "base_run": base_run,
                "mean_validated_link_mm": sum(
                    row["validated_link_mm"] for row in members
                ) / 2.0,
                "members": members,
            })
        else:
            incomplete_groups.append({
                "base_run": base_run,
                "expected_members": list(names),
                "present_members": [row["run"] for row in members],
                "eligible_members": [row["run"] for row in members if row["eligible"]],
            })
if groups:
    groups.sort(key=lambda group: group["mean_validated_link_mm"])
    selected_group = groups[0]
    selected_member = min(
        selected_group["members"], key=lambda row: row["validated_link_mm"]
    )
    selection = {
        "selection_rule": "lowest two-seed mean; extend best member of winning group",
        "groups": groups,
        "incomplete_or_ineligible_groups": incomplete_groups,
        "selected_base_run": selected_group["base_run"],
        "winner": selected_member,
    }
else:
    eligible = [row for row in ranking if row["eligible"]]
    selected_member = eligible[0] if eligible else ranking[0]
    selection = {
        "selection_rule": "lowest individual validated link error",
        "groups": [],
        "incomplete_or_ineligible_groups": incomplete_groups,
        "selected_base_run": selected_member["run"],
        "winner": selected_member,
    }
path = selection_path
fd, temporary = tempfile.mkstemp(prefix=".selection.", dir=root)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(selection, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
os.replace(temporary, path)
print(selected_member["output_dir"])
PY
}

for config in "${MPPI_CONFIGS[@]}"; do
  if ! run_fresh_trial "${config}"; then
    break
  fi
  rank_trials >/dev/null
done

mppi_winner="$(select_mppi_winner)"
if [[ -n "${mppi_winner}" ]]; then
  read -r best_sigma0 best_temperature best_smoothing <<<"$(
    "${PYTHON_BIN}" - "${mppi_winner}/run_config.json" <<'PY'
import json, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
print(config["sigma0"], config["temperature"], config["smoothing_window"])
PY
  )"
  for reward_config in "${REWARD_CONFIGS[@]}"; do
    read -r reward_name reward_weight reward_sigma <<<"${reward_config}"
    reward_name="${reward_name}__from_$(basename "${mppi_winner}")"
    if ! run_fresh_trial \
      "${reward_name} ${reward_weight} ${reward_sigma} ${best_sigma0} ${best_temperature} ${best_smoothing} ${SEED}"; then
      break
    fi
    rank_trials >/dev/null
  done
fi

rank_trials >/dev/null
while IFS= read -r replication_config; do
  if [[ -z "${replication_config}" ]]; then
    continue
  fi
  if ! run_fresh_trial "${replication_config}"; then
    break
  fi
  rank_trials >/dev/null
done < <(prepare_replication_configs)

rank_trials >/dev/null
winner="$(select_robust_winner)"
if [[ -z "${winner}" ]]; then
  write_status "failed" "no completed trial"
  exit 4
fi

remaining="$(remaining_seconds)"
if (( remaining <= 900 )); then
  write_status "complete" "screen complete; insufficient time for final extension"
  exit 0
fi

screen_winner="${winner}"
final_output_dir="${OUTPUT_ROOT}/final_best"
"${PYTHON_BIN}" - \
  "${screen_winner}" \
  "${final_output_dir}" \
  "${FINAL_PARTICLES}" \
  "${FINAL_TOTAL_ITERATIONS}" \
  "${VIDEO_INTERVAL}" <<'PY'
import json, os, shutil, sys, tempfile
from pathlib import Path

source = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
if not target.exists():
    temporary = Path(tempfile.mkdtemp(prefix=".final_best.", dir=target.parent))
    temporary.rmdir()
    try:
        shutil.copytree(source, temporary)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
source_path = target / "source_screen.json"
if source_path.exists():
    recorded = Path(json.loads(source_path.read_text())["output_dir"]).resolve()
    if recorded != source:
        raise RuntimeError(f"Existing final directory came from {recorded}, expected {source}")
else:
    fd, marker_temporary = tempfile.mkstemp(prefix=".source_screen.", dir=target)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump({"output_dir": str(source)}, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(marker_temporary, source_path)
path = target / "run_config.json"
config = json.loads(path.read_text())
config.update({
    "output_dir": str(target),
    "particles": int(sys.argv[3]),
    "iterations": int(sys.argv[4]),
    "save_video": True,
    "video_interval": int(sys.argv[5]),
    "video_best_kind": "link",
    "validation_interval": 0,
    "tensorboard": True,
    "history_format": "jsonl",
    "progress": False,
})
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(config, stream, ensure_ascii=False, indent=2, default=str)
    stream.write("\n")
os.replace(temporary, path)
PY

winner="${final_output_dir}"
write_status "final" "extending ${winner}; screen source=${screen_winner}; until ${FINAL_RUN_STOP_AT}"
remaining="$(final_run_remaining_seconds)"
set +e
timeout --signal=INT --kill-after=120 "${remaining}" \
  "${PYTHON_BIN}" -m experiments.bfm_trajectory_latent_adaptation.adapt_trajectory \
    --resume-dir "${winner}" \
    --resume-total-iterations "${FINAL_TOTAL_ITERATIONS}" \
    --resume-reset-mean-to-link-best \
    --history-format jsonl \
    --no-progress \
    >>"${winner}/run.log" 2>&1
status=$?
set -e

remaining="$(remaining_seconds)"
if [[ -f "${winner}/checkpoint.pt" ]] && (( remaining > 120 )); then
  completed_steps="$("${PYTHON_BIN}" - "${winner}/checkpoint.pt" <<'PY'
import sys, torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint["iteration"]) + 1)
PY
)"
  write_status "finalize" "validating ${winner} at step ${completed_steps}"
  remaining="$(remaining_seconds)"
  set +e
  timeout --signal=INT --kill-after=60 "${remaining}" \
    "${PYTHON_BIN}" -m experiments.bfm_trajectory_latent_adaptation.adapt_trajectory \
      --resume-dir "${winner}" \
      --resume-total-iterations "${FINAL_TOTAL_ITERATIONS}" \
      --stop-after-iteration "${completed_steps}" \
      --history-format jsonl \
      --no-progress \
      >>"${winner}/run.log" 2>&1
  validation_status=$?
  set -e
  if [[ ${validation_status} -ne 0 ]]; then
    write_status "finalize" "validation exited ${validation_status}; rendering saved checkpoint"
  fi
  remaining="$(remaining_seconds)"
  if (( remaining > 60 )); then
    timeout --signal=INT --kill-after=30 "${remaining}" \
      "${PYTHON_BIN}" -m experiments.bfm_trajectory_latent_adaptation.adapt_trajectory \
        --resume-dir "${winner}" \
        --render-only \
        >>"${winner}/run.log" 2>&1 || true
  fi
fi

rank_trials >/dev/null
if [[ ${status} -eq 0 || ${status} -eq 124 || ${status} -eq 130 ]]; then
  write_status "complete" "deadline reached; best=${winner}"
  exit 0
fi
write_status "failed" "final extension exited ${status}; best=${winner}"
exit "${status}"
