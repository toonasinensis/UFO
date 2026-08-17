#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# =============================================================================
# Promoted body-weight 0.95 overnight continuation. Edit this block to retarget.
# =============================================================================
PYTHON_BIN=".venv/bin/python"
OUTPUT_DIR="experiments/bfm_trajectory_latent_adaptation/outputs/overnight_link_tuning_20260813/promoted_best_weight095"
TOTAL_ITERATIONS=50000
LONG_RUN_STOP_AT="2026-08-14 07:38:00"
FINAL_DEADLINE_AT="2026-08-14 07:44:00"
HISTORY_FORMAT="jsonl"
PROGRESS_FLAG="--no-progress"
RENDER_FINAL_VIDEO=true

long_stop_epoch="$(date -d "${LONG_RUN_STOP_AT}" +%s)"
final_deadline_epoch="$(date -d "${FINAL_DEADLINE_AT}" +%s)"
remaining=$(( long_stop_epoch - $(date +%s) ))
if (( remaining <= 0 )); then
  echo "Long-run deadline has already passed" >&2
  exit 2
fi

set +e
timeout --signal=INT --kill-after=120 "${remaining}" \
  "${PYTHON_BIN}" -m experiments.bfm_trajectory_latent_adaptation.adapt_trajectory \
    --resume-dir "${OUTPUT_DIR}" \
    --resume-total-iterations "${TOTAL_ITERATIONS}" \
    --history-format "${HISTORY_FORMAT}" \
    "${PROGRESS_FLAG}" \
    >>"${OUTPUT_DIR}/run.log" 2>&1
long_status=$?
set -e

completed_steps="$(
  "${PYTHON_BIN}" - "${OUTPUT_DIR}/checkpoint.pt" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint["iteration"]) + 1)
PY
)"

remaining=$(( final_deadline_epoch - $(date +%s) ))
if (( remaining <= 120 )); then
  echo "Insufficient time for final paired validation at step ${completed_steps}" >&2
  exit 3
fi

timeout --signal=INT --kill-after=60 "${remaining}" \
  "${PYTHON_BIN}" -m experiments.bfm_trajectory_latent_adaptation.adapt_trajectory \
    --resume-dir "${OUTPUT_DIR}" \
    --resume-total-iterations "${TOTAL_ITERATIONS}" \
    --stop-after-iteration "${completed_steps}" \
    --history-format "${HISTORY_FORMAT}" \
    "${PROGRESS_FLAG}" \
    >>"${OUTPUT_DIR}/run.log" 2>&1

if [[ "${RENDER_FINAL_VIDEO}" == "true" ]]; then
  remaining=$(( final_deadline_epoch - $(date +%s) ))
  if (( remaining > 60 )); then
    timeout --signal=INT --kill-after=30 "${remaining}" \
      "${PYTHON_BIN}" -m experiments.bfm_trajectory_latent_adaptation.adapt_trajectory \
        --resume-dir "${OUTPUT_DIR}" \
        --render-only \
        >>"${OUTPUT_DIR}/run.log" 2>&1
  fi
fi

if [[ ${long_status} -ne 0 && ${long_status} -ne 124 && ${long_status} -ne 130 ]]; then
  echo "Long run exited unexpectedly with status ${long_status}" >&2
  exit "${long_status}"
fi

echo "Promoted weight-0.95 run finalized at completed step ${completed_steps}"
