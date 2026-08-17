"""Compare baseline/adapted MuJoCo diagnostics and enforce no hard-limit violation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--adapted-summary", type=Path, required=True)
    parser.add_argument("--baseline-timeseries", type=Path, required=True)
    parser.add_argument("--adapted-timeseries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-adapted-hard-violation-rad", type=float, default=0.0)
    parser.add_argument(
        "--adapted-joint-limit-guard-fraction",
        type=float,
        required=True,
        help=(
            "Require the adapted rollout's worst normalized hard-limit utilization "
            "to be at or below this value; the value itself must be below one."
        ),
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read MuJoCo diagnostics summary {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"MuJoCo diagnostics summary must be an object: {path}")
    return value


def _joint_limit_timeseries_stats(
    joint_pos: np.ndarray,
    hard_limits: np.ndarray,
    joint_names: np.ndarray | list[str],
    motion_frames: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure normalized hard-limit utilization, including exact contact."""
    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    hard_limits = np.asarray(hard_limits, dtype=np.float64)
    joint_names = np.asarray(joint_names).astype(str)
    if joint_pos.ndim != 2 or not joint_pos.shape[0] or not joint_pos.shape[1]:
        raise ValueError(f"joint_pos must be a non-empty [frames,joints] array, got {joint_pos.shape}")
    if hard_limits.shape != (joint_pos.shape[1], 2):
        raise ValueError(
            "hard_joint_limits must have shape "
            f"({joint_pos.shape[1]},2), got {hard_limits.shape}"
        )
    if joint_names.shape != (joint_pos.shape[1],):
        raise ValueError(
            f"joint_names must have shape ({joint_pos.shape[1]},), got {joint_names.shape}"
        )
    if not np.isfinite(joint_pos).all() or not np.isfinite(hard_limits).all():
        raise ValueError("MuJoCo joint positions/limits contain NaN or Inf")

    lower, upper = hard_limits.T
    half_range = 0.5 * (upper - lower)
    if np.any(half_range <= 0.0):
        raise ValueError("MuJoCo hard joint limits must have positive width")
    center = 0.5 * (lower + upper)
    utilization = np.abs((joint_pos - center[None, :]) / half_range[None, :])
    contact = utilization >= 1.0
    margin = np.minimum(joint_pos - lower[None, :], upper[None, :] - joint_pos)

    if motion_frames is None:
        motion_frames = np.arange(joint_pos.shape[0], dtype=np.int64)
    else:
        motion_frames = np.asarray(motion_frames)
        if motion_frames.shape != (joint_pos.shape[0],):
            raise ValueError(
                f"motion_frame must have shape ({joint_pos.shape[0]},), got {motion_frames.shape}"
            )

    worst_row, worst_joint = np.unravel_index(np.argmax(utilization), utilization.shape)
    closest_row, closest_joint = np.unravel_index(np.argmin(margin), margin.shape)
    contact_frames = np.any(contact, axis=1)
    return {
        "frames": int(joint_pos.shape[0]),
        "max_joint_limit_utilization": float(utilization[worst_row, worst_joint]),
        "worst_utilization_joint": str(joint_names[worst_joint]),
        "worst_utilization_row": int(worst_row),
        "worst_utilization_motion_frame": int(motion_frames[worst_row]),
        "minimum_hard_limit_margin_rad": float(margin[closest_row, closest_joint]),
        "closest_limit_joint": str(joint_names[closest_joint]),
        "closest_limit_row": int(closest_row),
        "closest_limit_motion_frame": int(motion_frames[closest_row]),
        "hard_limit_contact_frame_count": int(np.count_nonzero(contact_frames)),
        "hard_limit_contact_joint_sample_count": int(np.count_nonzero(contact)),
        "hard_limit_contact_fraction": float(np.mean(contact_frames)),
    }


def _load_timeseries(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    try:
        with np.load(path, allow_pickle=False) as arrays:
            required = {"joint_pos", "hard_joint_limits", "joint_names"}
            missing = sorted(required.difference(arrays.files))
            if missing:
                raise ValueError(f"missing arrays: {', '.join(missing)}")
            stats = _joint_limit_timeseries_stats(
                arrays["joint_pos"],
                arrays["hard_joint_limits"],
                arrays["joint_names"],
                arrays["motion_frame"] if "motion_frame" in arrays.files else None,
            )
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read MuJoCo diagnostics timeseries {path}: {error}") from error
    return {"timeseries": str(path), **stats}


def _variant_stats(summary: dict[str, Any], timeseries_path: Path) -> dict[str, Any]:
    try:
        activity = summary["joint_position_limit_activity"]["per_joint"]
        joint_mae = float(summary["mean_joint_pos_error_rad"]["mean"])
        local_link = float(summary["mean_local_link_pos_error_m"]["mean"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Malformed MuJoCo diagnostics summary: {error}") from error
    if not isinstance(activity, dict) or not activity:
        raise ValueError("MuJoCo diagnostics has no per-joint limit activity")
    violations: dict[str, dict[str, float]] = {}
    max_violation = 0.0
    for joint_name, values in activity.items():
        try:
            fraction = float(values["hard_limit_exceeded_fraction"])
            violation = float(values["max_hard_limit_violation_rad"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Malformed limit activity for {joint_name}: {error}") from error
        if fraction < 0.0 or violation < 0.0:
            raise ValueError(f"Negative limit statistic for {joint_name}")
        max_violation = max(max_violation, violation)
        if fraction > 0.0 or violation > 0.0:
            violations[str(joint_name)] = {
                "hard_limit_exceeded_fraction": fraction,
                "max_hard_limit_violation_rad": violation,
            }
    stats = {
        "frames": int(summary["frames"]),
        "duration_s": float(summary["duration_s"]),
        "mean_joint_pos_error_rad": joint_mae,
        "mean_local_link_pos_error_m": local_link,
        "max_hard_limit_violation_rad": max_violation,
        "violating_joints": violations,
    }
    timeseries = _load_timeseries(timeseries_path)
    if stats["frames"] != timeseries["frames"]:
        raise ValueError("MuJoCo summary/timeseries frame counts differ")
    return {**stats, **timeseries}


def _passes_limit_gate(
    stats: dict[str, Any],
    *,
    max_hard_violation_rad: float,
    guard_fraction: float,
) -> bool:
    """Require clearance from the limit, not merely absence of overshoot."""
    return bool(
        stats["max_hard_limit_violation_rad"] <= max_hard_violation_rad
        and not stats["violating_joints"]
        and stats["hard_limit_contact_frame_count"] == 0
        and stats["hard_limit_contact_joint_sample_count"] == 0
        and stats["max_joint_limit_utilization"] <= guard_fraction
    )


def main() -> None:
    args = parse_args()
    if args.max_adapted_hard_violation_rad < 0.0:
        raise ValueError("max-adapted-hard-violation-rad must be non-negative")
    guard = float(args.adapted_joint_limit_guard_fraction)
    if not 0.0 < guard < 1.0:
        raise ValueError("adapted-joint-limit-guard-fraction must be strictly in (0,1)")
    baseline_path = args.baseline_summary.expanduser().resolve()
    adapted_path = args.adapted_summary.expanduser().resolve()
    baseline = _variant_stats(_load(baseline_path), args.baseline_timeseries)
    adapted = _variant_stats(_load(adapted_path), args.adapted_timeseries)
    if baseline["frames"] != adapted["frames"]:
        raise ValueError("Baseline and adapted diagnostics frame counts differ")

    tolerance = float(args.max_adapted_hard_violation_rad)
    adapted_safe = _passes_limit_gate(
        adapted,
        max_hard_violation_rad=tolerance,
        guard_fraction=guard,
    )
    comparison = {
        "baseline_summary": str(baseline_path),
        "adapted_summary": str(adapted_path),
        "max_adapted_hard_violation_rad": tolerance,
        "adapted_joint_limit_guard_fraction": guard,
        "adapted_no_hard_limit_violation": adapted_safe,
        "baseline": baseline,
        "adapted": adapted,
        "changes": {
            "joint_mae_percent": 100.0
            * (
                adapted["mean_joint_pos_error_rad"]
                / max(baseline["mean_joint_pos_error_rad"], 1.0e-12)
                - 1.0
            ),
            "local_link_error_percent": 100.0
            * (
                adapted["mean_local_link_pos_error_m"]
                / max(baseline["mean_local_link_pos_error_m"], 1.0e-12)
                - 1.0
            ),
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    if not adapted_safe:
        raise SystemExit(
            "Adapted MuJoCo rollout contacts or approaches a hard joint limit "
            f"beyond the {guard:.6f} utilization guard; "
            f"see {output}"
        )


if __name__ == "__main__":
    main()
