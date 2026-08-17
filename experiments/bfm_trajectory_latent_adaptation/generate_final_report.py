"""Build an auditable final report for a completed trajectory-adaptation run.

The command is deliberately read-only unless ``--apply`` is supplied.  Before
building a report it runs the repository's final-run audit, so a live or
partially finalized checkpoint cannot be presented as a final result.  On
apply, ``final_report.md`` is installed first and ``final_report.json`` last;
the JSON file is therefore the commit record for the report pair.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo


SCHEMA_VERSION = 1
DEFAULT_WINDOW_SIZE = 50
LOCAL_TIME_ZONE = ZoneInfo("Asia/Shanghai")


class ReportValidationError(ValueError):
    """Raised when report inputs are missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    data: bytes
    sha256: str
    signature: tuple[int, int, int, int]


@dataclass(frozen=True)
class ReportPlan:
    output_root: Path
    run_dir: Path
    report: dict[str, Any]
    markdown: str
    json_bytes: bytes
    markdown_bytes: bytes
    source_snapshots: tuple[FileSnapshot, ...]


AuditRunner = Callable[..., Mapping[str, Any]]
TensorBoardReader = Callable[[Path], tuple[dict[str, Any], tuple[FileSnapshot, ...]]]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _compact_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _snapshot(path: Path) -> FileSnapshot:
    try:
        before = _signature(path)
        data = path.read_bytes()
        after = _signature(path)
    except FileNotFoundError as error:
        raise ReportValidationError(f"required report input is missing: {path}") from error
    if before != after or len(data) != after[2]:
        raise ReportValidationError(f"report input changed while being read: {path}")
    return FileSnapshot(path, data, _sha256(data), after)


def _ensure_unchanged(snapshots: Sequence[FileSnapshot]) -> None:
    for snapshot in snapshots:
        try:
            current = _signature(snapshot.path)
        except FileNotFoundError as error:
            raise ReportValidationError(
                f"report input disappeared after validation: {snapshot.path}"
            ) from error
        if current != snapshot.signature:
            raise ReportValidationError(
                f"report input changed after validation: {snapshot.path}"
            )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReportValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _assert_finite(value: Any, label: str) -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReportValidationError(f"{label} contains NaN or Inf")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{label}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{label}[{index}]")
        return
    raise ReportValidationError(f"{label} contains unsupported {type(value).__name__}")


def _load_json_snapshot(path: Path) -> tuple[Any, FileSnapshot]:
    snapshot = _snapshot(path)
    try:
        value = json.loads(
            snapshot.data.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReportValidationError(f"invalid JSON in {path}: {error}") from error
    _assert_finite(value, path.name)
    return value, snapshot


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportValidationError(f"{label} must be a JSON object")
    return value


def _number(mapping: Mapping[str, Any], key: str, label: str) -> float:
    if key not in mapping:
        raise ReportValidationError(f"{label} is missing {key!r}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportValidationError(f"{label}.{key} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ReportValidationError(f"{label}.{key} must be finite")
    return result


def _integer(mapping: Mapping[str, Any], key: str, label: str) -> int:
    value = _number(mapping, key, label)
    result = int(value)
    if result != value:
        raise ReportValidationError(f"{label}.{key} must be an integer")
    return result


def _optional_number(mapping: Mapping[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _path_entry(path: Path, *, relative_to: Path | None = None) -> dict[str, str]:
    result = {"absolute": str(path.resolve())}
    if relative_to is not None:
        try:
            result["relative_to_output_root"] = str(path.resolve().relative_to(relative_to))
        except ValueError:
            pass
    return result


def _metric_view(
    metrics: Mapping[str, Any],
    *,
    std: Mapping[str, Any] | None,
    sample_count: int | None,
    provenance: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "provenance": provenance,
        "sample_count": sample_count,
        "link_relative_position_mm": 1000.0
        * _number(metrics, "mean_body_position_error", provenance),
        "mpjpe_mm": 1000.0 * _number(metrics, "mpjpe", provenance),
        "root_position_mm": 1000.0
        * _number(metrics, "mean_root_position_error", provenance),
        "objective": _number(metrics, "objective", provenance),
    }
    if std is not None:
        result["population_std"] = {
            "link_relative_position_mm": 1000.0
            * _number(std, "mean_body_position_error", f"{provenance}.std"),
            "mpjpe_mm": 1000.0 * _number(std, "mpjpe", f"{provenance}.std"),
            "root_position_mm": 1000.0
            * _number(std, "mean_root_position_error", f"{provenance}.std"),
            "objective": _number(std, "objective", f"{provenance}.std"),
        }
    return result


def _reduction(reference: float, candidate: float) -> dict[str, float]:
    reduction = reference - candidate
    return {
        "reference": reference,
        "candidate": candidate,
        "reduction": reduction,
        "reduction_percent": 100.0 * reduction / reference if reference else 0.0,
    }


def _comparison_delta(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "link_relative_position_mm": _reduction(
            float(reference["link_relative_position_mm"]),
            float(candidate["link_relative_position_mm"]),
        ),
        "mpjpe_mm": _reduction(
            float(reference["mpjpe_mm"]), float(candidate["mpjpe_mm"])
        ),
        "root_position_mm": _reduction(
            float(reference["root_position_mm"]),
            float(candidate["root_position_mm"]),
        ),
        "objective_delta": float(candidate["objective"]) - float(reference["objective"]),
    }


def _parse_history(
    path: Path,
) -> tuple[list[Mapping[str, Any]], FileSnapshot]:
    snapshot = _snapshot(path)
    try:
        lines = snapshot.data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ReportValidationError(f"invalid UTF-8 in {path}") from error
    if not lines:
        raise ReportValidationError("history.jsonl is empty")
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ReportValidationError(f"history.jsonl has blank line {line_number}")
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as error:
            raise ReportValidationError(
                f"invalid history.jsonl line {line_number}: {error}"
            ) from error
        record = _mapping(value, f"history line {line_number}")
        _assert_finite(record, f"history[{line_number}]")
        records.append(record)
    iterations = [
        _integer(record, "iteration", f"history[{index}]")
        for index, record in enumerate(records)
    ]
    if iterations != list(range(iterations[0], iterations[-1] + 1)):
        raise ReportValidationError("history iterations contain a gap or duplicate")
    return records, snapshot


def _load_history(
    path: Path, completed_iterations: int
) -> tuple[list[Mapping[str, Any]], FileSnapshot]:
    records, snapshot = _parse_history(path)
    last_iteration = _integer(records[-1], "iteration", "history last")
    if last_iteration + 1 != completed_iterations:
        raise ReportValidationError(
            "history/final validation mismatch: "
            f"history completed={last_iteration + 1}, summary={completed_iterations}"
        )
    return records, snapshot


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _history_summary(
    history: Sequence[Mapping[str, Any]], window_size: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    if window_size <= 0:
        raise ReportValidationError("history window size must be positive")
    windows: list[dict[str, Any]] = []
    all_ess: list[float] = []
    total_seconds = 0.0
    for start in range(0, len(history), window_size):
        records = history[start : start + window_size]
        ess = [
            _number(_mapping(record.get("mppi"), "history.mppi"), "ess_ratio", "mppi")
            for record in records
        ]
        eligible = [
            _number(record, "mppi_eligible_ratio", "history") for record in records
        ]
        link = [
            1000.0
            * _number(
                _mapping(record.get("link_best_metrics"), "history.link_best_metrics"),
                "mean_body_position_error",
                "history.link_best_metrics",
            )
            for record in records
        ]
        seconds = [
            _optional_number(record, "iteration_seconds") or 0.0 for record in records
        ]
        all_ess.extend(ess)
        total_seconds += sum(seconds)
        first_iteration = _integer(records[0], "iteration", "history")
        last_iteration = _integer(records[-1], "iteration", "history")
        windows.append(
            {
                "iteration_range": [first_iteration, last_iteration],
                "completed_step_range": [first_iteration + 1, last_iteration + 1],
                "record_count": len(records),
                "complete_window": len(records) == window_size,
                "ess_ratio": {
                    "mean": statistics.fmean(ess),
                    "min": min(ess),
                    "p10": _percentile(ess, 0.10),
                    "median": statistics.median(ess),
                    "max": max(ess),
                },
                "eligible_ratio_mean": statistics.fmean(eligible),
                "skipped_updates": sum(bool(record.get("mppi_update_skipped")) for record in records),
                "link_best_mm_start": link[0],
                "link_best_mm_end": link[-1],
                "link_best_gain_mm": link[0] - link[-1],
                "link_best_update_count": sum(
                    bool(record.get("link_best_updated")) for record in records
                ),
                "iteration_seconds_sum": sum(seconds),
            }
        )

    first_window = windows[0]
    last_window = windows[-1]
    ess_delta = (
        float(last_window["ess_ratio"]["mean"])
        - float(first_window["ess_ratio"]["mean"])
    )
    threshold = 0.02
    trend = "stable"
    if ess_delta > threshold:
        trend = "increased"
    elif ess_delta < -threshold:
        trend = "decreased"
    summary = {
        "history_record_count": len(history),
        "window_size": window_size,
        "window_count": len(windows),
        "first_window_mean_ess_ratio": first_window["ess_ratio"]["mean"],
        "last_window_mean_ess_ratio": last_window["ess_ratio"]["mean"],
        "first_to_last_window_mean_delta": ess_delta,
        "trend": trend,
        "overall_mean_ess_ratio": statistics.fmean(all_ess),
        "overall_min_ess_ratio": min(all_ess),
        "overall_max_ess_ratio": max(all_ess),
        "iteration_seconds_sum": total_seconds,
    }
    return summary, {"summary": summary, "windows": windows}


def _search_estimate(
    history: Sequence[Mapping[str, Any]], candidate_kind: str = "link"
) -> dict[str, Any]:
    if candidate_kind not in {"link", "objective"}:
        raise ReportValidationError(
            f"candidate_kind must be 'link' or 'objective', got {candidate_kind!r}"
        )
    last = history[-1]
    metrics_key = (
        "link_best_metrics" if candidate_kind == "link" else "global_best_metrics"
    )
    objective_key = (
        "link_best_objective" if candidate_kind == "link" else "global_best_objective"
    )
    metrics = dict(
        _mapping(last.get(metrics_key), f"history last {metrics_key}")
    )
    # Search history keeps the scalar objective beside the metric mapping,
    # whereas final metrics JSON keeps it inside ``metrics``.
    if "objective" not in metrics:
        metrics["objective"] = _number(last, objective_key, "history last")
    estimate = _metric_view(
        metrics,
        std=None,
        sample_count=1,
        provenance=(
            f"optimizer history {candidate_kind}-best estimate; one search evaluation, "
            "not the final n=64 paired validation"
        ),
    )
    final_link = float(estimate["link_relative_position_mm"])
    first_seen = None
    for record in history:
        # Older history rows predate link-best tracking.  They remain valid
        # optimizer records but cannot establish when this link best appeared.
        raw_record_metrics = record.get(metrics_key)
        if not isinstance(raw_record_metrics, Mapping):
            continue
        record_metrics = raw_record_metrics
        value = 1000.0 * _number(
            record_metrics,
            "mean_body_position_error",
            "history.link_best_metrics",
        )
        if math.isclose(value, final_link, rel_tol=0.0, abs_tol=1.0e-9):
            first_seen = _integer(record, "iteration", "history")
            break
    last_iteration = _integer(last, "iteration", "history")
    estimate.update(
        {
            "history_iteration": last_iteration,
            "completed_step": last_iteration + 1,
            "best_first_seen_iteration": first_seen,
            "best_first_seen_completed_step": (
                first_seen + 1 if first_seen is not None else None
            ),
            "candidate_kind": candidate_kind,
        }
    )
    return estimate


def _select_warm_validation_files(warm_start_dir: Path) -> tuple[Path, Path, int | None]:
    direct_summary = warm_start_dir / "summary.json"
    direct_metrics = warm_start_dir / "metrics_adapted.json"
    if direct_summary.is_file() and direct_metrics.is_file():
        return direct_summary, direct_metrics, None
    candidates: list[tuple[int, Path, Path]] = []
    for summary_path in warm_start_dir.glob("summary_iter*.json"):
        suffix = summary_path.stem.removeprefix("summary_iter")
        try:
            step = int(suffix)
        except ValueError:
            continue
        metrics_path = warm_start_dir / f"metrics_adapted_iter{suffix}.json"
        if metrics_path.is_file():
            candidates.append((step, summary_path, metrics_path))
    if not candidates:
        raise ReportValidationError(
            "warm-start directory has no matching summary/metrics_adapted validation pair: "
            f"{warm_start_dir}"
        )
    step, summary_path, metrics_path = max(candidates, key=lambda item: item[0])
    return summary_path, metrics_path, step


def _load_pre_overnight_validated_candidate(
    warm_start_dir: Path,
) -> tuple[dict[str, Any], list[FileSnapshot]]:
    summary_path, metrics_path, filename_step = _select_warm_validation_files(
        warm_start_dir
    )
    summary_value, summary_snapshot = _load_json_snapshot(summary_path)
    metrics_value, metrics_snapshot = _load_json_snapshot(metrics_path)
    summary = _mapping(summary_value, "warm summary")
    metrics_file = _mapping(metrics_value, "warm metrics_adapted")
    metrics = _mapping(metrics_file.get("metrics"), "warm metrics_adapted.metrics")
    std = _mapping(
        metrics_file.get("population_std"), "warm metrics_adapted.population_std"
    )
    counts = _mapping(
        metrics_file.get("validation_counts"), "warm metrics_adapted.validation_counts"
    )
    sample_count = _integer(counts, "adapted", "warm validation_counts")
    completed = summary.get("completed_iterations", filename_step)
    if completed is not None:
        if isinstance(completed, bool) or not isinstance(completed, int):
            raise ReportValidationError("warm completed step must be an integer")

    # metrics_adapted belongs to validated_objective_best.  Preserve any
    # same-step validated link artifact as additional evidence, but never bind
    # these step-N metrics to the later live link_best_z.npy.
    validated_artifacts: list[FileSnapshot] = []
    for filename in ("validated_objective_best_z.npy", "validated_link_best_z.npy"):
        path = warm_start_dir / filename
        if path.is_file():
            validated_artifacts.append(_snapshot(path))
    primary_artifact = next(
        (
            artifact
            for artifact in validated_artifacts
            if artifact.path.name == "validated_objective_best_z.npy"
        ),
        validated_artifacts[0] if validated_artifacts else None,
    )

    result = _metric_view(
        metrics,
        std=std,
        sample_count=sample_count,
        provenance=(
            "pre-overnight candidate at the saved validation step; independently "
            "validated, but not the latent actually used to initialize overnight tuning"
        ),
    )
    result.update(
        {
            "completed_step": completed,
            "independently_validated": True,
            "validation_kind": f"n={sample_count} independent rollout validation",
            "summary_path": str(summary_path.resolve()),
            "metrics_path": str(metrics_path.resolve()),
            "validated_latent_path": (
                str(primary_artifact.path.resolve()) if primary_artifact else None
            ),
            "validated_latent_sha256": (
                primary_artifact.sha256 if primary_artifact else None
            ),
            "validated_latent_hash_available": primary_artifact is not None,
            "validated_latent_artifacts": [
                {
                    "path": str(artifact.path.resolve()),
                    "sha256": artifact.sha256,
                }
                for artifact in validated_artifacts
            ],
            "objective_config": summary.get("config"),
        }
    )
    snapshots = [summary_snapshot, metrics_snapshot, *validated_artifacts]
    return result, snapshots


def _load_actual_overnight_initial_mean(
    warm_start_dir: Path,
    final_initial_mean: FileSnapshot,
) -> tuple[dict[str, Any], list[FileSnapshot]]:
    warm_link_best = _snapshot(warm_start_dir / "link_best_z.npy")
    if warm_link_best.sha256 != final_initial_mean.sha256:
        raise ReportValidationError(
            "final initial_mean_z does not match warm-start link_best_z.npy"
        )
    history, history_snapshot = _parse_history(warm_start_dir / "history.jsonl")
    result = _search_estimate(history)
    result.update(
        {
            "provenance": (
                "actual latent used to initialize overnight tuning; metrics are the "
                "source run's final history link_best search estimate"
            ),
            "independently_validated": False,
            "validation_kind": "single search evaluation; not independently validated",
            "latent_source_path": str(warm_link_best.path.resolve()),
            "latent_sha256": warm_link_best.sha256,
            "initial_mean_path": str(final_initial_mean.path.resolve()),
            "initial_mean_sha256": final_initial_mean.sha256,
            "hash_match": True,
            "history_path": str(history_snapshot.path.resolve()),
            "history_sha256": history_snapshot.sha256,
        }
    )
    return result, [warm_link_best, history_snapshot]


def _normalized_comparison(
    comparison: Mapping[str, Any], relative_path: str
) -> dict[str, Any]:
    decision = str(comparison.get("decision", ""))
    outcome = "promoted" if decision.lower().startswith("promote") else "rejected"
    link_gain = _optional_number(comparison, "validated_link_gain_mm")
    if link_gain is None:
        link_gain = _optional_number(comparison, "trial_rescored_link_gain_mm")
    if link_gain is None:
        control_minus_trial = _optional_number(
            comparison, "control_minus_trial_validated_link_mm"
        )
        link_gain = control_minus_trial
    mpjpe_regression = _optional_number(comparison, "mpjpe_regression_mm")
    if mpjpe_regression is None:
        control = _optional_number(comparison, "control_mpjpe_mm")
        trial = _optional_number(comparison, "trial_mpjpe_mm")
        if control is not None and trial is not None:
            mpjpe_regression = trial - control
    root_regression = _optional_number(comparison, "root_position_regression_mm")
    if root_regression is None:
        control = _optional_number(comparison, "control_root_position_mm")
        trial = _optional_number(comparison, "trial_root_position_mm")
        if control is not None and trial is not None:
            root_regression = trial - control
    return {
        "path": relative_path,
        "fork_completed_step": comparison.get("fork_completed_steps"),
        "comparison_completed_step": comparison.get(
            "comparison_completed_steps",
            comparison.get("planned_comparison_completed_steps"),
        ),
        "changed_parameter": comparison.get(
            "changed_parameter", comparison.get("trial_score", "algorithmic diagnostic")
        ),
        "winner": comparison.get("winner"),
        "outcome": outcome,
        "validated_link_gain_mm": link_gain,
        "mpjpe_regression_mm": mpjpe_regression,
        "root_position_regression_mm": root_regression,
        "decision": decision,
        "raw": dict(comparison),
    }


def _load_comparisons(
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[FileSnapshot]]:
    comparisons: list[dict[str, Any]] = []
    snapshots: list[FileSnapshot] = []
    refinements = output_root / "refinements"
    if not refinements.is_dir():
        return comparisons, snapshots
    for path in sorted(refinements.glob("**/comparison.json")):
        value, snapshot = _load_json_snapshot(path)
        comparison = _mapping(value, str(path))
        relative_path = str(path.relative_to(output_root))
        comparisons.append(_normalized_comparison(comparison, relative_path))
        snapshots.append(snapshot)
    comparisons.sort(
        key=lambda row: (
            row["fork_completed_step"]
            if isinstance(row["fork_completed_step"], int)
            else 10**12,
            row["path"],
        )
    )
    return comparisons, snapshots


def _load_promotions(
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[FileSnapshot]]:
    promotions: list[dict[str, Any]] = []
    snapshots: list[FileSnapshot] = []
    for path in sorted(output_root.glob("*/promotion.json")):
        value, snapshot = _load_json_snapshot(path)
        promotion = dict(_mapping(value, str(path)))
        promotion["path"] = str(path.relative_to(output_root))
        promotion["promoted_run"] = path.parent.name
        promotions.append(promotion)
        snapshots.append(snapshot)
    promotions.sort(key=lambda item: (str(item.get("promoted_at", "")), item["path"]))
    return promotions, snapshots


def _iso_wall_time(value: float) -> dict[str, str]:
    utc = datetime.fromtimestamp(value, timezone.utc)
    return {
        "utc": utc.isoformat(),
        "asia_shanghai": utc.astimezone(LOCAL_TIME_ZONE).isoformat(),
    }


def _read_tensorboard_metadata(
    tensorboard_dir: Path,
) -> tuple[dict[str, Any], tuple[FileSnapshot, ...]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as error:
        raise ReportValidationError("tensorboard package is required") from error

    event_paths = sorted(tensorboard_dir.glob("events.out.tfevents.*"))
    if not event_paths:
        raise ReportValidationError(f"no TensorBoard event files in {tensorboard_dir}")
    snapshots = tuple(_snapshot(path) for path in event_paths)
    accumulator = EventAccumulator(
        str(tensorboard_dir), size_guidance={"scalars": 0, "tensors": 0}
    )
    accumulator.Reload()
    tags = accumulator.Tags()
    scalar_metadata: dict[str, dict[str, Any]] = {}
    all_wall_times: list[float] = []
    for tag in sorted(tags.get("scalars", [])):
        events = list(accumulator.Scalars(tag))
        if not events:
            continue
        values = [float(event.value) for event in events]
        if not all(math.isfinite(value) for value in values):
            raise ReportValidationError(f"TensorBoard scalar {tag} contains NaN or Inf")
        wall_times = [float(event.wall_time) for event in events]
        all_wall_times.extend(wall_times)
        scalar_metadata[tag] = {
            "event_count": len(events),
            "first_step": int(events[0].step),
            "last_step": int(events[-1].step),
            "first_wall_time": _iso_wall_time(wall_times[0]),
            "last_wall_time": _iso_wall_time(wall_times[-1]),
        }
    canonical_tag = "tracking/link_relative_position_mm/link_best"
    if canonical_tag not in scalar_metadata:
        raise ReportValidationError(
            f"TensorBoard is missing canonical scalar {canonical_tag}"
        )
    metadata: dict[str, Any] = {
        "logdir": str(tensorboard_dir.resolve()),
        "event_files": [
            {
                "path": str(snapshot.path.resolve()),
                "size_bytes": len(snapshot.data),
                "sha256": snapshot.sha256,
            }
            for snapshot in snapshots
        ],
        "scalar_tag_count": len(scalar_metadata),
        "tensor_tag_count": len(tags.get("tensors", [])),
        "scalar_tags": scalar_metadata,
        "canonical_training_tag": canonical_tag,
        "canonical_training_series": scalar_metadata[canonical_tag],
    }
    if all_wall_times:
        metadata["wall_time_range"] = {
            "first": _iso_wall_time(min(all_wall_times)),
            "last": _iso_wall_time(max(all_wall_times)),
            "elapsed_seconds": max(all_wall_times) - min(all_wall_times),
        }
    return metadata, snapshots


def _default_audit_runner(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    from experiments.bfm_trajectory_latent_adaptation.audit_final_run import (
        audit_final_run,
    )

    return audit_final_run(*args, **kwargs)


def _validate_audit(
    audit: Mapping[str, Any], run_dir: Path, output_root: Path, completed: int
) -> None:
    if audit.get("status") != "ready" or audit.get("failure_count") != 0:
        raise ReportValidationError(
            "final audit is not ready; refusing to create a final report"
        )
    if Path(str(audit.get("run_dir", ""))).resolve() != run_dir:
        raise ReportValidationError("audit run_dir does not match report run_dir")
    if Path(str(audit.get("output_root", ""))).resolve() != output_root:
        raise ReportValidationError("audit output_root does not match report output_root")
    observed = _mapping(audit.get("observed"), "audit.observed")
    if _integer(observed, "completed_iterations", "audit.observed") != completed:
        raise ReportValidationError("audit completed step does not match summary")


def _source_manifest(
    snapshots: Sequence[FileSnapshot], output_root: Path
) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        key = str(snapshot.path.resolve())
        if key in entries:
            continue
        item: dict[str, Any] = {
            "sha256": snapshot.sha256,
            "size_bytes": len(snapshot.data),
        }
        try:
            item["relative_to_output_root"] = str(
                snapshot.path.resolve().relative_to(output_root)
            )
        except ValueError:
            item["absolute_path"] = key
        entries[key] = item
    return entries


def _parameter_summary(
    run_config: Mapping[str, Any], summary: Mapping[str, Any]
) -> dict[str, Any]:
    summary_config = _mapping(summary.get("config"), "summary.config")
    weights = dict(_mapping(summary_config.get("weights"), "summary.config.weights"))
    sigmas = dict(_mapping(summary_config.get("sigmas"), "summary.config.sigmas"))
    for name, value in weights.items():
        config_value = _number(run_config, f"{name}_weight", "run_config")
        if not math.isclose(float(value), config_value, rel_tol=0.0, abs_tol=1.0e-12):
            raise ReportValidationError(f"weight mismatch for {name}")
    for name, value in sigmas.items():
        config_value = _number(run_config, f"{name}_sigma", "run_config")
        if not math.isclose(float(value), config_value, rel_tol=0.0, abs_tol=1.0e-12):
            raise ReportValidationError(f"sigma mismatch for {name}")
    return {
        "algorithm": "MPPI trajectory-latent adaptation",
        "particles": _integer(run_config, "particles", "run_config"),
        "sigma0": _number(run_config, "sigma0", "run_config"),
        "temperature": _number(run_config, "temperature", "run_config"),
        "beta_iteration": _number(run_config, "beta_iteration", "run_config"),
        "beta_horizon": _number(run_config, "beta_horizon", "run_config"),
        "smoothing_window": _integer(run_config, "smoothing_window", "run_config"),
        "score_mode": run_config.get("mppi_score"),
        "link_guard": run_config.get("mppi_link_guard"),
        "link_blend_alpha": _number(
            run_config, "mppi_link_blend_alpha", "run_config"
        ),
        "seed": _integer(run_config, "seed", "run_config"),
        "dial_schedule_iterations": _integer(
            run_config, "iterations", "run_config"
        ),
        "weights": weights,
        "sigmas": sigmas,
    }


def _conclusions(
    measurements: Mapping[str, Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    ess_summary: Mapping[str, Any],
    candidate_kind: str,
) -> list[str]:
    raw = measurements["raw_paired_baseline"]
    pre_validated = measurements["pre_overnight_validated_candidate"]
    actual_initial = measurements["actual_overnight_initial_mean"]
    validated = measurements[f"n64_validated_{candidate_kind}_best"]
    candidate_label = f"{candidate_kind}-best"
    raw_gain = _reduction(
        float(raw["link_relative_position_mm"]),
        float(validated["link_relative_position_mm"]),
    )
    pre_validated_gain = _reduction(
        float(pre_validated["link_relative_position_mm"]),
        float(validated["link_relative_position_mm"]),
    )
    actual_initial_gain = _reduction(
        float(actual_initial["link_relative_position_mm"]),
        float(validated["link_relative_position_mm"]),
    )
    promoted = [row for row in comparisons if row.get("outcome") == "promoted"]
    rejected = [row for row in comparisons if row.get("outcome") == "rejected"]
    conclusions = [
        (
            f"最终选定的 n=64 {candidate_label} 相对同批 raw baseline 将 "
            "local link-relative "
            f"position error 降低 {raw_gain['reduction']:.3f} mm "
            f"({raw_gain['reduction_percent']:.2f}%)."
        ),
        (
            "相对 step50 的 n=64 前置验证候选，local link error 降低 "
            f"{pre_validated_gain['reduction']:.3f} mm "
            f"({pre_validated_gain['reduction_percent']:.2f}%)."
        ),
        (
            "实际夜间 initial_mean 来自 source step"
            f"{actual_initial['completed_step']} 的 link_best_z；其单次搜索估计到最终 "
            f"n=64 值的描述性差值为 {actual_initial_gain['reduction']:.3f} mm。"
        ),
        (
            f"受控 A/B 共 {len(comparisons)} 项：{len(promoted)} 项晋级、"
            f"{len(rejected)} 项拒绝；拒绝项保留在报告中以记录 MPJPE/root 权衡。"
        ),
        (
            "MPPI ESS 的 promoted-run history 窗口趋势为 "
            f"{ess_summary['trend']}，首/末窗口均值分别为 "
            f"{ess_summary['first_window_mean_ess_ratio']:.3f}/"
            f"{ess_summary['last_window_mean_ess_ratio']:.3f}。"
        ),
        (
            f"最终 headline 采用 {candidate_label} 的 n=64 paired validation；"
            "history search estimate "
            "仅用于描述优化器搜索过程，不能替代重复验证。"
        ),
    ]
    if bool(validated.get("same_validated_latent")):
        conclusions.insert(
            3,
            (
                "validated_objective_best_z.npy 与 validated_link_best_z.npy "
                "SHA256 相同；两套 n=64 指标差仅表示同一 latent 在独立 rollout "
                "验证批次间的波动，不是算法 improvement。"
            ),
        )
    return conclusions


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.{digits}f}"
    return _md_escape(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    measurements = _mapping(report.get("measurements"), "report.measurements")
    selected = _mapping(report.get("selected_candidate"), "report.selected_candidate")
    candidate_kind = str(selected["kind"])
    candidate_label = f"{candidate_kind}-best"
    other_kind = "objective" if candidate_kind == "link" else "link"
    rows = (
        ("Raw paired baseline", measurements["raw_paired_baseline"]),
        (
            "Pre-overnight validated candidate",
            measurements["pre_overnight_validated_candidate"],
        ),
        (
            "Actual overnight initial_mean (search estimate)",
            measurements["actual_overnight_initial_mean"],
        ),
        ("Final-run search estimate", measurements["search_estimate"]),
        ("n=64 objective-best", measurements["n64_validated_objective_best"]),
        ("n=64 link-best", measurements["n64_validated_link_best"]),
    )
    lines = [
        "# BFM trajectory adaptation 最终报告",
        "",
        f"状态：**{_md_escape(report['status'])}**  ",
        f"报告 generation：`{_md_escape(report['report_generation'])}`  ",
        f"生成时间：{_md_escape(report['generated_at']['asia_shanghai'])}  ",
        f"Motion：`{_md_escape(report['run']['motion'])}`  ",
        f"最终 completed step：**{report['run']['completed_iterations']}**",
        f"Headline / selected candidate：**{candidate_label} (n=64)**",
        "",
        "## 结论",
        "",
    ]
    for conclusion in report["conclusions"]:
        lines.append(f"- {_md_escape(conclusion)}")
    lines.extend(
        [
            "",
            "## 测量层级",
            "",
            "| 层级 | n | Local link (mm) | MPJPE (mm) | Root pos (mm) | Objective |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    _md_escape(label),
                    _fmt(row.get("sample_count"), 0),
                    _fmt(row.get("link_relative_position_mm")),
                    _fmt(row.get("mpjpe_mm")),
                    _fmt(row.get("root_position_mm")),
                    _fmt(row.get("objective"), 6),
                )
            )
            + " |"
        )
    improvements = _mapping(report.get("improvements"), "report.improvements")
    lines.extend(
        [
            "",
            f"说明：step50 的 n=64 候选与实际 step153 initial_mean 是不同 latent；两个 search estimate 都是 history 单次估计，最终 headline 使用 {candidate_label} 的 n=64 paired validation。",
            "",
            "## 改善与权衡",
            "",
            "| 对比 | Link 降低 (mm) | Link 降低 (%) | MPJPE 降低 (mm) | Root 降低 (mm) | Objective Δ |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    improvement_rows = [
        (f"Raw baseline → final {candidate_label}", "raw_baseline_to_final"),
        (
            f"Step50 validated candidate → final {candidate_label}",
            "pre_overnight_validated_candidate_to_final",
        ),
        (
            f"Actual initial_mean search estimate → final n64 {candidate_label}*",
            "actual_overnight_initial_mean_to_final",
        ),
    ]
    if not bool(selected.get("same_validated_latent")):
        improvement_rows.append(
            (
                f"{other_kind}-best → selected {candidate_label}",
                "other_n64_to_selected_candidate",
            )
        )
    for label, key in improvement_rows:
        delta = improvements[key]
        lines.append(
            "| "
            + " | ".join(
                (
                    label,
                    _fmt(delta["link_relative_position_mm"]["reduction"]),
                    _fmt(delta["link_relative_position_mm"]["reduction_percent"], 2),
                    _fmt(delta["mpjpe_mm"]["reduction"]),
                    _fmt(delta["root_position_mm"]["reduction"]),
                    _fmt(delta["objective_delta"], 6),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "\\* 该行跨 measurement level，仅作描述，不是 paired validation。",
        ]
    )
    if bool(selected.get("same_validated_latent")):
        relationship = _mapping(
            report.get("validated_candidate_relationship"),
            "report.validated_candidate_relationship",
        )
        delta = _mapping(
            relationship.get("selected_minus_other"),
            "validated_candidate_relationship.selected_minus_other",
        )
        lines.extend(
            [
                "",
                "### 同一 latent 的独立验证波动",
                "",
                "`validated_objective_best_z.npy` 与 `validated_link_best_z.npy` 的 SHA256 完全相同。以下差值为两个独立 n=64 rollout 批次的 selected-minus-other 波动，仅作描述，不属于算法 improvement。",
                "",
                f"- Local link：{_fmt(delta['link_relative_position_mm'])} mm",
                f"- MPJPE：{_fmt(delta['mpjpe_mm'])} mm",
                f"- Root position：{_fmt(delta['root_position_mm'])} mm",
                f"- Objective：{_fmt(delta['objective'], 6)}",
            ]
        )

    ess = _mapping(report.get("mppi_ess"), "report.mppi_ess")
    lines.extend(
        [
            "",
            f"## MPPI ESS（{ess['summary']['window_size']}-step 窗口）",
            "",
            "| Completed steps | n | ESS mean | ESS min | ESS max | Link start→end (mm) | Link gain (mm) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for window in ess["windows"]:
        step_range = window["completed_step_range"]
        ess_ratio = window["ess_ratio"]
        lines.append(
            "| "
            + " | ".join(
                (
                    f"{step_range[0]}–{step_range[1]}",
                    str(window["record_count"]),
                    _fmt(ess_ratio["mean"]),
                    _fmt(ess_ratio["min"]),
                    _fmt(ess_ratio["max"]),
                    f"{_fmt(window['link_best_mm_start'])}→{_fmt(window['link_best_mm_end'])}",
                    _fmt(window["link_best_gain_mm"]),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 已完成的受控 A/B",
            "",
            "| Step | 测试 | Link gain (mm) | MPJPE regression (mm) | Root regression (mm) | 结论 |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in report["controlled_ab_comparisons"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    _fmt(row.get("comparison_completed_step"), 0),
                    _md_escape(row.get("changed_parameter")),
                    _fmt(row.get("validated_link_gain_mm")),
                    _fmt(row.get("mpjpe_regression_mm")),
                    _fmt(row.get("root_position_regression_mm")),
                    _md_escape(row.get("outcome")),
                )
            )
            + " |"
        )

    lines.extend(["", "决策明细：", ""])
    for row in report["controlled_ab_comparisons"]:
        lines.append(
            f"- `{_md_escape(row['path'])}`：{_md_escape(row['decision'])}"
        )

    params = report["best_parameters"]
    lines.extend(
        [
            "",
            "## 最优参数",
            "",
            f"- MPPI：particles={params['particles']}, sigma0={params['sigma0']}, "
            f"temperature={params['temperature']}, beta_iteration={params['beta_iteration']}, "
            f"beta_horizon={params['beta_horizon']}, smoothing={params['smoothing_window']}",
            f"- Score/guard：{params['score_mode']} / {params['link_guard']}，blend alpha={params['link_blend_alpha']}",
            f"- Reward weights：`{json.dumps(params['weights'], ensure_ascii=False, sort_keys=True)}`",
            f"- Reward sigmas：`{json.dumps(params['sigmas'], ensure_ascii=False, sort_keys=True)}`",
            "",
            "## 产物",
            "",
            f"- TensorBoard：{report['artifacts']['tensorboard']['url']}  ",
            f"  Logdir：`{_md_escape(report['artifacts']['tensorboard']['logdir'])}`",
            f"- REF | BASELINE | BEST ({candidate_label}) 视频：`{_md_escape(report['artifacts']['comparison_video']['absolute'])}`",
            f"  说明：{_md_escape(report['artifacts']['comparison_video']['description'])}",
            f"- 最终 run：`{_md_escape(report['run']['run_dir'])}`",
            "",
            "## 血缘与审计",
            "",
            f"- 实际 initial_mean SHA256：`{report['lineage']['actual_overnight_initial_mean']['initial_mean_sha256']}`",
            f"- 实际来源：`{_md_escape(report['lineage']['actual_overnight_initial_mean']['latent_source_path'])}`，source step {report['lineage']['actual_overnight_initial_mean']['completed_step']}",
            f"- step50 validated latent：`{_md_escape(report['lineage']['pre_overnight_validated_candidate']['validated_latent_path'])}`",
            f"- Final audit：**{report['audit']['status']}**，{len(report['audit']['passed_checks'])} 项通过",
            "- `final_report.json` 为提交记录，并保存实际 Markdown 字节的 SHA256。",
            "",
        ]
    )
    return "\n".join(lines)


def build_report_plan(
    output_root: Path | str,
    run_dir: Path | str,
    warm_start_dir: Path | str,
    *,
    candidate_kind: str = "link",
    tensorboard_url: str = "http://localhost:6007",
    window_size: int = DEFAULT_WINDOW_SIZE,
    expected_frame_count: int = 500,
    expected_validation_count: int = 64,
    audit_runner: AuditRunner = _default_audit_runner,
    tensorboard_reader: TensorBoardReader = _read_tensorboard_metadata,
    now: datetime | None = None,
) -> ReportPlan:
    output_path = Path(output_root).expanduser().resolve()
    run_path = Path(run_dir).expanduser().resolve()
    warm_path = Path(warm_start_dir).expanduser().resolve()
    if candidate_kind not in {"link", "objective"}:
        raise ReportValidationError(
            f"candidate_kind must be 'link' or 'objective', got {candidate_kind!r}"
        )
    if not output_path.is_dir():
        raise ReportValidationError(f"output root does not exist: {output_path}")
    if run_path.parent != output_path or not run_path.is_dir():
        raise ReportValidationError("run_dir must be a direct child of output_root")
    if not warm_path.is_dir():
        raise ReportValidationError(f"warm-start directory does not exist: {warm_path}")

    required_json: dict[str, Mapping[str, Any]] = {}
    source_snapshots: list[FileSnapshot] = []
    for name in (
        "summary.json",
        "metrics_baseline.json",
        "metrics_adapted.json",
        "metrics_link_best.json",
        "run_config.json",
        "promotion.json",
        "video_status.json",
    ):
        value, snapshot = _load_json_snapshot(run_path / name)
        required_json[name] = _mapping(value, name)
        source_snapshots.append(snapshot)

    summary = required_json["summary.json"]
    adapted_file = required_json["metrics_adapted.json"]
    link_file = required_json["metrics_link_best.json"]
    baseline_file = required_json["metrics_baseline.json"]
    run_config = required_json["run_config.json"]
    completed = _integer(summary, "completed_iterations", "summary")
    frame_count = _integer(summary, "frame_count", "summary")
    if frame_count != expected_frame_count:
        raise ReportValidationError(
            f"frame_count={frame_count}, expected {expected_frame_count}"
        )

    audit = dict(
        audit_runner(
            run_path,
            output_root=output_path,
            expected_frame_count=expected_frame_count,
            expected_validation_count=expected_validation_count,
        )
    )
    _assert_finite(audit, "audit")
    _validate_audit(audit, run_path, output_path, completed)

    history, history_snapshot = _load_history(run_path / "history.jsonl", completed)
    source_snapshots.append(history_snapshot)
    ess_summary, ess = _history_summary(history, window_size)
    search = _search_estimate(history, candidate_kind)

    final_initial_mean = _snapshot(Path(str(run_config.get("initial_mean_z", ""))))
    source_snapshots.append(final_initial_mean)
    pre_validated, pre_validated_snapshots = (
        _load_pre_overnight_validated_candidate(warm_path)
    )
    actual_initial, actual_initial_snapshots = _load_actual_overnight_initial_mean(
        warm_path, final_initial_mean
    )
    source_snapshots.extend(pre_validated_snapshots)
    source_snapshots.extend(actual_initial_snapshots)
    validated_objective_latent = _snapshot(
        run_path / "validated_objective_best_z.npy"
    )
    validated_link_latent = _snapshot(run_path / "validated_link_best_z.npy")
    source_snapshots.extend(
        (validated_objective_latent, validated_link_latent)
    )
    same_validated_latent = (
        validated_objective_latent.sha256 == validated_link_latent.sha256
    )

    comparisons, comparison_snapshots = _load_comparisons(output_path)
    source_snapshots.extend(comparison_snapshots)
    promotions, promotion_snapshots = _load_promotions(output_path)
    source_snapshots.extend(promotion_snapshots)

    tensorboard, tensorboard_snapshots = tensorboard_reader(run_path / "tensorboard")
    source_snapshots.extend(tensorboard_snapshots)
    canonical_series = _mapping(
        tensorboard.get("canonical_training_series"),
        "tensorboard.canonical_training_series",
    )
    if _integer(canonical_series, "last_step", "canonical_training_series") + 1 != completed:
        raise ReportValidationError("TensorBoard training step does not match summary")

    adapted_metrics = _mapping(adapted_file.get("metrics"), "metrics_adapted.metrics")
    adapted_std = _mapping(
        adapted_file.get("population_std"), "metrics_adapted.population_std"
    )
    link_metrics = _mapping(link_file.get("metrics"), "metrics_link_best.metrics")
    link_std = _mapping(
        link_file.get("population_std"), "metrics_link_best.population_std"
    )
    link_counts = _mapping(
        link_file.get("validation_counts"), "metrics_link_best.validation_counts"
    )
    adapted_counts = _mapping(
        adapted_file.get("validation_counts"), "metrics_adapted.validation_counts"
    )
    link_count = _integer(link_counts, "adapted", "link validation_counts")
    adapted_count = _integer(adapted_counts, "adapted", "adapted validation_counts")
    if link_count != expected_validation_count or adapted_count != expected_validation_count:
        raise ReportValidationError("final validation count differs from expected n")

    raw_baselines: dict[str, dict[str, Any]] = {}
    for kind, metrics_file, counts, label in (
        ("objective", adapted_file, adapted_counts, "metrics_adapted"),
        ("link", link_file, link_counts, "metrics_link_best"),
    ):
        paired_metrics = _mapping(
            metrics_file.get("paired_baseline_metrics"),
            f"{label}.paired_baseline_metrics",
        )
        paired_std = _mapping(
            metrics_file.get("paired_baseline_std"), f"{label}.paired_baseline_std"
        )
        baseline_count = _integer(counts, "baseline", f"{kind} validation_counts")
        raw_baselines[kind] = _metric_view(
            paired_metrics,
            std=paired_std,
            sample_count=baseline_count,
            provenance=(
                f"raw baseline_z rollout paired with final {kind}-best validation"
            ),
        )
    raw_baseline = raw_baselines[candidate_kind]
    objective_best = _metric_view(
        adapted_metrics,
        std=adapted_std,
        sample_count=adapted_count,
        provenance="final n=64 independently validated objective-best latent",
    )
    link_best = _metric_view(
        link_metrics,
        std=link_std,
        sample_count=link_count,
        provenance="final n=64 independently validated link-best latent",
    )
    validated_candidates = {"objective": objective_best, "link": link_best}
    objective_best["same_validated_latent"] = same_validated_latent
    objective_best["validated_latent_sha256"] = validated_objective_latent.sha256
    link_best["same_validated_latent"] = same_validated_latent
    link_best["validated_latent_sha256"] = validated_link_latent.sha256
    selected_candidate_metrics = validated_candidates[candidate_kind]
    other_candidate_kind = "objective" if candidate_kind == "link" else "link"
    other_candidate_metrics = validated_candidates[other_candidate_kind]
    diagnostic_metrics = _mapping(
        baseline_file.get("metrics"), "metrics_baseline.metrics"
    )
    diagnostic_std = _mapping(
        baseline_file.get("population_std"), "metrics_baseline.population_std"
    )
    baseline_diagnostic = _metric_view(
        diagnostic_metrics,
        std=diagnostic_std,
        sample_count=expected_validation_count,
        provenance="separate raw-baseline determinism diagnostic; not the paired headline",
    )
    baseline_diagnostic["repeat_objective_delta"] = _number(
        diagnostic_metrics, "repeat_objective_delta", "metrics_baseline.metrics"
    )
    baseline_diagnostic["repeat_mpjpe_delta_mm"] = 1000.0 * _number(
        diagnostic_metrics, "repeat_mpjpe_delta", "metrics_baseline.metrics"
    )

    measurements: dict[str, Mapping[str, Any]] = {
        "raw_paired_baseline": raw_baseline,
        "raw_paired_baseline_by_candidate": raw_baselines,
        "pre_overnight_validated_candidate": pre_validated,
        "actual_overnight_initial_mean": actual_initial,
        "search_estimate": search,
        "n64_validated_objective_best": objective_best,
        "n64_validated_link_best": link_best,
        "selected_n64_candidate": selected_candidate_metrics,
        "raw_baseline_repeat_diagnostic": baseline_diagnostic,
    }
    pre_validated_to_final = _comparison_delta(
        pre_validated, selected_candidate_metrics
    )
    pre_validated_to_final.update(
        {
            "objective_delta": None,
            "comparison_semantics": (
                "both endpoints are independently validated, but not paired; reward "
                "configurations differ, so objective delta is intentionally omitted"
            ),
        }
    )
    actual_initial_to_final = _comparison_delta(
        actual_initial, selected_candidate_metrics
    )
    actual_initial_to_final.update(
        {
            "objective_delta": None,
            "comparison_semantics": (
                "descriptive cross-level comparison: source search estimate versus "
                "final n=64 validation; reward configurations differ and this is not "
                "a paired validation"
            ),
        }
    )
    improvements: dict[str, Any] = {
        "selected_candidate_kind": candidate_kind,
        "raw_baseline_to_final": _comparison_delta(
            raw_baseline, selected_candidate_metrics
        ),
        "pre_overnight_validated_candidate_to_final": pre_validated_to_final,
        "actual_overnight_initial_mean_to_final": actual_initial_to_final,
        "search_estimate_to_validated_gap": {
            "validated_minus_search_link_mm": float(
                selected_candidate_metrics["link_relative_position_mm"]
            )
            - float(search["link_relative_position_mm"]),
            "validated_minus_search_mpjpe_mm": float(
                selected_candidate_metrics["mpjpe_mm"]
            )
            - float(search["mpjpe_mm"]),
            "validated_minus_search_root_position_mm": float(
                selected_candidate_metrics["root_position_mm"]
            )
            - float(search["root_position_mm"]),
            "candidate_kind": candidate_kind,
        },
    }
    candidate_relationship: dict[str, Any] = {
        "same_validated_latent": same_validated_latent,
        "objective_validated_latent_sha256": validated_objective_latent.sha256,
        "link_validated_latent_sha256": validated_link_latent.sha256,
    }
    if same_validated_latent:
        candidate_relationship.update(
            {
                "relationship": "same_latent_independent_validation_delta",
                "descriptive_only": True,
                "is_algorithm_improvement": False,
                "selected_candidate_kind": candidate_kind,
                "other_candidate_kind": other_candidate_kind,
                "selected_minus_other": {
                    "link_relative_position_mm": float(
                        selected_candidate_metrics["link_relative_position_mm"]
                    )
                    - float(other_candidate_metrics["link_relative_position_mm"]),
                    "mpjpe_mm": float(selected_candidate_metrics["mpjpe_mm"])
                    - float(other_candidate_metrics["mpjpe_mm"]),
                    "root_position_mm": float(
                        selected_candidate_metrics["root_position_mm"]
                    )
                    - float(other_candidate_metrics["root_position_mm"]),
                    "objective": float(selected_candidate_metrics["objective"])
                    - float(other_candidate_metrics["objective"]),
                },
                "explanation": (
                    "objective-best and link-best files are byte-identical; metric "
                    "differences come from separate n=64 rollout validation batches"
                ),
            }
        )
    else:
        improvements["other_n64_to_selected_candidate"] = _comparison_delta(
            other_candidate_metrics, selected_candidate_metrics
        )
        candidate_relationship.update(
            {
                "relationship": "distinct_validated_latents",
                "descriptive_only": False,
                "is_algorithm_improvement": True,
                "selected_candidate_kind": candidate_kind,
                "other_candidate_kind": other_candidate_kind,
            }
        )

    parameters = _parameter_summary(run_config, summary)
    configured_video_kind = str(run_config.get("video_best_kind", candidate_kind))
    if configured_video_kind not in {"link", "objective"}:
        raise ReportValidationError(
            f"run_config.video_best_kind is invalid: {configured_video_kind!r}"
        )
    if configured_video_kind != candidate_kind:
        raise ReportValidationError(
            "selected candidate does not match rendered BEST panel: "
            f"candidate_kind={candidate_kind}, video_best_kind={configured_video_kind}"
        )
    selected_latent_filename = (
        "validated_link_best_z.npy"
        if candidate_kind == "link"
        else "validated_objective_best_z.npy"
    )
    generated = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    event_files = tensorboard.get("event_files", [])
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "READY",
        "commit_protocol": "final_report.md_then_final_report.json_commit_record",
        "generated_at": {
            "utc": generated.isoformat(),
            "asia_shanghai": generated.astimezone(LOCAL_TIME_ZONE).isoformat(),
        },
        "run": {
            "run_dir": str(run_path),
            "output_root": str(output_path),
            "motion": summary.get("motion"),
            "model_folder": run_config.get("model_folder"),
            "frame_count": frame_count,
            "fps": run_config.get("fps"),
            "completed_iterations": completed,
            "history_first_iteration": _integer(history[0], "iteration", "history"),
            "history_last_iteration": _integer(history[-1], "iteration", "history"),
            "history_record_count": len(history),
            "promoted_history_compute_seconds": ess_summary["iteration_seconds_sum"],
        },
        "measurement_semantics": {
            "raw_paired_baseline": (
                f"unadapted baseline_z, n=64 and paired with selected "
                f"{candidate_kind}-best candidate"
            ),
            "pre_overnight_validated_candidate": (
                "step-specific n=64 validation from before overnight tuning; its "
                "validated latent is not the actual overnight initial_mean"
            ),
            "actual_overnight_initial_mean": (
                "hash-matched source link_best_z used by overnight tuning; values are "
                "the source history's final single search estimate, not n=64 validation"
            ),
            "search_estimate": "single optimizer/search evaluation from history; no repeat std",
            "n64_validated": "independent 64-rollout paired validation; headline result",
            "population_std_note": "reported std is population std across validation rollouts, not standard error",
        },
        "selected_candidate": {
            "kind": candidate_kind,
            "label": f"{candidate_kind}-best",
            "measurement_key": f"n64_validated_{candidate_kind}_best",
            "sample_count": selected_candidate_metrics["sample_count"],
            "validated_latent_path": str(
                (run_path / selected_latent_filename).resolve()
            ),
            "paired_baseline_provenance": raw_baseline["provenance"],
            "headline_uses_n64_validation": True,
            "same_validated_latent": same_validated_latent,
            "validated_latent_sha256": selected_candidate_metrics[
                "validated_latent_sha256"
            ],
        },
        "measurements": measurements,
        "improvements": improvements,
        "validated_candidate_relationship": candidate_relationship,
        "mppi_ess": ess,
        "best_parameters": parameters,
        "controlled_ab_comparisons": comparisons,
        "promotion_chain": promotions,
        "lineage": {
            "warm_start_dir": str(warm_path),
            "pre_overnight_validated_candidate": {
                "completed_step": pre_validated["completed_step"],
                "validated_latent_path": pre_validated["validated_latent_path"],
                "validated_latent_sha256": pre_validated[
                    "validated_latent_sha256"
                ],
                "independently_validated": True,
                "is_actual_overnight_initial_mean": (
                    pre_validated["validated_latent_sha256"]
                    == actual_initial["initial_mean_sha256"]
                    if pre_validated["validated_latent_sha256"] is not None
                    else None
                ),
            },
            "actual_overnight_initial_mean": {
                "completed_step": actual_initial["completed_step"],
                "history_iteration": actual_initial["history_iteration"],
                "initial_mean_path": actual_initial["initial_mean_path"],
                "initial_mean_sha256": actual_initial["initial_mean_sha256"],
                "latent_source_path": actual_initial["latent_source_path"],
                "latent_sha256": actual_initial["latent_sha256"],
                "hash_match": actual_initial["hash_match"],
                "independently_validated": False,
            },
        },
        "artifacts": {
            "comparison_video": {
                **_path_entry(
                    run_path / "comparison_ref_baseline_best.mp4",
                    relative_to=output_path,
                ),
                "panel_order": ["REF", "BASELINE", "BEST"],
                "best_panel_candidate_kind": candidate_kind,
                "render_config_video_best_kind": configured_video_kind,
                "description": (
                    f"MuJoCo REF | BASELINE | BEST; BEST panel is the selected "
                    f"n=64 {candidate_kind}-best latent"
                ),
            },
            "checkpoint": _path_entry(run_path / "checkpoint.pt", relative_to=output_path),
            "validated_objective_latent": _path_entry(
                run_path / "validated_objective_best_z.npy", relative_to=output_path
            ),
            "validated_link_latent": _path_entry(
                run_path / "validated_link_best_z.npy", relative_to=output_path
            ),
            "selected_validated_latent": _path_entry(
                run_path / selected_latent_filename, relative_to=output_path
            ),
            "tensorboard": {
                "url": tensorboard_url,
                "logdir": tensorboard.get("logdir"),
                "event_file_count": len(event_files),
            },
        },
        "tensorboard_metadata": tensorboard,
        "audit": {
            "status": audit["status"],
            "schema_version": audit.get("schema_version"),
            "observed": audit["observed"],
            "passed_checks": [
                check.get("name")
                for check in audit.get("checks", [])
                if isinstance(check, Mapping) and check.get("status") == "pass"
            ],
            "failure_count": audit["failure_count"],
            "catalog_apply_performed": audit.get("catalog_apply_performed"),
        },
    }
    report["conclusions"] = _conclusions(
        measurements, comparisons, ess_summary, candidate_kind
    )

    source_snapshots_tuple = tuple(
        {str(snapshot.path.resolve()): snapshot for snapshot in source_snapshots}.values()
    )
    report["source_artifacts"] = _source_manifest(
        source_snapshots_tuple, output_path
    )
    generation_basis = {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_path),
        "completed_iterations": completed,
        "candidate_kind": candidate_kind,
        "source_artifacts": report["source_artifacts"],
        "audit_status": audit["status"],
        "measurements": measurements,
        "best_parameters": parameters,
    }
    report["report_generation"] = _sha256(_compact_bytes(generation_basis))
    report["report_files"] = {
        "final_report.md": {
            "path": str((output_path / "final_report.md").resolve()),
        },
        "final_report.json": {
            "path": str((output_path / "final_report.json").resolve()),
            "role": "commit record written last",
        },
    }
    markdown = render_markdown(report)
    markdown_bytes = markdown.encode("utf-8")
    report["report_files"]["final_report.md"]["sha256"] = _sha256(markdown_bytes)
    report["report_files"]["final_report.md"]["size_bytes"] = len(markdown_bytes)
    json_bytes = _canonical_bytes(report)
    _ensure_unchanged(source_snapshots_tuple)
    return ReportPlan(
        output_path,
        run_path,
        report,
        markdown,
        json_bytes,
        markdown_bytes,
        source_snapshots_tuple,
    )


def _atomic_install_if_changed(path: Path, data: bytes) -> bool:
    try:
        if path.read_bytes() == data:
            return False
    except FileNotFoundError:
        pass
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return True


def apply_report_plan(plan: ReportPlan) -> list[str]:
    """Atomically install Markdown, then the JSON commit record."""
    changed: list[str] = []
    _ensure_unchanged(plan.source_snapshots)
    markdown_path = plan.output_root / "final_report.md"
    json_path = plan.output_root / "final_report.json"
    if _atomic_install_if_changed(markdown_path, plan.markdown_bytes):
        changed.append(markdown_path.name)
    _ensure_unchanged(plan.source_snapshots)
    if _atomic_install_if_changed(json_path, plan.json_bytes):
        changed.append(json_path.name)
    directory_descriptor = os.open(plan.output_root, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    if markdown_path.read_bytes() != plan.markdown_bytes:
        raise ReportValidationError("committed final_report.md differs from report plan")
    if json_path.read_bytes() != plan.json_bytes:
        raise ReportValidationError("committed final_report.json differs from report plan")
    _ensure_unchanged(plan.source_snapshots)
    return changed


def generate_final_report(
    output_root: Path | str,
    run_dir: Path | str,
    warm_start_dir: Path | str,
    *,
    apply: bool,
    candidate_kind: str = "link",
    tensorboard_url: str = "http://localhost:6007",
    window_size: int = DEFAULT_WINDOW_SIZE,
    expected_frame_count: int = 500,
    expected_validation_count: int = 64,
    audit_runner: AuditRunner = _default_audit_runner,
    tensorboard_reader: TensorBoardReader = _read_tensorboard_metadata,
    now: datetime | None = None,
) -> tuple[ReportPlan, list[str]]:
    output_path = Path(output_root).expanduser().resolve()
    if not apply:
        return (
            build_report_plan(
                output_path,
                run_dir,
                warm_start_dir,
                candidate_kind=candidate_kind,
                tensorboard_url=tensorboard_url,
                window_size=window_size,
                expected_frame_count=expected_frame_count,
                expected_validation_count=expected_validation_count,
                audit_runner=audit_runner,
                tensorboard_reader=tensorboard_reader,
                now=now,
            ),
            [],
        )
    lock_path = output_path / ".final_report.lock"
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        plan = build_report_plan(
            output_path,
            run_dir,
            warm_start_dir,
            candidate_kind=candidate_kind,
            tensorboard_url=tensorboard_url,
            window_size=window_size,
            expected_frame_count=expected_frame_count,
            expected_validation_count=expected_validation_count,
            audit_runner=audit_runner,
            tensorboard_reader=tensorboard_reader,
            now=now,
        )
        return plan, apply_report_plan(plan)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a finalized promoted run and generate final_report.json/.md. "
            "Defaults to a no-write preview."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--warm-start-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-kind",
        choices=("link", "objective"),
        default="link",
        help="Candidate used for headline, paired baseline, improvements, and BEST video panel.",
    )
    parser.add_argument("--tensorboard-url", default="http://localhost:6007")
    parser.add_argument("--history-window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--expected-frame-count", type=int, default=500)
    parser.add_argument("--expected-validation-count", type=int, default=64)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Atomically write final_report.md, then final_report.json.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        plan, changed = generate_final_report(
            args.output_root,
            args.run_dir,
            args.warm_start_dir,
            apply=args.apply,
            candidate_kind=args.candidate_kind,
            tensorboard_url=args.tensorboard_url,
            window_size=args.history_window_size,
            expected_frame_count=args.expected_frame_count,
            expected_validation_count=args.expected_validation_count,
        )
    except ReportValidationError as error:
        print(
            json.dumps(
                {"status": "error", "error": str(error)}, ensure_ascii=False
            ),
            file=sys.stderr,
        )
        return 2
    preview = {
        "mode": "apply" if args.apply else "dry-run",
        "status": "written" if args.apply else "validated",
        "report_generation": plan.report["report_generation"],
        "completed_iterations": plan.report["run"]["completed_iterations"],
        "selected_candidate_kind": plan.report["selected_candidate"]["kind"],
        "selected_validated_link_mm": plan.report["measurements"]
        ["selected_n64_candidate"]["link_relative_position_mm"],
        "changed_files": changed,
        "would_write": []
        if args.apply
        else [
            str(plan.output_root / "final_report.md"),
            str(plan.output_root / "final_report.json"),
        ],
        "catalog_apply_performed": False,
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
