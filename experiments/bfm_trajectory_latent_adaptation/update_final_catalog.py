"""Validate a finalized promoted run and publish catalog schema version 2.

The command is intentionally dry-run by default.  It only writes when ``--apply``
is supplied.  ``catalog_manifest.json`` is installed last and is the commit record
for the two mutable catalog views.  The original screening ranking is copied to
``ranking_screen.json`` once and is never overwritten.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

SCHEMA_VERSION = 2
HASH_ALGORITHM = "sha256"
SOURCE_FILENAMES = (
    "checkpoint.pt",
    "summary.json",
    "metrics_baseline.json",
    "metrics_adapted.json",
    "metrics_link_best.json",
    "run_config.json",
    "validated_objective_best_z.npy",
    "validated_link_best_z.npy",
)


class CatalogValidationError(ValueError):
    """Raised when a promoted directory is not a self-consistent final result."""


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    data: bytes
    sha256: str
    stat_signature: tuple[int, int, int, int]


@dataclass(frozen=True)
class CatalogPlan:
    output_root: Path
    promoted_dir: Path
    generation: str
    promoted_row: dict[str, Any]
    ranking_bytes: bytes
    latest_best_bytes: bytes
    manifest_bytes: bytes
    manifest: dict[str, Any]
    screen_bytes: bytes
    screen_snapshot: FileSnapshot
    create_screen_backup: bool
    source_snapshots: tuple[FileSnapshot, ...]


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


def _compact_canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _snapshot_file(path: Path) -> FileSnapshot:
    try:
        before = _stat_signature(path)
        data = path.read_bytes()
        after = _stat_signature(path)
    except FileNotFoundError as error:
        raise CatalogValidationError(f"required file is missing: {path}") from error
    if before != after or len(data) != after[2]:
        raise CatalogValidationError(f"file changed while being read: {path}")
    return FileSnapshot(
        path=path,
        data=data,
        sha256=_sha256(data),
        stat_signature=after,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CatalogValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(snapshot: FileSnapshot) -> Any:
    try:
        return json.loads(
            snapshot.data.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CatalogValidationError(f"invalid JSON in {snapshot.path}: {error}") from error


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CatalogValidationError(f"{label} must be a JSON object")
    return value


def _require_number(mapping: Mapping[str, Any], key: str, label: str) -> float:
    if key not in mapping:
        raise CatalogValidationError(f"{label} is missing {key!r}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CatalogValidationError(f"{label}.{key} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CatalogValidationError(f"{label}.{key} must be finite")
    return result


def _require_int(mapping: Mapping[str, Any], key: str, label: str) -> int:
    value = _require_number(mapping, key, label)
    result = int(value)
    if result != value:
        raise CatalogValidationError(f"{label}.{key} must be an integer")
    return result


def _assert_close(actual: Any, expected: Any, label: str) -> None:
    if isinstance(actual, bool) or isinstance(expected, bool):
        if actual is not expected:
            raise CatalogValidationError(
                f"{label} differs: {actual!r} != {expected!r}"
            )
        return
    if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
        if actual != expected:
            raise CatalogValidationError(
                f"{label} differs: {actual!r} != {expected!r}"
            )
        return
    actual_float = float(actual)
    expected_float = float(expected)
    if not (math.isfinite(actual_float) and math.isfinite(expected_float)):
        raise CatalogValidationError(f"{label} contains NaN or Inf")
    if not math.isclose(actual_float, expected_float, rel_tol=1.0e-7, abs_tol=1.0e-9):
        raise CatalogValidationError(
            f"{label} differs: {actual_float:.17g} != {expected_float:.17g}"
        )


def _assert_mapping_close(
    actual: Any, expected: Any, label: str, *, exact_keys: bool = True
) -> None:
    actual_mapping = _require_mapping(actual, f"{label} (checkpoint)")
    expected_mapping = _require_mapping(expected, f"{label} (JSON)")
    actual_keys = set(actual_mapping)
    expected_keys = set(expected_mapping)
    if exact_keys and actual_keys != expected_keys:
        raise CatalogValidationError(
            f"{label} keys differ: checkpoint={sorted(actual_keys)}, "
            f"JSON={sorted(expected_keys)}"
        )
    for key in sorted(expected_keys):
        if key not in actual_mapping:
            raise CatalogValidationError(f"{label} checkpoint is missing {key!r}")
        _assert_close(actual_mapping[key], expected_mapping[key], f"{label}.{key}")


def _load_checkpoint(snapshot: FileSnapshot) -> Mapping[str, Any]:
    try:
        try:
            value = torch.load(
                io.BytesIO(snapshot.data), map_location="cpu", weights_only=True
            )
        except TypeError:
            value = torch.load(io.BytesIO(snapshot.data), map_location="cpu")
    except Exception as error:
        raise CatalogValidationError(
            f"cannot load checkpoint {snapshot.path}: {error}"
        ) from error
    return _require_mapping(value, "checkpoint")


def _load_npy(snapshot: FileSnapshot) -> np.ndarray:
    try:
        value = np.load(io.BytesIO(snapshot.data), allow_pickle=False)
    except Exception as error:
        raise CatalogValidationError(
            f"cannot load NumPy artifact {snapshot.path}: {error}"
        ) from error
    if not isinstance(value, np.ndarray):
        raise CatalogValidationError(f"{snapshot.path} is not a NumPy array")
    if value.dtype != np.float32:
        raise CatalogValidationError(
            f"{snapshot.path} must have dtype float32, got {value.dtype}"
        )
    if not np.isfinite(value).all():
        raise CatalogValidationError(f"{snapshot.path} contains NaN or Inf")
    return value


def _validate_tensor_array(
    checkpoint: Mapping[str, Any],
    checkpoint_key: str,
    array: np.ndarray,
    expected_shape: tuple[int, ...],
    label: str,
) -> None:
    if array.shape != expected_shape:
        raise CatalogValidationError(
            f"{label} shape {array.shape} does not match summary {expected_shape}"
        )
    tensor = checkpoint.get(checkpoint_key)
    if not isinstance(tensor, torch.Tensor):
        raise CatalogValidationError(f"checkpoint.{checkpoint_key} must be a tensor")
    tensor = tensor.detach().cpu()
    if tensor.dtype != torch.float32:
        raise CatalogValidationError(
            f"checkpoint.{checkpoint_key} must have dtype float32, got {tensor.dtype}"
        )
    tensor_array = tensor.numpy()
    if tensor_array.shape != expected_shape:
        raise CatalogValidationError(
            f"checkpoint.{checkpoint_key} shape {tensor_array.shape} does not match "
            f"summary {expected_shape}"
        )
    if not np.isfinite(tensor_array).all():
        raise CatalogValidationError(f"checkpoint.{checkpoint_key} contains NaN or Inf")
    if not np.array_equal(tensor_array, array):
        difference = float(np.max(np.abs(tensor_array - array)))
        raise CatalogValidationError(
            f"{label} differs from checkpoint.{checkpoint_key}; max_abs={difference:.9g}"
        )


def _check_summary_metric(
    summary: Mapping[str, Any],
    summary_key: str,
    metrics: Mapping[str, Any],
    metric_group: str,
    metric_key: str,
) -> None:
    group = _require_mapping(metrics.get(metric_group), f"metrics.{metric_group}")
    _assert_close(
        _require_number(summary, summary_key, "summary"),
        _require_number(group, metric_key, f"metrics.{metric_group}"),
        f"summary.{summary_key}",
    )


def _validate_source(
    output_root: Path, promoted_dir: Path, *, candidate_kind: str
) -> tuple[dict[str, Any], tuple[FileSnapshot, ...]]:
    if candidate_kind not in {"link", "objective"}:
        raise CatalogValidationError(
            f"candidate_kind must be 'link' or 'objective', got {candidate_kind!r}"
        )
    snapshots = tuple(
        _snapshot_file(promoted_dir / filename) for filename in SOURCE_FILENAMES
    )
    by_name = {snapshot.path.name: snapshot for snapshot in snapshots}
    checkpoint = _load_checkpoint(by_name["checkpoint.pt"])
    summary = _require_mapping(_load_json(by_name["summary.json"]), "summary")
    baseline = _require_mapping(
        _load_json(by_name["metrics_baseline.json"]), "metrics_baseline"
    )
    adapted = _require_mapping(
        _load_json(by_name["metrics_adapted.json"]), "metrics_adapted"
    )
    link = _require_mapping(
        _load_json(by_name["metrics_link_best.json"]), "metrics_link_best"
    )
    run_config = _require_mapping(_load_json(by_name["run_config.json"]), "run_config")

    completed_iterations = _require_int(summary, "completed_iterations", "summary")
    checkpoint_iteration = checkpoint.get("iteration")
    if isinstance(checkpoint_iteration, bool) or not isinstance(
        checkpoint_iteration, int
    ):
        raise CatalogValidationError("checkpoint.iteration must be an integer")
    if checkpoint_iteration + 1 != completed_iterations:
        raise CatalogValidationError(
            "checkpoint/summary are not finalized at the same step: "
            f"checkpoint iteration={checkpoint_iteration}, "
            f"summary completed_iterations={completed_iterations}"
        )

    latent_shape_value = summary.get("latent_shape")
    if (
        not isinstance(latent_shape_value, Sequence)
        or isinstance(latent_shape_value, (str, bytes))
        or len(latent_shape_value) != 2
    ):
        raise CatalogValidationError("summary.latent_shape must contain two dimensions")
    try:
        latent_shape = tuple(int(dimension) for dimension in latent_shape_value)
    except (TypeError, ValueError) as error:
        raise CatalogValidationError(
            "summary.latent_shape must contain integers"
        ) from error
    if any(dimension <= 0 for dimension in latent_shape) or list(latent_shape) != list(
        latent_shape_value
    ):
        raise CatalogValidationError("summary.latent_shape must contain positive integers")
    frame_count = _require_int(summary, "frame_count", "summary")
    if frame_count != latent_shape[0] + 1:
        raise CatalogValidationError(
            "summary.frame_count must equal latent horizon + 1: "
            f"{frame_count} != {latent_shape[0]} + 1"
        )

    objective_array = _load_npy(by_name["validated_objective_best_z.npy"])
    link_array = _load_npy(by_name["validated_link_best_z.npy"])
    _validate_tensor_array(
        checkpoint,
        "validated_objective_best",
        objective_array,
        latent_shape,
        "validated_objective_best_z.npy",
    )
    _validate_tensor_array(
        checkpoint,
        "validated_link_best",
        link_array,
        latent_shape,
        "validated_link_best_z.npy",
    )

    for checkpoint_prefix, metrics_file, label in (
        ("validated_objective", adapted, "objective-best"),
        ("validated_link", link, "link-best"),
    ):
        _assert_mapping_close(
            checkpoint.get(f"{checkpoint_prefix}_metrics"),
            metrics_file.get("metrics"),
            f"{label} metrics",
        )
        _assert_mapping_close(
            checkpoint.get(f"{checkpoint_prefix}_rewards"),
            metrics_file.get("rewards"),
            f"{label} rewards",
        )
        _assert_mapping_close(
            checkpoint.get(f"{checkpoint_prefix}_std"),
            metrics_file.get("population_std"),
            f"{label} population_std",
        )

    summary_checks = (
        ("adapted_objective", adapted, "metrics", "objective"),
        ("adapted_mpjpe", adapted, "metrics", "mpjpe"),
        (
            "adapted_link_relative_position_error",
            adapted,
            "metrics",
            "mean_body_position_error",
        ),
        ("adapted_objective_std", adapted, "population_std", "objective"),
        ("adapted_mpjpe_std", adapted, "population_std", "mpjpe"),
        (
            "adapted_link_relative_position_error_std",
            adapted,
            "population_std",
            "mean_body_position_error",
        ),
        ("baseline_objective", adapted, "paired_baseline_metrics", "objective"),
        ("baseline_mpjpe", adapted, "paired_baseline_metrics", "mpjpe"),
        (
            "baseline_link_relative_position_error",
            adapted,
            "paired_baseline_metrics",
            "mean_body_position_error",
        ),
        (
            "baseline_objective_std",
            adapted,
            "paired_baseline_std",
            "objective",
        ),
        ("baseline_mpjpe_std", adapted, "paired_baseline_std", "mpjpe"),
        (
            "baseline_link_relative_position_error_std",
            adapted,
            "paired_baseline_std",
            "mean_body_position_error",
        ),
        ("link_best_objective", link, "metrics", "objective"),
        ("link_best_mpjpe", link, "metrics", "mpjpe"),
        (
            "link_best_link_relative_position_error",
            link,
            "metrics",
            "mean_body_position_error",
        ),
        (
            "link_best_link_relative_position_error_std",
            link,
            "population_std",
            "mean_body_position_error",
        ),
        (
            "baseline_repeat_objective_delta",
            baseline,
            "metrics",
            "repeat_objective_delta",
        ),
        (
            "baseline_repeat_mpjpe_delta",
            baseline,
            "metrics",
            "repeat_mpjpe_delta",
        ),
    )
    for summary_key, metrics_file, metric_group, metric_key in summary_checks:
        _check_summary_metric(
            summary, summary_key, metrics_file, metric_group, metric_key
        )

    summary_counts = _require_mapping(
        summary.get("paired_validation_counts"), "summary.paired_validation_counts"
    )
    for metrics_file, label in ((adapted, "adapted"), (link, "link")):
        counts = _require_mapping(
            metrics_file.get("validation_counts"), f"{label}.validation_counts"
        )
        if dict(counts) != dict(summary_counts):
            raise CatalogValidationError(
                f"{label}.validation_counts does not match summary"
            )

    motion = summary.get("motion")
    if not isinstance(motion, str) or not motion:
        raise CatalogValidationError("summary.motion must be a non-empty path")
    if run_config.get("motion") != motion:
        raise CatalogValidationError("run_config.motion does not match summary.motion")
    if _require_int(run_config, "frame_count", "run_config") != frame_count:
        raise CatalogValidationError(
            "run_config.frame_count does not match summary.frame_count"
        )
    if _require_int(summary, "dial_schedule_iterations", "summary") != _require_int(
        run_config, "iterations", "run_config"
    ):
        raise CatalogValidationError(
            "run_config.iterations does not match summary.dial_schedule_iterations"
        )

    artifact_entries: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        artifact_entries[snapshot.path.name] = {
            "path": str(snapshot.path.relative_to(output_root)),
            "sha256": snapshot.sha256,
            "size_bytes": len(snapshot.data),
        }
    artifact_set_sha256 = _sha256(_compact_canonical_bytes(artifact_entries))

    adapted_metrics = _require_mapping(adapted.get("metrics"), "adapted.metrics")
    link_metrics = _require_mapping(link.get("metrics"), "link.metrics")
    if candidate_kind == "link":
        candidate_file = link
        candidate_label = "link"
        candidate_name = "validated_link_best"
        candidate_latent_artifact = "validated_link_best_z.npy"
    else:
        candidate_file = adapted
        candidate_label = "adapted"
        candidate_name = "validated_objective_best"
        candidate_latent_artifact = "validated_objective_best_z.npy"
    candidate_metrics = _require_mapping(
        candidate_file.get("metrics"), f"{candidate_label}.metrics"
    )
    candidate_std = _require_mapping(
        candidate_file.get("population_std"),
        f"{candidate_label}.population_std",
    )
    candidate_baseline_metrics = _require_mapping(
        candidate_file.get("paired_baseline_metrics"),
        f"{candidate_label}.paired_baseline_metrics",
    )
    candidate_baseline_std = _require_mapping(
        candidate_file.get("paired_baseline_std"),
        f"{candidate_label}.paired_baseline_std",
    )
    validated_link = 1000.0 * _require_number(
        candidate_metrics,
        "mean_body_position_error",
        f"{candidate_label}.metrics",
    )
    validated_mpjpe = 1000.0 * _require_number(
        candidate_metrics, "mpjpe", f"{candidate_label}.metrics"
    )
    validated_objective = _require_number(
        candidate_metrics, "objective", f"{candidate_label}.metrics"
    )
    baseline_link = 1000.0 * _require_number(
        candidate_baseline_metrics,
        "mean_body_position_error",
        f"{candidate_label}.paired_baseline_metrics",
    )
    baseline_mpjpe = 1000.0 * _require_number(
        candidate_baseline_metrics,
        "mpjpe",
        f"{candidate_label}.paired_baseline_metrics",
    )
    baseline_objective = _require_number(
        candidate_baseline_metrics,
        "objective",
        f"{candidate_label}.paired_baseline_metrics",
    )
    eligible = (
        validated_link < baseline_link
        and validated_mpjpe < baseline_mpjpe
        and validated_objective >= baseline_objective
    )
    if not eligible:
        raise CatalogValidationError(
            f"promoted {candidate_name} candidate is not eligible against its paired baseline"
        )

    row_without_generation: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "entry_type": "promoted",
        "rank": 1,
        "run": promoted_dir.name,
        "output_dir": str(promoted_dir),
        "motion": motion,
        "completed_iterations": completed_iterations,
        "candidate_kind": candidate_name,
        "candidate_latent_artifact": candidate_latent_artifact,
        "baseline_link_mm": baseline_link,
        "validated_link_mm": validated_link,
        "validated_link_std_mm": 1000.0
        * _require_number(
            candidate_std,
            "mean_body_position_error",
            f"{candidate_label}.population_std",
        ),
        "baseline_link_std_mm": 1000.0
        * _require_number(
            candidate_baseline_std,
            "mean_body_position_error",
            f"{candidate_label}.paired_baseline_std",
        ),
        "baseline_mpjpe_mm": baseline_mpjpe,
        "validated_mpjpe_mm": validated_mpjpe,
        "baseline_mpjpe_std_mm": 1000.0
        * _require_number(
            candidate_baseline_std,
            "mpjpe",
            f"{candidate_label}.paired_baseline_std",
        ),
        "validated_mpjpe_std_mm": 1000.0
        * _require_number(
            candidate_std, "mpjpe", f"{candidate_label}.population_std"
        ),
        "baseline_objective": baseline_objective,
        "validated_objective": validated_objective,
        "baseline_objective_std": _require_number(
            candidate_baseline_std,
            "objective",
            f"{candidate_label}.paired_baseline_std",
        ),
        "validated_objective_std": _require_number(
            candidate_std, "objective", f"{candidate_label}.population_std"
        ),
        "eligible": eligible,
        "objective_best": {
            "validated_link_mm": 1000.0
            * _require_number(
                adapted_metrics, "mean_body_position_error", "adapted.metrics"
            ),
            "validated_mpjpe_mm": 1000.0
            * _require_number(adapted_metrics, "mpjpe", "adapted.metrics"),
            "validated_objective": _require_number(
                adapted_metrics, "objective", "adapted.metrics"
            ),
        },
        "link_best": {
            "validated_link_mm": 1000.0
            * _require_number(
                link_metrics, "mean_body_position_error", "link.metrics"
            ),
            "validated_mpjpe_mm": 1000.0
            * _require_number(link_metrics, "mpjpe", "link.metrics"),
            "validated_objective": _require_number(
                link_metrics, "objective", "link.metrics"
            ),
        },
        "artifact_hash_algorithm": HASH_ALGORITHM,
        "artifact_set_sha256": artifact_set_sha256,
        "artifacts": artifact_entries,
    }
    return row_without_generation, snapshots


def _screen_snapshot(output_root: Path) -> tuple[FileSnapshot, bool]:
    screen_path = output_root / "ranking_screen.json"
    if screen_path.exists():
        snapshot = _snapshot_file(screen_path)
        create_backup = False
    else:
        snapshot = _snapshot_file(output_root / "ranking.json")
        create_backup = True
    screen_value = _load_json(snapshot)
    if not isinstance(screen_value, list):
        raise CatalogValidationError(
            f"screen ranking must be a JSON list: {snapshot.path}"
        )
    if create_backup and any(
        isinstance(row, Mapping)
        and (
            row.get("schema_version") == SCHEMA_VERSION
            or row.get("entry_type") == "promoted"
        )
        for row in screen_value
    ):
        raise CatalogValidationError(
            "ranking.json already looks promoted but ranking_screen.json is missing"
        )
    manifest_path = output_root / "catalog_manifest.json"
    if not create_backup and manifest_path.exists():
        manifest = _require_mapping(
            _load_json(_snapshot_file(manifest_path)), "catalog_manifest"
        )
        if manifest.get("schema_version") == SCHEMA_VERSION:
            files = _require_mapping(
                manifest.get("files"), "catalog_manifest.files"
            )
            screen_entry = _require_mapping(
                files.get("ranking_screen.json"),
                "catalog_manifest.files.ranking_screen.json",
            )
            if screen_entry.get("sha256") != snapshot.sha256:
                raise CatalogValidationError(
                    "immutable ranking_screen.json hash differs from the committed manifest"
                )
    return snapshot, create_backup


def _ensure_direct_child(output_root: Path, promoted_dir: Path) -> None:
    if promoted_dir.parent != output_root:
        raise CatalogValidationError(
            f"promoted directory must be a direct child of output root: {promoted_dir}"
        )
    if not promoted_dir.is_dir():
        raise CatalogValidationError(f"promoted directory does not exist: {promoted_dir}")


def build_catalog_plan(
    output_root: Path | str,
    promoted_dir: Path | str,
    *,
    candidate_kind: str = "link",
) -> CatalogPlan:
    output_root = Path(output_root).expanduser().resolve()
    promoted_path = Path(promoted_dir).expanduser()
    if not promoted_path.is_absolute() and len(promoted_path.parts) == 1:
        promoted_path = output_root / promoted_path
    promoted_path = promoted_path.resolve()
    if not output_root.is_dir():
        raise CatalogValidationError(f"output root does not exist: {output_root}")
    _ensure_direct_child(output_root, promoted_path)

    row_without_generation, source_snapshots = _validate_source(
        output_root, promoted_path, candidate_kind=candidate_kind
    )
    screen_snapshot, create_screen_backup = _screen_snapshot(output_root)
    generation_input = {
        "schema_version": SCHEMA_VERSION,
        "candidate_kind": row_without_generation["candidate_kind"],
        "promoted_row": row_without_generation,
        "screen_ranking_sha256": screen_snapshot.sha256,
    }
    generation = _sha256(_compact_canonical_bytes(generation_input))
    promoted_row = dict(row_without_generation)
    promoted_row["catalog_generation"] = generation
    ranking_bytes = _canonical_bytes([promoted_row])
    latest_best_bytes = _canonical_bytes(promoted_row)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "catalog_generation": generation,
        "hash_algorithm": HASH_ALGORITHM,
        "commit_protocol": "ranking_screen_once, ranking, latest_best, manifest_last",
        "promoted": {
            "run": promoted_row["run"],
            "completed_iterations": promoted_row["completed_iterations"],
            "candidate_kind": promoted_row["candidate_kind"],
            "artifact_set_sha256": promoted_row["artifact_set_sha256"],
        },
        "files": {
            "ranking_screen.json": {
                "sha256": screen_snapshot.sha256,
                "size_bytes": len(screen_snapshot.data),
                "immutable_after_first_publish": True,
            },
            "ranking.json": {
                "sha256": _sha256(ranking_bytes),
                "size_bytes": len(ranking_bytes),
            },
            "latest_best.json": {
                "sha256": _sha256(latest_best_bytes),
                "size_bytes": len(latest_best_bytes),
            },
        },
    }
    manifest_bytes = _canonical_bytes(manifest)
    return CatalogPlan(
        output_root=output_root,
        promoted_dir=promoted_path,
        generation=generation,
        promoted_row=promoted_row,
        ranking_bytes=ranking_bytes,
        latest_best_bytes=latest_best_bytes,
        manifest_bytes=manifest_bytes,
        manifest=manifest,
        screen_bytes=screen_snapshot.data,
        screen_snapshot=screen_snapshot,
        create_screen_backup=create_screen_backup,
        source_snapshots=source_snapshots,
    )


def _ensure_unchanged(snapshots: Iterable[FileSnapshot]) -> None:
    for snapshot in snapshots:
        try:
            current = _stat_signature(snapshot.path)
        except FileNotFoundError as error:
            raise CatalogValidationError(
                f"validated source disappeared before catalog commit: {snapshot.path}"
            ) from error
        if current != snapshot.stat_signature:
            raise CatalogValidationError(
                f"validated source changed before catalog commit: {snapshot.path}"
            )


def _atomic_install_if_changed(path: Path, data: bytes) -> bool:
    try:
        if path.read_bytes() == data:
            return False
    except FileNotFoundError:
        pass
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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


def _atomic_create_once(path: Path, data: bytes) -> bool:
    """Create an immutable snapshot without ever replacing an existing path."""
    try:
        existing = path.read_bytes()
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if existing != data:
            raise CatalogValidationError(
                f"refusing to overwrite existing immutable snapshot: {path}"
            )
        return False
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise CatalogValidationError(
                    f"immutable snapshot appeared concurrently with different content: {path}"
                )
            return False
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return True


def _verify_committed_catalog(plan: CatalogPlan) -> None:
    ranking = _snapshot_file(plan.output_root / "ranking.json")
    latest = _snapshot_file(plan.output_root / "latest_best.json")
    screen = _snapshot_file(plan.output_root / "ranking_screen.json")
    manifest_snapshot = _snapshot_file(plan.output_root / "catalog_manifest.json")
    manifest = _require_mapping(_load_json(manifest_snapshot), "catalog_manifest")
    if manifest.get("catalog_generation") != plan.generation:
        raise CatalogValidationError("committed manifest generation differs from plan")
    files = _require_mapping(manifest.get("files"), "catalog_manifest.files")
    for name, snapshot in (
        ("ranking.json", ranking),
        ("latest_best.json", latest),
        ("ranking_screen.json", screen),
    ):
        entry = _require_mapping(files.get(name), f"catalog_manifest.files.{name}")
        if entry.get("sha256") != snapshot.sha256:
            raise CatalogValidationError(f"committed hash mismatch for {name}")
    ranking_value = _load_json(ranking)
    latest_value = _load_json(latest)
    if ranking_value != [latest_value]:
        raise CatalogValidationError("ranking.json and latest_best.json disagree")
    latest_mapping = _require_mapping(latest_value, "latest_best")
    if latest_mapping.get("catalog_generation") != plan.generation:
        raise CatalogValidationError("latest_best generation differs from manifest")


def apply_catalog_update(
    output_root: Path | str,
    promoted_dir: Path | str,
    *,
    candidate_kind: str = "link",
) -> tuple[CatalogPlan, list[str]]:
    output_root_path = Path(output_root).expanduser().resolve()
    if not output_root_path.is_dir():
        raise CatalogValidationError(f"output root does not exist: {output_root_path}")
    lock_path = output_root_path / ".catalog_update.lock"
    changed: list[str] = []
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        plan = build_catalog_plan(
            output_root_path, promoted_dir, candidate_kind=candidate_kind
        )
        _ensure_unchanged(plan.source_snapshots + (plan.screen_snapshot,))
        if plan.create_screen_backup:
            if _atomic_create_once(
                output_root_path / "ranking_screen.json", plan.screen_bytes
            ):
                changed.append("ranking_screen.json")
        else:
            current_screen = _snapshot_file(output_root_path / "ranking_screen.json")
            if current_screen.data != plan.screen_bytes:
                raise CatalogValidationError(
                    "ranking_screen.json changed after validation; refusing to overwrite it"
                )
        if _atomic_install_if_changed(output_root_path / "ranking.json", plan.ranking_bytes):
            changed.append("ranking.json")
        if _atomic_install_if_changed(
            output_root_path / "latest_best.json", plan.latest_best_bytes
        ):
            changed.append("latest_best.json")
        post_screen_sources = plan.source_snapshots
        if not plan.create_screen_backup:
            post_screen_sources = post_screen_sources + (plan.screen_snapshot,)
        _ensure_unchanged(post_screen_sources)
        if _atomic_install_if_changed(
            output_root_path / "catalog_manifest.json", plan.manifest_bytes
        ):
            changed.append("catalog_manifest.json")
        directory_descriptor = os.open(output_root_path, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        _verify_committed_catalog(plan)
        _ensure_unchanged(post_screen_sources)
        return plan, changed


def plan_preview(plan: CatalogPlan, *, applied: bool, changed: list[str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mode": "apply" if applied else "dry-run",
        "status": (
            "updated" if applied and changed else "unchanged" if applied else "validated"
        ),
        "output_root": str(plan.output_root),
        "promoted_dir": str(plan.promoted_dir),
        "catalog_generation": plan.generation,
        "screen_backup": {
            "would_create": plan.create_screen_backup,
            "sha256": _sha256(plan.screen_bytes),
            "size_bytes": len(plan.screen_bytes),
        },
        "catalog_file_hashes": {
            "ranking.json": _sha256(plan.ranking_bytes),
            "latest_best.json": _sha256(plan.latest_best_bytes),
            "catalog_manifest.json": _sha256(plan.manifest_bytes),
        },
        "promoted_row": plan.promoted_row,
        "catalog_manifest": plan.manifest,
    }
    if applied:
        result["changed_files"] = list(changed or [])
    else:
        result["apply_required_to_write"] = True
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a finalized promoted adaptation run and atomically publish "
            "catalog schema v2. Defaults to a no-write JSON preview."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--promoted-dir",
        type=Path,
        required=True,
        help="Promoted directory path, or its direct-child name under --output-root.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write catalog files. Without this flag, no filesystem changes are made.",
    )
    parser.add_argument(
        "--candidate-kind",
        choices=("link", "objective"),
        default="link",
        help=(
            "Validated latent to publish: link uses metrics_link_best and "
            "validated_link_best_z; objective uses metrics_adapted and "
            "validated_objective_best_z. Defaults to link."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.apply:
            plan, changed = apply_catalog_update(
                args.output_root,
                args.promoted_dir,
                candidate_kind=args.candidate_kind,
            )
            preview = plan_preview(plan, applied=True, changed=changed)
        else:
            plan = build_catalog_plan(
                args.output_root,
                args.promoted_dir,
                candidate_kind=args.candidate_kind,
            )
            preview = plan_preview(plan, applied=False)
    except CatalogValidationError as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
