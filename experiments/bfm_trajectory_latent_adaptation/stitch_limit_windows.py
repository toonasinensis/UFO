"""Stitch optimized latent windows into a full baseline latent trajectory.

The indexing convention is deliberately explicit: a trajectory-adaptation run
started at reference frame ``S`` produces a first latent that executes the
transition from reference frame ``S`` to ``S + 1``.  Consequently row zero of
that window replaces row ``S`` of the full latent array.

Windows are replaced directly; this tool does not blend, interpolate, project,
or otherwise modify latent values.  Every input is checked before any output is
written, and overlapping windows are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SUMMARY_SCHEMA_VERSION = 1
SUMMARY_FORMAT = "ufo_stitched_limit_latent_windows"


@dataclass(frozen=True)
class WindowSpec:
    """One optimized window and its global reference-frame start index."""

    start_frame: int
    latent_path: Path


def _resolved_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist or is not a file: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_latent(path: Path, *, label: str, latent_dim: int) -> np.ndarray:
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot load {label} as a non-pickle NPY array: {path}: {error}") from error
    if not isinstance(values, np.ndarray):
        raise ValueError(f"{label} is not a NumPy array: {path}")
    if values.ndim != 2 or values.shape[1] != latent_dim:
        raise ValueError(
            f"{label} must have shape [frames, {latent_dim}], got {values.shape}: {path}"
        )
    if values.shape[0] < 1:
        raise ValueError(f"{label} must contain at least one latent row: {path}")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"{label} must use a floating dtype, got {values.dtype}: {path}")
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"{label} contains NaN or Inf: {path}")
    return values


def _norm_statistics(values: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(np.asarray(values, dtype=np.float64), axis=1)
    return {
        "min": float(norms.min()),
        "max": float(norms.max()),
        "mean": float(norms.mean()),
        "max_abs_error": 0.0,
    }


def _validate_norms(
    values: np.ndarray,
    *,
    label: str,
    expected_norm: float,
    norm_atol: float,
    norm_rtol: float,
) -> dict[str, float]:
    norms = np.linalg.norm(np.asarray(values, dtype=np.float64), axis=1)
    errors = np.abs(norms - expected_norm)
    tolerance = norm_atol + norm_rtol * abs(expected_norm)
    bad = np.flatnonzero(errors > tolerance)
    stats = _norm_statistics(values)
    stats["max_abs_error"] = float(errors.max())
    if bad.size:
        row = int(bad[0])
        raise ValueError(
            f"{label} latent norm validation failed at row {row}: "
            f"norm={norms[row]:.9g}, expected={expected_norm:.9g}, "
            f"allowed_abs_error={tolerance:.9g}; bad_rows={bad.size}/{len(norms)}"
        )
    return stats


def _metadata_int(container: Any, key: str) -> int | None:
    if not isinstance(container, dict) or key not in container or container[key] is None:
        return None
    value = container[key]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Alignment metadata field {key!r} must be an integer, got {value!r}")
    return int(value)


def _alignment_values(document: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    """Return declared starts, reference-frame counts, and latent row counts."""

    containers = [document]
    for key in ("config", "args"):
        value = document.get(key)
        if isinstance(value, dict):
            containers.append(value)

    starts = [value for item in containers if (value := _metadata_int(item, "start_frame")) is not None]
    frame_counts = [
        value for item in containers if (value := _metadata_int(item, "frame_count")) is not None
    ]
    latent_rows: list[int] = []
    latent_shape = document.get("latent_shape")
    if latent_shape is not None:
        if (
            not isinstance(latent_shape, list)
            or len(latent_shape) != 2
            or isinstance(latent_shape[0], bool)
            or not isinstance(latent_shape[0], int)
        ):
            raise ValueError(f"Alignment metadata latent_shape must be [rows, dim], got {latent_shape!r}")
        latent_rows.append(int(latent_shape[0]))
    frame_range = document.get("frame_range")
    if frame_range is not None:
        if (
            not isinstance(frame_range, list)
            or len(frame_range) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in frame_range)
        ):
            raise ValueError(f"Alignment metadata frame_range must be [start, stop], got {frame_range!r}")
        starts.append(int(frame_range[0]))
        latent_rows.append(int(frame_range[1]) - int(frame_range[0]))
    return starts, frame_counts, latent_rows


def _discover_alignment_metadata(
    latent_path: Path,
    *,
    requested_start: int,
    latent_rows: int,
    require_metadata: bool,
) -> list[dict[str, Any]]:
    """Validate sibling run metadata when available and return an audit record."""

    records: list[dict[str, Any]] = []
    found_start = False
    for name in ("run_config.json", "summary.json"):
        path = latent_path.parent / name
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read alignment metadata {path}: {error}") from error
        if not isinstance(document, dict):
            raise ValueError(f"Alignment metadata must contain a JSON object: {path}")
        try:
            starts, frame_counts, declared_latent_rows = _alignment_values(document)
        except ValueError as error:
            raise ValueError(f"Invalid alignment metadata in {path}: {error}") from error

        if starts:
            found_start = True
            if len(set(starts)) != 1:
                raise ValueError(f"Conflicting start-frame declarations {starts} in {path}")
            if starts[0] != requested_start:
                raise ValueError(
                    f"Window start mismatch: CLI requested {requested_start}, "
                    f"but {path} declares {starts[0]}"
                )
        for frame_count in frame_counts:
            expected_rows = frame_count - 1
            if expected_rows != latent_rows:
                raise ValueError(
                    f"Window length mismatch: {path} declares frame_count={frame_count}, "
                    f"which requires {expected_rows} latent rows, but {latent_path} has {latent_rows}"
                )
        for declared_rows in declared_latent_rows:
            if declared_rows != latent_rows:
                raise ValueError(
                    f"Window length mismatch: {path} declares {declared_rows} latent rows, "
                    f"but {latent_path} has {latent_rows}"
                )
        records.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "declared_start_frames": starts,
                "declared_frame_counts": frame_counts,
                "declared_latent_rows": declared_latent_rows,
            }
        )
    if require_metadata and not found_start:
        raise ValueError(
            f"No sibling run_config.json or summary.json declares start_frame for {latent_path}; "
            "remove --require-alignment-metadata only if the explicit --window start is authoritative"
        )
    return records


def stitch_limit_windows(
    baseline_path: Path,
    windows: Sequence[WindowSpec],
    *,
    latent_dim: int = 256,
    expected_norm: float = 16.0,
    norm_atol: float = 1.0e-4,
    norm_rtol: float = 1.0e-5,
    require_alignment_metadata: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate and directly replace optimized windows in a full latent array."""

    if latent_dim < 1:
        raise ValueError(f"latent_dim must be positive, got {latent_dim}")
    if not math.isfinite(expected_norm) or expected_norm <= 0.0:
        raise ValueError(f"expected_norm must be finite and positive, got {expected_norm}")
    if not math.isfinite(norm_atol) or norm_atol < 0.0:
        raise ValueError(f"norm_atol must be finite and non-negative, got {norm_atol}")
    if not math.isfinite(norm_rtol) or norm_rtol < 0.0:
        raise ValueError(f"norm_rtol must be finite and non-negative, got {norm_rtol}")
    if not windows:
        raise ValueError("At least one optimized window is required")

    baseline_path = _resolved_file(baseline_path, label="Baseline latent")
    baseline = _load_latent(baseline_path, label="Baseline latent", latent_dim=latent_dim)
    baseline_norms = _validate_norms(
        baseline,
        label="Baseline",
        expected_norm=expected_norm,
        norm_atol=norm_atol,
        norm_rtol=norm_rtol,
    )
    result = np.array(baseline, copy=True, order="C")

    prepared: list[tuple[int, int, Path, np.ndarray, dict[str, float], list[dict[str, Any]]]] = []
    seen_sources: set[Path] = set()
    for index, spec in enumerate(windows):
        if isinstance(spec.start_frame, bool) or not isinstance(spec.start_frame, int):
            raise ValueError(f"Window {index} start_frame must be an integer, got {spec.start_frame!r}")
        if spec.start_frame < 0:
            raise ValueError(f"Window {index} start_frame must be non-negative, got {spec.start_frame}")
        latent_path = _resolved_file(spec.latent_path, label=f"Window {index} latent")
        if latent_path == baseline_path:
            raise ValueError(f"Window {index} cannot use the baseline itself as its source: {latent_path}")
        if latent_path in seen_sources:
            raise ValueError(f"Window latent source was specified more than once: {latent_path}")
        seen_sources.add(latent_path)
        latent = _load_latent(latent_path, label=f"Window {index}", latent_dim=latent_dim)
        if latent.dtype != baseline.dtype:
            raise ValueError(
                f"Window {index} dtype {latent.dtype} does not match baseline dtype {baseline.dtype}: {latent_path}"
            )
        end = spec.start_frame + latent.shape[0]
        if end > baseline.shape[0]:
            raise ValueError(
                f"Window {index} range [{spec.start_frame}, {end}) exceeds baseline rows "
                f"[0, {baseline.shape[0]}): {latent_path}"
            )
        norm_stats = _validate_norms(
            latent,
            label=f"Window {index}",
            expected_norm=expected_norm,
            norm_atol=norm_atol,
            norm_rtol=norm_rtol,
        )
        metadata = _discover_alignment_metadata(
            latent_path,
            requested_start=spec.start_frame,
            latent_rows=int(latent.shape[0]),
            require_metadata=require_alignment_metadata,
        )
        prepared.append((spec.start_frame, end, latent_path, latent, norm_stats, metadata))

    prepared.sort(key=lambda item: (item[0], item[1], str(item[2])))
    for previous, current in zip(prepared, prepared[1:]):
        if current[0] < previous[1]:
            raise ValueError(
                "Optimized windows overlap: "
                f"{previous[2]} covers [{previous[0]}, {previous[1]}) and "
                f"{current[2]} covers [{current[0]}, {current[1]})"
            )

    window_records: list[dict[str, Any]] = []
    replaced_rows = 0
    for start, end, path, latent, norm_stats, metadata in prepared:
        result[start:end] = latent
        replaced_rows += end - start
        window_records.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "shape": list(latent.shape),
                "dtype": str(latent.dtype),
                "start_frame": start,
                "start_latent_index": start,
                "stop_latent_index_exclusive": end,
                "end_reference_frame_inclusive": end,
                "latent_rows": end - start,
                "reference_frame_range_inclusive": [start, end],
                "norm": norm_stats,
                "alignment_metadata": metadata,
            }
        )

    result_norms = _validate_norms(
        result,
        label="Stitched output",
        expected_norm=expected_norm,
        norm_atol=norm_atol,
        norm_rtol=norm_rtol,
    )
    covered = np.zeros(baseline.shape[0], dtype=np.bool_)
    for start, end, *_rest in prepared:
        covered[start:end] = True
    if not np.array_equal(result[~covered], np.asarray(baseline)[~covered]):
        raise RuntimeError("Internal error: rows outside replacement windows were modified")

    audit = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "format": SUMMARY_FORMAT,
        "operation": "direct_replace",
        "indexing": {
            "baseline_start_reference_frame": 0,
            "window_row_zero_mapping": "full_latent[start_frame]",
            "latent_semantics": "latent[t] executes reference transition t -> t+1",
            "interval_convention": "half-open latent intervals [start, stop)",
        },
        "validation": {
            "latent_dim": latent_dim,
            "expected_latent_norm": expected_norm,
            "norm_atol": norm_atol,
            "norm_rtol": norm_rtol,
            "require_alignment_metadata": require_alignment_metadata,
            "overlap_allowed": False,
            "dtype_cast_allowed": False,
        },
        "baseline": {
            "path": str(baseline_path),
            "sha256": _sha256(baseline_path),
            "shape": list(baseline.shape),
            "dtype": str(baseline.dtype),
            "norm": baseline_norms,
        },
        "windows": window_records,
        "result": {
            "shape": list(result.shape),
            "dtype": str(result.dtype),
            "replaced_rows": replaced_rows,
            "unchanged_rows": int(result.shape[0] - replaced_rows),
            "norm": result_norms,
        },
    }
    return result, audit


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, values, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_save_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _parse_windows(raw_windows: Sequence[Sequence[str]]) -> list[WindowSpec]:
    windows: list[WindowSpec] = []
    for index, pair in enumerate(raw_windows):
        start_text, path_text = pair
        try:
            start = int(start_text)
        except ValueError as error:
            raise ValueError(f"Window {index} START_FRAME must be an integer, got {start_text!r}") from error
        windows.append(WindowSpec(start_frame=start, latent_path=Path(path_text)))
    return windows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Full baseline latent NPY [N, D].")
    parser.add_argument(
        "--window",
        action="append",
        nargs=2,
        required=True,
        metavar=("START_FRAME", "LATENT_NPY"),
        help=(
            "Optimized latent window. Repeat for multiple non-overlapping windows. "
            "Its row 0 replaces baseline row START_FRAME."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="Output full stitched latent NPY.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Audit summary JSON; defaults to <output stem>.summary.json beside --output.",
    )
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--expected-norm", type=float, default=16.0)
    parser.add_argument("--norm-atol", type=float, default=1.0e-4)
    parser.add_argument("--norm-rtol", type=float, default=1.0e-5)
    parser.add_argument(
        "--require-alignment-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require each window's sibling run_config.json or summary.json to declare a "
            "start_frame matching START_FRAME (default: true). Available metadata is always "
            "validated even when explicitly disabled."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing --output and --summary files; source files are never overwritten.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        windows = _parse_windows(args.window)
        baseline_path = args.baseline.expanduser().resolve()
        window_paths = {spec.latent_path.expanduser().resolve() for spec in windows}
        output_path = args.output.expanduser().resolve()
        summary_path = (
            args.summary.expanduser().resolve()
            if args.summary is not None
            else output_path.with_name(f"{output_path.stem}.summary.json")
        )
        source_paths = {baseline_path, *window_paths}
        if output_path in source_paths or summary_path in source_paths:
            raise ValueError("Output and summary paths must not overwrite baseline/window source files")
        if output_path == summary_path:
            raise ValueError("Output NPY and summary JSON paths must be different")
        if not args.overwrite:
            existing = [path for path in (output_path, summary_path) if path.exists()]
            if existing:
                raise ValueError(
                    "Refusing to overwrite existing output(s) without --overwrite: "
                    + ", ".join(str(path) for path in existing)
                )

        result, audit = stitch_limit_windows(
            baseline_path,
            windows,
            latent_dim=args.latent_dim,
            expected_norm=args.expected_norm,
            norm_atol=args.norm_atol,
            norm_rtol=args.norm_rtol,
            require_alignment_metadata=args.require_alignment_metadata,
        )
        _atomic_save_npy(output_path, result)
        audit["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        audit["output"] = {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "summary_path": str(summary_path),
        }
        _atomic_save_json(summary_path, audit)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error

    print(f"[stitch-limit-windows] output={output_path}")
    print(f"[stitch-limit-windows] summary={summary_path}")
    print(
        "[stitch-limit-windows] "
        f"shape={tuple(result.shape)} replaced_rows={audit['result']['replaced_rows']} "
        f"windows={len(audit['windows'])}"
    )


if __name__ == "__main__":
    main()
