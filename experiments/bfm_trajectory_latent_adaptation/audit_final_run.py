"""Read-only consistency audit for a finalized trajectory-adaptation run.

The audit deliberately does not publish the final catalog.  A run is ``ready``
only when the durable optimizer state, final paired validation, TensorBoard
events, NumPy artifacts, and the final comparison video all describe the same
completed step.  An in-progress or partially finalized run is reported as
``not-ready`` with exit status 2.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from experiments.bfm_trajectory_latent_adaptation.update_final_catalog import (
    CatalogValidationError,
    build_catalog_plan,
)


@dataclass(frozen=True)
class FileSignature:
    device: int
    inode: int
    size: int
    mtime_ns: int


class NotReadyError(ValueError):
    """Raised when a final artifact is absent, stale, or inconsistent."""


class FinalRunAuditor:
    def __init__(
        self,
        *,
        run_dir: Path,
        output_root: Path,
        expected_frame_count: int,
        expected_validation_count: int,
        expected_video_codec: str,
        expected_video_pixel_format: str,
        ffprobe_bin: str,
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.output_root = output_root.resolve()
        self.expected_frame_count = expected_frame_count
        self.expected_validation_count = expected_validation_count
        self.expected_video_codec = expected_video_codec
        self.expected_video_pixel_format = expected_video_pixel_format
        self.ffprobe_bin = ffprobe_bin
        self.checks: list[dict[str, Any]] = []
        self.observed: dict[str, Any] = {}
        self.checkpoint: Mapping[str, Any] | None = None
        self.summary: Mapping[str, Any] | None = None
        self.run_config: Mapping[str, Any] | None = None
        self.metrics: dict[str, Mapping[str, Any]] = {}
        self.history: list[Mapping[str, Any]] = []
        self._initial_signatures: dict[Path, FileSignature] = {}

    @staticmethod
    def _signature(path: Path) -> FileSignature:
        stat = path.stat()
        return FileSignature(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def _artifact_paths(self) -> list[Path]:
        fixed_names = (
            "checkpoint.pt",
            "summary.json",
            "run_config.json",
            "metrics_baseline.json",
            "metrics_adapted.json",
            "metrics_link_best.json",
            "history.jsonl",
            "baseline_z.npy",
            "current_mean_z.npy",
            "best_z.npy",
            "link_best_z.npy",
            "validated_objective_best_z.npy",
            "validated_link_best_z.npy",
            "tracking_timeseries.npz",
            "comparison_ref_baseline_best.mp4",
            "video_status.json",
            "run.log",
        )
        paths = [self.run_dir / name for name in fixed_names]
        tensorboard_dir = self.run_dir / "tensorboard"
        if tensorboard_dir.is_dir():
            paths.extend(sorted(tensorboard_dir.glob("events.out.tfevents.*")))
        return paths

    def _capture_initial_signatures(self) -> None:
        self._initial_signatures = {
            path: self._signature(path) for path in self._artifact_paths() if path.exists()
        }

    def _check(self, name: str, function: Callable[[], Any]) -> None:
        try:
            detail = function()
        except (NotReadyError, CatalogValidationError) as error:
            self.checks.append(
                {"name": name, "status": "not-ready", "detail": str(error)}
            )
        except Exception as error:  # Surface malformed or unreadable artifacts clearly.
            self.checks.append(
                {
                    "name": name,
                    "status": "not-ready",
                    "detail": f"{type(error).__name__}: {error}",
                }
            )
        else:
            self.checks.append({"name": name, "status": "pass", "detail": detail})

    @staticmethod
    def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise NotReadyError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    @classmethod
    def _load_json(cls, path: Path) -> Mapping[str, Any]:
        try:
            value = json.loads(
                path.read_text(encoding="utf-8"), object_pairs_hook=cls._json_pairs
            )
        except FileNotFoundError as error:
            raise NotReadyError(f"missing {path.name}") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise NotReadyError(f"invalid JSON in {path.name}: {error}") from error
        if not isinstance(value, Mapping):
            raise NotReadyError(f"{path.name} must contain a JSON object")
        return value

    @classmethod
    def _assert_finite(cls, value: Any, label: str) -> None:
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return
        if isinstance(value, (int, np.integer)):
            return
        if isinstance(value, (float, np.floating)):
            if not math.isfinite(float(value)):
                raise NotReadyError(f"{label} contains NaN or Inf")
            return
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
                torch.isfinite(tensor).all()
            ):
                raise NotReadyError(f"{label} contains NaN or Inf")
            return
        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.inexact) and not np.isfinite(value).all():
                raise NotReadyError(f"{label} contains NaN or Inf")
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                cls._assert_finite(item, f"{label}.{key}")
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, item in enumerate(value):
                cls._assert_finite(item, f"{label}[{index}]")
            return
        raise NotReadyError(f"{label} has unsupported value type {type(value).__name__}")

    @staticmethod
    def _require_int(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise NotReadyError(f"{label} must be an integer")
        return int(value)

    @staticmethod
    def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise NotReadyError(f"{label} must be an object")
        return value

    @staticmethod
    def _close(actual: Any, expected: Any, label: str, *, scale: float = 1.0) -> None:
        if isinstance(actual, bool) or isinstance(expected, bool):
            if actual is not expected:
                raise NotReadyError(f"{label} differs: {actual!r} != {expected!r}")
            return
        try:
            actual_float = float(actual)
            expected_float = scale * float(expected)
        except (TypeError, ValueError) as error:
            raise NotReadyError(f"{label} is not numeric") from error
        if not (math.isfinite(actual_float) and math.isfinite(expected_float)):
            raise NotReadyError(f"{label} contains NaN or Inf")
        if not math.isclose(
            actual_float, expected_float, rel_tol=2.0e-5, abs_tol=2.0e-5
        ):
            raise NotReadyError(
                f"{label} differs: {actual_float:.9g} != {expected_float:.9g}"
            )

    def _require_core_alignment(self) -> int:
        if self.checkpoint is None or self.summary is None:
            raise NotReadyError("core artifacts did not load")
        checkpoint_iteration = self._require_int(
            self.checkpoint.get("iteration"), "checkpoint.iteration"
        )
        completed = self._require_int(
            self.summary.get("completed_iterations"), "summary.completed_iterations"
        )
        if checkpoint_iteration + 1 != completed:
            raise NotReadyError(
                "final validation is not aligned with the live checkpoint: "
                f"checkpoint={checkpoint_iteration + 1}, summary={completed}"
            )
        return completed

    def _load_core(self) -> dict[str, Any]:
        checkpoint_path = self.run_dir / "checkpoint.pt"
        try:
            checkpoint_bytes = checkpoint_path.read_bytes()
        except FileNotFoundError as error:
            raise NotReadyError("missing checkpoint.pt") from error
        try:
            try:
                checkpoint = torch.load(
                    io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
                )
            except TypeError:
                checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu")
        except Exception as error:
            raise NotReadyError(f"cannot load checkpoint.pt: {error}") from error
        if not isinstance(checkpoint, Mapping):
            raise NotReadyError("checkpoint.pt must contain a mapping")

        summary = self._load_json(self.run_dir / "summary.json")
        run_config = self._load_json(self.run_dir / "run_config.json")
        metrics = {
            name: self._load_json(self.run_dir / f"metrics_{name}.json")
            for name in ("baseline", "adapted", "link_best")
        }
        self._assert_finite(checkpoint, "checkpoint")
        self._assert_finite(summary, "summary")
        self._assert_finite(run_config, "run_config")
        self._assert_finite(metrics, "metrics")
        self.checkpoint = checkpoint
        self.summary = summary
        self.run_config = run_config
        self.metrics = metrics

        checkpoint_iteration = self._require_int(
            checkpoint.get("iteration"), "checkpoint.iteration"
        )
        completed = self._require_int(
            summary.get("completed_iterations"), "summary.completed_iterations"
        )
        if checkpoint_iteration + 1 != completed:
            raise NotReadyError(
                "final validation is stale or incomplete: "
                f"checkpoint={checkpoint_iteration + 1}, summary={completed}"
            )
        frame_count = self._require_int(summary.get("frame_count"), "summary.frame_count")
        if frame_count != self.expected_frame_count:
            raise NotReadyError(
                f"summary.frame_count={frame_count}, expected {self.expected_frame_count}"
            )
        if self._require_int(run_config.get("frame_count"), "run_config.frame_count") != frame_count:
            raise NotReadyError("run_config.frame_count differs from summary.frame_count")
        self.observed.update(
            {
                "completed_iterations": completed,
                "checkpoint_iteration": checkpoint_iteration,
                "frame_count": frame_count,
                "motion": summary.get("motion"),
            }
        )
        return {
            "completed_iterations": completed,
            "frame_count": frame_count,
            "metrics_files": sorted(metrics),
        }

    def _check_catalog_source_schema(self) -> dict[str, Any]:
        if self.checkpoint is None:
            raise NotReadyError("core artifacts did not load")
        plan = build_catalog_plan(self.output_root, self.run_dir)
        return {
            "schema_version": plan.promoted_row["schema_version"],
            "catalog_generation_preview": plan.generation,
            "apply_performed": False,
        }

    def _check_validation_counts(self) -> dict[str, int]:
        self._require_core_alignment()
        assert self.summary is not None
        expected = {
            "baseline": self.expected_validation_count,
            "adapted": self.expected_validation_count,
        }
        summary_counts = self._require_mapping(
            self.summary.get("paired_validation_counts"),
            "summary.paired_validation_counts",
        )
        if dict(summary_counts) != expected:
            raise NotReadyError(
                f"summary validation counts={dict(summary_counts)}, expected {expected}"
            )
        for name in ("adapted", "link_best"):
            counts = self._require_mapping(
                self.metrics[name].get("validation_counts"),
                f"metrics_{name}.validation_counts",
            )
            if dict(counts) != expected:
                raise NotReadyError(
                    f"metrics_{name} validation counts={dict(counts)}, expected {expected}"
                )
        return expected

    def _load_history(self) -> dict[str, Any]:
        path = self.run_dir / "history.jsonl"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError as error:
            raise NotReadyError("missing history.jsonl") from error
        if not lines:
            raise NotReadyError("history.jsonl is empty")
        history: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                raise NotReadyError(f"history.jsonl has blank line {line_number}")
            try:
                record = json.loads(line, object_pairs_hook=self._json_pairs)
            except json.JSONDecodeError as error:
                raise NotReadyError(
                    f"invalid history.jsonl line {line_number}: {error}"
                ) from error
            if not isinstance(record, Mapping):
                raise NotReadyError(f"history line {line_number} is not an object")
            self._assert_finite(record, f"history[{line_number}]")
            history.append(record)
        iterations = [
            self._require_int(record.get("iteration"), f"history[{index}].iteration")
            for index, record in enumerate(history)
        ]
        expected_iterations = list(range(iterations[0], iterations[-1] + 1))
        if iterations != expected_iterations:
            raise NotReadyError("history iterations contain a gap, duplicate, or reordering")
        if self.checkpoint is None:
            raise NotReadyError("core artifacts did not load")
        checkpoint_iteration = self._require_int(
            self.checkpoint.get("iteration"), "checkpoint.iteration"
        )
        if iterations[-1] != checkpoint_iteration:
            raise NotReadyError(
                f"history ends at {iterations[-1]}, checkpoint is {checkpoint_iteration}"
            )
        last = history[-1]
        checkpoint_link_metrics = self._require_mapping(
            self.checkpoint.get("link_best_metrics"), "checkpoint.link_best_metrics"
        )
        history_link_metrics = self._require_mapping(
            last.get("link_best_metrics"), "history last link_best_metrics"
        )
        self._close(
            history_link_metrics.get("mean_body_position_error"),
            checkpoint_link_metrics.get("mean_body_position_error"),
            "history/checkpoint link error",
        )
        self._close(
            last.get("global_best_objective"),
            self.checkpoint.get("best_objective"),
            "history/checkpoint objective best",
        )
        self.history = history
        self.observed.update(
            {
                "history_records": len(history),
                "history_first_iteration": iterations[0],
                "history_last_iteration": iterations[-1],
            }
        )
        return {
            "records": len(history),
            "first_iteration": iterations[0],
            "last_iteration": iterations[-1],
            "contiguous": True,
        }

    def _check_numpy(self) -> dict[str, Any]:
        if self.checkpoint is None or self.summary is None:
            raise NotReadyError("core artifacts did not load")
        latent_shape_value = self.summary.get("latent_shape")
        if (
            not isinstance(latent_shape_value, Sequence)
            or isinstance(latent_shape_value, (str, bytes))
            or len(latent_shape_value) != 2
        ):
            raise NotReadyError("summary.latent_shape must have two dimensions")
        latent_shape = tuple(
            self._require_int(value, f"summary.latent_shape[{index}]")
            for index, value in enumerate(latent_shape_value)
        )
        if latent_shape[0] + 1 != self.expected_frame_count:
            raise NotReadyError("latent horizon does not match expected frame count")
        checkpoint_keys = {
            "current_mean_z.npy": "mean",
            "best_z.npy": "best",
            "link_best_z.npy": "link_best",
            "validated_objective_best_z.npy": "validated_objective_best",
            "validated_link_best_z.npy": "validated_link_best",
        }
        checked: list[str] = []
        for filename in ("baseline_z.npy", *checkpoint_keys):
            path = self.run_dir / filename
            try:
                array = np.load(path, allow_pickle=False)
            except FileNotFoundError as error:
                raise NotReadyError(f"missing {filename}") from error
            except Exception as error:
                raise NotReadyError(f"cannot load {filename}: {error}") from error
            if array.dtype != np.float32 or array.shape != latent_shape:
                raise NotReadyError(
                    f"{filename} has dtype/shape {array.dtype}/{array.shape}, "
                    f"expected float32/{latent_shape}"
                )
            if not np.isfinite(array).all():
                raise NotReadyError(f"{filename} contains NaN or Inf")
            checkpoint_key = checkpoint_keys.get(filename)
            if checkpoint_key is not None:
                tensor = self.checkpoint.get(checkpoint_key)
                if not isinstance(tensor, torch.Tensor):
                    raise NotReadyError(f"checkpoint.{checkpoint_key} is not a tensor")
                checkpoint_array = tensor.detach().cpu().numpy()
                if not np.array_equal(array, checkpoint_array):
                    difference = float(np.max(np.abs(array - checkpoint_array)))
                    raise NotReadyError(
                        f"{filename} differs from checkpoint.{checkpoint_key}; "
                        f"max_abs={difference:.9g}"
                    )
            checked.append(filename)

        timeseries_path = self.run_dir / "tracking_timeseries.npz"
        try:
            with np.load(timeseries_path, allow_pickle=False) as archive:
                keys = sorted(archive.files)
                if not keys:
                    raise NotReadyError("tracking_timeseries.npz is empty")
                # Per-transition tracking traces have one sample for each latent
                # action, so their horizon is frame_count - 1.
                expected_timeseries_length = self.expected_frame_count - 1
                for key in keys:
                    value = archive[key]
                    actual_length = value.shape[0] if value.ndim >= 1 else None
                    if actual_length != expected_timeseries_length:
                        raise NotReadyError(
                            f"tracking_timeseries[{key}] has {actual_length} transitions, "
                            f"expected {expected_timeseries_length} transitions"
                        )
                    if np.issubdtype(value.dtype, np.inexact) and not np.isfinite(value).all():
                        raise NotReadyError(f"tracking_timeseries[{key}] contains NaN or Inf")
        except FileNotFoundError as error:
            raise NotReadyError("missing tracking_timeseries.npz") from error
        checked.append("tracking_timeseries.npz")
        return {"latent_shape": list(latent_shape), "artifacts": checked}

    def _tensorboard_events(self, tag: str) -> list[Any]:
        tensorboard_dir = self.run_dir / "tensorboard"
        if not tensorboard_dir.is_dir():
            raise NotReadyError("missing tensorboard directory")
        accumulator = EventAccumulator(
            str(tensorboard_dir), size_guidance={"scalars": 0, "tensors": 0}
        )
        accumulator.Reload()
        scalar_tags = accumulator.Tags().get("scalars", [])
        for scalar_tag in scalar_tags:
            for event in accumulator.Scalars(scalar_tag):
                if not math.isfinite(float(event.value)):
                    raise NotReadyError(
                        f"TensorBoard scalar {scalar_tag} contains NaN or Inf"
                    )
        if tag not in scalar_tags:
            raise NotReadyError(f"TensorBoard is missing scalar tag {tag}")
        self._tensorboard_accumulator = accumulator
        return list(accumulator.Scalars(tag))

    def _check_tensorboard(self) -> dict[str, Any]:
        if not self.history or self.summary is None:
            raise NotReadyError("history/core artifacts did not load")
        completed = self._require_core_alignment()
        canonical_tag = "tracking/link_relative_position_mm/link_best"
        training_events = self._tensorboard_events(canonical_tag)
        accumulator = self._tensorboard_accumulator
        expected_steps = [int(record["iteration"]) for record in self.history]
        actual_steps = [int(event.step) for event in training_events]
        if actual_steps != expected_steps:
            raise NotReadyError(
                "TensorBoard training steps do not exactly match history.jsonl: "
                f"events={len(actual_steps)}, history={len(expected_steps)}, "
                f"last_event={actual_steps[-1] if actual_steps else None}, "
                f"last_history={expected_steps[-1]}"
            )
        last_history_link = self._require_mapping(
            self.history[-1].get("link_best_metrics"), "history last link_best_metrics"
        ).get("mean_body_position_error")
        self._close(
            training_events[-1].value,
            last_history_link,
            "TensorBoard/history final link best",
            scale=1000.0,
        )

        variant_specs = {
            "objective_best": self.metrics["adapted"],
            "link_best": self.metrics["link_best"],
        }
        metric_specs = {
            "objective": ("objective", 1.0),
            "mpjpe_mm": ("mpjpe", 1000.0),
            "link_relative_position_mm": ("mean_body_position_error", 1000.0),
        }
        final_tags_checked: list[str] = []
        for variant, metrics_file in variant_specs.items():
            metrics = self._require_mapping(metrics_file.get("metrics"), f"{variant}.metrics")
            for tag_component, (metric_key, scale) in metric_specs.items():
                tag = f"final_validation/{tag_component}/{variant}"
                if tag not in accumulator.Tags().get("scalars", []):
                    raise NotReadyError(f"TensorBoard is missing {tag}")
                at_step = [
                    event for event in accumulator.Scalars(tag) if int(event.step) == completed
                ]
                if not at_step:
                    raise NotReadyError(
                        f"TensorBoard {tag} has no final value at step {completed}"
                    )
                self._close(
                    at_step[-1].value,
                    metrics.get(metric_key),
                    f"TensorBoard {tag}",
                    scale=scale,
                )
                final_tags_checked.append(tag)

        # The logger writes the objective-best paired baseline first and the
        # link-best paired baseline second under the shared baseline tags.
        for tag_component, (metric_key, scale) in metric_specs.items():
            tag = f"final_validation/{tag_component}/baseline"
            if tag not in accumulator.Tags().get("scalars", []):
                raise NotReadyError(f"TensorBoard is missing {tag}")
            at_step = [
                event for event in accumulator.Scalars(tag) if int(event.step) == completed
            ]
            if len(at_step) < 2:
                raise NotReadyError(
                    f"TensorBoard {tag} needs two paired-baseline values at step {completed}"
                )
            expected_files = (self.metrics["adapted"], self.metrics["link_best"])
            for event, metrics_file in zip(at_step[-2:], expected_files):
                baseline_metrics = self._require_mapping(
                    metrics_file.get("paired_baseline_metrics"),
                    "paired_baseline_metrics",
                )
                self._close(
                    event.value,
                    baseline_metrics.get(metric_key),
                    f"TensorBoard {tag}",
                    scale=scale,
                )
            final_tags_checked.append(tag)
        self.observed["tensorboard_final_step"] = completed
        return {
            "training_scalar_count": len(training_events),
            "training_step_range": [actual_steps[0], actual_steps[-1]],
            "final_step": completed,
            "final_tags_checked": len(final_tags_checked),
        }

    def _check_freshness(self) -> dict[str, Any]:
        summary_path = self.run_dir / "summary.json"
        video_path = self.run_dir / "comparison_ref_baseline_best.mp4"
        summary_mtime = self._signature(summary_path).mtime_ns
        predecessors = (
            "checkpoint.pt",
            "history.jsonl",
            "metrics_adapted.json",
            "metrics_link_best.json",
            "validated_objective_best_z.npy",
            "validated_link_best_z.npy",
            "tracking_timeseries.npz",
        )
        newer_predecessors = [
            name
            for name in predecessors
            if self._signature(self.run_dir / name).mtime_ns > summary_mtime
        ]
        if newer_predecessors:
            raise NotReadyError(
                "summary.json predates final artifacts: " + ", ".join(newer_predecessors)
            )
        video_mtime = self._signature(video_path).mtime_ns
        if video_mtime < summary_mtime:
            raise NotReadyError("final video predates final summary/paired validation")
        return {
            "summary_is_newest_validation_manifest": True,
            "video_rendered_after_summary": True,
        }

    def _check_video(self) -> dict[str, Any]:
        if self.summary is None or self.run_config is None:
            raise NotReadyError("core artifacts did not load")
        completed = self._require_core_alignment()
        video_path = self.run_dir / "comparison_ref_baseline_best.mp4"
        if not video_path.is_file():
            raise NotReadyError(f"missing {video_path.name}")
        command = [
            self.ffprobe_bin,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=codec_name,pix_fmt,width,height,avg_frame_rate,"
                "nb_frames,nb_read_frames,duration"
            ),
            "-of",
            "json",
            str(video_path),
        ]
        try:
            completed_process = subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=120
            )
        except FileNotFoundError as error:
            raise NotReadyError(f"ffprobe executable not found: {self.ffprobe_bin}") from error
        except (subprocess.SubprocessError, OSError) as error:
            raise NotReadyError(f"ffprobe failed: {error}") from error
        try:
            probe = json.loads(completed_process.stdout)
            streams = probe["streams"]
            stream = streams[0]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise NotReadyError("ffprobe did not return one readable video stream") from error
        codec = str(stream.get("codec_name", ""))
        if codec != self.expected_video_codec:
            raise NotReadyError(
                f"video codec={codec!r}, expected {self.expected_video_codec!r}"
            )
        pixel_format = str(stream.get("pix_fmt", ""))
        if pixel_format != self.expected_video_pixel_format:
            raise NotReadyError(
                f"video pixel format={pixel_format!r}, "
                f"expected {self.expected_video_pixel_format!r}"
            )
        frame_text = stream.get("nb_read_frames", stream.get("nb_frames"))
        try:
            frame_count = int(frame_text)
        except (TypeError, ValueError) as error:
            raise NotReadyError(f"ffprobe returned invalid frame count {frame_text!r}") from error
        if frame_count != self.expected_frame_count:
            raise NotReadyError(
                f"video has {frame_count} frames, expected {self.expected_frame_count}"
            )
        render_size = self._require_int(
            self.run_config.get("render_size"), "run_config.render_size"
        )
        width = self._require_int(stream.get("width"), "video.width")
        height = self._require_int(stream.get("height"), "video.height")
        expected_width = 3 * render_size + 8
        if (width, height) != (expected_width, render_size):
            raise NotReadyError(
                f"video size={width}x{height}, expected {expected_width}x{render_size}"
            )
        expected_fps = self._require_int(self.run_config.get("fps"), "run_config.fps")
        try:
            fps = float(Fraction(str(stream.get("avg_frame_rate"))))
        except (ValueError, ZeroDivisionError) as error:
            raise NotReadyError("video has invalid avg_frame_rate") from error
        if not math.isclose(fps, expected_fps, rel_tol=0.0, abs_tol=1.0e-6):
            raise NotReadyError(f"video fps={fps}, expected {expected_fps}")
        expected_duration = self.expected_frame_count / expected_fps
        try:
            duration = float(stream.get("duration"))
        except (TypeError, ValueError) as error:
            raise NotReadyError("video has invalid duration") from error
        if not math.isclose(duration, expected_duration, rel_tol=0.0, abs_tol=1.0 / expected_fps):
            raise NotReadyError(
                f"video duration={duration}, expected about {expected_duration}"
            )

        interval = self._require_int(
            self.run_config.get("video_interval"), "run_config.video_interval"
        )
        status = self._load_json(self.run_dir / "video_status.json")
        status_step = self._require_int(status.get("step"), "video_status.step")
        if interval <= 0:
            raise NotReadyError("run_config.video_interval must be positive")
        if (
            status_step < 0
            or status_step > completed
            or status_step % interval != 0
            or completed - status_step >= interval
        ):
            raise NotReadyError(
                f"video_status.step={status_step} is not the latest {interval}-step "
                f"milestone before completed step {completed}"
            )
        self.observed.update(
            {
                "video_frames": frame_count,
                "video_codec": codec,
                "video_pixel_format": pixel_format,
                "video_status_step": status_step,
            }
        )
        return {
            "codec": codec,
            "pixel_format": pixel_format,
            "frames": frame_count,
            "size": [width, height],
            "fps": fps,
            "duration_seconds": duration,
            "periodic_status_step": status_step,
        }

    def _check_stability(self) -> dict[str, Any]:
        final_paths = {path for path in self._artifact_paths() if path.exists()}
        initial_paths = set(self._initial_signatures)
        if final_paths != initial_paths:
            appeared = sorted(path.name for path in final_paths - initial_paths)
            disappeared = sorted(path.name for path in initial_paths - final_paths)
            raise NotReadyError(
                f"artifact set changed during audit; appeared={appeared}, "
                f"disappeared={disappeared}"
            )
        changed = [
            str(path.relative_to(self.run_dir))
            for path, signature in self._initial_signatures.items()
            if self._signature(path) != signature
        ]
        if changed:
            raise NotReadyError("artifacts changed during audit: " + ", ".join(changed))
        return {"stable_files": len(final_paths)}

    def run(self) -> dict[str, Any]:
        self._capture_initial_signatures()
        self._check("core_checkpoint_summary_metrics", self._load_core)
        self._check("catalog_source_schema_v2_dry_run", self._check_catalog_source_schema)
        self._check("paired_validation_counts", self._check_validation_counts)
        self._check("history_checkpoint_consistency", self._load_history)
        self._check("numpy_checkpoint_consistency", self._check_numpy)
        self._check("tensorboard_history_and_final_validation", self._check_tensorboard)
        self._check("final_artifact_freshness", self._check_freshness)
        self._check("comparison_video", self._check_video)
        self._check("stable_read_snapshot", self._check_stability)
        failures = [check for check in self.checks if check["status"] != "pass"]
        return {
            "schema_version": 1,
            "mode": "read-only",
            "status": "ready" if not failures else "not-ready",
            "run_dir": str(self.run_dir),
            "output_root": str(self.output_root),
            "observed": self.observed,
            "checks": self.checks,
            "failure_count": len(failures),
            "catalog_apply_performed": False,
        }


def audit_final_run(
    run_dir: Path | str,
    *,
    output_root: Path | str | None = None,
    expected_frame_count: int = 500,
    expected_validation_count: int = 64,
    expected_video_codec: str = "h264",
    expected_video_pixel_format: str = "yuv420p",
    ffprobe_bin: str = "ffprobe",
) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    output_path = (
        run_path.parent
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    auditor = FinalRunAuditor(
        run_dir=run_path,
        output_root=output_path,
        expected_frame_count=expected_frame_count,
        expected_validation_count=expected_validation_count,
        expected_video_codec=expected_video_codec,
        expected_video_pixel_format=expected_video_pixel_format,
        ffprobe_bin=ffprobe_bin,
    )
    return auditor.run()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only final audit of checkpoint, validation, TensorBoard, NumPy, "
            "and REF|BASELINE|BEST video artifacts."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Catalog output root; defaults to the parent of --run-dir.",
    )
    parser.add_argument("--expected-frame-count", type=int, default=500)
    parser.add_argument("--expected-validation-count", type=int, default=64)
    parser.add_argument("--expected-video-codec", default="h264")
    parser.add_argument("--expected-video-pixel-format", default="yuv420p")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit_final_run(
        args.run_dir,
        output_root=args.output_root,
        expected_frame_count=args.expected_frame_count,
        expected_validation_count=args.expected_validation_count,
        expected_video_codec=args.expected_video_codec,
        expected_video_pixel_format=args.expected_video_pixel_format,
        ffprobe_bin=args.ffprobe_bin,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    sys.exit(main())
