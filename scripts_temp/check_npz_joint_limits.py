#!/usr/bin/env python3
"""Check RobotState NPZ joint positions against actuator-joint limits in an MJCF.

Directories are scanned completely. Zip archives are sampled without extraction.
When an NPZ has no joint_names/names field, its columns are explicitly assumed to
already follow the MJCF actuator-joint order and this is reported in the result.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import mujoco
import numpy as np


ISAAC_BFS_JOINT_NAMES = [
    "waist_yaw_joint",
    "zarm_l1_joint", "zarm_r1_joint", "leg_l1_joint", "leg_r1_joint",
    "zarm_l2_joint", "zarm_r2_joint", "leg_l2_joint", "leg_r2_joint",
    "zarm_l3_joint", "zarm_r3_joint", "leg_l3_joint", "leg_r3_joint",
    "zarm_l4_joint", "zarm_r4_joint", "leg_l4_joint", "leg_r4_joint",
    "leg_l5_joint", "leg_r5_joint", "leg_l6_joint", "leg_r6_joint",
]


@dataclass
class JointStats:
    minimum: float = float("inf")
    maximum: float = float("-inf")
    violation_frames: int = 0
    violation_files: int = 0
    worst_excess: float = 0.0
    worst_file: str = ""


@dataclass
class CheckStats:
    source: str
    requested_files: int
    checked_files: int = 0
    checked_frames: int = 0
    invalid_files: list[dict[str, str]] = field(default_factory=list)
    unnamed_files: int = 0
    violating_files: int = 0
    joints: dict[str, JointStats] = field(default_factory=dict)


def _actuated_joint_limits(xml_path: Path) -> tuple[list[str], np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    names: list[str] = []
    ranges: list[np.ndarray] = []
    seen: set[int] = set()
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0 or joint_id in seen:
            continue
        seen.add(joint_id)
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None:
            raise ValueError(f"Actuator {actuator_id} targets an unnamed joint")
        if not bool(model.jnt_limited[joint_id]):
            raise ValueError(f"Actuated joint has no position limit: {name}")
        names.append(name)
        ranges.append(model.jnt_range[joint_id].copy())
    if not names:
        raise ValueError(f"No actuated joints found in {xml_path}")
    return names, np.asarray(ranges, dtype=np.float64)


def _npz_joint_data(
    source: str | Path | BinaryIO,
    expected_names: list[str],
    unnamed_source_names: list[str],
) -> tuple[np.ndarray, bool]:
    with np.load(source, allow_pickle=True) as data:
        position_key = "joint_pos" if "joint_pos" in data else "dof_pos" if "dof_pos" in data else None
        if position_key is None:
            raise ValueError("missing joint_pos/dof_pos")
        positions = np.asarray(data[position_key], dtype=np.float64)
        if positions.ndim != 2:
            raise ValueError(f"{position_key} must be 2-D, got {positions.shape}")
        name_key = "joint_names" if "joint_names" in data else "names" if "names" in data else None
        unnamed = name_key is None
        if unnamed:
            if positions.shape[1] != len(unnamed_source_names):
                raise ValueError(
                    f"unnamed {position_key} width {positions.shape[1]} != assumed order width "
                    f"{len(unnamed_source_names)}"
                )
            index = {name: i for i, name in enumerate(unnamed_source_names)}
            missing = [name for name in expected_names if name not in index]
            if missing:
                raise ValueError(f"unnamed source order is missing expected joints: {missing}")
            return positions[:, [index[name] for name in expected_names]], True
        source_names = [str(value) for value in np.asarray(data[name_key]).tolist()]
        index = {name: i for i, name in enumerate(source_names)}
        missing = [name for name in expected_names if name not in index]
        if missing:
            raise ValueError(f"missing expected joints: {missing}")
        return positions[:, [index[name] for name in expected_names]], False


def _update(
    stats: CheckStats,
    file_name: str,
    positions: np.ndarray,
    unnamed: bool,
    joint_names: list[str],
    limits: np.ndarray,
    tolerance: float,
) -> None:
    if not np.isfinite(positions).all():
        raise ValueError("joint positions contain NaN/Inf")
    stats.checked_files += 1
    stats.checked_frames += len(positions)
    stats.unnamed_files += int(unnamed)
    low_excess = np.maximum(limits[:, 0] - positions, 0.0)
    high_excess = np.maximum(positions - limits[:, 1], 0.0)
    excess = np.maximum(low_excess, high_excess)
    file_violates = bool(np.any(excess > tolerance))
    stats.violating_files += int(file_violates)
    for joint_index, joint_name in enumerate(joint_names):
        joint = stats.joints[joint_name]
        values = positions[:, joint_index]
        joint.minimum = min(joint.minimum, float(values.min()))
        joint.maximum = max(joint.maximum, float(values.max()))
        mask = excess[:, joint_index] > tolerance
        count = int(mask.sum())
        joint.violation_frames += count
        joint.violation_files += int(count > 0)
        maximum = float(excess[:, joint_index].max())
        if maximum > joint.worst_excess:
            joint.worst_excess = maximum
            joint.worst_file = file_name


def _check_directory(
    path: Path, joint_names: list[str], limits: np.ndarray, tolerance: float
) -> CheckStats:
    files = sorted(path.rglob("*.npz"))
    stats = CheckStats(str(path), len(files), joints={name: JointStats() for name in joint_names})
    for file_path in files:
        try:
            positions, unnamed = _npz_joint_data(file_path, joint_names, joint_names)
            _update(stats, str(file_path), positions, unnamed, joint_names, limits, tolerance)
        except Exception as exc:  # Continue to report all bad files.
            stats.invalid_files.append({"file": str(file_path), "error": str(exc)})
    return stats


def _check_zip_sample(
    path: Path,
    sample_size: int,
    seed: int,
    joint_names: list[str],
    limits: np.ndarray,
    tolerance: float,
    unnamed_source_names: list[str],
) -> CheckStats:
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            info.filename for info in archive.infolist() if not info.is_dir() and info.filename.lower().endswith(".npz")
        )
        if not members:
            raise ValueError(f"Zip contains no NPZ files: {path}")
        selected = random.Random(seed).sample(members, min(sample_size, len(members)))
        stats = CheckStats(str(path), len(selected), joints={name: JointStats() for name in joint_names})
        for member in selected:
            try:
                with archive.open(member) as stream:
                    positions, unnamed = _npz_joint_data(
                        io.BytesIO(stream.read()), joint_names, unnamed_source_names
                    )
                _update(stats, member, positions, unnamed, joint_names, limits, tolerance)
            except Exception as exc:
                stats.invalid_files.append({"file": member, "error": str(exc)})
        return stats


def _serialise(stats: CheckStats, joint_names: list[str], limits: np.ndarray) -> dict[str, object]:
    joint_results = {}
    for index, name in enumerate(joint_names):
        value = stats.joints[name]
        joint_results[name] = {
            "xml_limit_rad": limits[index].tolist(),
            "xml_limit_deg": np.degrees(limits[index]).tolist(),
            "observed_rad": [value.minimum, value.maximum],
            "observed_deg": np.degrees([value.minimum, value.maximum]).tolist(),
            "violation_frames": value.violation_frames,
            "violation_files": value.violation_files,
            "worst_excess_rad": value.worst_excess,
            "worst_excess_deg": float(np.degrees(value.worst_excess)),
            "worst_file": value.worst_file,
        }
    return {
        "source": stats.source,
        "requested_files": stats.requested_files,
        "checked_files": stats.checked_files,
        "checked_frames": stats.checked_frames,
        "invalid_file_count": len(stats.invalid_files),
        "invalid_files": stats.invalid_files,
        "unnamed_files_reordered_from_assumed_order": stats.unnamed_files,
        "violating_files": stats.violating_files,
        "joints": joint_results,
    }


def _print_summary(result: dict[str, object]) -> None:
    print(f"\n=== {result['source']} ===")
    print(
        f"checked={result['checked_files']}/{result['requested_files']} files, "
        f"frames={result['checked_frames']}, invalid={result['invalid_file_count']}, "
        f"unnamed_reordered_from_assumed_order={result['unnamed_files_reordered_from_assumed_order']}, "
        f"violating_files={result['violating_files']}"
    )
    print("violating joints:")
    found = False
    for name, joint in result["joints"].items():
        if joint["violation_frames"]:
            found = True
            print(
                f"  {name}: files={joint['violation_files']}, frames={joint['violation_frames']}, "
                f"worst={joint['worst_excess_rad']:.8f} rad/{joint['worst_excess_deg']:.5f} deg, "
                f"file={joint['worst_file']}"
            )
    if not found:
        print("  none")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Directory scanned recursively for all NPZ files")
    parser.add_argument("--seed-archive", type=Path, required=True, help="Zip archive sampled without extraction")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument(
        "--seed-unnamed-order",
        choices=("isaac_bfs", "xml"),
        default="isaac_bfs",
        help="Column order assumed for unnamed NPZ files inside the seed archive",
    )
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Allowed numerical excess in radians")
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args()
    if args.sample_size <= 0 or args.tolerance < 0:
        parser.error("sample-size must be positive and tolerance non-negative")
    dataset = args.dataset.expanduser().resolve()
    archive = args.seed_archive.expanduser().resolve()
    xml = args.xml.expanduser().resolve()
    if not dataset.is_dir():
        parser.error(f"dataset is not a directory: {dataset}")
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        parser.error(f"seed-archive is not a readable zip: {archive}")
    if not xml.is_file():
        parser.error(f"xml does not exist: {xml}")

    joint_names, limits = _actuated_joint_limits(xml)
    print(f"XML: {xml}")
    print(f"Actuated limited joints ({len(joint_names)}): {joint_names}")
    dataset_stats = _check_directory(dataset, joint_names, limits, args.tolerance)
    archive_stats = _check_zip_sample(
        archive,
        args.sample_size,
        args.seed,
        joint_names,
        limits,
        args.tolerance,
        ISAAC_BFS_JOINT_NAMES if args.seed_unnamed_order == "isaac_bfs" else joint_names,
    )
    report = {
        "xml": str(xml),
        "tolerance_rad": args.tolerance,
        "archive_sample_seed": args.seed,
        "seed_archive_unnamed_order": args.seed_unnamed_order,
        "dataset": _serialise(dataset_stats, joint_names, limits),
        "seed_archive_sample": _serialise(archive_stats, joint_names, limits),
    }
    _print_summary(report["dataset"])
    _print_summary(report["seed_archive_sample"])
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON report: {output}")


if __name__ == "__main__":
    main()
