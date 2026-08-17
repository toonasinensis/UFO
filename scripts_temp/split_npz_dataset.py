#!/usr/bin/env python3
"""Split metadata-rich Roban NPZ motions into bounded-duration clips."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

FRAME_FIELDS = (
    "data",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def positive_scalar(archive: np.lib.npyio.NpzFile, name: str, path: Path) -> float:
    if name not in archive:
        raise ValueError(f"{path}: missing {name}")
    value = np.asarray(archive[name])
    if value.size != 1:
        raise ValueError(f"{path}: {name} must contain one value, got {value.shape}")
    result = float(value.reshape(-1)[0])
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{path}: invalid {name}={result}")
    return result


def load_and_validate(path: Path) -> tuple[dict[str, np.ndarray], int, float]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
        fps = positive_scalar(archive, "fps", path)
    missing = [name for name in FRAME_FIELDS if name not in arrays]
    if missing:
        raise ValueError(f"{path}: missing frame fields {missing}")
    frames = int(arrays["joint_pos"].shape[0])
    if frames <= 0:
        raise ValueError(f"{path}: motion contains no frames")
    mismatched = {name: arrays[name].shape for name in FRAME_FIELDS if arrays[name].shape[0] != frames}
    if mismatched:
        raise ValueError(f"{path}: frame field length mismatch: {mismatched}")
    return arrays, frames, fps


def write_clip(
    destination: Path,
    arrays: dict[str, np.ndarray],
    start: int,
    end: int,
) -> None:
    output = {name: value[start:end] if name in FRAME_FIELDS else value for name, value in arrays.items()}
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=0.0,
        help="Discard clips shorter than this duration, including final remainders",
    )
    parser.add_argument(
        "--filename-prefix",
        default="",
        help="Prefix added to every output filename, useful when mixing datasets",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and skip output clips that already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if input_dir == output_dir:
        raise ValueError("--input-dir and --output-dir must differ")
    if not np.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise ValueError("--max-seconds must be positive")
    if not np.isfinite(args.min_seconds) or args.min_seconds < 0:
        raise ValueError("--min-seconds must be non-negative")
    if args.min_seconds > args.max_seconds:
        raise ValueError("--min-seconds must not exceed --max-seconds")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if "/" in args.filename_prefix or "\\" in args.filename_prefix:
        raise ValueError("--filename-prefix must not contain path separators")
    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files under {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = total_input_frames = total_output_frames = discarded_frames = 0
    print(f"input={input_dir}")
    print(f"output={output_dir}")
    print(f"source_files={len(files)} max_seconds={args.max_seconds:g} min_seconds={args.min_seconds:g}")
    for source_index, source_path in enumerate(files, start=1):
        arrays, frames, fps = load_and_validate(source_path)
        max_frames = int(np.floor(fps * args.max_seconds + 1.0e-9))
        min_frames = int(np.ceil(fps * args.min_seconds - 1.0e-9))
        if max_frames <= 0:
            raise ValueError(f"{source_path}: max duration is shorter than one frame")
        total_input_frames += frames
        clip_index = 0
        for start in range(0, frames, max_frames):
            end = min(start + max_frames, frames)
            if end - start < min_frames:
                discarded_frames += end - start
                continue
            destination = output_dir / (f"{args.filename_prefix}{source_path.stem}_{clip_index:04d}.npz")
            if destination.exists() and args.resume:
                with np.load(destination, allow_pickle=False) as existing:
                    existing_frames = int(np.asarray(existing["joint_pos"]).shape[0])
                expected_frames = end - start
                if existing_frames != expected_frames:
                    raise ValueError(f"Existing clip has {existing_frames} frames, expected {expected_frames}: {destination}")
                output_files += 1
                total_output_frames += expected_frames
                clip_index += 1
                continue
            if destination.exists() and not args.overwrite:
                raise FileExistsError(f"Output exists; pass --overwrite to replace it: {destination}")
            write_clip(destination, arrays, start, end)
            output_files += 1
            total_output_frames += end - start
            clip_index += 1
        if source_index == 1 or source_index % 250 == 0 or source_index == len(files):
            print(f"progress={source_index}/{len(files)} clips={output_files} frames={total_output_frames}")
    if total_input_frames != total_output_frames + discarded_frames:
        raise RuntimeError(
            f"Frame accounting failed: input={total_input_frames}, output={total_output_frames}, discarded={discarded_frames}"
        )
    print(f"done: source_files={len(files)} clips={output_files} frames={total_output_frames} discarded_frames={discarded_frames}")


if __name__ == "__main__":
    main()
