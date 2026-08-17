#!/usr/bin/env python3
"""Inspect frame counts and durations of NPZ motion files in a directory."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/roban_mixed_8s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", nargs="?", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(data_dir)

    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files under {data_dir}")

    records: list[tuple[float, int, float, Path]] = []
    errors: list[tuple[Path, str]] = []
    fps_counts: Counter[float] = Counter()
    frame_counts: Counter[int] = Counter()
    for path in files:
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "fps" not in archive or "joint_pos" not in archive:
                    raise ValueError("missing fps or joint_pos")
                fps_array = np.asarray(archive["fps"])
                if fps_array.size != 1:
                    raise ValueError(f"fps must be scalar, got {fps_array.shape}")
                fps = float(fps_array.reshape(-1)[0])
                frames = int(np.asarray(archive["joint_pos"]).shape[0])
                if not np.isfinite(fps) or fps <= 0 or frames <= 0:
                    raise ValueError(f"invalid fps={fps} or frames={frames}")
            duration = frames / fps
            records.append((duration, frames, fps, path))
            fps_counts[fps] += 1
            frame_counts[frames] += 1
        except Exception as exc:
            errors.append((path, str(exc)))

    if not records:
        raise RuntimeError("No valid NPZ files found")
    records.sort(key=lambda item: (item[0], item[3].name))
    minimum = records[0]
    maximum = records[-1]
    under_two = sum(frames < 2 for _, frames, _, _ in records)

    print(f"directory={data_dir}")
    print(f"files={len(files)} valid={len(records)} errors={len(errors)}")
    print(f"fps_distribution={dict(sorted(fps_counts.items()))}")
    print(f"minimum: frames={minimum[1]} fps={minimum[2]:g} seconds={minimum[0]:.6f} file={minimum[3].name}")
    print(f"maximum: frames={maximum[1]} fps={maximum[2]:g} seconds={maximum[0]:.6f} file={maximum[3].name}")
    print(f"clips_with_fewer_than_2_frames={under_two}")
    print("frame_count_distribution:")
    for frames, count in sorted(frame_counts.items()):
        print(f"  frames={frames}: clips={count}")
    if errors:
        print("errors:")
        for path, message in errors:
            print(f"  {path.name}: {message}")


if __name__ == "__main__":
    main()
