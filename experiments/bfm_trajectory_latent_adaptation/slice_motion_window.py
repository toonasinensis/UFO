"""Create an auditable RobotState-NPZ frame window for closed-loop playback."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

FRAME_KEY = "joint_pos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output instead of refusing to overwrite it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.start_frame < 0 or args.frame_count < 2:
        raise ValueError("start-frame>=0 and frame-count>=2 are required")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if output == source:
        raise ValueError("Input and output paths must differ")

    with np.load(source, allow_pickle=False) as archive:
        if FRAME_KEY not in archive:
            raise KeyError(f"Motion has no required {FRAME_KEY!r} array")
        total_frames = int(archive[FRAME_KEY].shape[0])
        end_frame = args.start_frame + args.frame_count
        if end_frame > total_frames:
            raise ValueError(
                f"Requested [{args.start_frame}, {end_frame}), motion has {total_frames} frames"
            )
        values: dict[str, np.ndarray] = {}
        sliced_keys: list[str] = []
        static_keys: list[str] = []
        for key in archive.files:
            value = np.asarray(archive[key])
            if value.ndim > 0 and value.shape[0] == total_frames:
                values[key] = value[args.start_frame:end_frame].copy()
                sliced_keys.append(key)
            else:
                values[key] = value.copy()
                static_keys.append(key)

    for key, value in values.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"Output field {key!r} contains NaN/Inf")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, output)
    summary = {
        "input": str(source),
        "output": str(output),
        "source_total_frames": total_frames,
        "source_frame_interval": [args.start_frame, end_frame],
        "output_frame_count": args.frame_count,
        "latent_step_interval": [args.start_frame, end_frame - 1],
        "expected_latent_rows": args.frame_count - 1,
        "sliced_keys": sliced_keys,
        "static_keys": static_keys,
        "shapes": {key: list(value.shape) for key, value in values.items()},
    }
    summary_path = output.with_suffix(".summary.json")
    summary_tmp = summary_path.with_name(f".{summary_path.name}.tmp")
    summary_tmp.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(summary_tmp, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
