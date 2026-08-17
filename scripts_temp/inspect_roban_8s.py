#!/usr/bin/env python3
"""Interactively sample and inspect 8-second Roban NPZ clips."""

from __future__ import annotations

import argparse
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
from view_roban_npz import DEFAULT_XML, load_qpos

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/roban_mixed_8s"


@dataclass
class PlaybackControl:
    paused: bool = False
    pending_action: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def on_key(self, keycode: int) -> None:
        key = chr(keycode).upper() if 0 <= keycode < 128 else ""
        with self.lock:
            if keycode == 32:
                self.paused = not self.paused
                print("[paused]" if self.paused else "[resumed]", flush=True)
            elif key in {"N", "D"} or keycode == 262:
                self.pending_action = "next"
            elif key in {"P", "A"} or keycode == 263:
                self.pending_action = "previous"
            elif key == "R":
                self.pending_action = "restart"
            elif key in {"Q"} or keycode == 256:
                self.pending_action = "quit"

    def consume_action(self) -> str | None:
        with self.lock:
            action = self.pending_action
            self.pending_action = None
            return action

    def is_paused(self) -> bool:
        with self.lock:
            return self.paused


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument(
        "--source",
        choices=("all", "dataset1", "dataset2"),
        default="all",
        help="Limit sampling to one source dataset",
    )
    parser.add_argument("--samples", type=int, default=20, help="Number of random clips; 0 means all clips")
    parser.add_argument("--seed", type=int, default=None, help="Fixed seed for reproducible sampling")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop-batch", action="store_true", help="Loop after the final sampled clip")
    parser.add_argument("--list-only", action="store_true", help="Print sampled clips without opening a GUI")
    return parser.parse_args()


def select_clips(data_dir: Path, source: str, samples: int, seed: int | None) -> list[Path]:
    prefix = "*.npz" if source == "all" else f"{source}__*.npz"
    candidates = sorted(data_dir.glob(prefix))
    if not candidates:
        raise FileNotFoundError(f"No clips matching {prefix!r} under {data_dir}")
    if samples < 0:
        raise ValueError("--samples must be non-negative")
    count = len(candidates) if samples == 0 else min(samples, len(candidates))
    generator = random.Random(seed)
    return generator.sample(candidates, count)


def print_selection(clips: list[Path], seed: int | None) -> None:
    print(f"selected_clips={len(clips)} seed={seed}")
    for index, path in enumerate(clips, start=1):
        print(f"  {index:04d}: {path.name}")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    xml_path = args.xml.expanduser().resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(data_dir)
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)
    if args.speed <= 0:
        raise ValueError("--speed must be positive")

    clips = select_clips(data_dir, args.source, args.samples, args.seed)
    print_selection(clips, args.seed)
    if args.list_only:
        return

    from mujoco import viewer

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    control = PlaybackControl()
    clip_index = 0
    frame = 0
    qpos, fps = load_qpos(clips[clip_index], model)

    def announce() -> None:
        duration = len(qpos) / fps
        print(
            f"\n[{clip_index + 1}/{len(clips)}] {clips[clip_index].name} frames={len(qpos)} duration={duration:.3f}s fps={fps:g}",
            flush=True,
        )

    def change_clip(new_index: int) -> None:
        nonlocal clip_index, frame, qpos, fps
        clip_index = new_index
        frame = 0
        qpos, fps = load_qpos(clips[clip_index], model)
        announce()

    print("\nControls: SPACE pause | N/RIGHT next | P/LEFT previous | R restart | Q/ESC quit")
    announce()
    with viewer.launch_passive(model, data, key_callback=control.on_key) as window:
        window.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        window.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        window.cam.distance = 3.0
        window.cam.azimuth = 135.0
        window.cam.elevation = -18.0
        next_tick = time.perf_counter()
        while window.is_running():
            action = control.consume_action()
            if action == "quit":
                break
            if action == "next":
                change_clip((clip_index + 1) % len(clips))
                next_tick = time.perf_counter()
            elif action == "previous":
                change_clip((clip_index - 1) % len(clips))
                next_tick = time.perf_counter()
            elif action == "restart":
                frame = 0
                next_tick = time.perf_counter()

            if control.is_paused():
                window.sync()
                time.sleep(0.02)
                next_tick = time.perf_counter()
                continue

            data.qpos[:] = qpos[frame]
            mujoco.mj_forward(model, data)
            window.sync()
            frame += 1
            if frame >= len(qpos):
                next_index = clip_index + 1
                if next_index >= len(clips):
                    if not args.loop_batch:
                        print("\nAll sampled clips finished.", flush=True)
                        break
                    next_index = 0
                change_clip(next_index)
                next_tick = time.perf_counter()
                continue

            next_tick += 1.0 / (fps * args.speed)
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.5:
                next_tick = time.perf_counter()


if __name__ == "__main__":
    main()
