#!/usr/bin/env python3
"""Play a Roban RobotState/fk50 NPZ trajectory in the native MuJoCo viewer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path("/home/thl/Downloads/roban_npz")
DEFAULT_XML = (
    PROJECT_ROOT / "humanoidverse/data/robots/roban_s22_handball/roban_s22_handball/xml" / "biped_s17_verified_handball_fixedhead.xml"
)


def text_scalar(archive: np.lib.npyio.NpzFile, name: str, default: str) -> str:
    if name not in archive:
        return default
    value = np.asarray(archive[name])
    if value.size != 1:
        raise ValueError(f"{name} must be a scalar, got {value.shape}")
    return str(value.reshape(-1)[0])


def float_scalar(archive: np.lib.npyio.NpzFile, name: str) -> float:
    if name not in archive:
        raise ValueError(f"NPZ is missing {name}")
    value = np.asarray(archive[name])
    if value.size != 1:
        raise ValueError(f"{name} must be a scalar, got {value.shape}")
    result = float(value.reshape(-1)[0])
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"invalid {name}={result}")
    return result


def resolve_motion(path: Path) -> Path:
    path = path.expanduser()
    if path.is_file():
        return path.resolve()
    candidate = DEFAULT_DATA_DIR / path
    if candidate.is_file():
        return candidate.resolve()
    matches = sorted(DEFAULT_DATA_DIR.glob(str(path)))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        names = "\n  ".join(item.name for item in matches[:20])
        raise ValueError(f"Motion pattern is ambiguous ({len(matches)} matches):\n  {names}")
    raise FileNotFoundError(path)


def load_qpos(path: Path, model: mujoco.MjModel) -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=False) as archive:
        fps = float_scalar(archive, "fps")
        if "data" in archive:
            packed = np.asarray(archive["data"], dtype=np.float64)
            if packed.ndim != 2 or packed.shape[1] < 8:
                raise ValueError(f"data must have shape [T, 7 + J], got {packed.shape}")
            root_pos = packed[:, :3]
            root_quat = packed[:, 3:7]
            joint_pos = packed[:, 7:]
            quat_order = text_scalar(archive, "quat_order", "xyzw")
        else:
            required = ("joint_pos", "body_pos_w", "body_quat_w", "body_names")
            missing = [name for name in required if name not in archive]
            if missing:
                raise ValueError(f"NPZ is missing {missing}")
            body_names = [str(item) for item in np.asarray(archive["body_names"]).tolist()]
            if "base_link" not in body_names:
                raise ValueError("body_names does not contain base_link")
            base_index = body_names.index("base_link")
            root_pos = np.asarray(archive["body_pos_w"], dtype=np.float64)[:, base_index]
            root_quat = np.asarray(archive["body_quat_w"], dtype=np.float64)[:, base_index]
            joint_pos = np.asarray(archive["joint_pos"], dtype=np.float64)
            quat_order = text_scalar(archive, "body_quat_order", "wxyz")

        if "joint_names" not in archive:
            raise ValueError("NPZ is missing joint_names; refusing to guess the joint order")
        joint_names = [str(item) for item in np.asarray(archive["joint_names"]).tolist()]

    frames = root_pos.shape[0]
    if root_pos.shape != (frames, 3) or root_quat.shape != (frames, 4):
        raise ValueError("root position/quaternion shapes are inconsistent")
    if joint_pos.shape != (frames, len(joint_names)):
        raise ValueError("joint_pos shape does not match joint_names")
    if quat_order == "xyzw":
        root_quat_wxyz = root_quat[:, [3, 0, 1, 2]]
    elif quat_order == "wxyz":
        root_quat_wxyz = root_quat
    else:
        raise ValueError(f"unsupported quaternion order {quat_order!r}")

    xml_joints = [str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)) for index in range(1, model.njnt)]
    input_index = {name: index for index, name in enumerate(joint_names)}
    missing_joints = [name for name in xml_joints if name not in input_index]
    if missing_joints:
        raise ValueError(f"NPZ is missing XML joints {missing_joints}")

    qpos = np.empty((frames, model.nq), dtype=np.float64)
    qpos[:, :3] = root_pos
    qpos[:, 3:7] = root_quat_wxyz
    qpos[:, 7:] = joint_pos[:, [input_index[name] for name in xml_joints]]
    if not np.all(np.isfinite(qpos)):
        raise ValueError("trajectory contains NaN or infinity")
    return qpos, fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "motion",
        type=Path,
        help="NPZ path, filename under ~/Downloads/roban_npz, or a unique glob pattern",
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive end frame")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Validate and print metadata without opening a GUI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.speed <= 0:
        raise ValueError("--speed must be positive")
    motion_path = resolve_motion(args.motion)
    xml_path = args.xml.expanduser().resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    qpos, fps = load_qpos(motion_path, model)
    start = args.start_frame
    end = len(qpos) if args.end_frame is None else args.end_frame
    if not 0 <= start < end <= len(qpos):
        raise ValueError(f"frame range must satisfy 0 <= start < end <= {len(qpos)}")

    print(f"motion={motion_path}")
    print(f"xml={xml_path}")
    print(f"frames={len(qpos)} fps={fps:g} duration={len(qpos) / fps:.3f}s")
    print(f"playback=[{start}, {end}) speed={args.speed:g}x loop={args.loop}")
    if args.check_only:
        return

    from mujoco import viewer

    data = mujoco.MjData(model)
    with viewer.launch_passive(model, data) as window:
        window.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        window.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        window.cam.distance = 3.0
        window.cam.azimuth = 135.0
        window.cam.elevation = -18.0
        frame = start
        next_tick = time.perf_counter()
        while window.is_running():
            data.qpos[:] = qpos[frame]
            mujoco.mj_forward(model, data)
            window.sync()
            frame += 1
            if frame >= end:
                if not args.loop:
                    break
                frame = start
            next_tick += 1.0 / (fps * args.speed)
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.5:
                next_tick = time.perf_counter()


if __name__ == "__main__":
    main()
