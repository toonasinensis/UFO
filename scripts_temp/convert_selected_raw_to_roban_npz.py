#!/usr/bin/env python3
"""Convert stripped Roban NPZ files to the metadata-rich fk50 layout.

The source files contain joint trajectories and unnamed body FK arrays.  This
script preserves the root and joint trajectory, then recomputes all body FK
against the authoritative Roban MuJoCo model so that body_names and body arrays
cannot disagree.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = Path("/home/thl/Downloads/selected_raw")
DEFAULT_OUTPUT_DIR = Path("/home/thl/Downloads/selected_raw_aligned")
DEFAULT_ASSET_DIR = PROJECT_ROOT / "humanoidverse/data/robots/roban_s22_handball/roban_s22_handball"
DEFAULT_XML = DEFAULT_ASSET_DIR / "xml/biped_s17_verified_handball_fixedhead.xml"
DEFAULT_URDF = DEFAULT_ASSET_DIR / "urdf/biped_s17_verified_handball_fixedhead.urdf"

# selected_raw was logged from Isaac/PhysX in breadth-first articulation order.
SOURCE_JOINT_NAMES = (
    "waist_yaw_joint",
    "zarm_l1_joint",
    "zarm_r1_joint",
    "leg_l1_joint",
    "leg_r1_joint",
    "zarm_l2_joint",
    "zarm_r2_joint",
    "leg_l2_joint",
    "leg_r2_joint",
    "zarm_l3_joint",
    "zarm_r3_joint",
    "leg_l3_joint",
    "leg_r3_joint",
    "zarm_l4_joint",
    "zarm_r4_joint",
    "leg_l4_joint",
    "leg_r4_joint",
    "leg_l5_joint",
    "leg_r5_joint",
    "leg_l6_joint",
    "leg_r6_joint",
)

# This is the exact joint order stored by dataset 2.
TARGET_JOINT_NAMES = (
    "waist_yaw_joint",
    "leg_l1_joint",
    "leg_l2_joint",
    "leg_l3_joint",
    "leg_l4_joint",
    "leg_l5_joint",
    "leg_l6_joint",
    "leg_r1_joint",
    "leg_r2_joint",
    "leg_r3_joint",
    "leg_r4_joint",
    "leg_r5_joint",
    "leg_r6_joint",
    "zarm_l1_joint",
    "zarm_l2_joint",
    "zarm_l3_joint",
    "zarm_l4_joint",
    "zarm_r1_joint",
    "zarm_r2_joint",
    "zarm_r3_joint",
    "zarm_r4_joint",
)

# Isaac/PhysX body order from the source biped_s17_hands articulation.
SOURCE_BODY_NAMES = (
    "base_link",
    "waist_yaw_link",
    "zarm_l1_link",
    "zarm_r1_link",
    "zhead_1_link",
    "leg_l1_link",
    "leg_r1_link",
    "zarm_l2_link",
    "zarm_r2_link",
    "head_radar",
    "zhead_2_link",
    "leg_l2_link",
    "leg_r2_link",
    "zarm_l3_link",
    "zarm_r3_link",
    "camera_base",
    "leg_l3_link",
    "leg_r3_link",
    "zarm_l4_link",
    "zarm_r4_link",
    "leg_l4_link",
    "leg_r4_link",
    "zarm_l5_link",
    "zarm_r5_link",
    "leg_l5_link",
    "leg_r5_link",
    "leg_l6_link",
    "leg_r6_link",
)

REQUIRED_SOURCE_FIELDS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value: np.ndarray, name: str, path: Path) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{path}: {name} must contain one value, got {array.shape}")
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{path}: invalid {name}={result}")
    return result


def model_names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> list[str]:
    names = [mujoco.mj_id2name(model, object_type, index) for index in range(count)]
    if any(name is None for name in names):
        raise ValueError(f"MuJoCo model contains unnamed {object_type.name} objects")
    return [str(name) for name in names]


def angular_velocity_world(quat_wxyz: np.ndarray, fps: float) -> np.ndarray:
    """Differentiate wxyz quaternions into world-frame angular velocity."""
    quat = quat_wxyz.astype(np.float64, copy=True)
    for frame in range(1, quat.shape[0]):
        flip = np.sum(quat[frame] * quat[frame - 1], axis=-1) < 0
        quat[frame, flip] *= -1

    quat_dot = np.gradient(quat, 1.0 / fps, axis=0, edge_order=1)
    w, xyz = quat[..., :1], quat[..., 1:]
    dw, dxyz = quat_dot[..., :1], quat_dot[..., 1:]
    # Vector part of q_dot * conjugate(q), multiplied by two.
    omega = 2.0 * (-dw * xyz + w * dxyz - np.cross(dxyz, xyz))
    return omega.astype(np.float32)


def validate_source(path: Path, source: dict[str, np.ndarray]) -> tuple[int, float]:
    missing = [name for name in REQUIRED_SOURCE_FIELDS if name not in source]
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")
    joint_pos = np.asarray(source["joint_pos"])
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(SOURCE_JOINT_NAMES):
        raise ValueError(f"{path}: joint_pos must have shape [T, {len(SOURCE_JOINT_NAMES)}], got {joint_pos.shape}")
    frames = joint_pos.shape[0]
    if frames < 2:
        raise ValueError(f"{path}: at least two frames are required")
    if np.asarray(source["joint_vel"]).shape != joint_pos.shape:
        raise ValueError(f"{path}: joint_vel shape does not match joint_pos")
    for field, width in (("body_pos_w", 3), ("body_quat_w", 4)):
        array = np.asarray(source[field])
        expected_shape = (frames, len(SOURCE_BODY_NAMES), width)
        if array.shape != expected_shape:
            raise ValueError(f"{path}: invalid {field} shape {array.shape}")
    root_quat = np.asarray(source["body_quat_w"])[:, 0]
    norm_error = np.max(np.abs(np.linalg.norm(root_quat, axis=1) - 1.0))
    if norm_error > 1.0e-3:
        raise ValueError(f"{path}: root quaternion norm error is {norm_error:.6g}")
    return frames, scalar(source["fps"], "fps", path)


def validate_fk_alignment(
    source_path: Path,
    source: dict[str, np.ndarray],
    target_body_names: list[str],
    target_body_pos: np.ndarray,
    target_body_quat: np.ndarray,
) -> None:
    """Prove the inferred source joint mapping against the source body FK."""
    target_index = {name: index for index, name in enumerate(target_body_names)}
    common = [(source_index, target_index[name]) for source_index, name in enumerate(SOURCE_BODY_NAMES) if name in target_index]
    if len(common) < 20:
        raise ValueError("Too few common source/target bodies for FK validation")
    frame_indices = np.unique(np.linspace(0, target_body_pos.shape[0] - 1, num=min(8, target_body_pos.shape[0]), dtype=int))
    source_body_pos = np.asarray(source["body_pos_w"], dtype=np.float32)
    source_body_quat = np.asarray(source["body_quat_w"], dtype=np.float32)
    max_position_error = 0.0
    max_quaternion_error = 0.0
    for frame in frame_indices:
        for source_index, target_index_value in common:
            max_position_error = max(
                max_position_error,
                float(np.linalg.norm(source_body_pos[frame, source_index] - target_body_pos[frame, target_index_value])),
            )
            source_quat = source_body_quat[frame, source_index]
            target_quat = target_body_quat[frame, target_index_value]
            max_quaternion_error = max(
                max_quaternion_error,
                float(
                    min(
                        np.linalg.norm(source_quat - target_quat),
                        np.linalg.norm(source_quat + target_quat),
                    )
                ),
            )
    tolerance = 2.0e-4
    if max_position_error > tolerance or max_quaternion_error > tolerance:
        raise ValueError(
            f"{source_path}: inferred Isaac joint mapping failed FK validation: "
            f"max_position_error={max_position_error:.6g}, "
            f"max_quaternion_error={max_quaternion_error:.6g}"
        )


def convert_one(
    source_path: Path,
    output_path: Path,
    model: mujoco.MjModel,
    xml_path: Path,
    urdf_path: Path,
    asset_hash: str,
    urdf_hash: str,
) -> tuple[int, float]:
    with np.load(source_path, allow_pickle=False) as archive:
        source = {name: np.asarray(archive[name]) for name in archive.files}
    frames, fps = validate_source(source_path, source)

    xml_joint_names = model_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)[1:]
    if tuple(xml_joint_names) != TARGET_JOINT_NAMES:
        raise ValueError("XML actuated joint order does not match TARGET_JOINT_NAMES")
    source_index = {name: index for index, name in enumerate(SOURCE_JOINT_NAMES)}
    target_from_source = np.asarray([source_index[name] for name in TARGET_JOINT_NAMES])

    source_joint_pos = np.asarray(source["joint_pos"], dtype=np.float32)
    joint_pos = source_joint_pos[:, target_from_source]
    root_pos = np.asarray(source["body_pos_w"], dtype=np.float32)[:, 0]
    root_quat_wxyz = np.asarray(source["body_quat_w"], dtype=np.float32)[:, 0]
    qpos = np.empty((frames, model.nq), dtype=np.float64)
    qpos[:, :3] = root_pos
    qpos[:, 3:7] = root_quat_wxyz
    qpos[:, 7:] = joint_pos

    body_names = model_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)[1:]
    body_pos = np.empty((frames, len(body_names), 3), dtype=np.float32)
    body_quat = np.empty((frames, len(body_names), 4), dtype=np.float32)
    data = mujoco.MjData(model)
    for frame in range(frames):
        data.qpos[:] = qpos[frame]
        mujoco.mj_kinematics(model, data)
        body_pos[frame] = data.xpos[1:]
        body_quat[frame] = data.xquat[1:]

    validate_fk_alignment(source_path, source, body_names, body_pos, body_quat)

    joint_vel = np.gradient(joint_pos, 1.0 / fps, axis=0, edge_order=1).astype(np.float32)
    body_lin_vel = np.gradient(body_pos, 1.0 / fps, axis=0, edge_order=1).astype(np.float32)
    body_ang_vel = angular_velocity_world(body_quat, fps)
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    compact_data = np.concatenate((root_pos, root_quat_xyzw, joint_pos), axis=1).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            temporary_path,
            schema_version=np.asarray(1, dtype=np.int64),
            data=compact_data,
            fps=np.asarray(fps, dtype=np.float64),
            source_fps=np.asarray(fps, dtype=np.float64),
            joint_names=np.asarray(TARGET_JOINT_NAMES),
            body_names=np.asarray(body_names),
            source_file=np.asarray(str(source_path.resolve())),
            source_g11_file=np.asarray(str(source_path.resolve())),
            source_g11_sha256=np.asarray(sha256(source_path)),
            quat_order=np.asarray("xyzw"),
            body_quat_order=np.asarray("wxyz"),
            asset_path=np.asarray(str(xml_path.resolve())),
            asset_hash=np.asarray(asset_hash),
            urdf_path=np.asarray(str(urdf_path.resolve())),
            urdf_hash=np.asarray(urdf_hash),
            scaler_version=np.asarray("selected_raw_alignment_v2"),
            retargeter_version=np.asarray("isaac_bfs_to_mujoco_fk_recomputed_v2"),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos,
            body_quat_w=body_quat,
            body_lin_vel_w=body_lin_vel,
            body_ang_vel_w=body_ang_vel,
        )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return frames, fps


def output_name(source_path: Path) -> str:
    suffix = ".roban_s22.fk50"
    stem = source_path.stem
    if stem.endswith(suffix):
        return f"{stem}.npz"
    return f"{stem}{suffix}.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--pattern", default="*.npz")
    parser.add_argument("--limit", type=int, default=None, help="Convert only the first N files")
    parser.add_argument(
        "--preserve-filenames",
        action="store_true",
        help="Keep source filenames instead of appending .roban_s22.fk50",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    xml_path = args.xml.expanduser().resolve()
    urdf_path = args.urdf.expanduser().resolve()
    for path in (input_dir, xml_path, urdf_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if input_dir == output_dir:
        raise ValueError("--output-dir must differ from --input-dir")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    files = sorted(input_dir.glob(args.pattern))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No files matching {args.pattern!r} under {input_dir}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    asset_hash = sha256(xml_path)
    urdf_hash = sha256(urdf_path)
    converted = skipped = total_frames = 0
    print(f"input={input_dir}")
    print(f"output={output_dir}")
    print(f"xml={xml_path}")
    print(f"files={len(files)}")
    for index, source_path in enumerate(files, start=1):
        destination = output_dir / (source_path.name if args.preserve_filenames else output_name(source_path))
        if destination.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(files)}] skip existing: {destination.name}")
            continue
        frames, fps = convert_one(
            source_path,
            destination,
            model,
            xml_path,
            urdf_path,
            asset_hash,
            urdf_hash,
        )
        converted += 1
        total_frames += frames
        print(f"[{index}/{len(files)}] {source_path.name} -> {destination.name} ({frames} frames @ {fps:g} Hz)")
    print(f"done: converted={converted} skipped={skipped} frames={total_frames}")


if __name__ == "__main__":
    main()
