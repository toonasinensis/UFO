"""Minimal native-MuJoCo runner for an exported UFO FB policy.

This entry point intentionally does not construct the training environment and
does not load a PyTorch checkpoint.  It only needs:

* the robot MJCF;
* a RobotState NPZ motion;
* ``FBcprAuxModel.onnx``;
* the precomputed tracking latent ``tracking_inference/zs_0.npy``.

Example:

    python -m humanoidverse.onnx_mujoco_sim \
      --model-folder runs/新数据addlelay_onnx \
      --motion humanoidverse/data/roban/named_roban_lafan_10s/aiming1_subject1_0000.npz
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/新数据addlelay_onnx"
DEFAULT_MOTION = ROOT / "humanoidverse/data/roban/named_roban_lafan_10s/aiming1_subject1_0000.npz"
DEFAULT_SCENE = ROOT / "humanoidverse/data/robots/roban_s22_handball/roban_s22_handball/xml/scene.xml"

# Roban S2.2 motor no-load speeds from wbc_parkour/robots/roban_s22.py.
# Its actuator uses the linear T-N envelope implemented in
# wbc_parkour/actuators/actuator_pd.py::_clip_effort.
ROBAN_TN_VELOCITY_LIMITS = {
    "waist_yaw_joint": 12.0,
    **{f"leg_{side}{joint}_joint": 14.6 for side in "lr" for joint in (1, 2, 4)},
    **{f"leg_{side}3_joint": 12.0 for side in "lr"},
    **{f"leg_{side}{joint}_joint": 17.0 for side in "lr" for joint in (5, 6)},
    **{f"zarm_{side}1_joint": 10.5 for side in "lr"},
    **{f"zarm_{side}{joint}_joint": 15.0 for side in "lr" for joint in (2, 3, 4)},
}

def _torque_limits(
    joint_vel: np.ndarray,
    effort_limits: np.ndarray,
    velocity_limits: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lower/upper motor torque bounds, matching wbc_parkour's T-N clip."""
    if velocity_limits is None:
        return -effort_limits, effort_limits
    upper = effort_limits * (1.0 - joint_vel / velocity_limits)
    lower = effort_limits * (-1.0 - joint_vel / velocity_limits)
    return np.clip(lower, -effort_limits, 0.0), np.clip(upper, 0.0, effort_limits)


def _safe_output_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    return cleaned.strip("._-") or "unnamed"


def _new_diagnostics_dir(root: Path, motion_path: Path, model_folder: Path) -> Path:
    """Create a non-overwriting diagnostics folder named for motion and run."""
    root = root.expanduser().resolve()
    base_name = f"{_safe_output_name(motion_path.stem)}__{_safe_output_name(model_folder.name)}"
    output = root / base_name
    if output.exists():
        output = root / f"{base_name}__{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def _joint_groups(joint_names: list[str]) -> dict[str, list[int]]:
    requested = {
        "left_leg": [f"leg_l{i}_joint" for i in range(1, 5)],
        "left_ankle": [f"leg_l{i}_joint" for i in range(5, 7)],
        "right_leg": [f"leg_r{i}_joint" for i in range(1, 5)],
        "right_ankle": [f"leg_r{i}_joint" for i in range(5, 7)],
        "waist": ["waist_yaw_joint"],
        "left_arm": [f"zarm_l{i}_joint" for i in range(1, 5)],
        "right_arm": [f"zarm_r{i}_joint" for i in range(1, 5)],
    }
    index = {name: i for i, name in enumerate(joint_names)}
    groups = {group: [index[name] for name in names if name in index] for group, names in requested.items()}
    assigned = {joint_index for indices in groups.values() for joint_index in indices}
    unassigned = [i for i in range(len(joint_names)) if i not in assigned]
    if unassigned:
        groups["other"] = unassigned
    return {name: indices for name, indices in groups.items() if indices}


def _link_groups(link_names: list[str]) -> dict[str, list[int]]:
    requested = {
        "left_leg": [f"leg_l{i}_link" for i in range(1, 5)],
        "left_ankle": [f"leg_l{i}_link" for i in range(5, 7)],
        "right_leg": [f"leg_r{i}_link" for i in range(1, 5)],
        "right_ankle": [f"leg_r{i}_link" for i in range(5, 7)],
        "waist": ["waist_yaw_link"],
        "left_arm": [f"zarm_l{i}_link" for i in range(1, 5)],
        "right_arm": [f"zarm_r{i}_link" for i in range(1, 5)],
        "head": ["zhead_1_link", "zhead_2_link"],
    }
    index = {name: i for i, name in enumerate(link_names)}
    groups = {group: [index[name] for name in names if name in index] for group, names in requested.items()}
    assigned = {link_index for indices in groups.values() for link_index in indices}
    unassigned = [i for i in range(len(link_names)) if i not in assigned]
    if unassigned:
        groups["other"] = unassigned
    return {name: indices for name, indices in groups.items() if indices}


def _local_link_positions(body_pos: np.ndarray, body_quat_xyzw: np.ndarray) -> np.ndarray:
    heading_inv = _heading_inverse(body_quat_xyzw[0])
    heading_all = np.broadcast_to(heading_inv, body_pos.shape[:-1] + (4,))
    return _quat_rotate(heading_all, body_pos - body_pos[0])


def _reference_joint_velocities(
    model: mujoco.MjModel, qpos_frames: np.ndarray, joint_dof_adr: np.ndarray, fps: float
) -> np.ndarray:
    velocities = np.empty((len(qpos_frames), len(joint_dof_adr)), dtype=np.float64)
    qvel = np.empty(model.nv, dtype=np.float64)
    for frame in range(len(qpos_frames)):
        previous = max(frame - 1, 0)
        following = min(frame + 1, len(qpos_frames) - 1)
        span = max(following - previous, 1) / fps
        mujoco.mj_differentiatePos(model, qvel, span, qpos_frames[previous], qpos_frames[following])
        velocities[frame] = qvel[joint_dof_adr]
    return velocities


def _series_stats(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(finite)),
        "rmse": float(np.sqrt(np.mean(np.square(finite)))),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _joint_position_limit_activity(
    joint_pos: np.ndarray,
    hard_limits: np.ndarray,
    soft_limits: np.ndarray | None,
    margin_rad: float,
) -> dict[str, np.ndarray]:
    """Classify per-frame joint positions near or beyond soft/hard limits."""
    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    hard_limits = np.asarray(hard_limits, dtype=np.float64)
    if joint_pos.ndim != 2 or hard_limits.shape != (joint_pos.shape[1], 2):
        raise ValueError(
            "Invalid joint position limit shapes: "
            f"joint_pos={joint_pos.shape}, hard_limits={hard_limits.shape}"
        )
    if margin_rad < 0.0 or not np.isfinite(margin_rad):
        raise ValueError(f"joint limit highlight margin must be finite and non-negative, got {margin_rad}")

    hard_lower = hard_limits[:, 0]
    hard_upper = hard_limits[:, 1]
    finite_hard = np.isfinite(hard_lower) & np.isfinite(hard_upper)
    hard_exceeded = finite_hard[None, :] & (
        (joint_pos < hard_lower[None, :]) | (joint_pos > hard_upper[None, :])
    )
    hard_near = finite_hard[None, :] & ~hard_exceeded & (
        (joint_pos <= hard_lower[None, :] + margin_rad)
        | (joint_pos >= hard_upper[None, :] - margin_rad)
    )
    hard_violation = np.maximum(
        np.maximum(hard_lower[None, :] - joint_pos, joint_pos - hard_upper[None, :]),
        0.0,
    )
    hard_violation[:, ~finite_hard] = 0.0

    soft_reached = np.zeros_like(hard_exceeded)
    if soft_limits is not None:
        soft_limits = np.asarray(soft_limits, dtype=np.float64)
        if soft_limits.shape != hard_limits.shape:
            raise ValueError(
                f"Invalid soft joint limit shape: {soft_limits.shape}, expected={hard_limits.shape}"
            )
        soft_lower = soft_limits[:, 0]
        soft_upper = soft_limits[:, 1]
        finite_soft = np.isfinite(soft_lower) & np.isfinite(soft_upper)
        soft_reached = finite_soft[None, :] & (
            (joint_pos <= soft_lower[None, :] + margin_rad)
            | (joint_pos >= soft_upper[None, :] - margin_rad)
        )

    return {
        "soft_reached": soft_reached,
        "hard_near": hard_near,
        "hard_exceeded": hard_exceeded,
        "hard_violation": hard_violation,
    }


def _write_diagnostics(
    output_dir: Path,
    records: dict[str, list[np.ndarray | float | int]],
    joint_names: list[str],
    link_names: list[str],
    effort_limits: np.ndarray,
    hard_joint_limits: np.ndarray,
    soft_joint_limits: np.ndarray | None,
    joint_limit_highlight_margin_rad: float,
    motion_path: Path,
    model_folder: Path,
    torque_clip_mode: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = {key: np.asarray(values) for key, values in records.items()}
    np.savez_compressed(
        output_dir / "timeseries.npz",
        **arrays,
        joint_names=np.asarray(joint_names),
        link_names=np.asarray(link_names),
        effort_limits=np.asarray(effort_limits),
        hard_joint_limits=np.asarray(hard_joint_limits),
        soft_joint_limits=(
            np.asarray(soft_joint_limits)
            if soft_joint_limits is not None
            else np.full_like(hard_joint_limits, np.nan)
        ),
        joint_limit_highlight_margin_rad=np.asarray(joint_limit_highlight_margin_rad),
    )
    times = arrays["time_s"]
    joint_error = np.abs(arrays["joint_pos_error"])
    link_error = arrays["local_link_pos_error"]
    mean_joint_error = joint_error.mean(axis=1)
    mean_link_error = link_error.mean(axis=1)
    limit_activity = _joint_position_limit_activity(
        arrays["joint_pos"],
        hard_joint_limits,
        soft_joint_limits,
        joint_limit_highlight_margin_rad,
    )

    summary = {
        "motion": str(motion_path),
        "run": str(model_folder),
        "torque_clip_mode": torque_clip_mode,
        "frames": int(len(times)),
        "duration_s": float(times[-1]) if len(times) else 0.0,
        "mean_joint_pos_error_rad": _series_stats(mean_joint_error),
        "mean_local_link_pos_error_m": _series_stats(mean_link_error),
        "local_link_pos_error_m": {
            name: _series_stats(link_error[:, index]) for index, name in enumerate(link_names)
        },
        "joint_pos_error_rad": {
            name: _series_stats(joint_error[:, index]) for index, name in enumerate(joint_names)
        },
        "torque_nm": {
            name: {
                **_series_stats(np.abs(arrays["torque"][:, index])),
                "rms": float(np.sqrt(np.mean(np.square(arrays["torque"][:, index])))),
            }
            for index, name in enumerate(joint_names)
        },
        "torque_clipped_fraction": float(np.mean(arrays["torque_was_clipped"])),
        "joint_position_limit_activity": {
            "highlight_margin_rad": float(joint_limit_highlight_margin_rad),
            "highlight_margin_deg": float(np.rad2deg(joint_limit_highlight_margin_rad)),
            "per_joint": {
                name: {
                    "hard_lower_rad": float(hard_joint_limits[index, 0]),
                    "hard_upper_rad": float(hard_joint_limits[index, 1]),
                    "soft_lower_rad": (
                        float(soft_joint_limits[index, 0])
                        if soft_joint_limits is not None
                        else None
                    ),
                    "soft_upper_rad": (
                        float(soft_joint_limits[index, 1])
                        if soft_joint_limits is not None
                        else None
                    ),
                    "soft_limit_reached_fraction": float(
                        np.mean(limit_activity["soft_reached"][:, index])
                    ),
                    "hard_limit_near_fraction": float(
                        np.mean(limit_activity["hard_near"][:, index])
                    ),
                    "hard_limit_exceeded_fraction": float(
                        np.mean(limit_activity["hard_exceeded"][:, index])
                    ),
                    "max_hard_limit_violation_rad": float(
                        np.max(limit_activity["hard_violation"][:, index])
                    ),
                }
                for index, name in enumerate(joint_names)
            },
        },
    }
    link_groups = _link_groups(link_names)
    summary["local_link_pos_error_groups_m"] = {
        group_name: _series_stats(link_error[:, indices].mean(axis=1))
        for group_name, indices in link_groups.items()
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(times, mean_link_error, color="tab:blue")
    axes[0].set_ylabel("Mean local link error [m]")
    axes[1].plot(times, mean_joint_error, color="tab:orange")
    axes[1].set_ylabel("Mean joint position error [rad]")
    axes[1].set_xlabel("Time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "overall_errors.png", dpi=160)
    plt.close(figure)

    plot_specs = {
        "joint_pos": ("joint_pos", "reference_joint_pos", "Position [rad]"),
        "joint_vel": ("joint_vel", "reference_joint_vel", "Velocity [rad/s]"),
    }
    groups = _joint_groups(joint_names)
    for folder_name, (policy_key, reference_key, ylabel) in plot_specs.items():
        folder = output_dir / folder_name
        folder.mkdir()
        for group_name, indices in groups.items():
            figure, axes = plt.subplots(len(indices), 1, figsize=(12, max(3.0, 2.4 * len(indices))), sharex=True)
            axes = np.atleast_1d(axes)
            for axis, index in zip(axes, indices):
                policy_values = arrays[policy_key][:, index]
                axis.plot(times, policy_values, label="policy", color="tab:blue", linewidth=1.2)
                axis.plot(
                    times,
                    arrays[reference_key][:, index],
                    label="reference",
                    color="0.35",
                    linestyle="--",
                    linewidth=1.1,
                )
                if folder_name == "joint_pos":
                    hard_lower, hard_upper = hard_joint_limits[index]
                    axis.axhline(
                        hard_lower,
                        color="tab:red",
                        linestyle="-",
                        alpha=0.8,
                        linewidth=1.0,
                        label="XML hard limit",
                    )
                    axis.axhline(
                        hard_upper,
                        color="tab:red",
                        linestyle="-",
                        alpha=0.8,
                        linewidth=1.0,
                        label="_nolegend_",
                    )
                    if soft_joint_limits is not None:
                        soft_lower, soft_upper = soft_joint_limits[index]
                        axis.axhline(
                            soft_lower,
                            color="tab:orange",
                            linestyle=":",
                            alpha=0.9,
                            linewidth=1.2,
                            label="0.95 soft limit",
                        )
                        axis.axhline(
                            soft_upper,
                            color="tab:orange",
                            linestyle=":",
                            alpha=0.9,
                            linewidth=1.2,
                            label="_nolegend_",
                        )
                    soft_mask = limit_activity["soft_reached"][:, index]
                    hard_near_mask = limit_activity["hard_near"][:, index]
                    exceeded_mask = limit_activity["hard_exceeded"][:, index]
                    axis.scatter(
                        times[soft_mask],
                        policy_values[soft_mask],
                        color="tab:orange",
                        s=7,
                        alpha=0.65,
                        zorder=4,
                        label="at/outside soft limit",
                    )
                    axis.scatter(
                        times[hard_near_mask],
                        policy_values[hard_near_mask],
                        color="tab:red",
                        s=10,
                        alpha=0.8,
                        zorder=5,
                        label="within margin of hard limit",
                    )
                    axis.scatter(
                        times[exceeded_mask],
                        policy_values[exceeded_mask],
                        color="black",
                        marker="x",
                        s=18,
                        linewidths=0.9,
                        zorder=6,
                        label="exceeds XML hard limit",
                    )
                axis.set_ylabel(joint_names[index])
                axis.grid(True, alpha=0.3)
            axes[0].legend(loc="upper right", ncol=3, fontsize=8)
            axes[-1].set_xlabel("Time [s]")
            figure.supylabel(ylabel)
            figure.tight_layout()
            figure.savefig(folder / f"{group_name}.png", dpi=150)
            plt.close(figure)

    error_folder = output_dir / "joint_pos_error"
    torque_folder = output_dir / "torque"
    error_folder.mkdir()
    torque_folder.mkdir()
    for group_name, indices in groups.items():
        figure, axis = plt.subplots(figsize=(12, 4.8))
        for index in indices:
            axis.plot(times, arrays["joint_pos_error"][:, index], label=joint_names[index])
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set(xlabel="Time [s]", ylabel="Policy - reference [rad]")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", ncol=2)
        figure.tight_layout()
        figure.savefig(error_folder / f"{group_name}.png", dpi=150)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(12, 4.8))
        for index in indices:
            line = axis.plot(times, arrays["torque"][:, index], label=joint_names[index])[0]
            axis.plot(
                times, arrays["torque_limit_max"][:, index], color=line.get_color(),
                linestyle=":", alpha=0.6, linewidth=1.0,
            )
            axis.plot(
                times, arrays["torque_limit_min"][:, index], color=line.get_color(),
                linestyle=":", alpha=0.6, linewidth=1.0,
            )
        axis.set(xlabel="Time [s]", ylabel="Applied torque [Nm]")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", ncol=2)
        figure.tight_layout()
        figure.savefig(torque_folder / f"{group_name}.png", dpi=150)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 4.8))
    axis.plot(times, np.mean(np.abs(arrays["torque"]), axis=1))
    axis.set(xlabel="Time [s]", ylabel="Mean absolute torque [Nm]")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "mean_abs_torque.png", dpi=160)
    plt.close(figure)

    link_folder = output_dir / "local_link_pos_error"
    link_folder.mkdir()
    for group_name, indices in link_groups.items():
        figure, axis = plt.subplots(figsize=(12, 4.8))
        for index in indices:
            axis.plot(times, link_error[:, index], label=link_names[index], linewidth=1.1)
        axis.plot(
            times,
            link_error[:, indices].mean(axis=1),
            label=f"{group_name}_mean",
            color="black",
            linewidth=2.0,
        )
        axis.set(xlabel="Time [s]", ylabel="Root-frame position error [m]")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", ncol=2)
        figure.tight_layout()
        figure.savefig(link_folder / f"{group_name}.png", dpi=150)
        plt.close(figure)

# Deployment-only arm PD overrides. These do not modify the training YAML or
# the exported policy. Right-arm damping remains whatever was exported.
 


def _resolve_metadata_path(override: Path | None, metadata_value: str, label: str) -> Path:
    """Resolve export paths after moving a run between server and workstation."""
    if override is not None:
        path = override.expanduser().resolve()
    else:
        path = Path(metadata_value).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if not path.exists():
            parts = Path(metadata_value).parts
            if "UFO" in parts:
                suffix = parts[parts.index("UFO") + 1 :]
                local_path = ROOT.joinpath(*suffix).resolve()
                if local_path.exists():
                    path = local_path
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path

 
def _quat_mul(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Quaternion multiply for xyzw quaternions."""
    x0, y0, z0, w0 = np.moveaxis(q0, -1, 0)
    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)
    return np.stack(
        (
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        ),
        axis=-1,
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors by xyzw quaternions."""
    q_xyz = q[..., :3]
    uv = np.cross(q_xyz, v)
    uuv = np.cross(q_xyz, uv)
    return v + 2.0 * (q[..., 3:4] * uv + uuv)


def _quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_inv = q.copy()
    q_inv[..., :3] *= -1.0
    return _quat_rotate(q_inv, v)


def _heading_inverse(q: np.ndarray) -> np.ndarray:
    forward = _quat_rotate(q, np.broadcast_to(np.array([1.0, 0.0, 0.0]), q.shape[:-1] + (3,)))
    half = -0.5 * np.arctan2(forward[..., 1], forward[..., 0])
    zeros = np.zeros_like(half)
    return np.stack((zeros, zeros, np.sin(half), np.cos(half)), axis=-1)


def _max_local_observation(
    body_pos: np.ndarray,
    body_quat_xyzw: np.ndarray,
    body_lin_vel: np.ndarray,
    body_ang_vel: np.ndarray,
) -> np.ndarray:
    """NumPy equivalent of the body-count-dependent max-local observation."""
    root_pos = body_pos[0]
    heading_inv = _heading_inverse(body_quat_xyzw[0])
    heading_all = np.broadcast_to(heading_inv, body_quat_xyzw.shape)

    local_pos = _quat_rotate(heading_all, body_pos - root_pos)[1:].reshape(-1)
    local_quat = _quat_mul(heading_all, body_quat_xyzw)
    unit_x = np.broadcast_to(np.array([1.0, 0.0, 0.0]), body_pos.shape)
    unit_z = np.broadcast_to(np.array([0.0, 0.0, 1.0]), body_pos.shape)
    tan_norm = np.concatenate((_quat_rotate(local_quat, unit_x), _quat_rotate(local_quat, unit_z)), axis=-1).reshape(-1)
    local_vel = _quat_rotate(heading_all, body_lin_vel).reshape(-1)
    local_ang_vel = _quat_rotate(heading_all, body_ang_vel).reshape(-1)
    return np.concatenate(([root_pos[2]], local_pos, tan_norm, local_vel, local_ang_vel)).astype(np.float32)


def _prefix_reference_tree(root_body: ET.Element) -> ET.Element:
    reference = copy.deepcopy(root_body)
    for element in reference.iter():
        name = element.get("name")
        if name:
            element.set("name", f"reference_{name}")
        if element.tag == "body":
            element.set("gravcomp", "1")
        elif element.tag == "geom":
            element.set("contype", "0")
            element.set("conaffinity", "0")
            element.set("group", "1")
            element.set("rgba", "0.15 0.85 0.25 0.38")
    return reference


def _xml_body_names(xml_path: Path) -> list[str]:
    worldbody = ET.parse(xml_path).getroot().find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF has no <worldbody>: {xml_path}")
    return [str(body.get("name")) for body in worldbody.iter("body") if body.get("name")]


def _load_scene(
    scene_path: Path,
    robot_xml_path: Path,
    timestep: float,
    show_reference: bool,
) -> mujoco.MjModel:
    if not show_reference:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        model.opt.timestep = timestep
        return model

    scene_root = ET.parse(scene_path).getroot()
    worldbody = scene_root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Scene MJCF has no <worldbody>: {scene_path}")

    robot_worldbody = ET.parse(robot_xml_path).getroot().find("worldbody")
    if robot_worldbody is None:
        raise ValueError(f"Robot MJCF has no <worldbody>: {robot_xml_path}")
    robot_roots = robot_worldbody.findall("body")
    if len(robot_roots) != 1:
        raise ValueError(f"Expected one root robot body in {robot_xml_path}, found {len(robot_roots)}")
    worldbody.append(_prefix_reference_tree(robot_roots[0]))

    # Keep the temporary scene beside the real scene so relative <include>
    # and mesh paths retain normal MuJoCo resolution semantics.
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".xml",
            prefix=".onnx_mujoco_scene_",
            dir=scene_path.parent,
            delete=False,
            encoding="utf-8",
        ) as temporary:
            temporary.write(ET.tostring(scene_root, encoding="unicode"))
            temporary_path = Path(temporary.name)
        model = mujoco.MjModel.from_xml_path(str(temporary_path))
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    model.opt.timestep = timestep
    return model


def _name(model: mujoco.MjModel, obj: mujoco.mjtObj, index: int) -> str:
    value = mujoco.mj_id2name(model, obj, index)
    if value is None:
        raise ValueError(f"Unnamed {obj} at index {index}")
    return value


def _load_motion(
    path: Path, joint_names: list[str], model: mujoco.MjModel
) -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=True) as motion:
        required = {"fps", "joint_pos", "joint_names", "body_pos_w", "body_quat_w", "body_names"}
        missing = sorted(required.difference(motion.files))
        if missing:
            raise ValueError(f"Motion {path} is missing fields: {missing}")
        source_joints = [str(x) for x in motion["joint_names"].tolist()]
        body_names = [str(x) for x in motion["body_names"].tolist()]
        if "base_link" not in body_names:
            raise ValueError(f"Motion {path} has no base_link body")
        joint_index = {name: i for i, name in enumerate(source_joints)}
        absent = [name for name in joint_names if name not in joint_index]
        if absent:
            raise ValueError(f"Motion {path} is missing joints: {absent}")
        base_index = body_names.index("base_link")
        root_pos = np.asarray(motion["body_pos_w"][:, base_index], dtype=np.float64)
        # BFM/IsaacLab named NPZ stores body_quat_w in wxyz order.
        root_quat_wxyz = np.asarray(motion["body_quat_w"][:, base_index], dtype=np.float64)
        joints = np.asarray(motion["joint_pos"][:, [joint_index[name] for name in joint_names]], dtype=np.float64)
        free_joint_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free_joint_ids) != 1:
            raise ValueError(f"Expected one free joint in robot MJCF, found {len(free_joint_ids)}")
        free_qpos_adr = int(model.jnt_qposadr[int(free_joint_ids[0])])
        joint_ids = np.asarray(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names],
            dtype=int,
        )
        if np.any(joint_ids < 0):
            unresolved = [name for name, joint_id in zip(joint_names, joint_ids) if joint_id < 0]
            raise ValueError(f"Robot MJCF is missing motion joints: {unresolved}")
        joint_qpos_adr = model.jnt_qposadr[joint_ids].astype(int)

        # qpos must follow the MJCF's native address order, which is not
        # necessarily the policy/control order in ``joint_names``.
        qpos = np.zeros((len(root_pos), model.nq), dtype=np.float64)
        qpos[:, free_qpos_adr : free_qpos_adr + 3] = root_pos
        qpos[:, free_qpos_adr + 3 : free_qpos_adr + 7] = root_quat_wxyz
        qpos[:, joint_qpos_adr] = joints
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    if qpos.shape[0] < 2 or fps <= 0:
        raise ValueError(f"Motion must have at least two frames and positive fps, got shape={qpos.shape}, fps={fps}")
    return qpos, fps


def _reference_kinematics(
    model: mujoco.MjModel,
    qpos_frames: np.ndarray,
    body_ids: np.ndarray,
    frame: int,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate one reference frame and its finite-difference velocity."""
    data = mujoco.MjData(model)
    frame %= len(qpos_frames)
    prev_frame = (frame - 1) % len(qpos_frames)
    next_frame = (frame + 1) % len(qpos_frames)
    data.qpos[:] = qpos_frames[frame]
    if frame == 0:
        mujoco.mj_differentiatePos(model, data.qvel, 1.0 / fps, qpos_frames[frame], qpos_frames[next_frame])
    elif frame == len(qpos_frames) - 1:
        mujoco.mj_differentiatePos(model, data.qvel, 1.0 / fps, qpos_frames[prev_frame], qpos_frames[frame])
    else:
        mujoco.mj_differentiatePos(model, data.qvel, 2.0 / fps, qpos_frames[prev_frame], qpos_frames[next_frame])
    mujoco.mj_forward(model, data)

    body_pos = data.xpos[body_ids].copy()
    body_quat_xyzw = data.xquat[body_ids][:, [1, 2, 3, 0]].copy()
    body_lin_vel = np.empty((len(body_ids), 3), dtype=np.float64)
    body_ang_vel = np.empty_like(body_lin_vel)
    spatial = np.empty(6, dtype=np.float64)
    for out_index, body_id in enumerate(body_ids):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, int(body_id), spatial, 0)
        body_ang_vel[out_index] = spatial[:3]
        body_lin_vel[out_index] = spatial[3:]
    return body_pos, body_quat_xyzw, body_lin_vel, body_ang_vel


def _reference_encoder_inputs(
    model: mujoco.MjModel,
    qpos_frames: np.ndarray,
    body_ids: np.ndarray,
    qpos_adr: np.ndarray,
    dof_adr: np.ndarray,
    frame: int,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    body_pos, body_quat, body_vel, body_ang_vel = _reference_kinematics(model, qpos_frames, body_ids, frame, fps)
    ref_qpos = qpos_frames[frame % len(qpos_frames)]
    qvel = np.empty(model.nv, dtype=np.float64)
    next_frame = min(frame % len(qpos_frames) + 1, len(qpos_frames) - 1)
    prev_frame = max(frame % len(qpos_frames) - 1, 0)
    span = max(next_frame - prev_frame, 1) / fps
    mujoco.mj_differentiatePos(model, qvel, span, qpos_frames[prev_frame], qpos_frames[next_frame])
    dof_pos = ref_qpos[qpos_adr].astype(np.float32)
    dof_vel = qvel[dof_adr].astype(np.float32)
    projected_gravity = _quat_rotate_inverse(body_quat[0], np.array([0.0, 0.0, -1.0])).astype(np.float32)
    ref_ang_vel = (0.25 * _quat_rotate_inverse(body_quat[0], body_ang_vel[0])).astype(np.float32)
    state = np.concatenate((dof_pos, dof_vel, projected_gravity, ref_ang_vel)).astype(np.float32)
    privileged_state = _max_local_observation(body_pos, body_quat, body_vel, body_ang_vel)
    return state[None], privileged_state[None]


def _current_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_qpos_adr: np.ndarray,
    joint_dof_adr: np.ndarray,
    base_body_id: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    dof_pos = data.qpos[joint_qpos_adr].astype(np.float32).copy()
    dof_vel = data.qvel[joint_dof_adr].astype(np.float32).copy()
    base_quat = data.xquat[base_body_id, [1, 2, 3, 0]].copy()
    projected_gravity = _quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0])).astype(np.float32)
    spatial = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_body_id, spatial, 0)
    base_ang_vel = (0.25 * _quat_rotate_inverse(base_quat, spatial[:3])).astype(np.float32)
    state = np.concatenate((dof_pos, dof_vel, projected_gravity, base_ang_vel)).astype(np.float32)
    return state, {
        "base_ang_vel": base_ang_vel,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "projected_gravity": projected_gravity,
    }


def _config_values(
    config_path: Path, joint_names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    config = yaml.safe_load(config_path.read_text())
    if "training" in config:
        robot = config["training"]
        config_joint_names = list(config["control_joints"]["names"])
        effort_key = "effort_limit"
        actuator_joints = robot.get("actuator", {}).get("joints", {})
        armature = np.array([float(actuator_joints[name]["armature"]) for name in joint_names], dtype=np.float64)
        friction = np.array([float(actuator_joints[name]["friction"]) for name in joint_names], dtype=np.float64)
    else:
        robot = config["robot"]
        config_joint_names = list(robot["dof_names"])
        effort_key = "dof_effort_limit_list"
        armature = None
        friction = None
    control = robot["control"]
    kp = np.array([float(control["stiffness"][name]) for name in joint_names], dtype=np.float64)
    kd = np.array([float(control["damping"][name]) for name in joint_names], dtype=np.float64)
    effort = np.asarray(control[effort_key] if effort_key == "effort_limit" else robot[effort_key], dtype=np.float64)
    if config_joint_names != joint_names:
        index = {name: i for i, name in enumerate(config_joint_names)}
        effort = effort[[index[name] for name in joint_names]]
    scale = np.full(len(joint_names), float(control["action_scale"]), dtype=np.float64)
    if bool(control.get("action_rescale", True)):
        scale *= effort / kp
    return kp, kd, effort, scale, armature, friction


def _map_policy_action(
    raw_action: np.ndarray,
    *,
    action_mapping: str,
    default_joint_pos: np.ndarray,
    target_scale: np.ndarray,
    action_obs_scale: float,
    action_clip: float,
    mapping_bias: np.ndarray | None,
    mapping_range: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the training environment's actor-output to position-target transform."""
    raw_action = np.asarray(raw_action, dtype=np.float32)
    if action_mapping == "effort_kp":
        mapped_action = np.clip(
            raw_action * action_obs_scale, -action_clip, action_clip
        ).astype(np.float32)
    elif action_mapping == "soft_limit_bias":
        if mapping_bias is None or mapping_range is None:
            raise ValueError(
                "soft_limit_bias runtime metadata requires action_mapping_bias "
                "and action_mapping_range"
            )
        mapped_action = (
            mapping_bias + mapping_range * np.clip(raw_action, -1.0, 1.0)
        ).astype(np.float32)
    else:
        raise ValueError(f"Unsupported action_mapping in runtime metadata: {action_mapping!r}")

    target = default_joint_pos + mapped_action * target_scale
    return mapped_action, np.asarray(target, dtype=np.float64)


def _default_qpos(config_path: Path, joint_names: list[str], expected_nq: int) -> np.ndarray:
    config = yaml.safe_load(config_path.read_text())
    if "training" not in config:
        raise ValueError("--initial-pose default requires a robot config with a training section")
    training = config["training"]
    init_state = training["init_state"]
    root_pos = np.asarray(init_state["pos"], dtype=np.float64)
    root_quat_xyzw = np.asarray(init_state["rot"], dtype=np.float64)
    joint_values = init_state.get("default_joint_angles", {})
    missing = [name for name in joint_names if name not in joint_values]
    if missing:
        raise ValueError(f"Robot config default_joint_angles is missing joints: {missing}")
    joint_pos = np.asarray([joint_values[name] for name in joint_names], dtype=np.float64)
    qpos = np.concatenate((root_pos, root_quat_xyzw[[3, 0, 1, 2]], joint_pos))
    if qpos.shape != (expected_nq,) or not np.isfinite(qpos).all():
        raise ValueError(f"Invalid default qpos from {config_path}: shape={qpos.shape}, expected=({expected_nq},)")
    return qpos


def _ort_session(ort, path: Path, provider: str, intra_op_threads: int = 0):
    requested = "CUDAExecutionProvider" if provider == "cuda" else "CPUExecutionProvider"
    session_options = ort.SessionOptions()
    if intra_op_threads > 0:
        session_options.intra_op_num_threads = intra_op_threads
    session = ort.InferenceSession(str(path), sess_options=session_options, providers=[requested])
    if requested not in session.get_providers():
        raise RuntimeError(f"ONNX Runtime could not enable {requested}; active providers={session.get_providers()}")
    return session


def _inference_worker(connection, policy_path: str, provider: str, intra_op_threads: int) -> None:
    try:
        import onnxruntime as ort

        if provider == "cuda":
            ort.preload_dlls()
        policy = _ort_session(ort, Path(policy_path), provider, intra_op_threads)
        connection.send(("ready", policy.get_providers()[0]))
        while True:
            request = connection.recv()
            if request is None:
                break
            sequence, actor_obs = request
            infer_start = time.perf_counter()
            raw_action = policy.run(None, {"actor_obs": actor_obs})[0][0]
            infer_ms = (time.perf_counter() - infer_start) * 1000.0
            connection.send(("result", sequence, raw_action, infer_ms))
    except BaseException as exc:
        try:
            connection.send(("error", repr(exc)))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


def _measure_onnx(session, output_names, inputs: dict[str, np.ndarray], warmup: int, iterations: int) -> tuple[float, float]:
    for _ in range(warmup):
        session.run(output_names, inputs)
    start = time.perf_counter()
    for _ in range(iterations):
        session.run(output_names, inputs)
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / iterations, iterations / elapsed


def _benchmark_onnx(ort, exported: Path, warmup: int, iterations: int) -> None:
    available = set(ort.get_available_providers())
    providers = ["cpu"]
    if "CUDAExecutionProvider" in available:
        # Load CUDA/cuDNN libraries provided by this uv environment.
        ort.preload_dlls()
        providers.append("cuda")

    rng = np.random.default_rng(0)
    state = rng.standard_normal((1, 48), dtype=np.float32)
    privileged_state = rng.standard_normal((1, 358), dtype=np.float32)
    actor_obs = rng.standard_normal((1, 601), dtype=np.float32)

    print(f"[onnx-benchmark] warmup={warmup}, iterations={iterations}, batch=1")
    print("[onnx-benchmark] synchronous ORT run, including CPU<->GPU copies")
    print("[onnx-benchmark] current MuJoCo playback uses precomputed z, so actor timing is its actual ONNX cost")
    print(f"{'provider':<10} {'backward ms':>12} {'backward Hz':>12} {'actor ms':>12} {'actor Hz':>12} {'chain ms':>12} {'chain Hz':>12}")
    results: dict[str, tuple[float, float, float]] = {}
    for provider in providers:
        backward = _ort_session(ort, exported / "backward_encoder.onnx", provider)
        policy = _ort_session(ort, exported / "FBcprAuxModel.onnx", provider)
        backward_inputs = {"state": state, "privileged_state": privileged_state}
        backward_ms, backward_hz = _measure_onnx(backward, ["z"], backward_inputs, warmup, iterations)
        actor_ms, actor_hz = _measure_onnx(policy, ["action"], {"actor_obs": actor_obs}, warmup, iterations)

        for _ in range(warmup):
            z = backward.run(["z"], backward_inputs)[0]
            actor_obs[:, -z.shape[1] :] = z
            policy.run(["action"], {"actor_obs": actor_obs})
        start = time.perf_counter()
        for _ in range(iterations):
            z = backward.run(["z"], backward_inputs)[0]
            actor_obs[:, -z.shape[1] :] = z
            policy.run(["action"], {"actor_obs": actor_obs})
        elapsed = time.perf_counter() - start
        chain_ms = elapsed * 1000.0 / iterations
        chain_hz = iterations / elapsed
        results[provider] = (backward_ms, actor_ms, chain_ms)
        print(
            f"{provider:<10} {backward_ms:12.4f} {backward_hz:12.1f} "
            f"{actor_ms:12.4f} {actor_hz:12.1f} {chain_ms:12.4f} {chain_hz:12.1f}"
        )

    if "cuda" not in results:
        print(f"[onnx-benchmark] CUDA unavailable; available providers={ort.get_available_providers()}")
    else:
        cpu = results["cpu"]
        cuda = results["cuda"]
        print(
            "[onnx-benchmark] CUDA speedup over CPU: "
            f"backward={cpu[0] / cuda[0]:.2f}x, actor={cpu[1] / cuda[1]:.2f}x, chain={cpu[2] / cuda[2]:.2f}x"
        )


def run(args: argparse.Namespace) -> None:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise SystemExit("onnxruntime is required; install it with: uv add onnxruntime") from exc

    model_folder = args.model_folder.expanduser().resolve()
    exported = model_folder / "exported"
    meta_path = exported / "FBcprAuxModel.meta.json"
    meta = json.loads(meta_path.read_text())
    runtime_metadata_path = (
        args.runtime_metadata.expanduser().resolve()
        if args.runtime_metadata is not None
        else model_folder / "tracking_inference" / "metadata.json"
    )
    runtime_metadata = json.loads(runtime_metadata_path.read_text(encoding="utf-8"))
    if (
        runtime_metadata.get("schema_version") != 1
        or runtime_metadata.get("format") != "ufo_precomputed_onnx_latent"
    ):
        raise ValueError(f"Unsupported runtime metadata: {runtime_metadata_path}")
    runtime_dir = runtime_metadata_path.parent
    control_metadata = runtime_metadata["control"]
    latent_metadata = runtime_metadata["latent"]
    actor_path = (runtime_dir / runtime_metadata["actor"]["file"]).resolve()
    joint_names = [str(x) for x in control_metadata["joint_names"]]
    if joint_names != [str(x) for x in meta["control_joint_names"]]:
        raise ValueError("Runtime metadata and Actor export joint orders differ")

    if args.benchmark:
        _benchmark_onnx(ort, exported, args.benchmark_warmup, args.benchmark_iterations)
        return

    xml_path = _resolve_metadata_path(args.xml, str(meta["xml_path"]), "Robot XML")
    scene_path = args.scene.expanduser().resolve()
    config_path = _resolve_metadata_path(
        args.robot_config, str(meta["robot_config_path"]), "Robot config"
    )
    model = _load_scene(
        scene_path,
        robot_xml_path=xml_path,
        timestep=1.0 / args.sim_fps,
        show_reference=not args.no_reference,
    )
    reference_model = mujoco.MjModel.from_xml_path(str(xml_path))
    reference_model.opt.timestep = 1.0 / args.sim_fps
    data = mujoco.MjData(model)

    joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names])
    if np.any(joint_ids < 0):
        missing = [name for name, joint_id in zip(joint_names, joint_ids) if joint_id < 0]
        raise ValueError(f"MJCF is missing policy joints: {missing}")
    joint_qpos_adr = model.jnt_qposadr[joint_ids].astype(int)
    joint_dof_adr = model.jnt_dofadr[joint_ids].astype(int)
    hard_joint_limits = np.asarray(model.jnt_range[joint_ids], dtype=np.float64).copy()
    unlimited_joints = np.asarray(model.jnt_limited[joint_ids], dtype=bool) == 0
    hard_joint_limits[unlimited_joints] = np.nan

    actuator_by_joint = {int(model.actuator_trnid[i, 0]): i for i in range(model.nu)}
    actuator_ids = np.array([actuator_by_joint[int(joint_id)] for joint_id in joint_ids], dtype=int)
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    motion_path = args.motion.expanduser().resolve()
    qpos_frames, motion_fps = _load_motion(motion_path, joint_names, reference_model)
    if qpos_frames.shape[1] != reference_model.nq:
        raise ValueError(f"Motion qpos width {qpos_frames.shape[1]} does not match robot MJCF nq={reference_model.nq}")
    _config_kp, _config_kd, _config_effort, _config_scale, armature, friction = _config_values(
        config_path, joint_names
    )
    kp = np.asarray(control_metadata["p_gains"], dtype=np.float64).copy()
    kd = np.asarray(control_metadata["d_gains"], dtype=np.float64).copy()

    effort = np.asarray(control_metadata["effort_limits"], dtype=np.float64)
    tn_velocity_limits = None
    if args.torque_clip == "parkour-tn":
        missing_tn = [name for name in joint_names if name not in ROBAN_TN_VELOCITY_LIMITS]
        if missing_tn:
            raise ValueError(f"No wbc_parkour T-N velocity limit for joints: {missing_tn}")
        tn_velocity_limits = np.asarray(
            [ROBAN_TN_VELOCITY_LIMITS[name] for name in joint_names], dtype=np.float64
        )
    action_target_scale = np.asarray(control_metadata["target_scales"], dtype=np.float64)
    action_obs_scale = float(control_metadata["action_obs_scale"])
    action_clip = float(control_metadata["action_clip"])
    action_mapping = str(control_metadata.get("action_mapping", "effort_kp"))
    soft_joint_limits = None
    if "soft_joint_limits" in control_metadata:
        soft_joint_limits = np.asarray(control_metadata["soft_joint_limits"], dtype=np.float64)
        if soft_joint_limits.shape != (len(joint_names), 2) or not np.isfinite(soft_joint_limits).all():
            raise ValueError(
                "Invalid soft_joint_limits in runtime metadata: "
                f"shape={soft_joint_limits.shape}, expected=({len(joint_names)}, 2)"
            )
    default_joint_pos = np.asarray(
        control_metadata.get("default_joint_pos", np.zeros(len(joint_names))),
        dtype=np.float64,
    )
    action_mapping_bias = None
    action_mapping_range = None
    if action_mapping == "soft_limit_bias":
        try:
            action_mapping_bias = np.asarray(
                control_metadata["action_mapping_bias"], dtype=np.float32
            )
            action_mapping_range = np.asarray(
                control_metadata["action_mapping_range"], dtype=np.float32
            )
        except KeyError as exc:
            raise ValueError(
                "soft_limit_bias runtime metadata is missing its affine transform; "
                "regenerate it with humanoidverse.generate_onnx_latent"
            ) from exc
    expected_control_shape = (len(joint_names),)
    for label, values in (
        ("default_joint_pos", default_joint_pos),
        ("target_scales", action_target_scale),
        ("action_mapping_bias", action_mapping_bias),
        ("action_mapping_range", action_mapping_range),
    ):
        if values is None:
            continue
        if values.shape != expected_control_shape or not np.isfinite(values).all():
            raise ValueError(
                f"Invalid runtime control field {label}: "
                f"shape={values.shape}, expected={expected_control_shape}"
            )
    if action_mapping not in {"effort_kp", "soft_limit_bias"}:
        raise ValueError(f"Unsupported action_mapping in runtime metadata: {action_mapping!r}")
    if armature is not None:
        model.dof_armature[joint_dof_adr] = armature
    if friction is not None:
        model.dof_frictionloss[joint_dof_adr] = friction

    diagnostics_output: Path | None = None
    diagnostics_records: dict[str, list[np.ndarray | float | int]] | None = None
    diagnostic_body_names: list[str] = []
    diagnostic_policy_body_ids = np.empty(0, dtype=int)
    diagnostic_reference_body_ids = np.empty(0, dtype=int)
    diagnostic_reference_data: mujoco.MjData | None = None
    reference_joint_velocities: np.ndarray | None = None
    if not args.no_reference and not args.no_diagnostics:
        diagnostics_root = (
            args.diagnostics_root.expanduser().resolve()
            if args.diagnostics_root is not None
            else model_folder / "onnx_diagnostics"
        )
        diagnostics_output = _new_diagnostics_dir(diagnostics_root, motion_path, model_folder)
        diagnostic_body_names = _xml_body_names(xml_path)
        if "base_link" not in diagnostic_body_names:
            raise ValueError("Robot XML has no base_link body for local-link diagnostics")
        diagnostic_body_names.remove("base_link")
        diagnostic_body_names.insert(0, "base_link")
        diagnostic_policy_body_ids = np.asarray(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in diagnostic_body_names],
            dtype=int,
        )
        diagnostic_reference_body_ids = np.asarray(
            [mujoco.mj_name2id(reference_model, mujoco.mjtObj.mjOBJ_BODY, name) for name in diagnostic_body_names],
            dtype=int,
        )
        if np.any(diagnostic_policy_body_ids < 0) or np.any(diagnostic_reference_body_ids < 0):
            raise ValueError("Could not resolve all robot bodies for local-link diagnostics")
        diagnostic_reference_data = mujoco.MjData(reference_model)
        reference_joint_ids = np.asarray(
            [mujoco.mj_name2id(reference_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names],
            dtype=int,
        )
        reference_joint_velocities = _reference_joint_velocities(
            reference_model,
            qpos_frames,
            reference_model.jnt_dofadr[reference_joint_ids].astype(int),
            motion_fps,
        )
        diagnostics_records = {
            "time_s": [],
            "motion_frame": [],
            "joint_pos": [],
            "reference_joint_pos": [],
            "joint_pos_error": [],
            "joint_vel": [],
            "reference_joint_vel": [],
            "torque": [],
            "torque_requested": [],
            "torque_limit_min": [],
            "torque_limit_max": [],
            "torque_was_clipped": [],
            "local_link_pos_error_xyz": [],
            "local_link_pos_error": [],
        }

    # The scene supplies ground contact, while the robot YAML supplies passive
    # actuator dynamics.  Do not overwrite them with values from an unrelated
    # deployment adapter.
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise ValueError("MuJoCo scene has no floor geom")

    reference_free_qpos_adr: int | None = None
    reference_joint_qpos_adr: np.ndarray | None = None
    if not args.no_reference:
        free_joint_ids = np.flatnonzero(reference_model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free_joint_ids) != 1:
            raise ValueError(f"Expected one free joint in robot MJCF, found {len(free_joint_ids)}")
        free_joint_name = _name(reference_model, mujoco.mjtObj.mjOBJ_JOINT, int(free_joint_ids[0]))
        reference_free_joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"reference_{free_joint_name}"
        )
        if reference_free_joint_id < 0:
            raise ValueError(f"Duplicated reference robot has no reference_{free_joint_name} free joint")
        reference_free_qpos_adr = int(model.jnt_qposadr[reference_free_joint_id])
        reference_joint_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"reference_{name}") for name in joint_names]
        )
        reference_joint_qpos_adr = model.jnt_qposadr[reference_joint_ids].astype(int)

    policy = None
    inference_connection = None
    inference_process = None
    if args.inference_mode == "inline":
        if args.onnx_provider == "cuda":
            ort.preload_dlls()
        policy = _ort_session(ort, actor_path, args.onnx_provider, args.onnx_threads)
        active_provider = policy.get_providers()[0]
    else:
        context = mp.get_context("spawn")
        inference_connection, child_connection = context.Pipe()
        inference_process = context.Process(
            target=_inference_worker,
            args=(child_connection, str(actor_path), args.onnx_provider, args.onnx_threads),
            name="ufo-onnx-inference",
        )
        inference_process.start()
        child_connection.close()
        ready = inference_connection.recv()
        if ready[0] != "ready":
            raise RuntimeError(f"Inference worker failed during startup: {ready}")
        active_provider = ready[1]
    latent_path = (
        args.latent
        if args.latent is not None
        else runtime_dir / latent_metadata["file"]
    ).expanduser().resolve()
    if not latent_path.is_file():
        raise FileNotFoundError(f"Precomputed latent file does not exist: {latent_path}")
    latents = np.load(latent_path, allow_pickle=False)
    if latents.dtype != np.float32:
        raise ValueError(f"Expected float32 latent, got {latents.dtype} from {latent_path}")
    expected_z_dim = int(latent_metadata["shape"][1])
    if latents.ndim != 2 or latents.shape[1] != expected_z_dim:
        raise ValueError(
            f"Expected latent shape [frames, {expected_z_dim}], got {latents.shape} "
            f"from {latent_path}"
        )
    if len(latents) == 0 or not np.isfinite(latents).all():
        raise ValueError(f"Precomputed latent is empty or non-finite: {latent_path}")
    playback_frame_count = min(len(qpos_frames), len(latents))

    initial_motion_frame = args.initial_motion_frame % playback_frame_count
    initial_qpos = (
        _default_qpos(config_path, joint_names, reference_model.nq)
        if args.initial_pose == "default"
        else qpos_frames[initial_motion_frame]
    )
    data.qpos[: reference_model.nq] = initial_qpos
    if reference_free_qpos_adr is not None and reference_joint_qpos_adr is not None:
        source_free_joint_ids = np.flatnonzero(
            reference_model.jnt_type == mujoco.mjtJoint.mjJNT_FREE
        )
        source_free_qpos_adr = int(
            reference_model.jnt_qposadr[int(source_free_joint_ids[0])]
        )
        source_joint_ids = np.asarray(
            [
                mujoco.mj_name2id(reference_model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in joint_names
            ],
            dtype=int,
        )
        source_joint_qpos_adr = reference_model.jnt_qposadr[source_joint_ids].astype(int)
        data.qpos[reference_free_qpos_adr : reference_free_qpos_adr + 7] = qpos_frames[
            initial_motion_frame, source_free_qpos_adr : source_free_qpos_adr + 7
        ]
        data.qpos[reference_free_qpos_adr + 1] += args.reference_offset_y
        data.qpos[reference_joint_qpos_adr] = qpos_frames[
            initial_motion_frame, source_joint_qpos_adr
        ]
    mujoco.mj_forward(model, data)
    control_dt = args.decimation / args.sim_fps
    history = {
        "actions": np.zeros((4, len(joint_names)), dtype=np.float32),
        "base_ang_vel": np.zeros((4, 3), dtype=np.float32),
        "dof_pos": np.zeros((4, len(joint_names)), dtype=np.float32),
        "dof_vel": np.zeros((4, len(joint_names)), dtype=np.float32),
        "projected_gravity": np.zeros((4, 3), dtype=np.float32),
    }
    action = np.zeros(len(joint_names), dtype=np.float32)
    max_steps = args.max_steps if args.max_steps > 0 else math.inf

    print(f"[onnx-mujoco] policy={actor_path}")
    print(f"[onnx-mujoco] runtime_metadata={runtime_metadata_path}")
    print(
        f"[onnx-mujoco] inference_mode={args.inference_mode}, onnx_provider={active_provider}, "
        f"onnx_threads={args.onnx_threads or 'default'}"
    )
    print(f"[onnx-mujoco] latent={latent_path} ({len(latents)} frames x {latents.shape[1]})")
    print(f"[onnx-mujoco] xml={xml_path}")
    print(f"[onnx-mujoco] scene={scene_path}")
    print(f"[onnx-mujoco] motion={args.motion.resolve()} ({len(qpos_frames)} frames @ {motion_fps:g} Hz)")
    print(
        f"[onnx-mujoco] initial_pose={args.initial_pose}, "
        f"initial_motion_frame={initial_motion_frame}"
    )
    print(f"[onnx-mujoco] action_mapping={action_mapping}")
    print(f"[onnx-mujoco] torque_clip={args.torque_clip}")
    arm_names = [name for name in joint_names if name.startswith("zarm_")]
    arm_pd = ", ".join(
        f"{name}:Kp={kp[joint_names.index(name)]:.4f}/Kd={kd[joint_names.index(name)]:.4f}"
        for name in arm_names
    )
    print(f"[onnx-mujoco] arm_pd={arm_pd}")
    if len(qpos_frames) != len(latents):
        print(
            f"[onnx-mujoco] playback uses {playback_frame_count} frames "
            f"(motion={len(qpos_frames)}, latent={len(latents)})"
        )
    print(f"[onnx-mujoco] sim={args.sim_fps:g} Hz, policy={1.0 / control_dt:g} Hz, headless={args.headless}")
    if diagnostics_output is not None:
        print(f"[onnx-mujoco] diagnostics={diagnostics_output}")

    video_renderer = None
    video_writer = None
    video_camera = None
    if args.video is not None:
        import imageio.v2 as imageio

        video_path = args.video.expanduser().resolve()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.video_width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), args.video_height)
        video_renderer = mujoco.Renderer(model, height=args.video_height, width=args.video_width)
        video_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(video_camera)
        video_camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        video_camera.trackbodyid = base_body_id
        video_camera.distance = args.video_camera_distance
        video_camera.azimuth = args.video_camera_azimuth
        video_camera.elevation = args.video_camera_elevation
        video_writer = imageio.get_writer(
            str(video_path), fps=args.video_fps, codec="libx264", quality=8, macro_block_size=None
        )
        print(
            f"[onnx-mujoco] video={video_path} "
            f"({args.video_width}x{args.video_height} @ {args.video_fps:g} fps)"
        )

    def simulate(viewer=None) -> None:
        nonlocal action
        step = 0
        diagnostics_saved = False
        start = time.perf_counter()
        inference_samples: list[float] = []
        roundtrip_samples: list[float] = []
        total_inference_samples = 0
        # r_path = "/home/thl/wt_wbc/UFO/runs/新数据addlelay_onnx_153m_20260807/reward_inference/"

        # latents1 = np.load(r_path + "move-ego-0-0.5.npy", allow_pickle=False)
        # latents2 = np.load(r_path + "move-ego-90-0.5.npy", allow_pickle=False)
        # import ipdb;ipdb.set_trace()
        while step < max_steps and (viewer is None or viewer.is_running()):
            tick = time.perf_counter()
            motion_frame = initial_motion_frame + int(round(step * control_dt * motion_fps))
            if not args.loop and motion_frame >= playback_frame_count:
                break
            motion_frame %= playback_frame_count
            # motion_frame = 0 #TODO
            state, current = _current_observation(model, data, joint_qpos_adr, joint_dof_adr, base_body_id)
            history_actor = np.concatenate([history[key].reshape(-1) for key in sorted(history)]).astype(np.float32)
            z = latents[motion_frame]
            # z = (latents1[0]+latents2[0])/2
            actor_obs = np.concatenate((state, action, history_actor, z)).astype(np.float32)[None]
            roundtrip_start = time.perf_counter()
            if args.inference_mode == "inline":
                infer_start = time.perf_counter()
                raw_action = policy.run(None, {"actor_obs": actor_obs})[0][0]
                infer_ms = (time.perf_counter() - infer_start) * 1000.0
            else:
                inference_connection.send((step, actor_obs))
                response = inference_connection.recv()
                if response[0] == "error":
                    raise RuntimeError(f"Inference worker failed: {response[1]}")
                if response[0] != "result" or response[1] != step:
                    raise RuntimeError(f"Unexpected inference response: {response[:2]}")
                raw_action = response[2]
                infer_ms = float(response[3])
            roundtrip_ms = (time.perf_counter() - roundtrip_start) * 1000.0
            inference_samples.append(infer_ms)
            roundtrip_samples.append(roundtrip_ms)
            total_inference_samples += 1

            if len(inference_samples) >= args.latency_log_every:
                print(
                    "[onnx-mujoco] Actor inference latency: "
                    f"last={inference_samples[-1]:.3f} ms, avg={np.mean(inference_samples):.3f} ms, "
                    f"max={np.max(inference_samples):.3f} ms, samples={len(inference_samples)}, "
                    f"total_samples={total_inference_samples}"
                )
                if args.inference_mode == "process":
                    print(
                        "[onnx-mujoco] Actor IPC roundtrip latency: "
                        f"last={roundtrip_samples[-1]:.3f} ms, avg={np.mean(roundtrip_samples):.3f} ms, "
                        f"max={np.max(roundtrip_samples):.3f} ms, samples={len(roundtrip_samples)}, "
                        f"total_samples={total_inference_samples}"
                    )
                inference_samples.clear()
                roundtrip_samples.clear()

            # Match the training environment's observation timing: history is
            # queried before the current observation is inserted, so its newest
            # action is one policy step older than ``last_action``.  Insert the
            # action used in actor_obs before replacing it with the new output.
            for key, value in (("actions", action), *current.items()):
                history[key][1:] = history[key][:-1]
                history[key][0] = value

            action, target = _map_policy_action(
                raw_action,
                action_mapping=action_mapping,
                default_joint_pos=default_joint_pos,
                target_scale=action_target_scale,
                action_obs_scale=action_obs_scale,
                action_clip=action_clip,
                mapping_bias=action_mapping_bias,
                mapping_range=action_mapping_range,
            )
            applied_torques = []
            requested_torques = []
            torque_limit_mins = []
            torque_limit_maxs = []
            torque_was_clipped = []
            for _ in range(args.decimation):
                torque = kp * (target - data.qpos[joint_qpos_adr]) - kd * data.qvel[joint_dof_adr]
                limit_min, limit_max = _torque_limits(
                    data.qvel[joint_dof_adr], effort, tn_velocity_limits
                )
                clipped_torque = np.clip(torque, limit_min, limit_max)
                data.ctrl[actuator_ids] = clipped_torque
                applied_torques.append(clipped_torque.copy())
                requested_torques.append(torque.copy())
                torque_limit_mins.append(limit_min.copy())
                torque_limit_maxs.append(limit_max.copy())
                torque_was_clipped.append(np.abs(clipped_torque - torque) > 1e-9)
                mujoco.mj_step(model, data)

            if reference_free_qpos_adr is not None and reference_joint_qpos_adr is not None:
                data.qpos[reference_free_qpos_adr : reference_free_qpos_adr + 7] = qpos_frames[
                    motion_frame, source_free_qpos_adr : source_free_qpos_adr + 7
                ]
                data.qpos[reference_free_qpos_adr + 1] += args.reference_offset_y
                data.qpos[reference_joint_qpos_adr] = qpos_frames[
                    motion_frame, source_joint_qpos_adr
                ]
                mujoco.mj_forward(model, data)

            if diagnostics_records is not None and not diagnostics_saved:
                assert diagnostic_reference_data is not None
                assert reference_joint_velocities is not None
                reference_qpos = qpos_frames[motion_frame]
                diagnostic_reference_data.qpos[:] = reference_qpos
                mujoco.mj_forward(reference_model, diagnostic_reference_data)
                policy_joint_pos = data.qpos[joint_qpos_adr].copy()
                reference_joint_pos = reference_qpos[source_joint_qpos_adr].copy()
                joint_pos_error = (policy_joint_pos - reference_joint_pos + np.pi) % (2.0 * np.pi) - np.pi
                policy_local = _local_link_positions(
                    data.xpos[diagnostic_policy_body_ids].copy(),
                    data.xquat[diagnostic_policy_body_ids][:, [1, 2, 3, 0]].copy(),
                )
                reference_local = _local_link_positions(
                    diagnostic_reference_data.xpos[diagnostic_reference_body_ids].copy(),
                    diagnostic_reference_data.xquat[diagnostic_reference_body_ids][:, [1, 2, 3, 0]].copy(),
                )
                diagnostics_records["time_s"].append((step + 1) * control_dt)
                diagnostics_records["motion_frame"].append(motion_frame)
                diagnostics_records["joint_pos"].append(policy_joint_pos)
                diagnostics_records["reference_joint_pos"].append(reference_joint_pos)
                diagnostics_records["joint_pos_error"].append(joint_pos_error)
                diagnostics_records["joint_vel"].append(data.qvel[joint_dof_adr].copy())
                diagnostics_records["reference_joint_vel"].append(reference_joint_velocities[motion_frame].copy())
                diagnostics_records["torque"].append(np.mean(applied_torques, axis=0))
                diagnostics_records["torque_requested"].append(np.mean(requested_torques, axis=0))
                diagnostics_records["torque_limit_min"].append(np.mean(torque_limit_mins, axis=0))
                diagnostics_records["torque_limit_max"].append(np.mean(torque_limit_maxs, axis=0))
                diagnostics_records["torque_was_clipped"].append(np.any(torque_was_clipped, axis=0))
                local_link_error_xyz = policy_local[1:] - reference_local[1:]
                diagnostics_records["local_link_pos_error_xyz"].append(local_link_error_xyz)
                diagnostics_records["local_link_pos_error"].append(
                    np.linalg.norm(local_link_error_xyz, axis=1)
                )
                next_unwrapped_motion_frame = initial_motion_frame + int(
                    round((step + 1) * control_dt * motion_fps)
                )
                if next_unwrapped_motion_frame >= playback_frame_count:
                    assert diagnostics_output is not None
                    _write_diagnostics(
                        diagnostics_output,
                        diagnostics_records,
                        joint_names,
                        diagnostic_body_names[1:],
                        effort,
                        hard_joint_limits,
                        soft_joint_limits,
                        np.deg2rad(args.joint_limit_highlight_margin_deg),
                        motion_path,
                        model_folder,
                        args.torque_clip,
                    )
                    diagnostics_saved = True
                    print(
                        f"[onnx-mujoco] first playback complete; diagnostics saved once: "
                        f"{diagnostics_output}"
                    )

            if not np.isfinite(data.qpos).all():
                raise RuntimeError(f"Simulation became non-finite at policy step {step}")
            if viewer is not None:
                viewer.sync()
                remaining = control_dt - (time.perf_counter() - tick)
                if remaining > 0:
                    time.sleep(remaining)
            if video_renderer is not None and video_writer is not None:
                video_renderer.update_scene(data, camera=video_camera)
                video_writer.append_data(video_renderer.render())
            step += 1
            if args.log_every > 0 and step % args.log_every == 0:
                speed = step / max(time.perf_counter() - start, 1e-6)
                print(
                    f"[onnx-mujoco] step={step} motion_frame={motion_frame} base_z={data.qpos[2]:.3f} "
                    f"action=[{action.min():.2f}, {action.max():.2f}] "
                    f"speed={speed:.1f} policy_steps/s"
                )

    try:
        if args.headless:
            simulate()
        else:
            from mujoco import viewer as mj_viewer

            with mj_viewer.launch_passive(model, data) as viewer:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = base_body_id
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 135.0
                viewer.cam.elevation = -18.0
                viewer.sync()
                if args.initial_hold_seconds > 0:
                    time.sleep(args.initial_hold_seconds)
                simulate(viewer)
    finally:
        if video_writer is not None:
            video_writer.close()
        if video_renderer is not None:
            video_renderer.close()
        if inference_connection is not None:
            try:
                inference_connection.send(None)
            except (BrokenPipeError, EOFError):
                pass
            inference_connection.close()
        if inference_process is not None:
            inference_process.join(timeout=5.0)
            if inference_process.is_alive():
                inference_process.terminate()
                inference_process.join(timeout=1.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_RUN, help="Run folder containing exported/*.onnx")
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION, help="RobotState NPZ motion")
    parser.add_argument(
        "--runtime-metadata",
        type=Path,
        default=None,
        help="Override tracking runtime metadata.json (useful when comparing latent variants).",
    )
    parser.add_argument(
        "--latent",
        type=Path,
        default=None,
        help="Precomputed .npy latent; defaults to <model-folder>/tracking_inference/zs_0.npy",
    )
    parser.add_argument("--xml", type=Path, default=None, help="Override MJCF path from export metadata")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="MuJoCo scene XML containing the robot and world")
    parser.add_argument("--robot-config", type=Path, default=None, help="Override robot YAML path from export metadata")
    parser.add_argument("--sim-fps", type=float, default=200.0)
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument(
        "--torque-clip",
        choices=("fixed", "parkour-tn"),
        default="fixed",
        help="Fixed effort clip, or the velocity-dependent Roban T-N envelope used by wbc_parkour.",
    )
    parser.add_argument("--onnx-provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--onnx-threads", type=int, default=0, help="ORT intra-op threads; 0 uses the ORT default")
    parser.add_argument("--inference-mode", choices=("inline", "process"), default="inline")
    parser.add_argument("--latency-log-every", type=int, default=100)
    parser.add_argument("--benchmark", action="store_true", help="Benchmark CPU and CUDA ONNX inference, then exit")
    parser.add_argument("--benchmark-warmup", type=int, default=50)
    parser.add_argument("--benchmark-iterations", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=0, help="Policy steps; 0 runs until the window closes")
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--video", type=Path, default=None, help="Write an offscreen MP4 recording.")
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-fps", type=float, default=50.0)
    parser.add_argument("--video-camera-distance", type=float, default=4.0)
    parser.add_argument("--video-camera-azimuth", type=float, default=135.0)
    parser.add_argument("--video-camera-elevation", type=float, default=-18.0)
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-reference", action="store_true", help="Hide the translucent reference robot")
    parser.add_argument("--reference-offset-y", type=float, default=1.2, help="Side-by-side reference robot offset")
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        default=None,
        help=(
            "Root for per-run diagnostic folders. Defaults to <model-folder>/onnx_diagnostics. "
            "Each run creates <motion-name>__<run-name>, with a timestamp suffix on collisions."
        ),
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Disable automatic reference-tracking curves and timeseries output.",
    )
    parser.add_argument(
        "--joint-limit-highlight-margin-deg",
        type=float,
        default=1.0,
        help=(
            "Highlight policy positions within this many degrees of a hard limit; "
            "soft-limit markers use the same margin."
        ),
    )
    parser.add_argument(
        "--initial-pose",
        choices=("motion", "default"),
        default="motion",
        help="Initialize the controlled robot from the motion or robot-config default pose.",
    )
    parser.add_argument(
        "--initial-motion-frame",
        type=int,
        default=0,
        help="Motion/latent frame used at startup; playback continues from this frame.",
    )
    parser.add_argument(
        "--initial-hold-seconds",
        type=float,
        default=1.0,
        help="Show the untouched initial pose for this long before the first policy step.",
    )
    args = parser.parse_args()
    if (
        args.sim_fps <= 0
        or args.decimation <= 0
        or args.onnx_threads < 0
        or args.latency_log_every <= 0
        or args.benchmark_warmup < 0
        or args.benchmark_iterations <= 0
        or args.video_width <= 0
        or args.video_height <= 0
        or args.video_fps <= 0
        or args.video_camera_distance <= 0
        or args.joint_limit_highlight_margin_deg < 0
        or args.initial_motion_frame < 0
        or args.initial_hold_seconds < 0
    ):
        parser.error(
            "fps, dimensions, camera distance, decimation and benchmark iterations must be positive; "
            "warmup, initial frame/hold, and joint-limit highlight margin "
            "must be valid and non-negative"
        )
    return args


if __name__ == "__main__":
    run(_parse_args())
