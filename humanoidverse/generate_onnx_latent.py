"""Generate a precomputed tracking latent using an exported backward ONNX."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import yaml

from humanoidverse.mjlab_inference_utils import load_mjlab_env_cfg
from humanoidverse.utils.helpers import get_backward_observation
from humanoidverse.utils.motion_data import (
    prepare_manifest_dataset_path,
    prepare_manifest_robot_config_path,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_CONFIG = ROOT / "configs/robots/roban_s22.yaml"


def _train_aligned_latents(frame_z: np.ndarray, seq_length: int) -> np.ndarray:
    """Match FB expert tracking: future-window mean followed by z projection."""
    if frame_z.ndim != 2 or len(frame_z) == 0:
        raise ValueError(f"Expected non-empty [frames, z_dim] latent, got {frame_z.shape}")
    if seq_length <= 0:
        raise ValueError(f"seq_length must be positive, got {seq_length}")

    aligned = np.empty_like(frame_z, dtype=np.float32)
    for step in range(len(frame_z)):
        end = min(step + seq_length, len(frame_z))
        aligned[step] = frame_z[step:end].mean(axis=0, dtype=np.float32)

    # FBModel.project_z(): sqrt(z_dim) * normalize(z). The backward ONNX
    # already projects each frame; applying this after the window mean matches
    # the training-time expert/tracking aggregation order for this export.
    norms = np.linalg.norm(aligned, axis=1, keepdims=True)
    aligned *= np.sqrt(aligned.shape[1], dtype=np.float32) / np.maximum(norms, 1.0e-12)
    return np.ascontiguousarray(aligned, dtype=np.float32)


def _ordered(values, joint_names: list[str], label: str) -> list[float]:
    if isinstance(values, dict):
        missing = [name for name in joint_names if name not in values]
        if missing:
            raise ValueError(f"{label} is missing joints: {missing}")
        return [float(values[name]) for name in joint_names]
    result = [float(value) for value in values]
    if len(result) != len(joint_names):
        raise ValueError(f"{label} has {len(result)} values, expected {len(joint_names)}")
    return result


def _relative_path(path: Path, directory: Path) -> str:
    return os.path.relpath(path.resolve(), directory.resolve())


def _prepare_motion_input(
    args: argparse.Namespace,
    model_folder: Path,
) -> tuple[Path, Path, str, str, Path | None]:
    """Build a one-motion cache for --motion, or use a normal manifest."""
    if args.motion is None:
        data_path = Path(
            prepare_manifest_dataset_path(
                args.data_manifest,
                args.dataset,
                split="inference",
                rebuild_cache=bool(args.rebuild_motion_cache),
            )
        )
        robot_config = Path(prepare_manifest_robot_config_path(args.data_manifest))
        return data_path, robot_config, str(args.data_manifest.expanduser().resolve()), str(args.dataset), None

    motion_path = args.motion.expanduser().resolve()
    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion NPZ does not exist: {motion_path}")
    if motion_path.suffix.lower() != ".npz":
        raise ValueError(f"--motion must point to a .npz file: {motion_path}")
    robot_config = args.robot_config.expanduser().resolve()
    if not robot_config.is_file():
        raise FileNotFoundError(f"Robot config does not exist: {robot_config}")

    # Keep direct-motion caches distinct without coupling runtime inference to
    # file-content hashes. Size and mtime also invalidate normal replacements.
    motion_stat = motion_path.stat()
    safe_stem = "".join(char if char.isalnum() else "_" for char in motion_path.stem)
    dataset_name = f"single_motion_{safe_stem}_{motion_stat.st_size}_{motion_stat.st_mtime_ns}"
    cache_root = model_folder / "tracking_inference" / "motion_cache"
    manifest_data = {
        "robot_config": str(robot_config),
        "datasets": [
            {
                "name": dataset_name,
                "format": "robot_state_npz",
                "weight": 1.0,
                "train_path": str(motion_path),
            }
        ],
    }
    with tempfile.TemporaryDirectory(prefix="ufo_onnx_motion_") as temporary_dir:
        manifest_path = Path(temporary_dir) / "single_motion.yaml"
        manifest_path.write_text(
            yaml.safe_dump(manifest_data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        data_path = Path(
            prepare_manifest_dataset_path(
                manifest_path,
                dataset_name,
                split="inference",
                rebuild_cache=bool(args.rebuild_motion_cache),
                cache_root=cache_root,
            )
        )
    return data_path, robot_config, "direct_npz", dataset_name, motion_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--motion", type=Path, default=None, help="Generate z directly from one RobotState NPZ.")
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--motion-id", type=int, default=0)
    parser.add_argument(
        "--reference-index",
        type=int,
        default=None,
        help="Optional index used by an external reference-robot motion list",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument(
        "--latent-mode",
        choices=("single_frame", "train_aligned"),
        default="single_frame",
        help="single_frame preserves the old per-frame z; train_aligned uses the training seq_length future window.",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=None,
        help="Override training model.seq_length for train_aligned mode.",
    )
    parser.add_argument("--rebuild-motion-cache", action="store_true")
    args = parser.parse_args()
    if args.motion is None and (args.data_manifest is None or args.dataset is None):
        parser.error("provide --motion, or provide both --data-manifest and --dataset")
    if args.motion is not None and (args.data_manifest is not None or args.dataset is not None):
        parser.error("--motion cannot be combined with --data-manifest/--dataset")

    import onnxruntime as ort

    model_folder = args.model_folder.expanduser().resolve()
    backward_path = model_folder / "exported" / "backward_encoder.onnx"
    if not backward_path.is_file():
        raise FileNotFoundError(f"Missing backward ONNX: {backward_path}")

    data_path, robot_config, source_manifest, source_dataset, direct_motion_path = _prepare_motion_input(
        args,
        model_folder,
    )
    env_cfg, use_root_height_obs = load_mjlab_env_cfg(
        model_folder,
        data_path=data_path,
        robot_config=robot_config,
        device=args.device,
        headless=True,
        disable_dr=True,
        disable_obs_noise=True,
        max_episode_length_s=10000.0,
    )
    wrapped_env, _ = env_cfg.build(num_envs=1)
    env = wrapped_env._env
    env._motion_lib.load_all_motions()

    backward_obs, reference = get_backward_observation(
        env,
        args.motion_id,
        use_root_height_obs=use_root_height_obs,
    )
    candidate_inputs = {
        key: value[1:].detach().cpu().numpy().astype(np.float32)
        for key, value in backward_obs.items()
    }

    providers = ["CPUExecutionProvider"]
    if str(args.device).startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
        ort.preload_dlls()
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(str(backward_path), providers=providers)
    input_names = {item.name for item in session.get_inputs()}
    unknown = input_names.difference(candidate_inputs)
    if unknown:
        raise RuntimeError(f"Backward ONNX has unsupported inputs: {sorted(unknown)}")
    frame_z = session.run(["z"], {name: candidate_inputs[name] for name in input_names})[0]
    if frame_z.ndim != 2 or not np.isfinite(frame_z).all():
        raise ValueError(f"Invalid latent generated from {backward_path}: shape={frame_z.shape}")

    run_config_path = model_folder / "config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"Missing runtime input: {run_config_path}")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    training_seq_length = int(run_config["agent"]["model"]["seq_length"])
    seq_length = training_seq_length if args.seq_length is None else int(args.seq_length)
    if seq_length <= 0:
        parser.error("--seq-length must be positive")
    z = (
        _train_aligned_latents(frame_z, seq_length)
        if args.latent_mode == "train_aligned"
        else np.ascontiguousarray(frame_z, dtype=np.float32)
    )

    output = (
        model_folder / "tracking_inference" / f"zs_{args.motion_id}.npy"
        if args.output is None
        else args.output.expanduser().resolve()
    )
    if output.suffix != ".npy":
        raise ValueError(f"Latent output must use the .npy extension: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    z = np.ascontiguousarray(z, dtype=np.float32)
    np.save(output, z, allow_pickle=False)

    metadata_output = (
        output.parent / "metadata.json"
        if args.metadata_output is None
        else args.metadata_output.expanduser().resolve()
    )
    export_meta_path = model_folder / "exported" / "FBcprAuxModel.meta.json"
    actor_path = model_folder / "exported" / "FBcprAuxModel.onnx"
    for required in (export_meta_path, actor_path, run_config_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing runtime input: {required}")
    export_meta = json.loads(export_meta_path.read_text(encoding="utf-8"))
    robot_training = run_config["env"]["robot_training"]
    joint_names = [str(name) for name in export_meta["control_joint_names"]]
    if len(joint_names) != 21:
        raise ValueError(f"Expected 21 control joints, got {len(joint_names)}")
    env_joint_names = [str(name) for name in env.dof_names]
    if env_joint_names != joint_names:
        raise ValueError(
            "Runtime environment and Actor export joint orders differ: "
            f"env={env_joint_names}, actor={joint_names}"
        )
    kp = np.asarray(_ordered(robot_training["stiffness"], joint_names, "stiffness"))
    kd = np.asarray(_ordered(robot_training["damping"], joint_names, "damping"))
    effort = np.asarray(_ordered(robot_training["effort_limits"], joint_names, "effort_limits"))

    # Read the action transform from the fully composed training environment.
    # In particular, soft_limit_bias depends on the resolved reward soft-limit
    # factor and joint limits, so recomputing it from the deployment YAML can
    # silently diverge from training.
    default_joint_pos = (
        env.default_dof_pos[0].detach().cpu().numpy().astype(np.float64)
    )
    target_scales = (
        env.action_target_scale[0].detach().cpu().numpy().astype(np.float64)
    )
    action_mapping = str(env.action_mapping)
    action_mapping_bias = (
        env.action_mapping_bias.detach().cpu().numpy().astype(np.float64)
    )
    action_mapping_range = (
        env.action_mapping_range.detach().cpu().numpy().astype(np.float64)
    )
    action_mapping_lower = (
        env.action_mapping_lower.detach().cpu().numpy().astype(np.float64)
    )
    action_mapping_upper = (
        env.action_mapping_upper.detach().cpu().numpy().astype(np.float64)
    )
    soft_joint_limits = env.dof_pos_limits.detach().cpu().numpy().astype(np.float64)
    expected_vector_shape = (len(joint_names),)
    for label, values in (
        ("default_joint_pos", default_joint_pos),
        ("target_scales", target_scales),
        ("action_mapping_bias", action_mapping_bias),
        ("action_mapping_range", action_mapping_range),
        ("action_mapping_lower", action_mapping_lower),
        ("action_mapping_upper", action_mapping_upper),
    ):
        if values.shape != expected_vector_shape or not np.isfinite(values).all():
            raise ValueError(
                f"Invalid {label} from runtime environment: "
                f"shape={values.shape}, expected={expected_vector_shape}"
            )
    if soft_joint_limits.shape != (len(joint_names), 2) or not np.isfinite(soft_joint_limits).all():
        raise ValueError(
            "Invalid soft joint limits from runtime environment: "
            f"shape={soft_joint_limits.shape}"
        )
    reference_state = backward_obs["state"].detach().cpu().numpy().astype(np.float32)
    if reference_state.ndim != 2 or reference_state.shape[1] < len(joint_names):
        raise ValueError(f"Invalid reference state shape: {reference_state.shape}")
    root_pos = (
        reference["ref_body_pos"][0, 0].detach().cpu().numpy().astype(np.float32)
    )
    root_quat_xyzw = (
        reference["ref_body_rots"][0, 0].detach().cpu().numpy().astype(np.float32)
    )
    if (
        root_pos.shape != (3,)
        or root_quat_xyzw.shape != (4,)
        or not np.isfinite(root_pos).all()
        or not np.isfinite(root_quat_xyzw).all()
        or not np.isclose(np.linalg.norm(root_quat_xyzw), 1.0, atol=1.0e-4)
    ):
        raise ValueError("Invalid initial root pose from motion library")
    # MotionLib uses xyzw; MuJoCo free-joint qpos uses wxyz.
    root_quat_wxyz = root_quat_xyzw[[3, 0, 1, 2]]
    motion_key = str(env._motion_lib._motion_data_keys[args.motion_id])
    motion_name = Path(motion_key).stem
    motion_fps = float(env._motion_lib._motion_fps[args.motion_id].detach().cpu().item())
    if not np.isclose(motion_fps, 50.0):
        raise ValueError(f"Runtime motion must be 50 Hz, got {motion_fps}")

    metadata = {
        "schema_version": 1,
        "format": "ufo_precomputed_onnx_latent",
        "actor": {
            "file": _relative_path(actor_path, metadata_output.parent),
            "input_dim": int(export_meta["actor_obs_dim"]),
            "output_dim": int(export_meta["output_action_dim"]),
        },
        "latent": {
            "file": _relative_path(output, metadata_output.parent),
            "dtype": "float32",
            "shape": [int(z.shape[0]), int(z.shape[1])],
            "frame_offset": 1,
            "mode": args.latent_mode,
            "aggregation": "future_window_mean" if args.latent_mode == "train_aligned" else "single_frame",
            "seq_length": seq_length if args.latent_mode == "train_aligned" else 1,
            "training_seq_length": training_seq_length,
        },
        "motion": {
            "id": int(args.motion_id),
            "name": motion_name,
            "fps": motion_fps,
            "frame_count": int(z.shape[0]),
            "reference_index": int(
                args.motion_id if args.reference_index is None else args.reference_index
            ),
            "initial_root_pos": root_pos.tolist(),
            "initial_root_quat_wxyz": root_quat_wxyz.tolist(),
            "initial_joint_pos": reference_state[0, : len(joint_names)].tolist(),
            "final_joint_pos": reference_state[-1, : len(joint_names)].tolist(),
        },
        "control": {
            "dt": 0.02,
            "joint_names": joint_names,
            "default_joint_pos": default_joint_pos.tolist(),
            "p_gains": kp.tolist(),
            "d_gains": kd.tolist(),
            "effort_limits": effort.tolist(),
            "target_scales": target_scales.tolist(),
            "action_clip": float(robot_training["action_clip_value"]),
            "action_obs_scale": float(robot_training["normalize_action_to"]),
            "action_mapping": action_mapping,
            "action_mapping_bias": action_mapping_bias.tolist(),
            "action_mapping_range": action_mapping_range.tolist(),
            "action_mapping_lower": action_mapping_lower.tolist(),
            "action_mapping_upper": action_mapping_upper.tolist(),
            "soft_joint_limits": soft_joint_limits.tolist(),
            "base_ang_vel_obs_scale": 0.25,
            "history_length": 4,
        },
        "source": {
            "model_folder": str(model_folder),
            "data_manifest": source_manifest,
            "dataset": source_dataset,
            "motion_key": motion_key,
            "motion_file": str(direct_motion_path) if direct_motion_path is not None else None,
        },
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[INFO] Saved ONNX latent: {output}")
    print(f"[INFO] Saved runtime metadata: {metadata_output}")
    print(
        f"[INFO] motion_id={args.motion_id} shape={z.shape} mode={args.latent_mode} "
        f"seq_length={seq_length if args.latent_mode == 'train_aligned' else 1} "
        f"provider={session.get_providers()[0]}"
    )


if __name__ == "__main__":
    main()
