"""Infer FB reward latents from a replay buffer using only ONNX Runtime."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import h5py
import joblib
import mujoco
import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm

from humanoidverse.mjlab_inference_utils import add_bool_arg, write_mjlab_relabel_xml
from humanoidverse.mjlab_reward_relabel import make_reward_from_name, relabel
from humanoidverse.reward_inference import _resolve_reward_tasks
from humanoidverse.utils.robot_spec import load_robot_training_spec


def _providers(device: str) -> list[str]:
    available = set(ort.get_available_providers())
    if str(device).startswith("cuda") and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _static_input_dim(value: Any) -> int | None:
    shape = list(value.shape)
    if len(shape) != 2 or not isinstance(shape[1], int):
        return None
    return int(shape[1])


def _read_sample_block(
    buffer_path: Path,
    num_samples: int,
    *,
    seed: int,
    required_observation_keys: set[str],
) -> dict[str, Any]:
    """Read a bounded time/env block instead of materializing the full HDF5 file."""

    hdf5_path = buffer_path / "buffer.hdf5"
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"Missing replay buffer HDF5: {hdf5_path}")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    requested_num_samples = int(num_samples)

    rng = np.random.default_rng(seed)
    with h5py.File(hdf5_path, "r") as buffer:
        required_datasets = {
            "action",
            "qpos",
            "qvel",
            "truncated",
            *(f"observation-{key}" for key in required_observation_keys),
        }
        missing = sorted(required_datasets.difference(buffer.keys()))
        if missing:
            raise KeyError(f"Replay buffer is missing datasets: {missing}")

        time_size, env_size = map(int, buffer["truncated"].shape[:2])
        if time_size < 2 or env_size < 1:
            raise ValueError(f"Invalid replay buffer shape: time={time_size}, envs={env_size}")

        max_transition_count = (time_size - 1) * env_size
        num_samples = min(requested_num_samples, max_transition_count)
        if requested_num_samples > max_transition_count:
            print(
                f"[INFO] Requested {requested_num_samples} reward samples, but the buffer has at most "
                f"{max_transition_count} one-step pairs; using all available non-terminal pairs."
            )

        # A rectangular block keeps HDF5 reads sequential and bounds peak RAM.
        env_count = min(env_size, max(32, min(256, math.ceil(num_samples / max(time_size - 1, 1) * 1.35))))
        time_count = min(time_size - 1, max(2, math.ceil(num_samples / env_count * 1.35)))
        while time_count * env_count < num_samples and env_count < env_size:
            env_count = min(env_size, env_count * 2)
            time_count = min(time_size - 1, max(2, math.ceil(num_samples / env_count * 1.35)))
        if time_count * env_count < num_samples:
            raise RuntimeError("Internal sample block sizing failed after clamping to buffer capacity")

        for _attempt in range(8):
            time_start = int(rng.integers(0, time_size - time_count)) if time_count < time_size - 1 else 0
            env_start = int(rng.integers(0, env_size - env_count + 1)) if env_count < env_size else 0
            current = np.s_[time_start : time_start + time_count, env_start : env_start + env_count]
            truncated = np.asarray(buffer["truncated"][current]).reshape(-1).astype(bool)
            valid = np.flatnonzero(~truncated)
            if valid.size >= num_samples:
                selected = valid if valid.size == num_samples else rng.choice(valid, size=num_samples, replace=False)
                break
            if time_count == time_size - 1 and env_count == env_size:
                if valid.size == 0:
                    raise ValueError("Replay buffer contains no non-terminal one-step pairs")
                print(
                    f"[INFO] Buffer contains {valid.size} non-terminal one-step pairs after filtering; "
                    "using all of them."
                )
                num_samples = int(valid.size)
                selected = valid
                break
            time_count = min(time_size - 1, max(time_count + 1, int(math.ceil(time_count * 1.5))))
            env_count = min(env_size, max(env_count + 1, int(math.ceil(env_count * 1.5))))
        else:
            raise ValueError(f"Could not find {num_samples} non-terminal transitions in a bounded replay-buffer block")

        next_time = np.s_[time_start + 1 : time_start + time_count + 1, env_start : env_start + env_count]

        def current_values(name: str) -> np.ndarray:
            values = np.asarray(buffer[name][current]).reshape(-1, *buffer[name].shape[2:])
            return np.ascontiguousarray(values[selected], dtype=np.float32)

        def next_values(name: str) -> np.ndarray:
            values = np.asarray(buffer[name][next_time]).reshape(-1, *buffer[name].shape[2:])
            return np.ascontiguousarray(values[selected], dtype=np.float32)

        observations = {
            key: next_values(f"observation-{key}")
            for key in required_observation_keys
        }
        return {
            "observation": observations,
            "action": current_values("action"),
            "qpos": next_values("qpos"),
            "qvel": next_values("qvel"),
            "source_block": {
                "requested_num_samples": requested_num_samples,
                "sample_count": int(len(selected)),
                "max_transition_count": max_transition_count,
                "time_start": time_start,
                "time_count": time_count,
                "env_start": env_start,
                "env_count": env_count,
            },
        }


def _run_backward_encoder(
    session: ort.InferenceSession,
    observations: dict[str, np.ndarray],
    *,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError(f"onnx_batch_size must be positive, got {batch_size}")
    inputs = session.get_inputs()
    input_names = {item.name for item in inputs}
    missing = sorted(input_names.difference(observations))
    if missing:
        raise KeyError(f"Replay observation is missing ONNX inputs: {missing}")
    for item in inputs:
        expected = _static_input_dim(item)
        actual = int(observations[item.name].shape[-1])
        if expected is not None and actual != expected:
            raise ValueError(f"ONNX input {item.name!r} expects dim {expected}, replay buffer has {actual}")

    sample_count = int(next(iter(observations.values())).shape[0])
    chunks = []
    for start in tqdm(
        range(0, sample_count, batch_size),
        total=math.ceil(sample_count / batch_size),
        desc="Backward ONNX",
        unit="batches",
        dynamic_ncols=True,
    ):
        stop = min(start + batch_size, sample_count)
        feed = {name: observations[name][start:stop] for name in input_names}
        chunks.append(np.asarray(session.run(["z"], feed)[0], dtype=np.float32))
    encoded = np.concatenate(chunks, axis=0)
    if encoded.ndim != 2 or encoded.shape[0] != sample_count or not np.isfinite(encoded).all():
        raise ValueError(f"Backward ONNX returned invalid output: shape={encoded.shape}")
    return encoded


def _reward_weighted_latent(encoded: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    if encoded.shape[0] != rewards.shape[0]:
        raise ValueError(f"Reward/encoding batch mismatch: {rewards.shape[0]} vs {encoded.shape[0]}")
    logits = 10.0 * rewards
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= np.sum(weights)
    latent = np.matmul((rewards * weights)[None, :], encoded).astype(np.float32)
    norm = float(np.linalg.norm(latent))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Reward inference produced a zero/non-finite latent norm: {norm}")
    return np.ascontiguousarray(math.sqrt(encoded.shape[1]) * latent / norm, dtype=np.float32)


def _source_stamp(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _cache_sources(backward_onnx: Path, buffer_path: Path, robot_config: Path) -> dict[str, dict[str, int | str]]:
    return {
        "backward_onnx": _source_stamp(backward_onnx),
        "buffer_hdf5": _source_stamp(buffer_path / "buffer.hdf5"),
        "robot_config": _source_stamp(robot_config),
    }


def _reuse_cached_reward_latent(
    *,
    output: Path,
    metadata_path: Path,
    latent_output: Path,
    backward_onnx: Path,
    buffer_path: Path,
    robot_config: Path,
    tasks: list[str],
    num_samples: int,
    n_inferences: int,
    seed: int,
    latent_frames: int,
) -> bool:
    if not output.is_file() or not metadata_path.is_file() or not latent_output.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_paths = {
            "backward_onnx": backward_onnx,
            "buffer_path": buffer_path,
            "robot_config": robot_config,
        }
        for key, expected in expected_paths.items():
            if Path(metadata[key]).expanduser().resolve() != expected:
                return False
        if list(metadata.get("tasks", [])) != tasks:
            return False
        if int(metadata.get("num_samples", -1)) != num_samples:
            return False
        if int(metadata.get("n_inferences", -1)) != n_inferences:
            return False
        if metadata.get("rollout_task") != tasks[0]:
            return False
        if Path(metadata.get("rollout_latent_npy", "")).expanduser().resolve() != latent_output:
            return False
        run_seeds = [int(run.get("seed", -1)) for run in metadata.get("runs", [])]
        if run_seeds != list(range(seed, seed + n_inferences)):
            return False

        current_sources = _cache_sources(backward_onnx, buffer_path, robot_config)
        recorded_sources = metadata.get("cache_sources")
        if recorded_sources is not None and recorded_sources != current_sources:
            return False

        latents = np.load(latent_output, allow_pickle=False, mmap_mode="r")
        if latents.dtype != np.float32 or latents.ndim != 2 or latents.shape[0] != latent_frames:
            return False
        expected_shape = [int(value) for value in metadata.get("rollout_latent_shape", [])]
        if list(latents.shape) != expected_shape:
            return False

        cached_z = joblib.load(output)
        if any(task not in cached_z or len(cached_z[task]) != n_inferences for task in tasks):
            return False

        # Upgrade metadata written by versions before source stamps were added,
        # so subsequent runs also invalidate when a model/buffer/config changes.
        if recorded_sources is None:
            metadata["cache_sources"] = current_sources
            metadata["seed"] = seed
            metadata["latent_frames"] = latent_frames
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    except (KeyError, OSError, TypeError, ValueError, EOFError):
        return False

    print(f"[INFO] Reusing cached reward embeddings: {output}")
    print(f"[INFO] Reusing cached interactive rollout latent: {latent_output}")
    print(f"[INFO] Cache metadata matched tasks={tasks}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True, help="Folder containing exported/backward_encoder.onnx.")
    parser.add_argument("--buffer-path", type=Path, required=True, help="Replay buffer folder containing buffer.hdf5.")
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robots/roban_s22.yaml"))
    parser.add_argument("--backward-onnx", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Task-named .pkl/.npy/.json output directory; defaults to <model-folder>/reward_inference.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--latent-npy-output",
        type=Path,
        default=None,
        help="Constant per-frame latent used by the interactive ONNX MuJoCo runner.",
    )
    parser.add_argument("--latent-frames", type=int, default=5000)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--n-inferences", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="ONNX device: cpu or cuda.")
    parser.add_argument("--onnx-batch-size", type=int, default=8192)
    parser.add_argument("--max-workers", type=int, default=24)
    add_bool_arg(parser, "--reuse-existing", False, "Reuse matching reward latent outputs when available.")
    add_bool_arg(parser, "--process-executor", True, "Use processes for MuJoCo reward relabeling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_folder = args.model_folder.expanduser().resolve()
    buffer_path = args.buffer_path.expanduser().resolve()
    robot_config = args.robot_config.expanduser().resolve()
    backward_onnx = (
        model_folder / "exported" / "backward_encoder.onnx"
        if args.backward_onnx is None
        else args.backward_onnx.expanduser().resolve()
    )
    if not backward_onnx.is_file():
        raise FileNotFoundError(f"Missing backward encoder ONNX: {backward_onnx}")
    if args.latent_frames <= 0:
        raise ValueError(f"latent_frames must be positive, got {args.latent_frames}")

    robot_training = load_robot_training_spec(robot_config)
    tasks, task_support_mode = _resolve_reward_tasks(args.tasks, robot_training)
    if len(tasks) != 1 and (args.output is not None or args.latent_npy_output is not None):
        raise ValueError("--output and --latent-npy-output can only be used with exactly one task")
    output_dir = (
        model_folder / "reward_inference"
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    task_paths: dict[str, tuple[Path, Path, Path]] = {}
    pending_tasks: list[str] = []
    for task in tasks:
        output = args.output.expanduser().resolve() if args.output is not None else output_dir / f"{task}.pkl"
        latent_output = (
            args.latent_npy_output.expanduser().resolve()
            if args.latent_npy_output is not None
            else output_dir / f"{task}.npy"
        )
        metadata_path = output.with_suffix(".json")
        task_paths[task] = (output, latent_output, metadata_path)
        if not args.reuse_existing or not _reuse_cached_reward_latent(
            output=output,
            metadata_path=metadata_path,
            latent_output=latent_output,
            backward_onnx=backward_onnx,
            buffer_path=buffer_path,
            robot_config=robot_config,
            tasks=[task],
            num_samples=args.num_samples,
            n_inferences=args.n_inferences,
            seed=args.seed,
            latent_frames=args.latent_frames,
        ):
            pending_tasks.append(task)
    if not pending_tasks:
        return

    relabel_xml = write_mjlab_relabel_xml(
        Path(robot_training.robot.xml_path),
        output_dir,
        list(robot_training.robot.control_joint_names),
        robot_training.robot.name,
        root_body_name=robot_training.robot.base_body,
    )
    relabel_model = mujoco.MjModel.from_xml_path(str(relabel_xml))

    providers = _providers(args.device)
    session = ort.InferenceSession(str(backward_onnx), providers=providers)
    observation_keys = {item.name for item in session.get_inputs()}
    print(f"[INFO] ONNX reward inference model={backward_onnx}")
    print(f"[INFO] ONNX providers={session.get_providers()}")
    print(f"[INFO] Replay buffer={buffer_path}")
    print(f"[INFO] Robot={robot_training.robot.name} task support={task_support_mode}")
    print(f"[INFO] Requested tasks={tasks}")
    print(f"[INFO] Tasks to compute={pending_tasks}")

    z_dict: dict[str, list[torch.Tensor]] = {}
    run_metadata: list[dict[str, Any]] = []
    for inference_index in range(args.n_inferences):
        print(
            f"[INFO] Reading replay samples for inference "
            f"{inference_index + 1}/{args.n_inferences}..."
        )
        sample = _read_sample_block(
            buffer_path,
            args.num_samples,
            seed=args.seed + inference_index,
            required_observation_keys=observation_keys,
        )
        print(f"[INFO] Loaded {sample['source_block']['sample_count']} replay samples")
        encoded = _run_backward_encoder(session, sample["observation"], batch_size=args.onnx_batch_size)
        for task in pending_tasks:
            rewards = relabel(
                relabel_model,
                sample["qpos"],
                sample["qvel"],
                sample["action"],
                make_reward_from_name(task),
                max_workers=args.max_workers,
                process_executor=args.process_executor,
                show_progress=True,
                progress_desc=f"Reward {task}",
            )
            latent = _reward_weighted_latent(encoded, rewards)
            z_dict.setdefault(task, []).append(torch.from_numpy(latent))
            print(
                f"[INFO] Inference {inference_index + 1}/{args.n_inferences} task={task} "
                f"reward_mean={float(np.mean(rewards)):.6f} reward_max={float(np.max(rewards)):.6f} "
                f"z_norm={float(np.linalg.norm(latent)):.6f}"
            )
        run_metadata.append({"seed": args.seed + inference_index, **sample["source_block"]})

    for task in pending_tasks:
        output, latent_output, metadata_path = task_paths[task]
        output.parent.mkdir(parents=True, exist_ok=True)
        latent_output.parent.mkdir(parents=True, exist_ok=True)
        task_z = z_dict[task]
        joblib.dump({task: task_z}, output)
        rollout_z = task_z[0].detach().cpu().numpy().astype(np.float32, copy=False)
        rollout_latents = np.repeat(rollout_z, args.latent_frames, axis=0)
        np.save(latent_output, np.ascontiguousarray(rollout_latents), allow_pickle=False)
        metadata_path.write_text(
            json.dumps(
                {
                    "backward_onnx": str(backward_onnx),
                    "buffer_path": str(buffer_path),
                    "robot_config": str(robot_config),
                    "tasks": [task],
                    "num_samples": args.num_samples,
                    "n_inferences": args.n_inferences,
                    "seed": args.seed,
                    "latent_frames": args.latent_frames,
                    "cache_sources": _cache_sources(backward_onnx, buffer_path, robot_config),
                    "rollout_task": task,
                    "rollout_latent_npy": str(latent_output),
                    "rollout_latent_shape": list(rollout_latents.shape),
                    "onnx_inputs": {item.name: item.shape for item in session.get_inputs()},
                    "runs": run_metadata,
                },
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"[INFO] Saved task={task} reward embeddings: {output}")
        print(f"[INFO] Saved task={task} rollout latent: {latent_output}")
        print(f"[INFO] Saved task={task} metadata: {metadata_path}")


if __name__ == "__main__":
    main()
