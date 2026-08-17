"""Adapt a BFM tracking latent sequence with batched MuJoCo rollouts.

The Actor stays frozen.  Starting from the zero-shot tracking sequence, the
script performs zero-order trajectory optimization over temporally correlated
latent residuals, following the dual-annealing setup described by BFM-Zero.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mediapy as media
import numpy as np
import onnxruntime as ort
import torch
from tqdm import trange

from humanoidverse.generate_onnx_latent import _prepare_motion_input, _train_aligned_latents
from humanoidverse.mjlab_inference_utils import (
    MujocoQposRenderer,
    load_mjlab_env_cfg,
    policy_qpos_from_env,
    resolve_inference_robot_config,
)
from humanoidverse.utils.helpers import get_backward_observation
from humanoidverse.utils.robot_spec import load_robot_training_spec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_FOLDER = PROJECT_ROOT / "runs/新数据addlelay_onnx_153m_20260807"
DEFAULT_MOTION = Path("/home/thl/Downloads/retargeter/dance1_subject2.roban_s22.fk50.npz")
DEFAULT_ROBOT_CONFIG = PROJECT_ROOT / "configs/robots/roban_s22.yaml"


@dataclass
class TrackingMetrics:
    score: float
    global_mpjpe_m: float
    local_mpjpe_m: float
    root_position_error_m: float
    dof_position_error_rad: float
    mean_upright: float
    survival_s: float
    latent_cosine_to_initial: float


class OnnxActor:
    """Exact deployment Actor input: state, previous action, history, latent."""

    def __init__(self, path: Path, provider: str, device: str) -> None:
        provider_name = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}[provider]
        if provider_name not in ort.get_available_providers():
            raise ValueError(f"Unavailable ONNX provider {provider_name}; available={ort.get_available_providers()}")
        self.session = ort.InferenceSession(str(path), providers=[provider_name])
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].name != "actor_obs":
            raise ValueError(f"Expected one actor_obs input, got {[item.name for item in inputs]}: {path}")
        self.device = device

    def act(self, observation: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        actor_obs = torch.cat(
            [observation["state"], observation["last_action"], observation["history_actor"], z], dim=-1
        )
        output = self.session.run(
            None,
            {"actor_obs": actor_obs.detach().cpu().numpy().astype(np.float32, copy=False)},
        )[0]
        return torch.from_numpy(output).to(device=self.device, dtype=torch.float32)


def project_latents(z: torch.Tensor, norm: float = 16.0) -> torch.Tensor:
    return torch.nn.functional.normalize(z, dim=-1) * float(norm)


def sample_trajectory_candidates(
    nominal: torch.Tensor,
    initial: torch.Tensor,
    *,
    particles: int,
    sigma: float,
    temporal_beta: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample sphere-projected residuals with AR(1) temporal correlation."""
    steps, z_dim = nominal.shape
    white = torch.randn(
        (particles, steps, z_dim),
        dtype=nominal.dtype,
        device=nominal.device,
        generator=generator,
    )
    beta = float(temporal_beta)
    innovation_scale = math.sqrt(max(1.0 - beta * beta, 1.0e-8))
    noise = torch.empty_like(white)
    noise[:, 0] = white[:, 0]
    for step in range(1, steps):
        noise[:, step] = beta * noise[:, step - 1] + innovation_scale * white[:, step]

    nominal_unit = torch.nn.functional.normalize(nominal, dim=-1).unsqueeze(0)
    noise -= (noise * nominal_unit).sum(dim=-1, keepdim=True) * nominal_unit
    candidates = project_latents(nominal.unsqueeze(0) + float(sigma) * noise)
    candidates[0] = initial
    if particles > 1:
        candidates[1] = nominal
    return candidates


def _target_states(reference: dict[str, torch.Tensor], frame: int, num_envs: int, env) -> dict[str, torch.Tensor]:
    root_pos = reference["ref_body_pos"][frame, 0].unsqueeze(0).repeat(num_envs, 1)
    root_pos = root_pos + env.env_origins
    root_state = torch.cat(
        [
            root_pos,
            reference["ref_body_rots"][frame, 0].unsqueeze(0).repeat(num_envs, 1),
            reference["ref_body_vels"][frame, 0].unsqueeze(0).repeat(num_envs, 1),
            reference["ref_body_angular_vels"][frame, 0].unsqueeze(0).repeat(num_envs, 1),
        ],
        dim=-1,
    )
    dof_state = torch.zeros((num_envs, env.num_dof, 2), dtype=torch.float32, device=env.device)
    dof_state[..., 0] = reference["dof_pos"][frame].unsqueeze(0)
    # get_backward_observation stores reference joint velocity under ref_dof_vel.
    dof_state[..., 1] = reference["ref_dof_vel"][frame].unsqueeze(0)
    return {"root_states": root_state, "dof_states": dof_state}


@torch.inference_mode()
def evaluate_candidates(
    wrapped_env,
    actor: OnnxActor,
    candidates: torch.Tensor,
    initial: torch.Tensor,
    reference: dict[str, torch.Tensor],
    *,
    start_frame: int,
    dt: float,
    position_sigma: float,
    local_position_sigma: float,
    root_sigma: float,
    dof_sigma: float,
    latent_reg: float,
    min_upright: float,
    min_root_height: float,
    fall_grace_s: float,
) -> tuple[torch.Tensor, list[TrackingMetrics]]:
    env = wrapped_env._env
    particles, steps, _ = candidates.shape
    observation, _ = wrapped_env.reset(
        to_numpy=False,
        target_states=_target_states(reference, start_frame, particles, env),
    )

    alive = torch.ones(particles, dtype=torch.bool, device=env.device)
    fall_count = torch.zeros(particles, dtype=torch.long, device=env.device)
    fall_grace_steps = max(1, int(round(float(fall_grace_s) / dt)))
    score_sum = torch.zeros(particles, device=env.device)
    global_error_sum = torch.zeros(particles, device=env.device)
    local_error_sum = torch.zeros(particles, device=env.device)
    root_error_sum = torch.zeros(particles, device=env.device)
    dof_error_sum = torch.zeros(particles, device=env.device)
    upright_sum = torch.zeros(particles, device=env.device)
    survival_steps = torch.zeros(particles, device=env.device)

    for step in trange(steps, desc="trajectory rollout", leave=False):
        action = actor.act(observation, candidates[:, step])
        observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)

        ref_frame = start_frame + step + 1
        ref_body = reference["ref_body_pos"][ref_frame]
        ref_root = ref_body[0]
        current_body = env.body_pos
        body_count = min(current_body.shape[1], ref_body.shape[0])
        current_body = current_body[:, :body_count]
        ref_body = ref_body[:body_count].unsqueeze(0) + env.env_origins.unsqueeze(1)
        global_error = torch.norm(current_body - ref_body, dim=-1).mean(dim=-1)
        current_local = current_body - current_body[:, :1]
        ref_local = ref_body - ref_body[:, :1]
        local_error = torch.norm(current_local - ref_local, dim=-1).mean(dim=-1)
        root_error = torch.norm(env.robot_root_states[:, :3] - (ref_root.unsqueeze(0) + env.env_origins), dim=-1)
        dof_error = torch.abs(env.dof_pos - reference["dof_pos"][ref_frame].unsqueeze(0)).mean(dim=-1)
        upright = torch.clamp(-env.projected_gravity[:, 2], 0.0, 1.0)

        done = torch.as_tensor(terminated, device=env.device).bool() | torch.as_tensor(truncated, device=env.device).bool()
        fallen = (upright < float(min_upright)) | (env.robot_root_states[:, 2] < float(min_root_height))
        fall_count = torch.where(fallen & alive, fall_count + 1, torch.zeros_like(fall_count))
        failed = done | (fall_count >= fall_grace_steps)
        valid = alive & ~failed
        weight = valid.float()

        pose_score = torch.exp(-0.5 * torch.square(global_error / float(position_sigma)))
        local_score = torch.exp(-0.5 * torch.square(local_error / float(local_position_sigma)))
        root_score = torch.exp(-0.5 * torch.square(root_error / float(root_sigma)))
        dof_score = torch.exp(-0.5 * torch.square(dof_error / float(dof_sigma)))
        instant_score = 0.35 * pose_score + 0.30 * local_score + 0.15 * root_score + 0.15 * dof_score + 0.05 * upright
        score_sum += instant_score * weight
        global_error_sum += global_error * weight
        local_error_sum += local_error * weight
        root_error_sum += root_error * weight
        dof_error_sum += dof_error * weight
        upright_sum += upright * weight
        survival_steps += weight
        alive &= ~failed

    raw_score = score_sum / float(steps)
    cosine = torch.nn.functional.cosine_similarity(candidates, initial.unsqueeze(0), dim=-1).mean(dim=-1)
    scores = raw_score - float(latent_reg) * (1.0 - cosine)
    denom = survival_steps.clamp_min(1.0)
    metrics = [
        TrackingMetrics(
            score=float(scores[index].cpu()),
            global_mpjpe_m=float((global_error_sum[index] / denom[index]).cpu()),
            local_mpjpe_m=float((local_error_sum[index] / denom[index]).cpu()),
            root_position_error_m=float((root_error_sum[index] / denom[index]).cpu()),
            dof_position_error_rad=float((dof_error_sum[index] / denom[index]).cpu()),
            mean_upright=float((upright_sum[index] / denom[index]).cpu()),
            survival_s=float((survival_steps[index] * dt).cpu()),
            latent_cosine_to_initial=float(cosine[index].cpu()),
        )
        for index in range(particles)
    ]
    return scores, metrics


def _load_reference_and_initial_latent(args: argparse.Namespace, model_folder: Path):
    data_args = argparse.Namespace(
        motion=args.motion,
        robot_config=args.robot_config,
        data_manifest=None,
        dataset=None,
        rebuild_motion_cache=bool(args.rebuild_motion_cache),
    )
    data_path, robot_config, _source_manifest, _source_dataset, _motion_path = _prepare_motion_input(data_args, model_folder)
    env_cfg, use_root_height_obs = load_mjlab_env_cfg(
        model_folder,
        data_path=data_path,
        robot_config=robot_config,
        device=args.device,
        headless=True,
        disable_dr=True,
        disable_obs_noise=True,
        max_episode_length_s=max(args.duration_s + 2.0, 10.0),
    )
    wrapped_env, _ = env_cfg.build(num_envs=1)
    try:
        env = wrapped_env._env
        env._motion_lib.load_all_motions()
        backward_obs, reference = get_backward_observation(env, 0, use_root_height_obs=use_root_height_obs)
        backward_path = model_folder / "exported/backward_encoder.onnx"
        if not backward_path.is_file():
            raise FileNotFoundError(f"Missing backward encoder: {backward_path}")
        providers = ["CPUExecutionProvider"]
        backward = ort.InferenceSession(str(backward_path), providers=providers)
        inputs = {
            key: value[1:].detach().cpu().numpy().astype(np.float32)
            for key, value in backward_obs.items()
        }
        frame_z = backward.run(
            ["z"],
            {item.name: inputs[item.name] for item in backward.get_inputs()},
        )[0]
        config = json.loads((model_folder / "config.json").read_text())
        seq_length = int(config["agent"]["model"]["seq_length"])
        latent = _train_aligned_latents(frame_z, seq_length) if args.latent_mode == "train_aligned" else frame_z
        reference = {key: value.detach().clone() if isinstance(value, torch.Tensor) else value for key, value in reference.items()}
        return env_cfg, robot_config, reference, np.ascontiguousarray(latent, dtype=np.float32), float(env.dt)
    finally:
        wrapped_env.close()


def _save_result(output_dir: Path, best: torch.Tensor, full_latent: np.ndarray, start_frame: int) -> Path:
    adapted_full = np.array(full_latent, dtype=np.float32, copy=True)
    adapted_full[start_frame : start_frame + best.shape[0]] = best.detach().cpu().numpy()
    output_path = output_dir / "adapted_latent.npy"
    np.save(output_path, adapted_full, allow_pickle=False)
    np.save(output_dir / "adapted_segment.npy", best.detach().cpu().numpy(), allow_pickle=False)
    return output_path


@torch.inference_mode()
def _rollout_qpos(wrapped_env, actor, latent, reference, start_frame, dt):
    env = wrapped_env._env
    observation, _ = wrapped_env.reset(
        to_numpy=False,
        target_states=_target_states(reference, start_frame, 1, env),
    )
    qposes = [policy_qpos_from_env(wrapped_env, expected_qpos_size=7 + env.num_dof)]
    for step in range(latent.shape[0]):
        action = actor.act(observation, latent[step : step + 1])
        observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)
        qposes.append(policy_qpos_from_env(wrapped_env, expected_qpos_size=7 + env.num_dof))
        if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
            break
    return qposes


def _reference_qpos(reference, start_frame, steps, joint_order_indices):
    frames = []
    for frame in range(start_frame, start_frame + steps + 1):
        root_pos = reference["ref_body_pos"][frame, 0].detach().cpu().numpy()
        quat_xyzw = reference["ref_body_rots"][frame, 0].detach().cpu().numpy()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        dof = reference["dof_pos"][frame].detach().cpu().numpy()[joint_order_indices]
        frames.append(np.concatenate([root_pos, quat_wxyz, dof]))
    return frames


def render_video(env_cfg, actor, baseline, adapted, reference, start_frame, robot_config, output_path, fps, size):
    wrapped_env, _ = env_cfg.build(num_envs=1)
    renderer = None
    try:
        dt = float(wrapped_env._env.dt)
        baseline_qpos = _rollout_qpos(wrapped_env, actor, baseline, reference, start_frame, dt)
        adapted_qpos = _rollout_qpos(wrapped_env, actor, adapted, reference, start_frame, dt)
        spec = load_robot_training_spec(robot_config)
        control_names = list(spec.robot.control_joint_names)
        qpos_names = sorted(control_names, key=lambda name: spec.robot.joint_qpos_addr[name])
        control_index = {name: index for index, name in enumerate(control_names)}
        qpos_indices = np.asarray([control_index[name] for name in qpos_names])
        reference_qpos = _reference_qpos(reference, start_frame, baseline.shape[0], qpos_indices)
        renderer = MujocoQposRenderer(
            Path(spec.robot.xml_path).expanduser().resolve(),
            render_size=size,
            expected_qpos_size=reference_qpos[0].size,
        )
        frame_count = max(len(reference_qpos), len(baseline_qpos), len(adapted_qpos))
        frames = []
        for index in trange(frame_count, desc="render reference/baseline/adapted", leave=False):
            panels = [
                renderer.render_qpos(reference_qpos[min(index, len(reference_qpos) - 1)]),
                renderer.render_qpos(baseline_qpos[min(index, len(baseline_qpos) - 1)]),
                renderer.render_qpos(adapted_qpos[min(index, len(adapted_qpos) - 1)]),
            ]
            separator = np.full((panels[0].shape[0], 4, 3), 255, dtype=np.uint8)
            frames.append(np.concatenate([panels[0], separator, panels[1], separator, panels[2]], axis=1))
        media.write_video(str(output_path), frames, fps=fps)
    finally:
        if renderer is not None:
            renderer.close()
        wrapped_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_MODEL_FOLDER)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--actor-onnx", type=Path, default=None)
    parser.add_argument("--onnx-provider", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--latent-mode", choices=("single_frame", "train_aligned"), default="train_aligned")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--particles", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--elite-fraction", type=float, default=0.0625)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--beta-iteration", type=float, default=0.85)
    parser.add_argument("--beta-temporal", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--position-sigma", type=float, default=0.20)
    parser.add_argument("--local-position-sigma", type=float, default=0.12)
    parser.add_argument("--root-sigma", type=float, default=0.25)
    parser.add_argument("--dof-sigma", type=float, default=0.50)
    parser.add_argument("--latent-reg", type=float, default=0.02)
    parser.add_argument("--min-upright", type=float, default=0.5)
    parser.add_argument("--min-root-height", type=float, default=0.45)
    parser.add_argument("--fall-grace-s", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--rebuild-motion-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.particles < 4:
        raise ValueError("--particles must be at least 4")
    if args.iterations < 1 or args.duration_s <= 0.0 or args.start_frame < 0:
        raise ValueError("iterations/duration must be positive and start-frame non-negative")
    if not 0.0 <= args.beta_temporal < 1.0 or not 0.0 < args.beta_iteration <= 1.0:
        raise ValueError("beta-temporal must be in [0,1), beta-iteration in (0,1]")

    model_folder = args.model_folder.expanduser().resolve()
    args.motion = args.motion.expanduser().resolve()
    args.robot_config = resolve_inference_robot_config(args.robot_config, None)
    actor_path = (args.actor_onnx or model_folder / "exported/FBcprAuxModel.onnx").expanduser().resolve()
    if not actor_path.is_file() or not args.motion.is_file():
        raise FileNotFoundError(f"Missing actor or motion: actor={actor_path}, motion={args.motion}")

    env_cfg, robot_config, reference, full_latent_np, dt = _load_reference_and_initial_latent(args, model_folder)
    steps = int(round(args.duration_s / dt))
    if args.start_frame + steps + 1 > len(reference["ref_body_pos"]):
        raise ValueError(
            f"Requested frames [{args.start_frame}, {args.start_frame + steps}] exceed reference length "
            f"{len(reference['ref_body_pos'])}"
        )
    initial = torch.from_numpy(full_latent_np[args.start_frame : args.start_frame + steps]).to(args.device)
    initial = project_latents(initial)
    nominal = initial.clone()
    actor = OnnxActor(actor_path, args.onnx_provider, args.device)
    wrapped_env, _ = env_cfg.build(num_envs=args.particles)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        or Path(__file__).resolve().parent / "outputs" / f"{args.motion.stem}_f{args.start_frame}_{timestamp}"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    elite_count = max(2, int(math.ceil(args.particles * args.elite_fraction)))
    sigma = float(args.sigma)
    best = initial.clone()
    best_metrics: TrackingMetrics | None = None
    history: list[dict[str, Any]] = []

    print(f"[trajectory-adapt] motion={args.motion}")
    print(f"[trajectory-adapt] actor={actor_path} provider={args.onnx_provider}")
    print(
        f"[trajectory-adapt] frames={args.start_frame}:{args.start_frame + steps} duration={args.duration_s:.2f}s "
        f"particles={args.particles} iterations={args.iterations} beta1={args.beta_iteration} beta2={args.beta_temporal}"
    )
    try:
        for iteration in range(args.iterations):
            candidates = sample_trajectory_candidates(
                nominal,
                initial,
                particles=args.particles,
                sigma=sigma,
                temporal_beta=args.beta_temporal,
                generator=generator,
            )
            scores, metrics = evaluate_candidates(
                wrapped_env,
                actor,
                candidates,
                initial,
                reference,
                start_frame=args.start_frame,
                dt=dt,
                position_sigma=args.position_sigma,
                local_position_sigma=args.local_position_sigma,
                root_sigma=args.root_sigma,
                dof_sigma=args.dof_sigma,
                latent_reg=args.latent_reg,
                min_upright=args.min_upright,
                min_root_height=args.min_root_height,
                fall_grace_s=args.fall_grace_s,
            )
            order = torch.argsort(scores, descending=True)
            elite_indices = order[:elite_count]
            elite_scores = scores[elite_indices]
            weights = torch.softmax((elite_scores - elite_scores.max()) / max(args.temperature, 1.0e-6), dim=0)
            nominal = project_latents((candidates[elite_indices] * weights[:, None, None]).sum(dim=0))
            index = int(order[0])
            iteration_best = metrics[index]
            if best_metrics is None or iteration_best.score > best_metrics.score:
                best = candidates[index].clone()
                best_metrics = iteration_best
                _save_result(output_dir, best, full_latent_np, args.start_frame)
            record = {
                "iteration": iteration,
                "sigma": sigma,
                "best": asdict(iteration_best),
                "baseline": asdict(metrics[0]),
                "score_mean": float(scores.mean().cpu()),
                "score_std": float(scores.std().cpu()),
            }
            history.append(record)
            (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n")
            print(
                f"[trajectory-adapt] iter={iteration + 1}/{args.iterations} sigma={sigma:.4f} "
                f"score={iteration_best.score:.4f} baseline={metrics[0].score:.4f} "
                f"MPJPE={iteration_best.global_mpjpe_m * 1000:.1f}mm "
                f"local={iteration_best.local_mpjpe_m * 1000:.1f}mm survive={iteration_best.survival_s:.2f}s"
            )
            sigma *= float(args.beta_iteration)
    finally:
        wrapped_env.close()

    assert best_metrics is not None
    adapted_path = _save_result(output_dir, best, full_latent_np, args.start_frame)
    baseline_metrics = history[-1]["baseline"]
    search_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    summary = {
        "method": "BFM-Zero-style dual-annealed zero-order latent trajectory adaptation",
        "actor_onnx": str(actor_path),
        "motion": str(args.motion),
        "frame_range": [args.start_frame, args.start_frame + steps],
        "baseline": baseline_metrics,
        "best": asdict(best_metrics),
        "adapted_latent": str(adapted_path),
        "args": search_args,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    if args.save_video:
        video_path = output_dir / "comparison_reference_baseline_adapted.mp4"
        render_video(
            env_cfg,
            actor,
            initial,
            best,
            reference,
            args.start_frame,
            robot_config,
            video_path,
            args.fps,
            args.render_size,
        )
        print(f"[trajectory-adapt] video={video_path}")
    print(f"[trajectory-adapt] adapted latent={adapted_path}")
    print(f"[trajectory-adapt] summary={output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
