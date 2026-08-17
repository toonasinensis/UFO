"""Diagonal-CEM search for a fixed BFM latent that tracks a planar velocity target.

This is deliberately an inference-only experiment.  The actor is frozen and
each candidate z is held constant for the whole MuJoCo rollout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mediapy as media
import numpy as np
import onnxruntime as ort
import torch
from tqdm import trange

from humanoidverse.mjlab_inference_utils import (
    MujocoQposRenderer,
    load_mjlab_env_cfg,
    policy_qpos_from_env,
    resolve_inference_robot_config,
)
from humanoidverse.reward_inference import _default_standing_target_states
from humanoidverse.utils.robot_spec import load_robot_training_spec
from humanoidverse.utils.torch_utils import quat_rotate_inverse

from .tensorboard_logger import LatentSearchTensorboardLogger
from .objective_registry import (
    MetricAccumulator,
    ObjectiveConfig,
    StepSignals,
    compute_failure,
    compute_instant_objective,
    registry_snapshot,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_FOLDER = PROJECT_ROOT / "runs/新数据addlelay_onnx_153m_20260807"
DEFAULT_TASK = "move-ego-90-0.5"
DEFAULT_ROBOT_CONFIG = PROJECT_ROOT / "configs/robots/roban_s22.yaml"
DEFAULT_DATA_PATH = PROJECT_ROOT / "cache/motion_data/roban_s22/roban_lafan_10s_inference_ufo.pkl"
MOVE_EGO_TASK_PATTERN = re.compile(r"^move-ego-(-?\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)$")

OBJECTIVE_DESCRIPTION = r"""
For each scored policy step (the first warm-up second is excluded):

`velocity = exp(-0.5 * (((vx - target_vx) / velocity_sigma)^2 + ((vy - target_vy) / velocity_sigma)^2))`

`yaw = exp(-0.5 * (yaw_rate / yaw_rate_sigma)^2)`

`stand = clamp((upright - 0.40) / 0.45, 0, 1) * clamp((root_height - 0.35) / 0.30, 0, 1)`

For each contacting foot, `foot_error` is the XY norm of world gravity expressed
in the foot frame, exactly as in the training aux reward `penalty_feet_ori`.

`feet_flat = exp(-0.5 * ((left_foot_error + right_foot_error) / foot_flatness_sigma)^2)`

`instant = stand * (velocity_weight * velocity + yaw_weight * yaw + foot_flatness_weight * feet_flat)`

The rollout objective is `0.8 * mean(instant, including zero after a fall) + 0.2 * survival_fraction`.
The final search score is `rollout_objective - latent_reg * (1 - cosine(z, initial_z))`.
""".strip()


@dataclass
class RolloutMetrics:
    score: float
    objective_without_latent_reg: float
    survival_s: float
    latent_cosine_to_initial: float
    step_metrics: dict[str, float]

    def __getattr__(self, name: str) -> float:
        metrics = self.__dict__.get("step_metrics", {})
        if name in metrics:
            return metrics[name]
        raise AttributeError(name)

    def to_dict(self) -> dict[str, float]:
        return {
            "score": self.score,
            "objective_without_latent_reg": self.objective_without_latent_reg,
            "survival_s": self.survival_s,
            "latent_cosine_to_initial": self.latent_cosine_to_initial,
            **self.step_metrics,
        }

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> "RolloutMetrics":
        fixed_names = {"score", "objective_without_latent_reg", "survival_s", "latent_cosine_to_initial"}
        return cls(
            score=float(values["score"]),
            objective_without_latent_reg=float(values["objective_without_latent_reg"]),
            survival_s=float(values["survival_s"]),
            latent_cosine_to_initial=float(values["latent_cosine_to_initial"]),
            step_metrics={key: float(value) for key, value in values.items() if key not in fixed_names},
        )


class OnnxActor:
    """Exact deployed Actor path: [state, last_action, history_actor, z]."""

    def __init__(self, path: Path, provider: str):
        providers = {
            "cpu": ["CPUExecutionProvider"],
            "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        }[provider]
        self.path = path
        self.session = ort.InferenceSession(str(path), providers=providers)
        actor_input = self.session.get_inputs()[0]
        if actor_input.name != "actor_obs":
            raise ValueError(f"Expected ONNX input actor_obs, got {actor_input.name}: {path}")
        if actor_input.shape[-1] not in (601, "601"):
            raise ValueError(f"Expected ONNX actor_obs width 601, got {actor_input.shape}: {path}")
        output_names = {output.name for output in self.session.get_outputs()}
        if "action" not in output_names:
            raise ValueError(f"Expected ONNX output action, got {sorted(output_names)}: {path}")

    def act(self, observation: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        actor_obs = torch.cat(
            [observation["state"], observation["last_action"], observation["history_actor"], z], dim=-1
        )
        output = self.session.run(["action"], {"actor_obs": actor_obs.detach().cpu().numpy()})[0]
        return torch.from_numpy(output).to(device=z.device, dtype=torch.float32)


def project_latents(z: torch.Tensor, norm: float) -> torch.Tensor:
    """Project latent rows onto the radius-``norm`` hypersphere."""
    return torch.nn.functional.normalize(z, dim=-1) * float(norm)


def planar_velocity_from_move_ego_task(task: str) -> tuple[float, float] | None:
    """Decode ``move-ego-ANGLE-SPEED`` using +X forward and +Y left."""
    match = MOVE_EGO_TASK_PATTERN.fullmatch(task)
    if match is None:
        return None
    angle_rad = math.radians(float(match.group(1)))
    speed = float(match.group(2))
    return speed * math.cos(angle_rad), speed * math.sin(angle_rad)


def load_initial_latent(path: Path, *, z_dim: int, device: str) -> torch.Tensor:
    values = np.load(path)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != z_dim or values.shape[0] < 1:
        raise ValueError(f"Expected latent shape [N,{z_dim}], got {values.shape}: {path}")
    first = np.asarray(values[0], dtype=np.float32)
    if not np.isfinite(first).all():
        raise ValueError(f"Latent contains NaN/Inf: {path}")
    return torch.from_numpy(first).to(device=device).unsqueeze(0)


def sample_candidates(
    mean: torch.Tensor,
    initial: torch.Tensor,
    *,
    population: int,
    std: torch.Tensor,
    latent_norm: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample a diagonal Gaussian on the sphere tangent plane.

    Candidate 0 is the original latent for a stable baseline and candidate 1 is
    the current distribution mean.  The remaining candidates are stochastic.
    """
    if std.shape != mean.shape:
        raise ValueError(f"std shape must match mean shape, got {std.shape} vs {mean.shape}")
    noise = torch.randn(
        (population, mean.shape[-1]),
        device=mean.device,
        dtype=mean.dtype,
        generator=generator,
    )
    noise = noise * std
    mean_unit = torch.nn.functional.normalize(mean, dim=-1)
    noise = noise - (noise * mean_unit).sum(dim=-1, keepdim=True) * mean_unit
    candidates = project_latents(mean + noise, latent_norm)
    candidates[0] = initial[0]
    if population > 1:
        candidates[1] = mean[0]
    return candidates


def update_diagonal_cem(
    elite: torch.Tensor,
    old_std: torch.Tensor,
    *,
    latent_norm: float,
    covariance_alpha: float,
    min_sigma: float,
    max_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit the next spherical mean and smoothed diagonal covariance to elites.

    ``covariance_alpha`` is the retained fraction of the previous covariance.
    Elite residuals are projected onto the new mean's tangent plane before the
    per-dimension variance is estimated, so radial spread introduced by the
    fixed-norm constraint is not learned as exploration noise.
    """
    if elite.ndim != 2 or elite.shape[0] < 2:
        raise ValueError(f"elite must contain at least two rows, got {elite.shape}")
    if old_std.shape != (1, elite.shape[1]):
        raise ValueError(f"old_std must have shape [1,{elite.shape[1]}], got {old_std.shape}")

    new_mean = project_latents(elite.mean(dim=0, keepdim=True), latent_norm)
    mean_unit = torch.nn.functional.normalize(new_mean, dim=-1)
    residual = elite - new_mean
    residual = residual - (residual * mean_unit).sum(dim=-1, keepdim=True) * mean_unit
    elite_variance = residual.square().mean(dim=0, keepdim=True)
    variance = float(covariance_alpha) * old_std.square() + (
        1.0 - float(covariance_alpha)
    ) * elite_variance
    new_std = torch.sqrt(torch.clamp_min(variance, 0.0))
    new_std = torch.clamp(new_std, min=float(min_sigma), max=float(max_sigma))
    return new_mean, new_std


def _contact_foot_flatness_error(core, foot_index: int) -> torch.Tensor:
    """Return contact-gated foot tilt using the existing aux-reward signals.

    The orientation error mirrors ``penalty_feet_ori``.  Contact uses the
    force-norm test from ``penalty_slippage`` because MJLab's signed world-Z
    force convention can make ``force_z > 1`` false for a valid contact.
    """
    body_index = core.feet_indices[foot_index]
    foot_quat = core.body_rot[:, body_index]
    gravity_in_foot = quat_rotate_inverse(foot_quat, core.gravity_vec, w_last=True)
    in_contact = torch.linalg.vector_norm(core.contact_forces[:, body_index, :], dim=1) > 1.0
    return torch.linalg.vector_norm(gravity_in_foot[:, :2], dim=1) * in_contact


@torch.inference_mode()
def evaluate_candidates(
    wrapped_env,
    model,
    candidates: torch.Tensor,
    initial: torch.Tensor,
    *,
    rollout_s: float,
    warmup_s: float,
    objective_config: ObjectiveConfig,
    latent_reg: float,
    fall_grace_s: float,
) -> tuple[torch.Tensor, list[RolloutMetrics]]:
    core = wrapped_env._env
    dt = float(core.dt)
    rollout_steps = max(1, int(round(rollout_s / dt)))
    warmup_steps = min(rollout_steps - 1, max(0, int(round(warmup_s / dt))))
    scoring_steps = max(1, rollout_steps - warmup_steps)
    target_states = _default_standing_target_states(wrapped_env, device=core.device)
    observation, _ = wrapped_env.reset(to_numpy=False, target_states=target_states)

    population = candidates.shape[0]
    alive = torch.ones(population, dtype=torch.bool, device=candidates.device)
    objective_sum = torch.zeros(population, device=candidates.device)
    metric_accumulator = MetricAccumulator(population, device=candidates.device)
    survival_steps = torch.zeros(population, device=candidates.device)
    fall_steps = torch.zeros(population, dtype=torch.long, device=candidates.device)
    fall_grace_steps = max(1, int(round(fall_grace_s / dt)))

    for step in trange(rollout_steps, desc="rollout", leave=False):
        action = model.act(observation, candidates)
        observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)
        done = torch.as_tensor(terminated, device=candidates.device).bool() | torch.as_tensor(
            truncated, device=candidates.device
        ).bool()
        signals = StepSignals(
            body_vx=core.base_lin_vel[:, 0],
            body_vy=core.base_lin_vel[:, 1],
            body_yaw_rate=core.base_ang_vel[:, 2],
            upright=torch.clamp(-core.projected_gravity[:, 2], 0.0, 1.0),
            root_height=core.robot_root_states[:, 2],
            left_foot_flatness_error=_contact_foot_flatness_error(core, 0),
            right_foot_flatness_error=_contact_foot_flatness_error(core, 1),
        )
        fallen_now = compute_failure(signals, objective_config)
        fall_steps = torch.where(fallen_now & alive, fall_steps + 1, torch.zeros_like(fall_steps))
        failed = done | (fall_steps >= fall_grace_steps)
        # A done environment has already been reset by the wrapper.  Exclude
        # that reset state.  MJLab has no fall termination in this config, so
        # also stop scoring candidates that remain tipped/low for the grace.
        valid = alive & ~failed
        survival_steps += valid.float()
        if step >= warmup_steps:
            instant_objective = compute_instant_objective(signals, objective_config)
            weight = valid.float()
            objective_sum += instant_objective * weight
            metric_accumulator.update(signals, objective_config, valid)
        alive &= ~failed

    # MJLab's current Roban config only terminates on timeout.  Make staying
    # upright for the whole horizon an explicit part of the optimization.
    survival_fraction = survival_steps / float(rollout_steps)
    raw_objective = 0.8 * objective_sum / float(scoring_steps) + 0.2 * survival_fraction
    initial_unit = torch.nn.functional.normalize(initial, dim=-1)
    candidate_unit = torch.nn.functional.normalize(candidates, dim=-1)
    cosine = (candidate_unit * initial_unit).sum(dim=-1).clamp(-1.0, 1.0)
    scores = raw_objective - float(latent_reg) * (1.0 - cosine)
    finalized_metrics = metric_accumulator.finalize()

    metrics = []
    for index in range(population):
        metrics.append(
            RolloutMetrics(
                score=float(scores[index].cpu()),
                objective_without_latent_reg=float(raw_objective[index].cpu()),
                survival_s=float((survival_steps[index] * dt).cpu()),
                latent_cosine_to_initial=float(cosine[index].cpu()),
                step_metrics={name: float(values[index].cpu()) for name, values in finalized_metrics.items()},
            )
        )
    return scores, metrics


def save_latents(output_dir: Path, best: torch.Tensor, *, rollout_frames: int) -> None:
    best_np = best.detach().cpu().numpy().astype(np.float32)
    np.save(output_dir / "best_z.npy", best_np)
    np.save(output_dir / "best_z_rollout.npy", np.repeat(best_np, int(rollout_frames), axis=0))


@torch.inference_mode()
def rollout_qpos(wrapped_env, model, z: torch.Tensor, *, rollout_s: float) -> list[np.ndarray]:
    core = wrapped_env._env
    observation, _ = wrapped_env.reset(
        to_numpy=False,
        target_states=_default_standing_target_states(wrapped_env, device=core.device),
    )
    qposes = [policy_qpos_from_env(wrapped_env, expected_qpos_size=7 + core.num_dof)]
    for _ in range(max(1, int(round(rollout_s / float(core.dt))))):
        action = model.act(observation, z)
        observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)
        qposes.append(policy_qpos_from_env(wrapped_env, expected_qpos_size=7 + core.num_dof))
        if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
            break
    return qposes


def render_comparison(
    env_cfg,
    model,
    initial: torch.Tensor,
    best: torch.Tensor,
    *,
    robot_xml: Path,
    rollout_s: float,
    fps: int,
    render_size: int,
    output_path: Path,
) -> None:
    wrapped_env, _ = env_cfg.build(num_envs=1)
    renderer = None
    try:
        baseline_qpos = rollout_qpos(wrapped_env, model, initial, rollout_s=rollout_s)
        best_qpos = rollout_qpos(wrapped_env, model, best, rollout_s=rollout_s)
        renderer = MujocoQposRenderer(
            robot_xml,
            render_size=render_size,
            expected_qpos_size=baseline_qpos[0].size,
        )
        frame_count = max(len(baseline_qpos), len(best_qpos))
        frames = []
        for index in trange(frame_count, desc="render comparison", leave=False):
            left = renderer.render_qpos(baseline_qpos[min(index, len(baseline_qpos) - 1)])
            right = renderer.render_qpos(best_qpos[min(index, len(best_qpos) - 1)])
            separator = np.full((left.shape[0], 4, 3), 255, dtype=np.uint8)
            frames.append(np.concatenate([left, separator, right], axis=1))
        media.write_video(str(output_path), frames, fps=fps)
    finally:
        if renderer is not None:
            renderer.close()
        wrapped_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_MODEL_FOLDER)
    parser.add_argument("--actor-onnx", type=Path, default=None, help="Deployed Actor ONNX; defaults below --model-folder.")
    parser.add_argument("--onnx-provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--initial-latent", type=Path, default=None)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--elite-fraction", type=float, default=0.125)
    parser.add_argument("--sigma", type=float, default=0.20, help="Initial per-dimension tangent standard deviation.")
    parser.add_argument(
        "--covariance-alpha",
        type=float,
        default=0.70,
        help="Fraction of the previous diagonal covariance retained after fitting elites.",
    )
    parser.add_argument("--min-sigma", type=float, default=0.03)
    parser.add_argument("--max-sigma", type=float, default=0.50)
    parser.add_argument("--rollout-s", type=float, default=5.0)
    parser.add_argument("--warmup-s", type=float, default=1.0)
    parser.add_argument("--target-forward-speed", type=float, default=0.0)
    parser.add_argument("--target-lateral-speed", type=float, default=0.5)
    parser.add_argument("--velocity-sigma", type=float, default=0.20)
    parser.add_argument("--yaw-rate-sigma", type=float, default=0.75)
    parser.add_argument(
        "--foot-flatness-sigma",
        type=float,
        default=0.20,
        help="Scale for contact-gated feet-orientation error; 0.20 is about 11.5 degrees for one foot.",
    )
    parser.add_argument("--velocity-weight", type=float, default=0.75)
    parser.add_argument("--yaw-weight", type=float, default=0.15)
    parser.add_argument("--foot-flatness-weight", type=float, default=0.10)
    parser.add_argument("--latent-reg", type=float, default=0.05)
    parser.add_argument("--min-upright", type=float, default=0.5)
    parser.add_argument("--min-root-height", type=float, default=0.5)
    parser.add_argument("--fall-grace-s", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rollout-frames", type=int, default=5000)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard-log-dir", type=Path, default=None)
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--fps", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.population < 4:
        raise ValueError("--population must be at least 4")
    if not 0.0 < args.elite_fraction <= 0.5:
        raise ValueError("--elite-fraction must be in (0, 0.5]")
    if args.warmup_s >= args.rollout_s:
        raise ValueError("--warmup-s must be smaller than --rollout-s")
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    if not 0.0 <= args.covariance_alpha < 1.0:
        raise ValueError("--covariance-alpha must be in [0, 1)")
    if not 0.0 < args.min_sigma <= args.sigma <= args.max_sigma:
        raise ValueError("sigmas must satisfy 0 < min-sigma <= sigma <= max-sigma")
    if args.rollout_frames < 1:
        raise ValueError("--rollout-frames must be at least 1")
    if args.velocity_sigma <= 0.0 or args.yaw_rate_sigma <= 0.0 or args.foot_flatness_sigma <= 0.0:
        raise ValueError("score sigmas must be positive")
    if not math.isfinite(args.target_forward_speed) or not math.isfinite(args.target_lateral_speed):
        raise ValueError("target planar speeds must be finite")
    objective_weights = {
        "velocity_tracking": float(args.velocity_weight),
        "yaw_stability": float(args.yaw_weight),
        "feet_flatness": float(args.foot_flatness_weight),
    }
    if any(weight < 0.0 for weight in objective_weights.values()):
        raise ValueError("objective weights must be non-negative")
    if not math.isclose(sum(objective_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"objective weights must sum to 1.0, got {objective_weights}")

    model_folder = args.model_folder.expanduser().resolve()
    robot_config = resolve_inference_robot_config(args.robot_config, None)
    initial_path = args.initial_latent
    if initial_path is None:
        initial_path = model_folder / "reward_inference" / f"{args.task}.npy"
    initial_path = initial_path.expanduser().resolve()
    if not initial_path.exists():
        raise FileNotFoundError(f"Missing initial latent: {initial_path}")
    expected_velocity = planar_velocity_from_move_ego_task(args.task)
    if expected_velocity is not None:
        configured_velocity = (float(args.target_forward_speed), float(args.target_lateral_speed))
        if not all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(expected_velocity, configured_velocity)):
            print(
                f"[latent-search] WARNING task {args.task!r} implies target velocity "
                f"({expected_velocity[0]:.3f}, {expected_velocity[1]:.3f}) m/s, but configured target is "
                f"({configured_velocity[0]:.3f}, {configured_velocity[1]:.3f}) m/s",
                flush=True,
            )
    if initial_path.stem != args.task:
        print(
            f"[latent-search] WARNING task name {args.task!r} differs from initial latent "
            f"task {initial_path.stem!r}; output naming follows --task",
            flush=True,
        )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or Path(__file__).resolve().parent / "outputs" / f"{args.task}_{timestamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    actor_onnx = (args.actor_onnx or model_folder / "exported/FBcprAuxModel.onnx").expanduser().resolve()
    if not actor_onnx.exists():
        raise FileNotFoundError(f"Missing deployed Actor ONNX: {actor_onnx}")
    model = OnnxActor(actor_onnx, args.onnx_provider)
    z_dim = 256
    norm_z = True
    print(f"[latent-search] actor={actor_onnx} provider={args.onnx_provider}")
    initial = load_initial_latent(initial_path, z_dim=z_dim, device=args.device)
    if norm_z:
        latent_norm = math.sqrt(z_dim)
        initial = project_latents(initial, latent_norm)
    else:
        latent_norm = float(initial.norm())
    initial_np = initial.detach().cpu().numpy().astype(np.float32)
    np.save(output_dir / "baseline_z.npy", initial_np)
    np.save(output_dir / "baseline_z_rollout.npy", np.repeat(initial_np, int(args.rollout_frames), axis=0))

    env_cfg, _ = load_mjlab_env_cfg(
        model_folder,
        data_path=args.data_path,
        robot_config=robot_config,
        device=args.device,
        headless=True,
        disable_dr=True,
        disable_obs_noise=True,
        max_episode_length_s=max(args.rollout_s + 1.0, 10.0),
    )
    wrapped_env, _ = env_cfg.build(num_envs=args.population)
    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)
    mean = initial.clone()
    search_std = torch.full_like(mean, float(args.sigma))
    elite_count = max(2, int(math.ceil(args.population * args.elite_fraction)))
    global_best = initial.clone()
    global_best_metrics: RolloutMetrics | None = None
    history: list[dict] = []
    objective_config = ObjectiveConfig(
        target_forward_speed=float(args.target_forward_speed),
        target_lateral_speed=float(args.target_lateral_speed),
        velocity_sigma=float(args.velocity_sigma),
        yaw_rate_sigma=float(args.yaw_rate_sigma),
        foot_flatness_sigma=float(args.foot_flatness_sigma),
        min_upright=float(args.min_upright),
        min_root_height=float(args.min_root_height),
        reward_weights=objective_weights,
    )
    tensorboard_log_dir = (args.tensorboard_log_dir or output_dir / "tensorboard").expanduser().resolve()
    tb_logger = LatentSearchTensorboardLogger(
        tensorboard_log_dir,
        enabled=args.tensorboard,
        config=vars(args),
        objective_text=(
            f"{OBJECTIVE_DESCRIPTION}\n\nConfigured reward weights: "
            f"{json.dumps(objective_weights, sort_keys=True)}"
        ),
    )

    print(
        f"[latent-search] task={args.task} target body velocity="
        f"({args.target_forward_speed:.3f}, {args.target_lateral_speed:.3f}) m/s"
    )
    print(f"[latent-search] objective weights={objective_weights}")
    print(f"[latent-search] initial={initial_path} norm={float(initial.norm()):.6f}")
    print(
        f"[latent-search] population={args.population} iterations={args.iterations} "
        f"rollout={args.rollout_s:.2f}s warmup={args.warmup_s:.2f}s device={args.device}"
    )
    try:
        for iteration in range(args.iterations):
            candidates = sample_candidates(
                mean,
                initial,
                population=args.population,
                std=search_std,
                latent_norm=latent_norm,
                generator=generator,
            )
            scores, metrics = evaluate_candidates(
                wrapped_env,
                model,
                candidates,
                initial,
                rollout_s=args.rollout_s,
                warmup_s=args.warmup_s,
                objective_config=objective_config,
                latent_reg=args.latent_reg,
                fall_grace_s=args.fall_grace_s,
            )
            order = torch.argsort(scores, descending=True)
            best_index = int(order[0])
            elite_indices = order[:elite_count]
            elite = candidates[elite_indices]
            mean, search_std = update_diagonal_cem(
                elite,
                search_std,
                latent_norm=latent_norm,
                covariance_alpha=float(args.covariance_alpha),
                min_sigma=float(args.min_sigma),
                max_sigma=float(args.max_sigma),
            )
            iteration_best = metrics[best_index]
            if global_best_metrics is None or iteration_best.score > global_best_metrics.score:
                global_best = candidates[best_index : best_index + 1].clone()
                global_best_metrics = iteration_best
                save_latents(output_dir, global_best, rollout_frames=args.rollout_frames)
            record = {
                "iteration": iteration,
                "search_std_mean": float(search_std.mean().cpu()),
                "search_std_min": float(search_std.min().cpu()),
                "search_std_max": float(search_std.max().cpu()),
                "best_candidate_index": best_index,
                "best": iteration_best.to_dict(),
                "baseline": metrics[0].to_dict(),
                "population_score_mean": float(scores.mean().cpu()),
                "population_score_std": float(scores.std().cpu()),
            }
            history.append(record)
            (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n")
            tb_logger.log_iteration(
                iteration=iteration,
                search_std=search_std,
                scores=scores,
                candidates=candidates,
                metrics=metrics,
                best_index=best_index,
                global_best_score=global_best_metrics.score,
            )
            print(
                f"[latent-search] iter={iteration + 1}/{args.iterations} "
                f"std(mean/min/max)={float(search_std.mean()):.4f}/"
                f"{float(search_std.min()):.4f}/{float(search_std.max()):.4f} "
                f"best={iteration_best.score:.4f} baseline={metrics[0].score:.4f} "
                f"vy={iteration_best.mean_body_vy:.3f} vx={iteration_best.mean_body_vx:.3f} "
                f"planar_err={iteration_best.rms_planar_velocity_error:.3f} "
                f"survive={iteration_best.survival_s:.2f}s"
            )
    finally:
        tb_logger.close()
        wrapped_env.close()

    assert global_best_metrics is not None
    search_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    search_args.update({"initial_latent": str(initial_path), "output_dir": str(output_dir)})
    summary = {
        "task": args.task,
        "target_forward_speed": args.target_forward_speed,
        "target_lateral_speed": args.target_lateral_speed,
        "initial_latent": str(initial_path),
        "model_folder": str(model_folder),
        "actor_onnx": str(actor_onnx),
        "robot_config": str(robot_config),
        "output_dir": str(output_dir),
        "best": global_best_metrics.to_dict(),
        "objective_weights": objective_weights,
        "objective_registry": registry_snapshot(),
        "search_args": search_args,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    if args.save_video:
        robot_xml = Path(load_robot_training_spec(robot_config).robot.xml_path).expanduser().resolve()
        comparison_path = output_dir / "comparison_baseline_left_best_right.mp4"
        render_comparison(
            env_cfg,
            model,
            initial,
            global_best,
            robot_xml=robot_xml,
            rollout_s=args.rollout_s,
            fps=args.fps,
            render_size=args.render_size,
            output_path=comparison_path,
        )
        print(f"[latent-search] comparison video={comparison_path}")

    print(f"[latent-search] best_z={output_dir / 'best_z.npy'}")
    print(f"[latent-search] onnx latent={output_dir / 'best_z_rollout.npy'}")
    print(f"[latent-search] summary={output_dir / 'summary.json'}")
    if args.tensorboard:
        print(f"[latent-search] tensorboard={tensorboard_log_dir}")


if __name__ == "__main__":
    main()
