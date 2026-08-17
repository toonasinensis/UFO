"""DIAL-style offline optimization of a complete BFM tracking latent sequence."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mediapy as media
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import trange

from humanoidverse.envs.motion_observations import (
    compute_humanoid_observations_max,
    compute_humanoid_observations_max_with_contact,
)
from humanoidverse.generate_onnx_latent import _prepare_motion_input, _train_aligned_latents
from humanoidverse.mjlab_inference_utils import (
    MujocoQposRenderer,
    load_mjlab_env_cfg,
    policy_qpos_from_env,
)
from humanoidverse.utils.reference_observations import reference_base_ang_vel
from humanoidverse.utils.robot_spec import load_robot_training_spec
from humanoidverse.utils.torch_utils import quat_rotate_inverse

from .objective_registry import (
    TrackingConfig,
    TrackingSignals,
    compute_metrics,
    compute_rewards,
    joint_limit_signals,
    metric_reductions,
    registry_snapshot,
    reward_reductions,
)
from .tensorboard_logger import AdaptationTensorboardLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_FOLDER = PROJECT_ROOT / "runs/新数据addlelay_onnx_153m_20260807"
DEFAULT_MOTION = PROJECT_ROOT / "humanoidverse/data/roban/named_roban_lafan_10s/dance1_subject2_0001.npz"
DEFAULT_ROBOT_CONFIG = PROJECT_ROOT / "configs/robots/roban_s22.yaml"
LATENT_DIM = 256
LATENT_NORM = math.sqrt(LATENT_DIM)
PROGRESS_ENABLED = True


@dataclass
class RolloutResult:
    objective: torch.Tensor
    reward_means: dict[str, torch.Tensor]
    metric_means: dict[str, torch.Tensor]
    timeseries: dict[str, np.ndarray] | None = None


class OnnxActor:
    def __init__(self, path: Path, provider: str):
        available = set(ort.get_available_providers())
        providers = ["CPUExecutionProvider"]
        if provider == "cuda" and "CUDAExecutionProvider" in available:
            try:
                ort.preload_dlls()
            except AttributeError:
                pass
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(str(path), providers=providers)
        actor_input = self.session.get_inputs()[0]
        if actor_input.name != "actor_obs" or actor_input.shape[-1] not in (601, "601"):
            raise ValueError(f"Expected actor_obs [batch,601], got {actor_input.name} {actor_input.shape}")
        if "action" not in {item.name for item in self.session.get_outputs()}:
            raise ValueError("Actor ONNX does not expose output 'action'")
        print(f"[adapt] actor provider={self.session.get_providers()[0]}")

    def act(self, observation: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        actor_obs = torch.cat(
            [observation["state"], observation["last_action"], observation["history_actor"], z], dim=-1
        )
        values = self.session.run(
            ["action"], {"actor_obs": actor_obs.detach().cpu().numpy().astype(np.float32, copy=False)}
        )[0]
        return torch.from_numpy(values).to(device=z.device, dtype=torch.float32)


def project_latents(z: torch.Tensor, norm: float = LATENT_NORM) -> torch.Tensor:
    if z.ndim < 2 or z.shape[-1] != LATENT_DIM:
        raise ValueError(f"Expected latent last dimension {LATENT_DIM}, got {tuple(z.shape)}")
    if not torch.isfinite(z).all():
        raise ValueError("Latent contains NaN/Inf")
    return F.normalize(z, dim=-1) * float(norm)


def dial_sigma_schedule(
    *, iteration: int, iterations: int, horizon: int, sigma0: float,
    beta_iteration: float, beta_horizon: float, device: str | torch.device,
) -> torch.Tensor:
    if not (0 <= iteration < iterations) or iterations < 1 or horizon < 1:
        raise ValueError("Invalid iteration/horizon for DIAL schedule")
    if min(sigma0, beta_iteration, beta_horizon) <= 0.0:
        raise ValueError("DIAL sigma and beta values must be positive")
    h = torch.arange(horizon, device=device, dtype=torch.float32)
    exponent = -float(iteration) / (float(beta_iteration) * iterations)
    exponent = exponent - (horizon - 1.0 - h) / (float(beta_horizon) * horizon)
    # The paper schedule is a covariance schedule. Sampling needs standard
    # deviation, hence sqrt(sigma0^2 * exp(exponent)).
    return float(sigma0) * torch.exp(0.5 * exponent)


def mppi_weights(
    scores: torch.Tensor,
    temperature: float,
    eligible: torch.Tensor | None = None,
) -> torch.Tensor:
    if scores.ndim != 1 or not torch.isfinite(scores).all():
        raise ValueError("MPPI scores must be a finite vector")
    if temperature <= 0.0:
        raise ValueError("MPPI temperature must be positive")
    normalized = (scores - scores.mean()) / scores.std(unbiased=False).clamp_min(1.0e-6)
    logits = normalized / float(temperature)
    if eligible is None:
        return torch.softmax(logits, dim=0)
    if eligible.shape != scores.shape or eligible.dtype != torch.bool:
        raise ValueError("MPPI eligibility mask must be a matching boolean vector")
    if not bool(eligible.any()):
        return torch.zeros_like(scores)
    masked_logits = torch.where(eligible, logits, torch.full_like(logits, -torch.inf))
    return torch.softmax(masked_logits, dim=0)


def objective_link_mppi_weights(
    objective_scores: torch.Tensor,
    link_errors: torch.Tensor,
    *,
    temperature: float,
    link_eligible: torch.Tensor,
    blend_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Blend objective and guarded direct-link MPPI weight distributions.

    The returned tuple is ``(mixed, objective, guarded_link, raw_link)``.  A
    missing guarded-link arm falls back to the objective arm, so the optimizer
    always retains a normalized update distribution.
    """

    if objective_scores.shape != link_errors.shape:
        raise ValueError("Objective scores and link errors must have matching shapes")
    if not math.isfinite(blend_alpha) or not 0.0 <= blend_alpha <= 1.0:
        raise ValueError("MPPI link blend alpha must be finite and in [0, 1]")
    objective_weights = mppi_weights(objective_scores, temperature)
    if blend_alpha == 0.0:
        zeros = torch.zeros_like(objective_weights)
        return objective_weights, objective_weights, zeros, zeros

    link_scores = -link_errors
    raw_link_weights = mppi_weights(link_scores, temperature)
    link_weights = mppi_weights(
        link_scores,
        temperature,
        eligible=link_eligible,
    )
    if not bool(link_eligible.any()):
        return objective_weights, objective_weights, link_weights, raw_link_weights
    mixed_weights = (
        (1.0 - blend_alpha) * objective_weights
        + blend_alpha * link_weights
    )
    return mixed_weights, objective_weights, link_weights, raw_link_weights


def mppi_link_arm_diagnostics(
    weights: torch.Tensor,
    *,
    baseline_feasible: torch.Tensor,
    link_eligible: torch.Tensor,
    score_mode: str,
    configured_blend_alpha: float,
    update_skipped: bool,
) -> dict[str, float | bool]:
    """Describe the link arm without changing the MPPI update distribution.

    ``effective_alpha`` is a blend-only quantity: pure-link updates therefore
    report zero even though their link arm is active.  For an objective/link
    blend, an empty link eligibility mask means the objective fallback was
    used and the effective alpha is also zero.
    """

    if weights.ndim != 1 or not torch.isfinite(weights).all():
        raise ValueError("MPPI diagnostic weights must be a finite vector")
    for name, mask in (
        ("baseline feasibility", baseline_feasible),
        ("link eligibility", link_eligible),
    ):
        if mask.shape != weights.shape or mask.dtype != torch.bool:
            raise ValueError(f"MPPI {name} mask must be a matching boolean vector")
    if score_mode not in {"objective", "link"}:
        raise ValueError(f"Unsupported MPPI score mode {score_mode!r}")
    if not math.isfinite(configured_blend_alpha) or not (
        0.0 <= configured_blend_alpha <= 1.0
    ):
        raise ValueError("MPPI configured blend alpha must be finite and in [0, 1]")

    has_link_eligible = bool(link_eligible.any())
    if score_mode == "objective":
        link_arm_active = configured_blend_alpha > 0.0 and has_link_eligible
        effective_alpha = configured_blend_alpha if link_arm_active else 0.0
    else:
        link_arm_active = has_link_eligible and not update_skipped
        effective_alpha = 0.0

    return {
        "link_arm_eligible_ratio": float(link_eligible.float().mean().cpu()),
        "mixed_feasible_mass": float(weights[baseline_feasible].sum().cpu()),
        "link_blend_effective_alpha": float(effective_alpha),
        "link_arm_active": bool(link_arm_active),
    }


def mppi_diagnostics(scores: torch.Tensor, weights: torch.Tensor) -> dict[str, float]:
    if scores.ndim != 1 or weights.shape != scores.shape:
        raise ValueError("MPPI scores and weights must be matching vectors")
    if not torch.isfinite(scores).all() or not torch.isfinite(weights).all():
        raise ValueError("MPPI scores and weights must be finite")
    weight_sum = weights.sum()
    if torch.isclose(weight_sum, torch.zeros_like(weight_sum), atol=1.0e-8):
        quantiles = torch.quantile(
            scores, torch.tensor([0.1, 0.5, 0.9], device=scores.device)
        )
        return {
            "effective_sample_size": 0.0,
            "ess_ratio": 0.0,
            "weight_entropy": 0.0,
            "normalized_weight_entropy": 0.0,
            "max_weight": 0.0,
            "score_p10": float(quantiles[0].cpu()),
            "score_p50": float(quantiles[1].cpu()),
            "score_p90": float(quantiles[2].cpu()),
        }
    if not torch.isclose(weight_sum, torch.ones_like(weight_sum), atol=1.0e-5):
        raise ValueError(f"MPPI weights must sum to one, got {float(weight_sum)}")
    effective_sample_size = torch.reciprocal(torch.square(weights).sum().clamp_min(1.0e-12))
    entropy = -(weights * torch.log(weights.clamp_min(1.0e-12))).sum()
    quantiles = torch.quantile(scores, torch.tensor([0.1, 0.5, 0.9], device=scores.device))
    return {
        "effective_sample_size": float(effective_sample_size.cpu()),
        "ess_ratio": float((effective_sample_size / scores.numel()).cpu()),
        "weight_entropy": float(entropy.cpu()),
        "normalized_weight_entropy": float((entropy / math.log(scores.numel())).cpu()),
        "max_weight": float(weights.max().cpu()),
        "score_p10": float(quantiles[0].cpu()),
        "score_p50": float(quantiles[1].cpu()),
        "score_p90": float(quantiles[2].cpu()),
    }


def sample_candidates(
    mean: torch.Tensor,
    baseline: torch.Tensor,
    *,
    population: int,
    sigma: torch.Tensor,
    smoothing_window: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if mean.shape != baseline.shape or mean.ndim != 2 or mean.shape[1] != LATENT_DIM:
        raise ValueError(f"Expected matching [H,{LATENT_DIM}] mean/baseline")
    horizon = mean.shape[0]
    if sigma.shape != (horizon,):
        raise ValueError(f"Expected sigma [{horizon}], got {tuple(sigma.shape)}")
    if population < 2:
        raise ValueError("Population must be at least 2")
    noise = torch.randn(
        (population, horizon, LATENT_DIM), device=mean.device, dtype=mean.dtype, generator=generator
    )
    if smoothing_window > 1:
        if smoothing_window % 2 == 0:
            raise ValueError("Temporal smoothing window must be odd")
        flat = noise.permute(0, 2, 1).reshape(-1, 1, horizon)
        flat = F.avg_pool1d(flat, smoothing_window, stride=1, padding=smoothing_window // 2)
        noise = flat.reshape(population, LATENT_DIM, horizon).permute(0, 2, 1)
        noise = noise * math.sqrt(smoothing_window)
    mean_unit = F.normalize(mean, dim=-1).unsqueeze(0)
    noise = noise - (noise * mean_unit).sum(dim=-1, keepdim=True) * mean_unit
    noise = noise * sigma.view(1, horizon, 1)
    candidates = project_latents(mean.unsqueeze(0) + noise)
    candidates[0] = baseline
    candidates[1] = mean
    return candidates


def update_mppi_mean(mean: torch.Tensor, candidates: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if candidates.shape[1:] != mean.shape or weights.shape != (candidates.shape[0],):
        raise ValueError("Incompatible mean/candidates/weights shapes")
    residual = candidates - mean.unsqueeze(0)
    mean_unit = F.normalize(mean, dim=-1).unsqueeze(0)
    tangent = residual - (residual * mean_unit).sum(dim=-1, keepdim=True) * mean_unit
    update = torch.sum(weights[:, None, None] * tangent, dim=0)
    return project_latents(mean + update)


def quaternion_geodesic(q: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    reference = F.normalize(reference, dim=-1)
    dot = torch.abs(torch.sum(q * reference, dim=-1)).clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def root_frame_vectors(vectors: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
    shape = vectors.shape
    roots = root_quat.unsqueeze(-2).expand(*shape[:-2], shape[-2], 4)
    return quat_rotate_inverse(roots.reshape(-1, 4), vectors.reshape(-1, 3), w_last=True).reshape(shape)


def tracking_signals(
    core: Any,
    reference: dict[str, torch.Tensor],
    step: int,
    config: TrackingConfig,
) -> TrackingSignals:
    body_pos, body_rot, body_vel, body_ang_vel = core._extend_body_state()
    ref_pos = reference["ref_body_pos"][step]
    ref_rot = reference["ref_body_rots"][step]
    ref_vel = reference["ref_body_vels"][step]
    ref_ang_vel = reference["ref_body_angular_vels"][step]
    ref_dof = reference["dof_pos"][step]
    current_relative = root_frame_vectors(body_pos - body_pos[:, :1], body_rot[:, 0])
    ref_relative = root_frame_vectors(ref_pos - ref_pos[:, :1], ref_rot[:, 0])
    (
        joint_limit_error,
        joint_limit_utilization,
        hard_limit_contact,
        soft_limit_contact,
    ) = joint_limit_signals(
        core.dof_pos,
        core.hard_dof_pos_limits,
        core.dof_pos_limits,
        safe_fraction=config.joint_limit_safe_fraction,
    )
    return TrackingSignals(
        root_position_error=torch.linalg.vector_norm(body_pos[:, 0] - ref_pos[:, 0], dim=-1),
        root_rotation_error=quaternion_geodesic(body_rot[:, 0], ref_rot[:, 0]),
        body_position_error=torch.linalg.vector_norm(current_relative - ref_relative, dim=-1).mean(dim=-1),
        body_rotation_error=quaternion_geodesic(body_rot, ref_rot).mean(dim=-1),
        joint_position_error=torch.abs(core.dof_pos - ref_dof).mean(dim=-1),
        body_linear_velocity_error=torch.linalg.vector_norm(body_vel - ref_vel, dim=-1).mean(dim=-1),
        body_angular_velocity_error=torch.linalg.vector_norm(body_ang_vel - ref_ang_vel, dim=-1).mean(dim=-1),
        global_body_position_error=torch.linalg.vector_norm(body_pos - ref_pos, dim=-1).mean(dim=-1),
        joint_limit_error=joint_limit_error,
        joint_limit_utilization=joint_limit_utilization,
        hard_limit_contact=hard_limit_contact,
        soft_limit_contact=soft_limit_contact,
    )


def _full_reference_observation(
    env: Any,
    frame_count: int,
    use_root_height_obs: bool,
    *,
    start_frame: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    times = (
        start_frame
        + torch.arange(frame_count, device=env.device, dtype=torch.float32)
    ) * float(env.dt)
    motion_ids = torch.zeros(frame_count, device=env.device, dtype=torch.long)
    motion_state = env._motion_lib.get_motion_state(motion_ids, times)
    pos = motion_state["rg_pos_t"]
    rot = motion_state["rg_rot_t"]
    vel = motion_state["body_vel_t"]
    ang_vel = motion_state["body_ang_vel_t"]
    dof_pos_relative = motion_state["dof_pos"] - env.default_dof_pos[0]
    dof_vel = motion_state["dof_vel"]
    if env.use_contact_in_obs_max:
        contact = env.foot_contact_detect(pos, vel)
        obs_parts = compute_humanoid_observations_max_with_contact(
            pos, rot, vel, ang_vel, local_root_obs=True,
            root_height_obs=use_root_height_obs, contact_binary=contact,
        )
    else:
        obs_parts = compute_humanoid_observations_max(
            pos, rot, vel, ang_vel, local_root_obs=True, root_height_obs=use_root_height_obs,
        )
    privileged = torch.cat(list(obs_parts.values()), dim=-1)
    root_ang_vel = reference_base_ang_vel(env, rot[:, 0], ang_vel[:, 0])
    gravity = quat_rotate_inverse(
        rot[:, 0], env.gravity_vec[:1].repeat(frame_count, 1), w_last=True
    )
    state = torch.cat([dof_pos_relative, dof_vel, gravity, root_ang_vel], dim=-1)
    backward = {"state": state, "privileged_state": privileged}
    reference = {
        "ref_body_pos": pos,
        "ref_body_rots": rot,
        "ref_body_vels": vel,
        "ref_body_angular_vels": ang_vel,
        "dof_pos": motion_state["dof_pos"],
        "dof_vel": dof_vel,
    }
    return backward, reference


def _expand_reference(reference: dict[str, torch.Tensor], origins: torch.Tensor) -> dict[str, torch.Tensor]:
    population = origins.shape[0]
    expanded: dict[str, torch.Tensor] = {}
    for name, values in reference.items():
        values = values[:, None].expand(values.shape[0], population, *values.shape[1:]).clone()
        expanded[name] = values
    return expanded


def generate_baseline(
    env: Any, *, frame_count: int, backward_onnx: Path, provider: str,
    use_root_height_obs: bool, seq_length: int, start_frame: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    backward, reference = _full_reference_observation(
        env,
        frame_count,
        use_root_height_obs,
        start_frame=start_frame,
    )
    providers = ["CPUExecutionProvider"]
    if provider == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            ort.preload_dlls()
        except AttributeError:
            pass
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(str(backward_onnx), providers=providers)
    inputs = {
        item.name: backward[item.name][1:].detach().cpu().numpy().astype(np.float32)
        for item in session.get_inputs()
    }
    frame_z = session.run(["z"], inputs)[0]
    z = _train_aligned_latents(frame_z, seq_length)
    if z.shape != (frame_count - 1, LATENT_DIM):
        raise ValueError(f"Expected baseline [{frame_count - 1},{LATENT_DIM}], got {z.shape}")
    return torch.from_numpy(z).to(env.device), reference


def initial_target_states(reference: dict[str, torch.Tensor], core: Any) -> dict[str, torch.Tensor]:
    root_state = torch.cat(
        [
            reference["ref_body_pos"][0, :, 0],
            reference["ref_body_rots"][0, :, 0],
            reference["ref_body_vels"][0, :, 0],
            reference["ref_body_angular_vels"][0, :, 0],
        ],
        dim=-1,
    ).to(device=core.device, dtype=torch.float32)
    dof_state = torch.stack(
        [reference["dof_pos"][0], reference["dof_vel"][0]], dim=-1
    ).to(device=core.device, dtype=torch.float32)
    return {"root_states": root_state, "dof_states": dof_state}


@torch.inference_mode()
def evaluate_sequences(
    wrapped_env: Any, actor: OnnxActor, sequences: torch.Tensor,
    reference_single: dict[str, torch.Tensor], config: TrackingConfig,
    *, collect_timeseries: bool = False,
) -> RolloutResult:
    core = wrapped_env._env
    population, horizon, latent_dim = sequences.shape
    if population != core.num_envs or latent_dim != LATENT_DIM:
        raise ValueError(f"Sequence/env mismatch: sequences={sequences.shape}, envs={core.num_envs}")
    reference = _expand_reference(reference_single, core.env_origins)
    target = initial_target_states(reference, core)
    observation, _ = wrapped_env.reset(to_numpy=False, target_states=target)
    reward_reduction_modes = reward_reductions()
    reward_accumulators = {
        name: (
            torch.zeros(population, device=core.device)
            if reward_reduction_modes[name] == "mean"
            else torch.full((population,), torch.inf, device=core.device)
        )
        for name in config.weights
    }
    metric_accumulators: dict[str, torch.Tensor] | None = None
    reductions = metric_reductions()
    time_values: dict[str, list[np.ndarray]] = {}
    for step in trange(
        horizon, desc="rollout", leave=False, disable=not PROGRESS_ENABLED
    ):
        action = actor.act(observation, sequences[:, step])
        observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)
        if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
            raise RuntimeError(f"Unexpected episode termination at tracking step {step}")
        signals = tracking_signals(core, reference, step + 1, config)
        _instant, rewards = compute_rewards(signals, config)
        metrics = compute_metrics(signals)
        for name, values in rewards.items():
            if reward_reduction_modes[name] == "mean":
                reward_accumulators[name] += values
            else:
                reward_accumulators[name] = torch.minimum(
                    reward_accumulators[name], values
                )
        if metric_accumulators is None:
            metric_accumulators = {
                name: (
                    torch.zeros_like(values)
                    if reductions[name] == "mean"
                    else torch.full_like(values, -torch.inf)
                )
                for name, values in metrics.items()
            }
        for name, values in metrics.items():
            if reductions[name] == "mean":
                metric_accumulators[name] += values
            else:
                metric_accumulators[name] = torch.maximum(
                    metric_accumulators[name], values
                )
        if collect_timeseries:
            for name, values in {**rewards, **metrics}.items():
                time_values.setdefault(name, []).append(values[:1].detach().cpu().numpy())
    assert metric_accumulators is not None
    reward_means = {
        name: (
            values / horizon
            if reward_reduction_modes[name] == "mean"
            else values
        )
        for name, values in reward_accumulators.items()
    }
    metric_means = {
        name: values / horizon if reductions[name] == "mean" else values
        for name, values in metric_accumulators.items()
    }
    objective = sum(config.weights[name] * values for name, values in reward_means.items())
    timeseries = (
        {name: np.concatenate(values, axis=0).astype(np.float32) for name, values in time_values.items()}
        if collect_timeseries else None
    )
    return RolloutResult(objective, reward_means, metric_means, timeseries)


def _one_metrics(result: RolloutResult, index: int = 0) -> tuple[dict[str, float], dict[str, float]]:
    rewards = {name: float(values[index].cpu()) for name, values in result.reward_means.items()}
    metrics = {name: float(values[index].cpu()) for name, values in result.metric_means.items()}
    metrics["objective"] = float(result.objective[index].cpu())
    return rewards, metrics


def _aggregate_metrics(
    result: RolloutResult, indices: slice | torch.Tensor | None = None
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    selected = slice(None) if indices is None else indices
    rewards = {
        name: float(values[selected].mean().cpu()) for name, values in result.reward_means.items()
    }
    metric_tensors = {name: values[selected] for name, values in result.metric_means.items()}
    metric_tensors["objective"] = result.objective[selected]
    reductions = metric_reductions()
    metrics = {
        name: float(
            (
                values.max()
                if reductions.get(name, "mean") == "max"
                else values.mean()
            ).cpu()
        )
        for name, values in metric_tensors.items()
    }
    std = {
        name: float(values.std(unbiased=False).cpu()) for name, values in metric_tensors.items()
    }
    return rewards, metrics, std


def _save_json(path: Path, values: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(values, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, values: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(values, ensure_ascii=False, default=str) + "\n")


def _write_jsonl_atomic(path: Path, values: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, path)


def _last_jsonl_step(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    last: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                last = json.loads(line)
    return None if last is None else int(last["step"])


def _repair_jsonl_tail(path: Path) -> None:
    """Discard only a crash-truncated final JSONL record, if one exists."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as stream:
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) == b"\n":
            return
        cursor = stream.tell() - 1
        while cursor > 0:
            cursor -= 1
            stream.seek(cursor)
            if stream.read(1) == b"\n":
                stream.truncate(cursor + 1)
                return
        stream.truncate(0)


def _atomic_torch_save(path: Path, values: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(values, temporary)
    os.replace(temporary, path)


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values)
    os.replace(temporary, path)


def _validation_bound(
    result: dict[str, Any], metric: str, *, upper: bool, z_score: float = 1.96
) -> float:
    # ``_aggregate_metrics`` deliberately reduces max-type metrics to the
    # worst observed replica.  A confidence interval for a sample *mean* is
    # not valid for that extreme value and can even reject a trajectory whose
    # observed worst case is below the hard safety guard.  Safety gates use
    # the observed maximum directly; mean-type tracking metrics retain the
    # paired normal-approximation bound used by the original experiment.
    if metric_reductions().get(metric, "mean") == "max":
        return float(result["metrics"][metric])
    standard_error = float(result["std"][metric]) / math.sqrt(int(result["count"]))
    direction = 1.0 if upper else -1.0
    return float(result["metrics"][metric]) + direction * z_score * standard_error


@torch.inference_mode()
def _evaluate_variants(
    wrapped_env: Any,
    actor: OnnxActor,
    variants: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    tracking_config: TrackingConfig,
    *,
    rotation: int,
) -> dict[str, dict[str, Any]]:
    """Repeatedly evaluate semantic variants in one balanced environment batch."""
    population = wrapped_env._env.num_envs
    unique_names: list[str] = []
    unique_sequences: list[torch.Tensor] = []
    aliases: dict[str, str] = {}
    for name, sequence in variants.items():
        canonical = next(
            (
                existing_name
                for existing_name, existing_sequence in zip(unique_names, unique_sequences)
                if torch.equal(sequence, existing_sequence)
            ),
            None,
        )
        if canonical is None:
            unique_names.append(name)
            unique_sequences.append(sequence)
            canonical = name
        aliases[name] = canonical
    if len(unique_names) > population:
        raise ValueError(
            f"Validation variants ({len(unique_names)}) exceed population ({population})"
        )
    assignments = (
        torch.arange(population, device=unique_sequences[0].device) + int(rotation)
    ).remainder(len(unique_names))
    batch = torch.empty(
        (population, *unique_sequences[0].shape),
        device=unique_sequences[0].device,
        dtype=unique_sequences[0].dtype,
    )
    canonical_results: dict[str, dict[str, Any]] = {}
    masks: dict[str, torch.Tensor] = {}
    for index, (name, sequence) in enumerate(zip(unique_names, unique_sequences)):
        mask = assignments.eq(index)
        masks[name] = mask
        batch[mask] = sequence
    rollout = evaluate_sequences(wrapped_env, actor, batch, reference, tracking_config)
    for name, mask in masks.items():
        rewards, metrics, std = _aggregate_metrics(rollout, mask)
        canonical_results[name] = {
            "count": int(mask.sum().item()),
            "rewards": rewards,
            "metrics": metrics,
            "std": std,
        }
    return {
        name: {**canonical_results[canonical], "canonical_variant": canonical}
        for name, canonical in aliases.items()
    }


def _latent_stats(z: torch.Tensor, baseline: torch.Tensor) -> tuple[float, float]:
    cosine = F.cosine_similarity(z, baseline, dim=-1).mean()
    delta = torch.linalg.vector_norm(z - baseline, dim=-1).mean()
    return float(cosine.cpu()), float(delta.cpu())


def _control_to_qpos_indices(robot_training: Any) -> np.ndarray:
    names = list(robot_training.robot.control_joint_names)
    ordered = sorted(names, key=lambda name: robot_training.robot.joint_qpos_addr[name])
    return np.asarray([names.index(name) for name in ordered], dtype=np.int64)


def _policy_to_mujoco_qpos(qpos: np.ndarray, joint_order: np.ndarray) -> np.ndarray:
    """Convert root + policy-DOF order into the MJCF qpos-address order."""
    values = np.asarray(qpos)
    if values.ndim != 1 or values.shape[0] != 7 + len(joint_order):
        raise ValueError(
            f"Expected one root+DOF qpos vector of size {7 + len(joint_order)}, "
            f"got shape={values.shape}"
        )
    return np.concatenate((values[:7], values[7:][joint_order]))


def _label_frame(image: np.ndarray, label: str) -> np.ndarray:
    frame = Image.fromarray(image)
    draw = ImageDraw.Draw(frame, "RGBA")
    font = ImageFont.load_default(size=24)
    box = draw.textbbox((0, 0), label, font=font)
    text_width = box[2] - box[0]
    draw.rectangle((0, 0, image.shape[1], 44), fill=(0, 0, 0, 150))
    draw.text(((image.shape[1] - text_width) / 2, 9), label, font=font, fill=(255, 255, 255, 255))
    return np.asarray(frame)


@torch.inference_mode()
def rollout_qpos(wrapped_env: Any, actor: OnnxActor, z: torch.Tensor, reference: dict[str, torch.Tensor]) -> list[np.ndarray]:
    core = wrapped_env._env
    expanded = _expand_reference(reference, core.env_origins)
    observation, _ = wrapped_env.reset(to_numpy=False, target_states=initial_target_states(expanded, core))
    qposes = [policy_qpos_from_env(wrapped_env, expected_qpos_size=7 + core.num_dof)]
    for step in range(z.shape[0]):
        action = actor.act(observation, z[step : step + 1])
        observation, _, terminated, truncated, _ = wrapped_env.step(action, to_numpy=False)
        qposes.append(policy_qpos_from_env(wrapped_env, expected_qpos_size=7 + core.num_dof))
        if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
            break
    return qposes


def render_comparison(
    args: argparse.Namespace,
    data_path: Path,
    baseline: torch.Tensor,
    best: torch.Tensor,
) -> None:
    env_cfg, use_root_height_obs = load_mjlab_env_cfg(
        args.model_folder, data_path=data_path, robot_config=args.robot_config,
        device=args.device, headless=True, disable_dr=True, disable_obs_noise=True,
        max_episode_length_s=10000.0,
    )
    wrapped_env, _ = env_cfg.build(num_envs=1)
    core = wrapped_env._env
    renderer = None
    try:
        core._motion_lib.load_all_motions()
        _, reference = _full_reference_observation(
            core,
            args.frame_count,
            use_root_height_obs,
            start_frame=args.start_frame,
        )
        actor = OnnxActor(args.actor_onnx, args.onnx_provider)
        baseline_qpos = rollout_qpos(wrapped_env, actor, baseline, reference)
        adapted_qpos = rollout_qpos(wrapped_env, actor, best, reference)
        robot_training = load_robot_training_spec(args.robot_config)
        order = _control_to_qpos_indices(robot_training)
        baseline_qpos = [_policy_to_mujoco_qpos(qpos, order) for qpos in baseline_qpos]
        adapted_qpos = [_policy_to_mujoco_qpos(qpos, order) for qpos in adapted_qpos]
        pos = reference["ref_body_pos"][:, 0].detach().cpu().numpy()
        quat = reference["ref_body_rots"][:, 0].detach().cpu().numpy()[:, [3, 0, 1, 2]]
        joints = reference["dof_pos"].detach().cpu().numpy()[:, order]
        expert_qpos = np.concatenate([pos, quat, joints], axis=-1)
        renderer = MujocoQposRenderer(
            Path(robot_training.robot.xml_path), render_size=args.render_size,
            expected_qpos_size=7 + core.num_dof,
        )
        frames = []
        count = min(len(expert_qpos), max(len(baseline_qpos), len(adapted_qpos)))
        for step in trange(
            count, desc="render", leave=False, disable=not PROGRESS_ENABLED
        ):
            images = [
                _label_frame(
                    renderer.render_qpos(expert_qpos[min(step, len(expert_qpos) - 1)]),
                    "REF",
                ),
                _label_frame(
                    renderer.render_qpos(baseline_qpos[min(step, len(baseline_qpos) - 1)]),
                    "BASELINE",
                ),
                _label_frame(
                    renderer.render_qpos(adapted_qpos[min(step, len(adapted_qpos) - 1)]),
                    "BEST",
                ),
            ]
            separator = np.full((images[0].shape[0], 4, 3), 255, dtype=np.uint8)
            frames.append(
                np.concatenate(
                    [images[0], separator, images[1], separator, images[2]],
                    axis=1,
                )
            )
        video_path = args.output_dir / "comparison_ref_baseline_best.mp4"
        temporary_video = video_path.with_name(f".{video_path.stem}.tmp.mp4")
        media.write_video(str(temporary_video), frames, fps=args.fps)
        os.replace(temporary_video, video_path)
    finally:
        if renderer is not None:
            renderer.close()
        wrapped_env.close()


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in ("model_folder", "motion", "robot_config"):
        value = Path(getattr(args, name)).expanduser()
        setattr(args, name, (PROJECT_ROOT / value).resolve() if not value.is_absolute() else value.resolve())
    args.actor_onnx = (args.model_folder / "exported/FBcprAuxModel.onnx") if args.actor_onnx is None else Path(args.actor_onnx).expanduser().resolve()
    args.backward_onnx = (args.model_folder / "exported/backward_encoder.onnx") if args.backward_onnx is None else Path(args.backward_onnx).expanduser().resolve()
    if args.initial_mean_z is not None:
        args.initial_mean_z = Path(args.initial_mean_z).expanduser().resolve()
    if args.resume_dir is not None:
        args.resume_dir = Path(args.resume_dir).expanduser().resolve()
        saved = json.loads((args.resume_dir / "run_config.json").read_text(encoding="utf-8"))
        preserved = {
            "resume_dir": args.resume_dir,
            "render_only": args.render_only,
            "resume_total_iterations": args.resume_total_iterations,
            "resume_use_cli_config": args.resume_use_cli_config,
            "resume_reset_objective_best": args.resume_reset_objective_best,
            "resume_reset_mean_to_link_best": args.resume_reset_mean_to_link_best,
            "stop_after_iteration": args.stop_after_iteration,
            "mppi_score": args.mppi_score,
            "mppi_link_guard": args.mppi_link_guard,
            "mppi_link_blend_alpha": args.mppi_link_blend_alpha,
            "history_format": args.history_format,
            "progress": args.progress,
        }
        if not args.resume_use_cli_config:
            for key, value in saved.items():
                if hasattr(args, key):
                    setattr(
                        args,
                        key,
                        Path(value)
                        if key
                        in {
                            "model_folder",
                            "motion",
                            "robot_config",
                            "actor_onnx",
                            "backward_onnx",
                            "output_dir",
                        }
                        else value,
                    )
        args.resume_dir = preserved["resume_dir"]
        args.render_only = preserved["render_only"]
        args.resume_use_cli_config = preserved["resume_use_cli_config"]
        if preserved["resume_reset_objective_best"]:
            args.resume_reset_objective_best = True
        if preserved["resume_reset_mean_to_link_best"]:
            args.resume_reset_mean_to_link_best = True
        if preserved["stop_after_iteration"] is not None:
            args.stop_after_iteration = preserved["stop_after_iteration"]
        if preserved["mppi_score"] is not None:
            args.mppi_score = preserved["mppi_score"]
        if preserved["mppi_link_guard"] is not None:
            args.mppi_link_guard = preserved["mppi_link_guard"]
        if preserved["mppi_link_blend_alpha"] is not None:
            args.mppi_link_blend_alpha = preserved["mppi_link_blend_alpha"]
        if preserved["history_format"] is not None:
            args.history_format = preserved["history_format"]
        if preserved["progress"] is not None:
            args.progress = preserved["progress"]
        if preserved["resume_total_iterations"] is not None:
            args.iterations = preserved["resume_total_iterations"]
        args.resume_total_iterations = preserved["resume_total_iterations"]
        args.output_dir = args.resume_dir
    elif args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = PROJECT_ROOT / "experiments/bfm_trajectory_latent_adaptation/outputs" / f"{args.motion.stem}_{stamp}"
    else:
        args.output_dir = Path(args.output_dir).expanduser().resolve()
    if args.history_format is None:
        args.history_format = "json"
    if args.progress is None:
        args.progress = True
    if args.mppi_score is None:
        args.mppi_score = "objective"
    if args.mppi_link_guard is None:
        args.mppi_link_guard = "baseline"
    if args.mppi_link_blend_alpha is None:
        args.mppi_link_blend_alpha = 0.0
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_MODEL_FOLDER)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--actor-onnx", type=Path, default=None)
    parser.add_argument("--backward-onnx", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--onnx-provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--particles", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Reference frame used for reset; optimization covers this local motion window.",
    )
    parser.add_argument("--frame-count", type=int, default=501)
    parser.add_argument("--sigma0", type=float, default=0.20)
    parser.add_argument("--beta-iteration", type=float, default=0.85)
    parser.add_argument("--beta-horizon", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--mppi-score",
        choices=("objective", "link"),
        default=None,
        help=(
            "Score used only for the MPPI mean update. 'link' maximizes negative "
            "local link-relative position error behind tracking-quality guards."
        ),
    )
    parser.add_argument(
        "--mppi-link-guard",
        choices=("baseline", "mean"),
        default=None,
        help=(
            "Quality-reference candidate for link-scored MPPI updates. 'mean' "
            "requires per-iteration non-regression versus candidate 1. The "
            "objective/link blend always uses the baseline-feasible mask."
        ),
    )
    parser.add_argument(
        "--mppi-link-blend-alpha",
        type=float,
        default=None,
        help=(
            "For objective-scored MPPI, convexly blend its weights with "
            "baseline-feasible direct-link weights. 0 preserves the original "
            "objective update and 1 uses the guarded direct-link update. If no "
            "link candidate is eligible, the objective fallback has effective "
            "alpha 0. The configured value is ignored by link-scored MPPI."
        ),
    )
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--require-action-mapping",
        choices=("effort_kp", "soft_limit_bias"),
        default=None,
        help="Fail if the model-folder environment does not use this action mapping.",
    )
    parser.add_argument(
        "--require-limit-safe-best",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Only accept/export candidates whose maximum XML hard-limit utilization "
            "does not exceed --joint-limit-guard-fraction."
        ),
    )
    parser.add_argument(
        "--joint-limit-guard-fraction",
        type=float,
        default=0.995,
        help="Absolute whole-window hard-limit utilization gate for exported candidates.",
    )
    parser.add_argument(
        "--joint-limit-safe-fraction",
        type=float,
        default=0.90,
        help="Start of the continuous joint-limit reward margin inside XML hard limits.",
    )
    parser.add_argument(
        "--selection-tracking-regression-fraction",
        type=float,
        default=0.0,
        help=(
            "Maximum relative MPJPE and root-relative link-error regression allowed "
            "when selecting a limit-safe best. Zero preserves strict non-regression."
        ),
    )
    parser.add_argument(
        "--determinism-tolerance", type=float, default=2.0e-2,
        help="Maximum repeated-baseline objective drift allowed for GPU MuJoCo.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--initial-mean-z",
        type=Path,
        default=None,
        help="Initialize a fresh run's MPPI mean from this [H,256] NPY latent sequence.",
    )
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument(
        "--resume-total-iterations",
        type=int,
        default=None,
        help="When resuming, extend the saved run to this total iteration count.",
    )
    parser.add_argument(
        "--resume-use-cli-config",
        action="store_true",
        help="Use all explicit CLI values while loading only optimizer state from resume-dir.",
    )
    parser.add_argument(
        "--resume-reset-objective-best",
        action="store_true",
        help="Reset the composite best after loading the optimizer mean; useful when objective parameters change.",
    )
    parser.add_argument(
        "--resume-reset-mean-to-link-best",
        action="store_true",
        help="Resume MPPI exploration around the saved link-error best instead of the saved population mean.",
    )
    parser.add_argument(
        "--stop-after-iteration",
        type=int,
        default=None,
        help=(
            "Stop and run final paired validation after this absolute completed-step "
            "count while retaining --iterations as the DIAL schedule horizon."
        ),
    )
    parser.add_argument("--history-format", choices=("json", "jsonl"), default=None)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--video-best-kind",
        choices=("objective", "link"),
        default="objective",
        help="Choose whether BEST in the comparison video uses composite-objective or link-error best.",
    )
    parser.add_argument(
        "--video-interval",
        type=int,
        default=50,
        help="Overwrite the REF|BASELINE|BEST video every N iterations; 0 disables periodic video.",
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=50,
        help="Fresh repeated validation interval; 0 disables periodic validation.",
    )
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--fps", type=int, default=50)
    defaults = {
        "root-position-weight": 0.15, "root-rotation-weight": 0.10,
        "body-position-weight": 0.30, "body-rotation-weight": 0.15,
        "joint-position-weight": 0.15, "body-linear-velocity-weight": 0.075,
        "body-angular-velocity-weight": 0.075, "joint-limit-weight": 0.0,
        "root-position-sigma": 0.15, "root-rotation-sigma": 0.35,
        "body-position-sigma": 0.10, "body-rotation-sigma": 0.35,
        "joint-position-sigma": 0.25, "body-linear-velocity-sigma": 0.60,
        "body-angular-velocity-sigma": 1.50, "joint-limit-sigma": 0.25,
    }
    for flag, default in defaults.items():
        parser.add_argument(f"--{flag}", type=float, default=default)
    return _resolve_args(parser.parse_args())


def _tracking_config(args: argparse.Namespace) -> TrackingConfig:
    names = (
        "root_position", "root_rotation", "body_position", "body_rotation",
        "joint_position", "body_linear_velocity", "body_angular_velocity",
        "joint_limit",
    )
    return TrackingConfig(
        weights={name: float(getattr(args, f"{name}_weight")) for name in names},
        sigmas={name: float(getattr(args, f"{name}_sigma")) for name in names},
        joint_limit_safe_fraction=float(args.joint_limit_safe_fraction),
    )


def main() -> None:
    global PROGRESS_ENABLED
    args = parse_args()
    PROGRESS_ENABLED = bool(args.progress)
    if (
        args.particles < 2
        or args.iterations < 1
        or args.frame_count < 2
        or args.start_frame < 0
    ):
        raise ValueError(
            "particles>=2, iterations>=1, frame-count>=2 and start-frame>=0 are required"
        )
    if args.video_interval < 0:
        raise ValueError("video-interval must be non-negative")
    if args.validation_interval < 0:
        raise ValueError("validation-interval must be non-negative")
    if not math.isfinite(args.mppi_link_blend_alpha) or not (
        0.0 <= args.mppi_link_blend_alpha <= 1.0
    ):
        raise ValueError("mppi-link-blend-alpha must be finite and in [0, 1]")
    if not math.isfinite(args.joint_limit_guard_fraction) or not (
        0.0 < args.joint_limit_guard_fraction < 1.0
    ):
        raise ValueError("joint-limit-guard-fraction must be finite and in (0,1)")
    if not math.isfinite(args.joint_limit_safe_fraction) or not (
        0.0 <= args.joint_limit_safe_fraction < args.joint_limit_guard_fraction
    ):
        raise ValueError(
            "joint-limit-safe-fraction must be finite, non-negative, and below "
            "joint-limit-guard-fraction"
        )
    if not math.isfinite(args.selection_tracking_regression_fraction) or not (
        0.0 <= args.selection_tracking_regression_fraction <= 1.0
    ):
        raise ValueError(
            "selection-tracking-regression-fraction must be finite and in [0,1]"
        )
    if args.stop_after_iteration is not None and not (
        1 <= args.stop_after_iteration <= args.iterations
    ):
        raise ValueError(
            "stop-after-iteration must be between 1 and the DIAL schedule "
            f"horizon ({args.iterations}), got {args.stop_after_iteration}"
        )
    optimization_stop = (
        args.iterations
        if args.stop_after_iteration is None
        else args.stop_after_iteration
    )
    if args.smoothing_window < 1 or args.smoothing_window % 2 == 0:
        raise ValueError("smoothing-window must be a positive odd number")
    actual_frames = int(np.load(args.motion, mmap_mode="r")["joint_pos"].shape[0])
    if args.start_frame + args.frame_count > actual_frames:
        raise ValueError(
            f"Requested frames [{args.start_frame}, "
            f"{args.start_frame + args.frame_count}), motion only has {actual_frames}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_values = vars(args).copy()
    config_values["resume_dir"] = None
    config_values["resume_use_cli_config"] = False
    config_values["resume_reset_objective_best"] = False
    config_values["resume_reset_mean_to_link_best"] = False
    config_values["stop_after_iteration"] = None
    if not args.render_only:
        _save_json(args.output_dir / "run_config.json", config_values)

    data_args = argparse.Namespace(
        motion=args.motion, robot_config=args.robot_config, data_manifest=None, dataset=None,
        rebuild_motion_cache=False,
    )
    data_path, _, _, _, _ = _prepare_motion_input(data_args, args.model_folder)
    if args.render_only:
        baseline = torch.from_numpy(np.load(args.output_dir / "baseline_z.npy")).to(args.device)
        checkpoint = torch.load(
            args.output_dir / "checkpoint.pt",
            map_location=args.device,
            weights_only=False,
        )
        if args.video_best_kind == "link":
            if args.validation_interval > 0:
                best = checkpoint.get(
                    "validated_link_best", checkpoint.get("link_best", checkpoint["best"])
                ).to(args.device)
            else:
                best = checkpoint.get("link_best", checkpoint["best"]).to(args.device)
        else:
            best = checkpoint["best"].to(args.device)
        render_comparison(args, data_path, baseline, best)
        return

    env_cfg, use_root_height_obs = load_mjlab_env_cfg(
        args.model_folder, data_path=data_path, robot_config=args.robot_config,
        device=args.device, headless=True, disable_dr=True, disable_obs_noise=True,
        max_episode_length_s=10000.0,
    )
    wrapped_env, _ = env_cfg.build(num_envs=args.particles)
    core = wrapped_env._env
    if (
        args.require_action_mapping is not None
        and core.action_mapping != args.require_action_mapping
    ):
        wrapped_env.close()
        raise ValueError(
            f"Expected action_mapping={args.require_action_mapping}, "
            f"model environment uses {core.action_mapping}"
        )
    print(
        f"[adapt] action_mapping={core.action_mapping} "
        f"motion_window=[{args.start_frame}, "
        f"{args.start_frame + args.frame_count})",
        flush=True,
    )
    logger = AdaptationTensorboardLogger(args.output_dir / "tensorboard", config_values, enabled=args.tensorboard)
    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)
    history: list[dict[str, Any]] = []
    try:
        core._motion_lib.load_all_motions()
        core.is_evaluating = True
        run_cfg = json.loads((args.model_folder / "config.json").read_text(encoding="utf-8"))
        seq_length = int(run_cfg["agent"]["model"]["seq_length"])
        baseline, reference = generate_baseline(
            core, frame_count=args.frame_count, backward_onnx=args.backward_onnx,
            provider=args.onnx_provider, use_root_height_obs=use_root_height_obs,
            seq_length=seq_length, start_frame=args.start_frame,
        )
        baseline = project_latents(baseline)
        _atomic_save_npy(
            args.output_dir / "baseline_z.npy", baseline.cpu().numpy().astype(np.float32)
        )
        actor = OnnxActor(args.actor_onnx, args.onnx_provider)
        tracking_config = _tracking_config(args)

        baseline_batch = baseline.unsqueeze(0).expand(args.particles, -1, -1)
        baseline_result = evaluate_sequences(wrapped_env, actor, baseline_batch, reference, tracking_config)
        baseline_rewards, baseline_metrics, baseline_std = _aggregate_metrics(baseline_result)
        repeated_baseline_result = evaluate_sequences(
            wrapped_env, actor, baseline_batch, reference, tracking_config
        )
        _repeated_rewards, repeated_baseline_metrics, repeated_baseline_std = _aggregate_metrics(
            repeated_baseline_result
        )
        deterministic_delta = abs(repeated_baseline_metrics["objective"] - baseline_metrics["objective"])
        if deterministic_delta > args.determinism_tolerance:
            print(
                "[adapt] WARNING: repeated baseline objective drift "
                f"objective delta={deterministic_delta:.8g} exceeds "
                f"{args.determinism_tolerance:.8g}; recording it as numerical uncertainty."
            )
        baseline_metrics["repeat_objective_delta"] = deterministic_delta
        baseline_metrics["repeat_mpjpe_delta"] = abs(
            repeated_baseline_metrics["mpjpe"] - baseline_metrics["mpjpe"]
        )
        _save_json(
            args.output_dir / "metrics_baseline.json",
            {
                "rewards": baseline_rewards,
                "metrics": baseline_metrics,
                "population_std": baseline_std,
                "repeated_population_std": repeated_baseline_std,
            },
        )
        logger.log_baseline({**baseline_metrics, **{f"reward_{k}": v for k, v in baseline_rewards.items()}})
        mean = baseline.clone()
        if args.initial_mean_z is not None:
            initial_mean = torch.from_numpy(np.load(args.initial_mean_z)).to(
                device=args.device, dtype=baseline.dtype
            )
            if initial_mean.shape != baseline.shape:
                raise ValueError(
                    f"Initial mean shape {tuple(initial_mean.shape)} does not match "
                    f"baseline shape {tuple(baseline.shape)}"
                )
            mean = project_latents(initial_mean)
            print(f"[adapt] initialized mean from {args.initial_mean_z}", flush=True)
        best = baseline.clone()
        best_objective = baseline_metrics["objective"]
        best_mpjpe = baseline_metrics["mpjpe"]
        global_best_metrics = dict(baseline_metrics)
        global_best_rewards = dict(baseline_rewards)
        baseline_limit_safe = (
            not args.require_limit_safe_best
            or baseline_metrics["max_joint_limit_utilization"]
            <= args.joint_limit_guard_fraction
        )
        safe_best_found = baseline_limit_safe
        link_best = baseline.clone()
        link_best_error = baseline_metrics["mean_body_position_error"]
        link_best_objective = baseline_metrics["objective"]
        link_best_metrics = dict(baseline_metrics)
        link_best_rewards = dict(baseline_rewards)
        link_safe_best_found = baseline_limit_safe
        validated_objective_best = best.clone()
        validated_objective_metrics = dict(global_best_metrics)
        validated_objective_rewards = dict(global_best_rewards)
        validated_objective_std = dict(baseline_std)
        validated_link_best = link_best.clone()
        validated_link_metrics = dict(link_best_metrics)
        validated_link_rewards = dict(link_best_rewards)
        validated_link_std = dict(baseline_std)
        start_iteration = 0
        checkpoint_path = args.output_dir / "checkpoint.pt"
        if args.resume_dir is not None and checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
            mean = checkpoint["mean"].to(args.device)
            best = checkpoint["best"].to(args.device)
            best_objective = float(checkpoint["best_objective"])
            best_mpjpe = float(checkpoint.get("best_mpjpe", baseline_metrics["mpjpe"]))
            safe_best_found = bool(
                checkpoint.get(
                    "safe_best_found",
                    not args.require_limit_safe_best
                    or checkpoint.get("global_best_metrics", baseline_metrics)[
                        "max_joint_limit_utilization"
                    ]
                    <= args.joint_limit_guard_fraction,
                )
            )
            previous_history: list[dict[str, Any]] = []
            legacy_history_path = args.output_dir / "history.json"
            if legacy_history_path.exists():
                previous_history = json.loads(legacy_history_path.read_text(encoding="utf-8"))
            if args.history_format == "json":
                history = previous_history
            else:
                jsonl_path = args.output_dir / "history.jsonl"
                if not jsonl_path.exists():
                    _write_jsonl_atomic(jsonl_path, previous_history)
                _repair_jsonl_tail(jsonl_path)
            global_best_metrics = dict(
                checkpoint.get(
                    "global_best_metrics",
                    previous_history[-1].get("global_best_metrics", baseline_metrics)
                    if previous_history
                    else baseline_metrics,
                )
            )
            global_best_rewards = dict(
                checkpoint.get(
                    "global_best_rewards",
                    previous_history[-1].get("global_best_rewards", baseline_rewards)
                    if previous_history
                    else baseline_rewards,
                )
            )
            link_best = checkpoint.get("link_best", checkpoint["best"]).to(args.device)
            link_best_metrics = dict(
                checkpoint.get("link_best_metrics", global_best_metrics)
            )
            link_best_rewards = dict(
                checkpoint.get("link_best_rewards", global_best_rewards)
            )
            link_best_error = float(
                checkpoint.get(
                    "link_best_error",
                    link_best_metrics["mean_body_position_error"],
                )
            )
            link_best_objective = float(
                checkpoint.get("link_best_objective", best_objective)
            )
            link_safe_best_found = bool(
                checkpoint.get(
                    "link_safe_best_found",
                    not args.require_limit_safe_best
                    or link_best_metrics["max_joint_limit_utilization"]
                    <= args.joint_limit_guard_fraction,
                )
            )
            if args.resume_reset_mean_to_link_best:
                mean = link_best.clone()
                print("[adapt] reset resumed MPPI mean to saved link best", flush=True)
            if args.resume_reset_objective_best:
                best = baseline.clone()
                best_objective = baseline_metrics["objective"]
                best_mpjpe = baseline_metrics["mpjpe"]
                global_best_metrics = dict(baseline_metrics)
                global_best_rewards = dict(baseline_rewards)
                safe_best_found = baseline_limit_safe
            validated_objective_best = checkpoint.get(
                "validated_objective_best", checkpoint["best"]
            ).to(args.device)
            validated_objective_metrics = dict(
                checkpoint.get("validated_objective_metrics", global_best_metrics)
            )
            validated_objective_rewards = dict(
                checkpoint.get("validated_objective_rewards", global_best_rewards)
            )
            validated_objective_std = dict(
                checkpoint.get("validated_objective_std", baseline_std)
            )
            validated_link_best = checkpoint.get(
                "validated_link_best", checkpoint.get("link_best", checkpoint["best"])
            ).to(args.device)
            validated_link_metrics = dict(
                checkpoint.get("validated_link_metrics", link_best_metrics)
            )
            validated_link_rewards = dict(
                checkpoint.get("validated_link_rewards", link_best_rewards)
            )
            validated_link_std = dict(
                checkpoint.get("validated_link_std", baseline_std)
            )
            start_iteration = int(checkpoint["iteration"]) + 1
            if start_iteration > args.iterations:
                raise ValueError(
                    f"Resume checkpoint starts at iteration {start_iteration}; "
                    f"target total iterations is {args.iterations}"
                )
            if start_iteration > optimization_stop:
                raise ValueError(
                    f"Resume checkpoint starts at iteration {start_iteration}; "
                    f"requested stop-after-iteration is {optimization_stop}"
                )
            # Final-only resumes still rewrite the durable checkpoint after paired
            # validation.  Always restore the saved generator state so that such a
            # validation pass cannot reset RNG continuity if the run is extended
            # again later.
            generator.set_state(checkpoint["rng_state"].cpu())
            _atomic_save_npy(
                args.output_dir / "current_mean_z.npy",
                mean.cpu().numpy().astype(np.float32),
            )
            _atomic_save_npy(
                args.output_dir / "best_z.npy", best.cpu().numpy().astype(np.float32)
            )
            _atomic_save_npy(
                args.output_dir / "link_best_z.npy",
                link_best.cpu().numpy().astype(np.float32),
            )
        _atomic_save_npy(
            args.output_dir / "current_mean_z.npy", mean.cpu().numpy().astype(np.float32)
        )
        _atomic_save_npy(
            args.output_dir / "best_z.npy", best.cpu().numpy().astype(np.float32)
        )
        _atomic_save_npy(
            args.output_dir / "link_best_z.npy", link_best.cpu().numpy().astype(np.float32)
        )
        _atomic_save_npy(
            args.output_dir / "validated_objective_best_z.npy",
            validated_objective_best.cpu().numpy().astype(np.float32),
        )
        _atomic_save_npy(
            args.output_dir / "validated_link_best_z.npy",
            validated_link_best.cpu().numpy().astype(np.float32),
        )

        def save_optimizer_state(iteration: int) -> None:
            _atomic_save_npy(
                args.output_dir / "current_mean_z.npy",
                mean.cpu().numpy().astype(np.float32),
            )
            _atomic_save_npy(
                args.output_dir / "best_z.npy", best.cpu().numpy().astype(np.float32)
            )
            _atomic_save_npy(
                args.output_dir / "link_best_z.npy",
                link_best.cpu().numpy().astype(np.float32),
            )
            _atomic_save_npy(
                args.output_dir / "validated_objective_best_z.npy",
                validated_objective_best.cpu().numpy().astype(np.float32),
            )
            _atomic_save_npy(
                args.output_dir / "validated_link_best_z.npy",
                validated_link_best.cpu().numpy().astype(np.float32),
            )
            _atomic_torch_save(
                checkpoint_path,
                {
                    "iteration": iteration,
                    "mean": mean.cpu(),
                    "best": best.cpu(),
                    "best_objective": best_objective,
                    "best_mpjpe": best_mpjpe,
                    "safe_best_found": safe_best_found,
                    "global_best_metrics": global_best_metrics,
                    "global_best_rewards": global_best_rewards,
                    "link_best": link_best.cpu(),
                    "link_best_error": link_best_error,
                    "link_best_objective": link_best_objective,
                    "link_best_metrics": link_best_metrics,
                    "link_best_rewards": link_best_rewards,
                    "link_safe_best_found": link_safe_best_found,
                    "validated_objective_best": validated_objective_best.cpu(),
                    "validated_objective_metrics": validated_objective_metrics,
                    "validated_objective_rewards": validated_objective_rewards,
                    "validated_objective_std": validated_objective_std,
                    "validated_link_best": validated_link_best.cpu(),
                    "validated_link_metrics": validated_link_metrics,
                    "validated_link_rewards": validated_link_rewards,
                    "validated_link_std": validated_link_std,
                    "rng_state": generator.get_state().cpu(),
                },
            )

        periodic_validation_path = args.output_dir / "periodic_validation.jsonl"
        _repair_jsonl_tail(periodic_validation_path)
        last_validation_step = _last_jsonl_step(periodic_validation_path)

        def run_periodic_validation(step: int) -> None:
            nonlocal validated_objective_best
            nonlocal validated_objective_metrics
            nonlocal validated_objective_rewards
            nonlocal validated_objective_std
            nonlocal validated_link_best
            nonlocal validated_link_metrics
            nonlocal validated_link_rewards
            nonlocal validated_link_std
            nonlocal last_validation_step

            variants = {
                "baseline": baseline,
                "current_mean": mean,
                "search_objective_best": best,
                "search_link_best": link_best,
                "validated_objective_best": validated_objective_best,
                "validated_link_best": validated_link_best,
            }
            evaluated = _evaluate_variants(
                wrapped_env,
                actor,
                variants,
                reference,
                tracking_config,
                rotation=step // max(args.validation_interval, 1),
            )
            baseline_validation = evaluated["baseline"]

            def feasible(name: str, *, require_link: bool) -> bool:
                candidate = evaluated[name]
                objective_ok = _validation_bound(
                    candidate, "objective", upper=False
                ) >= _validation_bound(baseline_validation, "objective", upper=True)
                tracking_multiplier = (
                    1.0 + args.selection_tracking_regression_fraction
                )
                mpjpe_ok = _validation_bound(
                    candidate, "mpjpe", upper=True
                ) <= tracking_multiplier * _validation_bound(
                    baseline_validation, "mpjpe", upper=False
                )
                link_ok = (
                    _validation_bound(
                        candidate, "mean_body_position_error", upper=True
                    )
                    <= tracking_multiplier * _validation_bound(
                        baseline_validation,
                        "mean_body_position_error",
                        upper=False,
                    )
                )
                limit_ok = (
                    not args.require_limit_safe_best
                    or _validation_bound(
                        candidate,
                        "max_joint_limit_utilization",
                        upper=True,
                    )
                    <= args.joint_limit_guard_fraction
                )
                return (
                    objective_ok
                    and mpjpe_ok
                    and (link_ok or not require_link)
                    and limit_ok
                )

            candidate_names = [
                "current_mean",
                "search_objective_best",
                "search_link_best",
                "validated_objective_best",
                "validated_link_best",
            ]
            feasible_objective_names = [
                name for name in candidate_names if feasible(name, require_link=True)
            ]
            objective_selection = "validated_objective_best"
            objective_updated = False
            if feasible_objective_names:
                objective_candidate = max(
                    feasible_objective_names,
                    key=lambda name: evaluated[name]["metrics"]["objective"],
                )
                current_objective_safe = feasible(
                    "validated_objective_best", require_link=True
                )
                if (
                    not current_objective_safe
                    or _validation_bound(
                        evaluated[objective_candidate], "objective", upper=False
                    )
                    > _validation_bound(
                        evaluated["validated_objective_best"],
                        "objective",
                        upper=True,
                    )
                ):
                    objective_selection = objective_candidate
                    objective_updated = objective_candidate != "validated_objective_best"

            feasible_link_names = [
                name for name in candidate_names if feasible(name, require_link=False)
            ]
            link_selection = "validated_link_best"
            link_updated = False
            if feasible_link_names:
                link_candidate = min(
                    feasible_link_names,
                    key=lambda name: evaluated[name]["metrics"][
                        "mean_body_position_error"
                    ],
                )
                current_link_safe = feasible(
                    "validated_link_best", require_link=False
                )
                if (
                    not current_link_safe
                    or _validation_bound(
                        evaluated[link_candidate],
                        "mean_body_position_error",
                        upper=True,
                    )
                    < _validation_bound(
                        evaluated["validated_link_best"],
                        "mean_body_position_error",
                        upper=False,
                    )
                ):
                    link_selection = link_candidate
                    link_updated = link_candidate != "validated_link_best"

            objective_result = evaluated[objective_selection]
            link_result = evaluated[link_selection]
            if objective_updated:
                validated_objective_best = variants[objective_selection].clone()
            if link_updated:
                validated_link_best = variants[link_selection].clone()
            validated_objective_metrics = dict(objective_result["metrics"])
            validated_objective_rewards = dict(objective_result["rewards"])
            validated_objective_std = dict(objective_result["std"])
            validated_link_metrics = dict(link_result["metrics"])
            validated_link_rewards = dict(link_result["rewards"])
            validated_link_std = dict(link_result["std"])

            logged_variants = dict(evaluated)
            logged_variants["validated_objective_best"] = dict(objective_result)
            logged_variants["validated_link_best"] = dict(link_result)
            validation_record = {
                "step": step,
                "confidence_z": 1.96,
                "objective_selection": objective_selection,
                "objective_updated": objective_updated,
                "link_selection": link_selection,
                "link_updated": link_updated,
                "variants": logged_variants,
            }
            save_optimizer_state(step - 1)
            _append_jsonl(periodic_validation_path, validation_record)
            _save_json(
                args.output_dir / "periodic_validation_latest.json",
                validation_record,
            )
            logger.log_validation(step, validation_record)
            last_validation_step = step
            print(
                f"[adapt] validation step={step} "
                f"objective={validated_objective_metrics['objective']:.6f} "
                f"mpjpe_mm={1000.0 * validated_objective_metrics['mpjpe']:.3f} "
                f"link_best_mm={1000.0 * validated_link_metrics['mean_body_position_error']:.3f}",
                flush=True,
            )

        def render_periodic_video(step: int) -> None:
            if not args.save_video or args.video_interval <= 0:
                return
            video_status_path = args.output_dir / "video_status.json"
            if video_status_path.exists():
                status = json.loads(video_status_path.read_text(encoding="utf-8"))
                if int(status.get("step", -1)) >= step:
                    return
            print(
                f"[adapt] rendering REF|BASELINE|BEST at step {step}",
                flush=True,
            )
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "experiments.bfm_trajectory_latent_adaptation.adapt_trajectory",
                        "--resume-dir",
                        str(args.output_dir),
                        "--render-only",
                    ],
                    cwd=PROJECT_ROOT,
                    check=True,
                    timeout=900,
                )
            except (OSError, subprocess.SubprocessError) as error:
                print(
                    f"[adapt] WARNING: periodic video failed at step {step}: {error}",
                    flush=True,
                )
                return
            _save_json(video_status_path, {"step": step})

        if (
            start_iteration > 0
            and args.validation_interval > 0
            and start_iteration % args.validation_interval == 0
            and last_validation_step != start_iteration
        ):
            run_periodic_validation(start_iteration)
        if (
            start_iteration > 0
            and args.video_interval > 0
            and start_iteration % args.video_interval == 0
        ):
            render_periodic_video(start_iteration)

        for iteration in range(start_iteration, optimization_stop):
            started = time.perf_counter()
            sigma = dial_sigma_schedule(
                iteration=iteration, iterations=args.iterations, horizon=baseline.shape[0],
                sigma0=args.sigma0, beta_iteration=args.beta_iteration,
                beta_horizon=args.beta_horizon, device=args.device,
            )
            candidates = sample_candidates(
                mean, baseline, population=args.particles, sigma=sigma,
                smoothing_window=args.smoothing_window, generator=generator,
            )
            result = evaluate_sequences(wrapped_env, actor, candidates, reference, tracking_config)
            objective_scores = result.objective
            population_mpjpe = result.metric_means["mpjpe"]
            population_link_relative = result.metric_means[
                "mean_body_position_error"
            ]
            population_limit_utilization = result.metric_means[
                "max_joint_limit_utilization"
            ]
            tracking_multiplier = 1.0 + args.selection_tracking_regression_fraction
            objective_guard = objective_scores >= objective_scores[0]
            mpjpe_guard = population_mpjpe <= tracking_multiplier * population_mpjpe[0]
            link_guard = (
                population_link_relative
                <= tracking_multiplier * population_link_relative[0]
            )
            limit_guard = (
                population_limit_utilization <= args.joint_limit_guard_fraction
                if args.require_limit_safe_best
                else torch.ones_like(population_limit_utilization, dtype=torch.bool)
            )
            feasible = objective_guard & mpjpe_guard & link_guard & limit_guard
            if args.mppi_score == "link":
                if args.mppi_link_guard == "mean":
                    update_objective_guard = (
                        objective_scores >= objective_scores[1]
                    )
                    update_mpjpe_guard = (
                        population_mpjpe
                        <= tracking_multiplier * population_mpjpe[1]
                    )
                    update_link_guard = (
                        population_link_relative
                        <= tracking_multiplier * population_link_relative[1]
                    )
                    mppi_eligible = (
                        update_objective_guard
                        & update_mpjpe_guard
                        & update_link_guard
                        & limit_guard
                    )
                else:
                    update_objective_guard = objective_guard
                    update_mpjpe_guard = mpjpe_guard
                    update_link_guard = link_guard
                    mppi_eligible = feasible
                mppi_scores = -population_link_relative
                raw_weights = mppi_weights(mppi_scores, args.temperature)
                raw_feasible_weight_mass = float(
                    raw_weights[mppi_eligible].sum().cpu()
                )
                weights = mppi_weights(
                    mppi_scores, args.temperature, eligible=mppi_eligible
                )
                mppi_update_skipped = not bool(mppi_eligible.any())
                mppi_best_index = (
                    int(
                        torch.argmax(
                            torch.where(
                                mppi_eligible,
                                mppi_scores,
                                torch.full_like(mppi_scores, -torch.inf),
                            )
                        ).item()
                    )
                    if not mppi_update_skipped
                    else None
                )
                mppi_eligible_count = int(mppi_eligible.sum().item())
                link_arm_eligible = mppi_eligible
            else:
                update_objective_guard = objective_guard
                update_mpjpe_guard = mpjpe_guard
                update_link_guard = link_guard
                mppi_scores = objective_scores
                if args.mppi_link_blend_alpha > 0.0:
                    (
                        weights,
                        objective_arm_weights,
                        link_arm_weights,
                        raw_link_arm_weights,
                    ) = objective_link_mppi_weights(
                        objective_scores,
                        population_link_relative,
                        temperature=args.temperature,
                        link_eligible=feasible,
                        blend_alpha=args.mppi_link_blend_alpha,
                    )
                    link_arm_feasible_mass = float(
                        raw_link_arm_weights[feasible].sum().cpu()
                    )
                else:
                    # Keep the default path identical to the original
                    # objective-scored MPPI update.
                    weights = mppi_weights(mppi_scores, args.temperature)
                    objective_arm_weights = weights
                    link_arm_weights = None
                    link_arm_feasible_mass = 0.0
                raw_feasible_weight_mass = float(weights[feasible].sum().cpu())
                mppi_update_skipped = False
                mppi_best_index = int(torch.argmax(objective_scores).item())
                mppi_eligible_count = args.particles
                link_arm_eligible = feasible
            link_arm_diagnostics = mppi_link_arm_diagnostics(
                weights,
                baseline_feasible=feasible,
                link_eligible=link_arm_eligible,
                score_mode=args.mppi_score,
                configured_blend_alpha=args.mppi_link_blend_alpha,
                update_skipped=mppi_update_skipped,
            )
            weight_diagnostics = mppi_diagnostics(mppi_scores, weights)
            if args.mppi_score == "link":
                objective_arm_ess_ratio = 0.0
                link_arm_ess_ratio = weight_diagnostics["ess_ratio"]
                link_arm_feasible_mass = raw_feasible_weight_mass
            elif args.mppi_link_blend_alpha > 0.0:
                objective_arm_ess_ratio = mppi_diagnostics(
                    objective_scores, objective_arm_weights
                )["ess_ratio"]
                link_arm_ess_ratio = mppi_diagnostics(
                    -population_link_relative, link_arm_weights
                )["ess_ratio"]
            else:
                objective_arm_ess_ratio = weight_diagnostics["ess_ratio"]
                link_arm_ess_ratio = 0.0
            previous_mean = mean
            if not mppi_update_skipped:
                mean = update_mppi_mean(mean, candidates, weights)
            mean_update_norm = float(
                torch.linalg.vector_norm(mean - previous_mean, dim=-1).mean().cpu()
            )
            best_index = int(torch.argmax(objective_scores).item())
            iteration_best = float(objective_scores[best_index].cpu())
            # The exported global best must satisfy both acceptance metrics in
            # the same vectorized rollout as that iteration's baseline. This
            # prevents a Gaussian composite reward from trading worse MPJPE for
            # better rotation/velocity terms while still calling it "best".
            feasible_scores = torch.where(
                feasible,
                objective_scores,
                torch.full_like(objective_scores, -torch.inf),
            )
            feasible_index = int(torch.argmax(feasible_scores).item())
            feasible_objective = float(feasible_scores[feasible_index].cpu())
            global_best_updated = False
            if (
                math.isfinite(feasible_objective)
                and (
                    not safe_best_found
                    or feasible_objective > best_objective
                )
            ):
                global_best_updated = True
                safe_best_found = True
                best_objective = feasible_objective
                best_mpjpe = float(population_mpjpe[feasible_index].cpu())
                best = candidates[feasible_index].clone()
                global_best_metrics = {
                    name: float(values[feasible_index].cpu())
                    for name, values in result.metric_means.items()
                }
                global_best_rewards = {
                    name: float(values[feasible_index].cpu())
                    for name, values in result.reward_means.items()
                }
            link_feasible = objective_guard & mpjpe_guard & limit_guard
            feasible_link_errors = torch.where(
                link_feasible,
                population_link_relative,
                torch.full_like(population_link_relative, torch.inf),
            )
            link_best_index = int(torch.argmin(feasible_link_errors).item())
            iteration_link_best_error = float(feasible_link_errors[link_best_index].cpu())
            link_best_updated = False
            if (
                math.isfinite(iteration_link_best_error)
                and (
                    not link_safe_best_found
                    or iteration_link_best_error < link_best_error
                )
            ):
                link_best_updated = True
                link_safe_best_found = True
                link_best = candidates[link_best_index].clone()
                link_best_error = iteration_link_best_error
                link_best_objective = float(objective_scores[link_best_index].cpu())
                link_best_metrics = {
                    name: float(values[link_best_index].cpu())
                    for name, values in result.metric_means.items()
                }
                link_best_rewards = {
                    name: float(values[link_best_index].cpu())
                    for name, values in result.reward_means.items()
                }
            elapsed = time.perf_counter() - started
            reward_values = {name: float(values[best_index].cpu()) for name, values in result.reward_means.items()}
            metric_values = {name: float(values[best_index].cpu()) for name, values in result.metric_means.items()}
            mppi_best_metrics = (
                {
                    name: float(values[mppi_best_index].cpu())
                    for name, values in result.metric_means.items()
                }
                if mppi_best_index is not None
                else None
            )
            link_quantiles_mm = 1000.0 * torch.quantile(
                population_link_relative,
                torch.tensor(
                    [0.1, 0.5, 0.9], device=population_link_relative.device
                ),
            )
            weighted_link_error_mm = (
                1000.0 * float((weights * population_link_relative).sum().cpu())
                if not mppi_update_skipped
                else None
            )
            limit_quantiles = torch.quantile(
                population_limit_utilization,
                torch.tensor(
                    [0.0, 0.1, 0.5, 0.9, 1.0],
                    device=population_limit_utilization.device,
                ),
            )
            cosine, delta = _latent_stats(candidates[best_index], baseline)
            record = {
                "iteration": iteration,
                "iteration_best_objective": iteration_best,
                "feasible_best_objective": feasible_objective if math.isfinite(feasible_objective) else None,
                "feasible_best_index": feasible_index if math.isfinite(feasible_objective) else None,
                "global_best_objective": best_objective,
                "baseline_objective": baseline_metrics["objective"],
                "baseline_mpjpe": baseline_metrics["mpjpe"],
                "global_best_mpjpe": best_mpjpe,
                "safe_best_found": safe_best_found,
                "link_safe_best_found": link_safe_best_found,
                "baseline_metrics": baseline_metrics,
                "baseline_rewards": baseline_rewards,
                "global_best_metrics": global_best_metrics,
                "global_best_rewards": global_best_rewards,
                "link_best_objective": link_best_objective,
                "link_best_metrics": link_best_metrics,
                "link_best_rewards": link_best_rewards,
                "objective_improvement_percent": 100.0 * (
                    best_objective - baseline_metrics["objective"]
                ) / max(abs(baseline_metrics["objective"]), 1.0e-12),
                "mpjpe_improvement_percent": 100.0 * (
                    baseline_metrics["mpjpe"] - best_mpjpe
                ) / max(abs(baseline_metrics["mpjpe"]), 1.0e-12),
                "link_relative_position_improvement_percent": 100.0 * (
                    baseline_metrics["mean_body_position_error"]
                    - global_best_metrics["mean_body_position_error"]
                ) / max(abs(baseline_metrics["mean_body_position_error"]), 1.0e-12),
                "link_best_improvement_percent": 100.0 * (
                    baseline_metrics["mean_body_position_error"]
                    - link_best_metrics["mean_body_position_error"]
                ) / max(abs(baseline_metrics["mean_body_position_error"]), 1.0e-12),
                "temperature": args.temperature,
                "population_mean": float(objective_scores.mean().cpu()),
                "population_std": float(objective_scores.std(unbiased=False).cpu()),
                "mean_update_norm": mean_update_norm,
                "feasible_ratio": float(feasible.float().mean().cpu()),
                "global_best_updated": global_best_updated,
                "link_best_updated": link_best_updated,
                "noise_mean": float(sigma.mean().cpu()), "noise_min": float(sigma.min().cpu()), "noise_max": float(sigma.max().cpu()),
                "best_cosine_to_baseline": cosine, "best_mean_frame_delta": delta,
                "iteration_seconds": elapsed, "rollouts_per_second": args.particles / elapsed,
                "iteration_best_rewards": reward_values, "iteration_best_metrics": metric_values,
                "current_mean_candidate_metrics": {
                    name: float(values[1].cpu())
                    for name, values in result.metric_means.items()
                },
                "population_limit_utilization": {
                    "min": float(limit_quantiles[0].cpu()),
                    "p10": float(limit_quantiles[1].cpu()),
                    "p50": float(limit_quantiles[2].cpu()),
                    "p90": float(limit_quantiles[3].cpu()),
                    "max": float(limit_quantiles[4].cpu()),
                },
                "mppi": weight_diagnostics,
                "mppi_score_mode": args.mppi_score,
                "mppi_link_guard": args.mppi_link_guard,
                "mppi_link_blend_alpha": args.mppi_link_blend_alpha,
                "mppi_link_blend_effective_alpha": link_arm_diagnostics[
                    "link_blend_effective_alpha"
                ],
                "mppi_link_arm_active": link_arm_diagnostics["link_arm_active"],
                "mppi_link_arm_eligible_ratio": link_arm_diagnostics[
                    "link_arm_eligible_ratio"
                ],
                "mppi_mixed_feasible_mass": link_arm_diagnostics[
                    "mixed_feasible_mass"
                ],
                "mppi_objective_arm_ess_ratio": objective_arm_ess_ratio,
                "mppi_link_arm_ess_ratio": link_arm_ess_ratio,
                "mppi_link_arm_feasible_mass": link_arm_feasible_mass,
                "mppi_update_skipped": mppi_update_skipped,
                "mppi_eligible_ratio": mppi_eligible_count / args.particles,
                "mppi_raw_feasible_weight_mass": raw_feasible_weight_mass,
                "mppi_ess_ratio_eligible": weight_diagnostics[
                    "effective_sample_size"
                ]
                / max(mppi_eligible_count, 1),
                "mppi_weighted_link_error_mm": weighted_link_error_mm,
                "mppi_link_error_mm_p10": float(link_quantiles_mm[0].cpu()),
                "mppi_link_error_mm_p50": float(link_quantiles_mm[1].cpu()),
                "mppi_link_error_mm_p90": float(link_quantiles_mm[2].cpu()),
                "mppi_best_metrics": mppi_best_metrics,
                "guards": {
                    "objective_pass_ratio": float(
                        update_objective_guard.float().mean().cpu()
                    ),
                    "mpjpe_pass_ratio": float(
                        update_mpjpe_guard.float().mean().cpu()
                    ),
                    "link_pass_ratio": float(
                        update_link_guard.float().mean().cpu()
                    ),
                    "limit_pass_ratio": float(
                        limit_guard.float().mean().cpu()
                    ),
                },
            }
            save_optimizer_state(iteration)
            if args.history_format == "json":
                history.append(record)
                _save_json(args.output_dir / "history.json", history)
            else:
                _append_jsonl(args.output_dir / "history.jsonl", record)
            logger.log_iteration(iteration, record)
            print(
                f"[adapt] iteration={iteration + 1}/{args.iterations} "
                f"best={iteration_best:.6f} global={best_objective:.6f} "
                f"link_best_mm={1000.0 * link_best_error:.3f} seconds={elapsed:.1f}",
                flush=True,
            )
            completed_steps = iteration + 1
            if (
                args.validation_interval > 0
                and completed_steps % args.validation_interval == 0
                and last_validation_step != completed_steps
            ):
                run_periodic_validation(completed_steps)
            if (
                args.video_interval > 0
                and completed_steps % args.video_interval == 0
            ):
                render_periodic_video(completed_steps)

        if args.validation_interval == 0:
            validated_objective_best = best
            validated_link_best = link_best
        elif last_validation_step != optimization_stop:
            # Always validate the newest search candidates.  Previously a run
            # shorter than ``validation_interval`` silently kept the initial
            # baseline in ``validated_*`` and discarded a safe search best.
            run_periodic_validation(optimization_stop)
        validation_indices = torch.arange(args.particles, device=args.device)
        baseline_validation_mask = validation_indices.remainder(2).eq(0)
        adapted_validation_mask = ~baseline_validation_mask
        baseline_validation_count = int(baseline_validation_mask.sum().item())
        adapted_validation_count = int(adapted_validation_mask.sum().item())
        paired_sequences = baseline.unsqueeze(0).expand(args.particles, -1, -1).clone()
        paired_sequences[adapted_validation_mask] = validated_objective_best
        paired_result = evaluate_sequences(
            wrapped_env, actor, paired_sequences, reference, tracking_config
        )
        paired_baseline_rewards, paired_baseline_metrics, paired_baseline_std = _aggregate_metrics(
            paired_result, baseline_validation_mask
        )
        adapted_rewards, adapted_metrics, adapted_std = _aggregate_metrics(
            paired_result, adapted_validation_mask
        )
        adapted_batch = validated_objective_best.unsqueeze(0).expand(
            args.particles, -1, -1
        )
        adapted_result = evaluate_sequences(
            wrapped_env, actor, adapted_batch, reference, tracking_config, collect_timeseries=True
        )
        link_paired_sequences = baseline.unsqueeze(0).expand(
            args.particles, -1, -1
        ).clone()
        link_paired_sequences[adapted_validation_mask] = validated_link_best
        link_paired_result = evaluate_sequences(
            wrapped_env, actor, link_paired_sequences, reference, tracking_config
        )
        (
            link_paired_baseline_rewards,
            link_paired_baseline_metrics,
            link_paired_baseline_std,
        ) = _aggregate_metrics(link_paired_result, baseline_validation_mask)
        link_adapted_rewards, link_adapted_metrics, link_adapted_std = _aggregate_metrics(
            link_paired_result, adapted_validation_mask
        )
        # Keep the durable "validated_*" artifacts aligned with the final paired
        # evaluation.  In validation_interval=0 screening runs these candidates are
        # promoted only after the search loop, so the files/checkpoint written by the
        # last optimization step would otherwise still contain the initialization.
        validated_objective_metrics = adapted_metrics
        validated_objective_rewards = adapted_rewards
        validated_objective_std = adapted_std
        validated_link_metrics = link_adapted_metrics
        validated_link_rewards = link_adapted_rewards
        validated_link_std = link_adapted_std
        save_optimizer_state(optimization_stop - 1)
        _save_json(
            args.output_dir / "metrics_adapted.json",
            {
                "rewards": adapted_rewards,
                "metrics": adapted_metrics,
                "population_std": adapted_std,
                "paired_baseline_rewards": paired_baseline_rewards,
                "paired_baseline_metrics": paired_baseline_metrics,
                "paired_baseline_std": paired_baseline_std,
                "validation_counts": {
                    "baseline": baseline_validation_count,
                    "adapted": adapted_validation_count,
                },
            },
        )
        _save_json(
            args.output_dir / "metrics_link_best.json",
            {
                "rewards": link_adapted_rewards,
                "metrics": link_adapted_metrics,
                "population_std": link_adapted_std,
                "paired_baseline_rewards": link_paired_baseline_rewards,
                "paired_baseline_metrics": link_paired_baseline_metrics,
                "paired_baseline_std": link_paired_baseline_std,
                "validation_counts": {
                    "baseline": baseline_validation_count,
                    "adapted": adapted_validation_count,
                },
            },
        )
        logger.log_final(
            baseline_metrics=paired_baseline_metrics,
            adapted_metrics=adapted_metrics,
            baseline_rewards=paired_baseline_rewards,
            adapted_rewards=adapted_rewards,
            step=optimization_stop,
            variant="objective_best",
        )
        logger.log_final(
            baseline_metrics=link_paired_baseline_metrics,
            adapted_metrics=link_adapted_metrics,
            baseline_rewards=link_paired_baseline_rewards,
            adapted_rewards=link_adapted_rewards,
            step=optimization_stop,
            variant="link_best",
        )
        # Save paired scalar traces so plotting does not need to join separate files.
        baseline_one = evaluate_sequences(
            wrapped_env, actor, baseline_batch, reference, tracking_config, collect_timeseries=True
        )
        series = {f"adapted_{k}": v for k, v in (adapted_result.timeseries or {}).items()}
        series.update({f"baseline_{k}": v for k, v in (baseline_one.timeseries or {}).items()})
        np.savez_compressed(args.output_dir / "tracking_timeseries.npz", **series)
        # This field reports physical safety independently of whether the
        # caller requested safety as a mandatory export gate.
        final_limit_safe = (
            adapted_metrics["max_joint_limit_utilization"]
            <= args.joint_limit_guard_fraction
            and adapted_metrics["hard_limit_contact_fraction"] == 0.0
        )
        final_tracking_multiplier = (
            1.0 + args.selection_tracking_regression_fraction
        )
        final_objective_ok = (
            adapted_metrics["objective"] + args.determinism_tolerance
            >= paired_baseline_metrics["objective"]
        )
        final_mpjpe_ok = (
            adapted_metrics["mpjpe"]
            <= final_tracking_multiplier * paired_baseline_metrics["mpjpe"]
        )
        final_link_ok = (
            adapted_metrics["mean_body_position_error"]
            <= final_tracking_multiplier
            * paired_baseline_metrics["mean_body_position_error"]
        )
        final_selection_ok = (
            final_objective_ok
            and final_mpjpe_ok
            and final_link_ok
            and (final_limit_safe or not args.require_limit_safe_best)
        )
        summary = {
            "motion": str(args.motion),
            "start_frame": args.start_frame,
            "frame_count": args.frame_count,
            "completed_iterations": optimization_stop,
            "dial_schedule_iterations": args.iterations,
            "latent_shape": list(validated_objective_best.shape),
            "latent_norm_mean": float(
                torch.linalg.vector_norm(validated_objective_best, dim=-1).mean().cpu()
            ),
            "baseline_objective": paired_baseline_metrics["objective"], "adapted_objective": adapted_metrics["objective"],
            "baseline_objective_std": paired_baseline_std["objective"], "adapted_objective_std": adapted_std["objective"],
            "baseline_repeat_objective_delta": baseline_metrics["repeat_objective_delta"],
            "baseline_repeat_mpjpe_delta": baseline_metrics["repeat_mpjpe_delta"],
            "determinism_within_tolerance": baseline_metrics["repeat_objective_delta"] <= args.determinism_tolerance,
            "baseline_mpjpe": paired_baseline_metrics["mpjpe"], "adapted_mpjpe": adapted_metrics["mpjpe"],
            "baseline_mpjpe_std": paired_baseline_std["mpjpe"], "adapted_mpjpe_std": adapted_std["mpjpe"],
            "baseline_link_relative_position_error": paired_baseline_metrics[
                "mean_body_position_error"
            ],
            "adapted_link_relative_position_error": adapted_metrics[
                "mean_body_position_error"
            ],
            "baseline_link_relative_position_error_std": paired_baseline_std[
                "mean_body_position_error"
            ],
            "adapted_link_relative_position_error_std": adapted_std[
                "mean_body_position_error"
            ],
            "link_best_objective": link_adapted_metrics["objective"],
            "link_best_mpjpe": link_adapted_metrics["mpjpe"],
            "link_best_link_relative_position_error": link_adapted_metrics[
                "mean_body_position_error"
            ],
            "link_best_link_relative_position_error_std": link_adapted_std[
                "mean_body_position_error"
            ],
            "paired_mpjpe_improvement_percent": 100.0 * (
                paired_baseline_metrics["mpjpe"] - adapted_metrics["mpjpe"]
            ) / max(abs(paired_baseline_metrics["mpjpe"]), 1.0e-12),
            "paired_objective_improvement_percent": 100.0 * (
                adapted_metrics["objective"] - paired_baseline_metrics["objective"]
            ) / max(abs(paired_baseline_metrics["objective"]), 1.0e-12),
            "paired_validation_counts": {"baseline": baseline_validation_count, "adapted": adapted_validation_count},
            "objective_non_regression": adapted_metrics["objective"] >= paired_baseline_metrics["objective"] - 1.0e-6,
            "mpjpe_improved": adapted_metrics["mpjpe"] < paired_baseline_metrics["mpjpe"],
            "require_limit_safe_best": args.require_limit_safe_best,
            "joint_limit_guard_fraction": args.joint_limit_guard_fraction,
            "selection_tracking_regression_fraction": (
                args.selection_tracking_regression_fraction
            ),
            "baseline_max_joint_limit_utilization": paired_baseline_metrics[
                "max_joint_limit_utilization"
            ],
            "adapted_max_joint_limit_utilization": adapted_metrics[
                "max_joint_limit_utilization"
            ],
            "baseline_hard_limit_contact_fraction": paired_baseline_metrics[
                "hard_limit_contact_fraction"
            ],
            "adapted_hard_limit_contact_fraction": adapted_metrics[
                "hard_limit_contact_fraction"
            ],
            "baseline_soft_limit_contact_fraction": paired_baseline_metrics[
                "soft_limit_contact_fraction"
            ],
            "adapted_soft_limit_contact_fraction": adapted_metrics[
                "soft_limit_contact_fraction"
            ],
            "limit_safe": final_limit_safe,
            "final_objective_ok": final_objective_ok,
            "final_mpjpe_ok": final_mpjpe_ok,
            "final_link_ok": final_link_ok,
            "final_selection_ok": final_selection_ok,
            "safe_best_found_during_search": safe_best_found,
            "registry": registry_snapshot(), "config": asdict(tracking_config),
        }
        _save_json(args.output_dir / "summary.json", summary)
        if args.require_limit_safe_best and not final_limit_safe:
            raise RuntimeError(
                "No validated limit-safe trajectory was produced: "
                f"max utilization={adapted_metrics['max_joint_limit_utilization']:.6f}, "
                f"required<={args.joint_limit_guard_fraction:.6f}. "
                "Artifacts were saved for resume/tuning."
            )
        if not final_selection_ok:
            raise RuntimeError(
                "Final paired validation rejected the selected trajectory: "
                f"objective_ok={final_objective_ok}, mpjpe_ok={final_mpjpe_ok}, "
                f"link_ok={final_link_ok}, limit_safe={final_limit_safe}. "
                "Artifacts were saved for resume/tuning."
            )
    finally:
        logger.close()
        wrapped_env.close()

    if args.save_video and (
        args.video_interval == 0 or optimization_stop % args.video_interval != 0
    ):
        video_best = link_best if args.video_best_kind == "link" else best
        render_comparison(args, data_path, baseline, video_best)
    print(f"[adapt] outputs={args.output_dir}")


if __name__ == "__main__":
    main()
