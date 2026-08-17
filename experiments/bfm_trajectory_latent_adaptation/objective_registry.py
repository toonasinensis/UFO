"""Registered tracking and joint-limit rewards for trajectory adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class TrackingSignals:
    root_position_error: torch.Tensor
    root_rotation_error: torch.Tensor
    body_position_error: torch.Tensor
    body_rotation_error: torch.Tensor
    joint_position_error: torch.Tensor
    body_linear_velocity_error: torch.Tensor
    body_angular_velocity_error: torch.Tensor
    global_body_position_error: torch.Tensor
    joint_limit_error: torch.Tensor
    joint_limit_utilization: torch.Tensor
    hard_limit_contact: torch.Tensor
    soft_limit_contact: torch.Tensor


@dataclass(frozen=True)
class TrackingConfig:
    weights: dict[str, float]
    sigmas: dict[str, float]
    joint_limit_safe_fraction: float = 0.90


@dataclass(frozen=True)
class RewardTerm:
    name: str
    signal: str
    function: Callable[[TrackingSignals, TrackingConfig], torch.Tensor]
    reduction: str


@dataclass(frozen=True)
class MetricTerm:
    name: str
    function: Callable[[TrackingSignals], torch.Tensor]
    reduction: str


REWARD_TERMS: dict[str, RewardTerm] = {}
METRIC_TERMS: dict[str, MetricTerm] = {}


def register_reward(name: str, *, signal: str, reduction: str = "mean"):
    if reduction not in {"mean", "min"}:
        raise ValueError(f"Unsupported reward reduction: {reduction}")

    def decorator(function):
        if name in REWARD_TERMS:
            raise ValueError(f"Duplicate reward term: {name}")
        REWARD_TERMS[name] = RewardTerm(
            name=name,
            signal=signal,
            function=function,
            reduction=reduction,
        )
        return function

    return decorator


def register_metric(name: str, *, reduction: str = "mean"):
    if reduction not in {"mean", "max"}:
        raise ValueError(f"Unsupported metric reduction: {reduction}")

    def decorator(function):
        if name in METRIC_TERMS:
            raise ValueError(f"Duplicate metric term: {name}")
        METRIC_TERMS[name] = MetricTerm(
            name=name, function=function, reduction=reduction
        )
        return function

    return decorator


def gaussian_reward(error: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        raise ValueError(f"Reward sigma must be positive, got {sigma}")
    return torch.exp(-0.5 * torch.square(error / float(sigma)))


def joint_limit_signals(
    dof_pos: torch.Tensor,
    hard_limits: torch.Tensor,
    soft_limits: torch.Tensor,
    *,
    safe_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return worst-joint limit risk and exact per-step safety signals.

    ``utilization`` is zero at each hard-range center and one at either XML
    hard limit.  The reward error is zero inside ``safe_fraction`` and rises
    linearly to one at the hard limit.  Taking the worst joint avoids hiding a
    single ankle collision behind the other 20 safe joints.
    """

    if dof_pos.ndim != 2:
        raise ValueError(f"Expected dof_pos [batch,joints], got {tuple(dof_pos.shape)}")
    expected_limits = (dof_pos.shape[1], 2)
    if hard_limits.shape != expected_limits or soft_limits.shape != expected_limits:
        raise ValueError(
            "Expected hard/soft limits "
            f"{expected_limits}, got hard={tuple(hard_limits.shape)}, "
            f"soft={tuple(soft_limits.shape)}"
        )
    if not 0.0 <= safe_fraction < 1.0:
        raise ValueError(
            f"joint limit safe fraction must be in [0,1), got {safe_fraction}"
        )
    if not torch.isfinite(dof_pos).all():
        raise ValueError("dof_pos contains NaN/Inf")
    if not torch.isfinite(hard_limits).all() or not torch.isfinite(soft_limits).all():
        raise ValueError("joint limits contain NaN/Inf")

    hard_lower, hard_upper = hard_limits.unbind(dim=-1)
    hard_half_range = 0.5 * (hard_upper - hard_lower)
    if bool(torch.any(hard_half_range <= 0.0)):
        raise ValueError("hard joint limits must have positive width")
    soft_lower, soft_upper = soft_limits.unbind(dim=-1)
    if bool(torch.any(soft_upper <= soft_lower)):
        raise ValueError("soft joint limits must have positive width")

    hard_center = 0.5 * (hard_lower + hard_upper)
    per_joint_utilization = torch.abs(
        (dof_pos - hard_center.unsqueeze(0)) / hard_half_range.unsqueeze(0)
    )
    utilization = torch.amax(per_joint_utilization, dim=-1)
    error = torch.amax(
        torch.relu(
            (per_joint_utilization - float(safe_fraction))
            / (1.0 - float(safe_fraction))
        ),
        dim=-1,
    )
    hard_contact = torch.any(per_joint_utilization >= 1.0, dim=-1).to(dof_pos.dtype)
    soft_contact = torch.any(
        (dof_pos <= soft_lower.unsqueeze(0))
        | (dof_pos >= soft_upper.unsqueeze(0)),
        dim=-1,
    ).to(dof_pos.dtype)
    return error, utilization, hard_contact, soft_contact


def _signal(signals: TrackingSignals, name: str) -> torch.Tensor:
    return getattr(signals, name)


@register_reward("root_position", signal="root_position_error")
def reward_root_position(signals: TrackingSignals, config: TrackingConfig) -> torch.Tensor:
    return gaussian_reward(signals.root_position_error, config.sigmas["root_position"])


@register_reward("root_rotation", signal="root_rotation_error")
def reward_root_rotation(signals: TrackingSignals, config: TrackingConfig) -> torch.Tensor:
    return gaussian_reward(signals.root_rotation_error, config.sigmas["root_rotation"])


@register_reward("body_position", signal="body_position_error")
def reward_body_position(signals: TrackingSignals, config: TrackingConfig) -> torch.Tensor:
    return gaussian_reward(signals.body_position_error, config.sigmas["body_position"])


@register_reward("body_rotation", signal="body_rotation_error")
def reward_body_rotation(signals: TrackingSignals, config: TrackingConfig) -> torch.Tensor:
    return gaussian_reward(signals.body_rotation_error, config.sigmas["body_rotation"])


@register_reward("joint_position", signal="joint_position_error")
def reward_joint_position(signals: TrackingSignals, config: TrackingConfig) -> torch.Tensor:
    return gaussian_reward(signals.joint_position_error, config.sigmas["joint_position"])


@register_reward("body_linear_velocity", signal="body_linear_velocity_error")
def reward_body_linear_velocity(signals: TrackingSignals, config: TrackingConfig) -> torch.Tensor:
    return gaussian_reward(signals.body_linear_velocity_error, config.sigmas["body_linear_velocity"])


@register_reward("body_angular_velocity", signal="body_angular_velocity_error")
def reward_body_angular_velocity(signals: TrackingSignals, config: TrackingConfig) -> torch.Tensor:
    return gaussian_reward(signals.body_angular_velocity_error, config.sigmas["body_angular_velocity"])


@register_reward("joint_limit", signal="joint_limit_error")
def reward_joint_limit(signals: TrackingSignals, config: TrackingConfig) -> torch.Tensor:
    return gaussian_reward(signals.joint_limit_error, config.sigmas["joint_limit"])


for _reward_name, _term in tuple(REWARD_TERMS.items()):
    register_metric(f"mean_{_term.signal}")(
        lambda signals, signal_name=_term.signal: _signal(signals, signal_name)
    )


@register_metric("mpjpe")
def metric_mpjpe(signals: TrackingSignals) -> torch.Tensor:
    return signals.global_body_position_error


@register_metric("joint_mae")
def metric_joint_mae(signals: TrackingSignals) -> torch.Tensor:
    return signals.joint_position_error


@register_metric("max_joint_limit_utilization", reduction="max")
def metric_max_joint_limit_utilization(signals: TrackingSignals) -> torch.Tensor:
    return signals.joint_limit_utilization


@register_metric("hard_limit_contact_fraction")
def metric_hard_limit_contact_fraction(signals: TrackingSignals) -> torch.Tensor:
    return signals.hard_limit_contact


@register_metric("soft_limit_contact_fraction")
def metric_soft_limit_contact_fraction(signals: TrackingSignals) -> torch.Tensor:
    return signals.soft_limit_contact


def validate_config(config: TrackingConfig) -> None:
    expected = set(REWARD_TERMS)
    if set(config.weights) != expected:
        raise ValueError(f"Tracking weights must have keys {sorted(expected)}, got {sorted(config.weights)}")
    if set(config.sigmas) != expected:
        raise ValueError(f"Tracking sigmas must have keys {sorted(expected)}, got {sorted(config.sigmas)}")
    if any(value < 0.0 for value in config.weights.values()):
        raise ValueError("Tracking weights must be non-negative")
    if not abs(sum(config.weights.values()) - 1.0) < 1.0e-6:
        raise ValueError(f"Tracking weights must sum to 1, got {sum(config.weights.values())}")
    if any(value <= 0.0 for value in config.sigmas.values()):
        raise ValueError("Tracking sigmas must be positive")
    if not 0.0 <= config.joint_limit_safe_fraction < 1.0:
        raise ValueError(
            "joint_limit_safe_fraction must be in [0,1), got "
            f"{config.joint_limit_safe_fraction}"
        )


def compute_rewards(
    signals: TrackingSignals, config: TrackingConfig
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    validate_config(config)
    components = {name: term.function(signals, config) for name, term in REWARD_TERMS.items()}
    objective = sum(config.weights[name] * value for name, value in components.items())
    return objective, components


def compute_metrics(signals: TrackingSignals) -> dict[str, torch.Tensor]:
    return {name: term.function(signals) for name, term in METRIC_TERMS.items()}


def metric_reductions() -> dict[str, str]:
    return {name: term.reduction for name, term in METRIC_TERMS.items()}


def reward_reductions() -> dict[str, str]:
    return {name: term.reduction for name, term in REWARD_TERMS.items()}


def registry_snapshot() -> dict[str, object]:
    return {
        "rewards": list(REWARD_TERMS),
        "reward_reductions": reward_reductions(),
        "metrics": list(METRIC_TERMS),
        "metric_reductions": metric_reductions(),
    }
