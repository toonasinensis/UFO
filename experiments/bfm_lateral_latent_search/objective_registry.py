"""Composable objective, failure, and metric registries for latent search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch


@dataclass(frozen=True)
class ObjectiveConfig:
    target_forward_speed: float
    target_lateral_speed: float
    velocity_sigma: float
    yaw_rate_sigma: float
    foot_flatness_sigma: float
    min_upright: float
    min_root_height: float
    reward_weights: dict[str, float] | None = None


@dataclass(frozen=True)
class StepSignals:
    body_vx: torch.Tensor
    body_vy: torch.Tensor
    body_yaw_rate: torch.Tensor
    upright: torch.Tensor
    root_height: torch.Tensor
    left_foot_flatness_error: torch.Tensor
    right_foot_flatness_error: torch.Tensor


@dataclass(frozen=True)
class WeightedTerm:
    name: str
    weight: float
    function: Callable[[StepSignals, ObjectiveConfig], torch.Tensor]


@dataclass(frozen=True)
class NamedTerm:
    name: str
    function: Callable[[StepSignals, ObjectiveConfig], torch.Tensor]


Reduction = Literal["mean", "rms", "min", "max"]


@dataclass(frozen=True)
class MetricTerm:
    name: str
    reduction: Reduction
    function: Callable[[StepSignals, ObjectiveConfig], torch.Tensor]


REWARD_TERMS: dict[str, WeightedTerm] = {}
GATE_TERMS: dict[str, NamedTerm] = {}
FAILURE_CONDITIONS: dict[str, NamedTerm] = {}
METRIC_TERMS: dict[str, MetricTerm] = {}


def _add_unique(registry: dict, name: str, value) -> None:
    if name in registry:
        raise ValueError(f"Duplicate registered term: {name}")
    registry[name] = value


def register_reward_term(name: str, *, weight: float):
    def decorator(function):
        _add_unique(REWARD_TERMS, name, WeightedTerm(name, float(weight), function))
        return function

    return decorator


def register_gate(name: str):
    def decorator(function):
        _add_unique(GATE_TERMS, name, NamedTerm(name, function))
        return function

    return decorator


def register_failure_condition(name: str):
    def decorator(function):
        _add_unique(FAILURE_CONDITIONS, name, NamedTerm(name, function))
        return function

    return decorator


def register_metric(name: str, *, reduction: Reduction = "mean"):
    def decorator(function):
        _add_unique(METRIC_TERMS, name, MetricTerm(name, reduction, function))
        return function

    return decorator


@register_reward_term("velocity_tracking", weight=0.75)
def velocity_tracking(signals: StepSignals, config: ObjectiveConfig) -> torch.Tensor:
    return torch.exp(
        -0.5
        * (
            torch.square((signals.body_vx - config.target_forward_speed) / config.velocity_sigma)
            + torch.square((signals.body_vy - config.target_lateral_speed) / config.velocity_sigma)
        )
    )


@register_reward_term("yaw_stability", weight=0.15)
def yaw_stability(signals: StepSignals, config: ObjectiveConfig) -> torch.Tensor:
    return torch.exp(-0.5 * torch.square(signals.body_yaw_rate / config.yaw_rate_sigma))


@register_reward_term("feet_flatness", weight=0.10)
def feet_flatness(signals: StepSignals, config: ObjectiveConfig) -> torch.Tensor:
    """Reward level contacting feet using the training aux-reward signal.

    Each per-foot error is the XY norm of world gravity expressed in the foot
    frame, multiplied by that foot's contact indicator.  Consequently zero is
    level and the error grows with roll/pitch.  The orientation signal matches
    ``penalty_feet_ori`` and its robust force-norm contact gate matches
    ``penalty_slippage`` in ``HumanoidVerseMjlabEnv._compute_reward``.
    """
    error = signals.left_foot_flatness_error + signals.right_foot_flatness_error
    return torch.exp(-0.5 * torch.square(error / config.foot_flatness_sigma))


def configured_reward_weight(config: ObjectiveConfig, name: str) -> float:
    """Return the effective weight for a registered reward term."""
    default = REWARD_TERMS[name].weight
    if config.reward_weights is None:
        return default
    return float(config.reward_weights.get(name, default))


@register_gate("upright")
def upright_gate(signals: StepSignals, _config: ObjectiveConfig) -> torch.Tensor:
    return torch.clamp((signals.upright - 0.40) / 0.45, 0.0, 1.0)


@register_gate("root_height")
def root_height_gate(signals: StepSignals, _config: ObjectiveConfig) -> torch.Tensor:
    return torch.clamp((signals.root_height - 0.35) / 0.30, 0.0, 1.0)


@register_failure_condition("low_upright")
def low_upright(signals: StepSignals, config: ObjectiveConfig) -> torch.Tensor:
    return signals.upright < config.min_upright


@register_failure_condition("low_root_height")
def low_root_height(signals: StepSignals, config: ObjectiveConfig) -> torch.Tensor:
    return signals.root_height < config.min_root_height


@register_metric("mean_body_vx")
def metric_body_vx(signals: StepSignals, _config: ObjectiveConfig) -> torch.Tensor:
    return signals.body_vx


@register_metric("mean_body_vy")
def metric_body_vy(signals: StepSignals, _config: ObjectiveConfig) -> torch.Tensor:
    return signals.body_vy


@register_metric("mean_abs_forward_error")
def metric_abs_forward_error(signals: StepSignals, config: ObjectiveConfig) -> torch.Tensor:
    return torch.abs(signals.body_vx - config.target_forward_speed)


@register_metric("rms_forward_error", reduction="rms")
def metric_rms_forward_error(signals: StepSignals, config: ObjectiveConfig) -> torch.Tensor:
    return signals.body_vx - config.target_forward_speed


@register_metric("mean_abs_lateral_error")
def metric_abs_lateral_error(signals: StepSignals, config: ObjectiveConfig) -> torch.Tensor:
    return torch.abs(signals.body_vy - config.target_lateral_speed)


@register_metric("rms_lateral_error", reduction="rms")
def metric_rms_lateral_error(signals: StepSignals, config: ObjectiveConfig) -> torch.Tensor:
    return signals.body_vy - config.target_lateral_speed


@register_metric("rms_planar_velocity_error", reduction="rms")
def metric_rms_planar_velocity_error(signals: StepSignals, config: ObjectiveConfig) -> torch.Tensor:
    return torch.sqrt(
        torch.square(signals.body_vx - config.target_forward_speed)
        + torch.square(signals.body_vy - config.target_lateral_speed)
    )


@register_metric("mean_abs_yaw_rate")
def metric_abs_yaw_rate(signals: StepSignals, _config: ObjectiveConfig) -> torch.Tensor:
    return torch.abs(signals.body_yaw_rate)


@register_metric("mean_signed_yaw_rate")
def metric_signed_yaw_rate(signals: StepSignals, _config: ObjectiveConfig) -> torch.Tensor:
    return signals.body_yaw_rate


@register_metric("mean_upright")
def metric_upright(signals: StepSignals, _config: ObjectiveConfig) -> torch.Tensor:
    return signals.upright


@register_metric("min_upright", reduction="min")
def metric_min_upright(signals: StepSignals, _config: ObjectiveConfig) -> torch.Tensor:
    return signals.upright


@register_metric("mean_root_height")
def metric_root_height(signals: StepSignals, _config: ObjectiveConfig) -> torch.Tensor:
    return signals.root_height


@register_metric("min_root_height", reduction="min")
def metric_min_root_height(signals: StepSignals, _config: ObjectiveConfig) -> torch.Tensor:
    return signals.root_height


@register_metric("mean_velocity_score")
def metric_velocity_score(signals: StepSignals, config: ObjectiveConfig) -> torch.Tensor:
    return velocity_tracking(signals, config)


@register_metric("mean_yaw_score")
def metric_yaw_score(signals: StepSignals, config: ObjectiveConfig) -> torch.Tensor:
    return yaw_stability(signals, config)


@register_metric("mean_stand_gate")
def metric_stand_gate(signals: StepSignals, config: ObjectiveConfig) -> torch.Tensor:
    return compute_gate(signals, config)


@register_metric("mean_left_foot_flatness_error")
def metric_left_foot_flatness_error(signals: StepSignals, _config: ObjectiveConfig) -> torch.Tensor:
    return signals.left_foot_flatness_error


@register_metric("mean_right_foot_flatness_error")
def metric_right_foot_flatness_error(signals: StepSignals, _config: ObjectiveConfig) -> torch.Tensor:
    return signals.right_foot_flatness_error


@register_metric("mean_feet_flatness_error")
def metric_feet_flatness_error(signals: StepSignals, _config: ObjectiveConfig) -> torch.Tensor:
    return signals.left_foot_flatness_error + signals.right_foot_flatness_error


@register_metric("mean_feet_flatness_score")
def metric_feet_flatness_score(signals: StepSignals, config: ObjectiveConfig) -> torch.Tensor:
    return feet_flatness(signals, config)


@register_metric("mean_velocity_weighted_contribution")
def metric_velocity_weighted_contribution(signals: StepSignals, config: ObjectiveConfig) -> torch.Tensor:
    return (
        compute_gate(signals, config)
        * configured_reward_weight(config, "velocity_tracking")
        * velocity_tracking(signals, config)
    )


@register_metric("mean_yaw_weighted_contribution")
def metric_yaw_weighted_contribution(signals: StepSignals, config: ObjectiveConfig) -> torch.Tensor:
    return (
        compute_gate(signals, config)
        * configured_reward_weight(config, "yaw_stability")
        * yaw_stability(signals, config)
    )


@register_metric("mean_feet_flatness_weighted_contribution")
def metric_feet_flatness_weighted_contribution(signals: StepSignals, config: ObjectiveConfig) -> torch.Tensor:
    return (
        compute_gate(signals, config)
        * configured_reward_weight(config, "feet_flatness")
        * feet_flatness(signals, config)
    )


@register_metric("mean_instant_objective")
def metric_instant_objective(signals: StepSignals, config: ObjectiveConfig) -> torch.Tensor:
    return compute_instant_objective(signals, config)


def compute_gate(signals: StepSignals, config: ObjectiveConfig) -> torch.Tensor:
    gate = torch.ones_like(signals.body_vx)
    for term in GATE_TERMS.values():
        gate = gate * term.function(signals, config)
    return gate


def compute_instant_objective(signals: StepSignals, config: ObjectiveConfig) -> torch.Tensor:
    reward = torch.zeros_like(signals.body_vx)
    for term in REWARD_TERMS.values():
        weight = configured_reward_weight(config, term.name)
        reward = reward + weight * term.function(signals, config)
    return compute_gate(signals, config) * reward


def compute_failure(signals: StepSignals, config: ObjectiveConfig) -> torch.Tensor:
    failure = torch.zeros_like(signals.body_vx, dtype=torch.bool)
    for term in FAILURE_CONDITIONS.values():
        failure = failure | term.function(signals, config).bool()
    return failure


def registry_snapshot() -> dict[str, object]:
    """Return an auditable description of all automatically invoked terms."""
    return {
        "reward_terms": {
            name: {"weight": term.weight, "function": term.function.__name__}
            for name, term in REWARD_TERMS.items()
        },
        "gates": {name: term.function.__name__ for name, term in GATE_TERMS.items()},
        "failure_conditions": {name: term.function.__name__ for name, term in FAILURE_CONDITIONS.items()},
        "metrics": {
            name: {"reduction": term.reduction, "function": term.function.__name__}
            for name, term in METRIC_TERMS.items()
        },
    }


class MetricAccumulator:
    """Automatically aggregate every function in ``METRIC_TERMS``."""

    def __init__(self, population: int, *, device: str):
        self.count = torch.zeros(population, device=device)
        self.values: dict[str, torch.Tensor] = {}
        for term in METRIC_TERMS.values():
            if term.reduction == "min":
                initial = torch.full((population,), torch.inf, device=device)
            elif term.reduction == "max":
                initial = torch.full((population,), -torch.inf, device=device)
            else:
                initial = torch.zeros(population, device=device)
            self.values[term.name] = initial

    def update(self, signals: StepSignals, config: ObjectiveConfig, valid: torch.Tensor) -> None:
        weight = valid.float()
        self.count += weight
        for term in METRIC_TERMS.values():
            value = term.function(signals, config)
            aggregate = self.values[term.name]
            if term.reduction == "mean":
                aggregate += value * weight
            elif term.reduction == "rms":
                aggregate += torch.square(value) * weight
            elif term.reduction == "min":
                self.values[term.name] = torch.where(valid, torch.minimum(aggregate, value), aggregate)
            elif term.reduction == "max":
                self.values[term.name] = torch.where(valid, torch.maximum(aggregate, value), aggregate)

    def finalize(self) -> dict[str, torch.Tensor]:
        denominator = self.count.clamp_min(1.0)
        result: dict[str, torch.Tensor] = {}
        for term in METRIC_TERMS.values():
            value = self.values[term.name]
            if term.reduction == "mean":
                value = value / denominator
            elif term.reduction == "rms":
                value = torch.sqrt(value / denominator)
            value = torch.where(self.count > 0, value, torch.zeros_like(value))
            result[term.name] = value
        return result
