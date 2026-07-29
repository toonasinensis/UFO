"""Data-driven MJLab domain-randomization helpers.

Robot-specific selectors and ranges live in robot profiles.  This module only
translates the generic profile schema into MJLab event terms.
"""

from __future__ import annotations

import math
import re
from typing import Any

import torch
from mjlab.envs import mdp as mjlab_mdp
from mjlab.envs.mdp import dr as mjlab_dr
from mjlab.envs.mdp.dr._core import _get_entity_indices
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from omegaconf import OmegaConf


def _plain(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def get_profile(config) -> dict[str, Any] | None:
    """Return the robot profile, or ``None`` for legacy global DR."""
    domain_rand = config.get("domain_rand")
    if domain_rand is None:
        return None
    profile = domain_rand.get("profile")
    if profile is None:
        return None
    result = _plain(profile)
    if not isinstance(result, dict):
        raise ValueError("domain_rand.profile must be a mapping")
    return dict(result)


def disable_profile(config) -> None:
    """Disable a configured robot profile without altering legacy settings."""
    if get_profile(config) is not None:
        config.domain_rand.profile.enabled = False


def default_joint_position_range(config) -> tuple[float, float] | None:
    """Resolve the active default-joint calibration offset range."""
    profile = get_profile(config)
    if profile is not None:
        if not bool(profile.get("enabled", False)):
            return None
        term = profile.get("joint_default_position")
        if not isinstance(term, dict) or not bool(term.get("enabled", True)):
            return None
        return _range(term.get("range"), "joint_default_position.range")

    domain_rand = config.domain_rand
    if not bool(domain_rand.get("randomize_default_dof_pos", False)):
        return None
    return _range(domain_rand.default_dof_pos_noise_range, "default_dof_pos_noise_range")


def actuator_delay_kwargs(config) -> dict[str, int | bool]:
    """Resolve a delay range expressed directly in MJLab physics steps."""
    profile = get_profile(config)
    if profile is None or not bool(profile.get("enabled", False)):
        return {}
    term = profile.get("control_delay")
    if not isinstance(term, dict) or not bool(term.get("enabled", True)):
        return {}

    low, high = _integer_range(term.get("control_step_range"), "control_delay.control_step_range")

    # DelayBuffer updates at physics rate.  Sample once on the first physics
    # step after reset, then hold the sampled lag for the whole episode.
    fps = float(config.simulator.config.sim.fps)
    episode_physics_steps = max(1, math.ceil(float(config.max_episode_length_s) * fps))
    return {
        "delay_min_lag": low,
        "delay_max_lag": high,
        "delay_update_period": episode_physics_steps + 1,
        "delay_per_env_phase": False,
    }


def build_profile_events(config) -> dict[str, EventTermCfg]:
    """Build all MJLab events for an enabled robot DR profile."""
    profile = get_profile(config)
    if profile is None or not bool(profile.get("enabled", False)):
        return {}

    body_names = tuple(str(name) for name in _plain(config.robot.body_names))
    joint_names = tuple(str(name) for name in _plain(config.robot.dof_names))
    events: dict[str, EventTermCfg] = {}

    material = profile.get("material")
    if isinstance(material, dict) and bool(material.get("enabled", True)):
        geom_selector = _selector(material.get("geom_names", ".*"), "material.geom_names")
        friction_range = _range(material.get("friction_range"), "material.friction_range")
        restitution_range = _range(material.get("restitution_range"), "material.restitution_range")
        geom_cfg = SceneEntityCfg("robot", geom_names=geom_selector)
        events["material_friction"] = EventTermCfg(
            mode="startup",
            func=mjlab_dr.geom_friction,
            params={
                "asset_cfg": geom_cfg,
                "operation": "abs",
                "axes": [0],
                "ranges": friction_range,
            },
        )
        events["material_restitution"] = EventTermCfg(
            mode="startup",
            func=randomize_geom_restitution,
            params={"asset_cfg": geom_cfg, "restitution_range": restitution_range},
        )

    for name, term in _named_terms(profile.get("body_com"), "body_com"):
        selector = _selector(term.get("body_names"), f"body_com.{name}.body_names")
        _validate_matches(selector, body_names, f"body_com.{name}.body_names")
        events[f"body_com_{name}"] = EventTermCfg(
            mode="startup",
            func=mjlab_dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=selector),
                "operation": "add",
                "ranges": _xyz_ranges(term.get("ranges"), f"body_com.{name}.ranges"),
            },
        )

    for name, term in _named_terms(profile.get("body_mass"), "body_mass"):
        selector = _selector(term.get("body_names"), f"body_mass.{name}.body_names")
        _validate_matches(selector, body_names, f"body_mass.{name}.body_names")
        events[f"body_mass_{name}"] = EventTermCfg(
            mode="startup",
            func=mjlab_dr.body_mass,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=selector),
                "operation": "scale",
                "ranges": _range(term.get("range"), f"body_mass.{name}.range"),
            },
        )

    actuator_gains = profile.get("actuator_gains")
    if isinstance(actuator_gains, dict) and bool(actuator_gains.get("enabled", True)):
        events["actuator_gains"] = EventTermCfg(
            mode="startup",
            func=mjlab_dr.pd_gains,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    actuator_names=_selector(
                        actuator_gains.get("actuator_names", ".*"),
                        "actuator_gains.actuator_names",
                    ),
                ),
                "operation": "scale",
                "kp_range": _range(actuator_gains.get("kp_range"), "actuator_gains.kp_range"),
                "kd_range": _range(actuator_gains.get("kd_range"), "actuator_gains.kd_range"),
            },
        )

    joint_parameters = profile.get("joint_parameters")
    if isinstance(joint_parameters, dict) and bool(joint_parameters.get("enabled", True)):
        selector = _selector(joint_parameters.get("joint_names", ".*"), "joint_parameters.joint_names")
        _validate_matches(selector, joint_names, "joint_parameters.joint_names")
        asset_cfg = SceneEntityCfg("robot", joint_names=selector)
        events["joint_friction"] = EventTermCfg(
            mode="startup",
            func=mjlab_dr.joint_friction,
            params={
                "asset_cfg": asset_cfg,
                "operation": "scale",
                "ranges": _range(
                    joint_parameters.get("friction_range"),
                    "joint_parameters.friction_range",
                ),
            },
        )
        events["joint_armature"] = EventTermCfg(
            mode="startup",
            func=mjlab_dr.joint_armature,
            params={
                "asset_cfg": asset_cfg,
                "operation": "scale",
                "ranges": _range(
                    joint_parameters.get("armature_range"),
                    "joint_parameters.armature_range",
                ),
            },
        )

    push = profile.get("push")
    if isinstance(push, dict) and bool(push.get("enabled", True)):
        velocity = push.get("velocity")
        if not isinstance(velocity, dict):
            raise ValueError("push.velocity must be a mapping")
        expected = ("x", "y", "z", "roll", "pitch", "yaw")
        velocity_range = {axis: _range(velocity.get(axis), f"push.velocity.{axis}") for axis in expected}
        events["push_robots"] = EventTermCfg(
            mode="interval",
            func=mjlab_mdp.push_by_setting_velocity,
            interval_range_s=_range(push.get("interval_s"), "push.interval_s"),
            params={"velocity_range": velocity_range},
        )

    return events


@requires_model_fields("geom_solref")
def randomize_geom_restitution(
    env,
    env_ids: torch.Tensor | None,
    restitution_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Approximate a restitution coefficient through MuJoCo's damping ratio."""
    low, high = restitution_range
    if low < 0.0 or high > 1.0:
        raise ValueError(f"restitution_range must be within [0, 1], got {restitution_range}")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)
    asset = env.scene[asset_cfg.name]
    geom_ids = _get_entity_indices(asset.indexing, asset_cfg, "geom", False)
    restitution = torch.empty(
        (len(env_ids), len(geom_ids)), device=env.device, dtype=torch.float32
    ).uniform_(low, high)
    log_e = torch.log(torch.clamp(restitution, min=1.0e-6))
    damping_ratio = -log_e / torch.sqrt(math.pi**2 + log_e.square())
    damping_ratio = torch.where(restitution <= 1.0e-6, torch.ones_like(damping_ratio), damping_ratio)
    env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")
    env.sim.model.geom_solref[env_grid, geom_grid, 1] = damping_ratio


def _named_terms(value: Any, field: str) -> list[tuple[str, dict[str, Any]]]:
    value = _plain(value)
    if value is None:
        return []
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    result = []
    for name, term in value.items():
        if not isinstance(term, dict):
            raise ValueError(f"{field}.{name} must be a mapping")
        if bool(term.get("enabled", True)):
            result.append((str(name), dict(term)))
    return result


def _range(value: Any, field: str) -> tuple[float, float]:
    value = _plain(value)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must be a two-element range")
    low, high = float(value[0]), float(value[1])
    if low > high:
        raise ValueError(f"{field} lower bound exceeds upper bound: {(low, high)}")
    return low, high


def _integer_range(value: Any, field: str) -> tuple[int, int]:
    low, high = _range(value, field)
    if not low.is_integer() or not high.is_integer():
        raise ValueError(f"{field} must contain integers")
    return int(low), int(high)


def _xyz_ranges(value: Any, field: str) -> dict[int, tuple[float, float]]:
    value = _plain(value)
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return {index: _range(value.get(axis), f"{field}.{axis}") for index, axis in enumerate(("x", "y", "z"))}


def _selector(value: Any, field: str) -> str | tuple[str, ...]:
    value = _plain(value)
    if isinstance(value, str) and value:
        return value
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, str) and item for item in value):
        return tuple(value)
    raise ValueError(f"{field} must be a regex string or a non-empty list of regex strings")


def _validate_matches(selector: str | tuple[str, ...], available: tuple[str, ...], field: str) -> None:
    patterns = (selector,) if isinstance(selector, str) else selector
    unmatched = [pattern for pattern in patterns if not any(re.fullmatch(pattern, name) for name in available)]
    if unmatched:
        raise ValueError(f"{field} patterns matched no names: {unmatched}; available={list(available)}")
