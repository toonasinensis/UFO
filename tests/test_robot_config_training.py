from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from humanoidverse.agents.envs.humanoidverse_mjlab import (
    HumanoidVerseMjlabCore,
    _compose_humanoidverse_config,
    make_mjlab_ufo_env_cfg,
)
from humanoidverse.agents.envs.mjlab_domain_randomization import (
    actuator_delay_kwargs,
    build_profile_events,
    default_joint_position_range,
)
from humanoidverse.tracking_inference import (
    _expert_qpos_from_obs,
    _resolve_tracking_robot_config,
    _target_states_from_obs,
)
from humanoidverse.tracking_inference import (
    parse_args as parse_tracking_args,
)
from humanoidverse.train import _resolve_training_robot_config, build_ufo_mjlab_config
from humanoidverse.train import parse_args as parse_train_args
from humanoidverse.utils.robot_spec import load_robot_training_spec


def _write_tiny_robot_with_training(root: Path, *, missing_actuator_joint: bool = False) -> Path:
    xml_path = root / "tiny_train.xml"
    xml_path.write_text(
        """
<mujoco model="tiny_train">
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="sphere" size="0.05" mass="1"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" mass="0.1"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-2 2"/>
          <geom type="sphere" size="0.03" mass="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="joint1_motor" joint="joint1"/>
    <motor name="joint2_motor" joint="joint2"/>
  </actuator>
</mujoco>
""".strip()
    )
    joint2_block = (
        []
        if missing_actuator_joint
        else [
            "        joint2:",
            "          effort_limit: 2.0",
            "          velocity_limit: 20.0",
            "          armature: 0.02",
            "          friction: 0.002",
        ]
    )
    robot_config = root / "tiny_train.yaml"
    robot_config.write_text(
        "\n".join(
            [
                "name: tiny_train",
                "xml_path: tiny_train.xml",
                "base_body: base",
                "root_quat_order: xyzw",
                "coordinate_system: z_up",
                "dof_unit: rad",
                "control_joints:",
                "  mode: all_actuated",
                "feet: [link2]",
                "hands: []",
                "key_bodies: [base, link1, link2]",
                "default_dof_pos: {}",
                "training:",
                "  hydra_robot: g1/g1_29dof_hard_waist",
                "  hydra_overrides: []",
                "  semantics:",
                "    contact_bodies: [link2]",
                "    undesired_contact_bodies: [base]",
                "    torso_name: base",
                "    left_ankle_dof_names: []",
                "    right_ankle_dof_names: []",
                "  init_state:",
                "    pos: [0.0, 0.0, 1.0]",
                "    rot: [0.0, 0.0, 0.0, 1.0]",
                "    lin_vel: [0.0, 0.0, 0.0]",
                "    ang_vel: [0.0, 0.0, 0.0]",
                "    default_joint_angles:",
                "      joint1: 0.0",
                "      joint2: 0.0",
                "  control:",
                "    action_scale: 0.25",
                "    action_clip_value: 5.0",
                "    normalize_action_to: 5.0",
                "    effort_limit: [1.0, 2.0]",
                "    velocity_limit: [10.0, 20.0]",
                "    stiffness: {joint1: 1.0, joint2: 2.0}",
                "    damping: {joint1: 0.1, joint2: 0.2}",
                "  actuator:",
                "    source: yaml",
                "    joints:",
                "      joint1:",
                "        effort_limit: 1.0",
                "        velocity_limit: 10.0",
                "        armature: 0.01",
                "        friction: 0.001",
                *joint2_block,
            ]
        )
    )
    return robot_config


class RobotConfigTrainingTest(unittest.TestCase):
    def _compose_roban(self, *, disable_dr: bool = False):
        train_cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_dr_unit",
            num_envs=2,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            robot_config="configs/robots/roban_s22.yaml",
            data_path="unused.pkl",
            disable_dr=disable_dr,
        )
        env_cfg = train_cfg.env
        hv_cfg, _ = _compose_humanoidverse_config(
            num_envs=2,
            relative_config_path=env_cfg.relative_config_path,
            hydra_overrides=list(env_cfg.hydra_overrides),
            headless=True,
            lafan_tail_path=env_cfg.lafan_tail_path,
            data_mix_weights=None,
            disable_obs_noise=False,
            disable_domain_randomization=disable_dr,
            max_episode_length_s=None,
            root_height_obs=True,
            robot_training=env_cfg.robot_training,
        )
        return env_cfg, hv_cfg

    def test_old_g1_default_builds_cfg(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_unit",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
        )
        self.assertTrue(str(cfg.env.robot_config_path).endswith("configs/robots/g1_29dof.yaml"))
        self.assertTrue(str(cfg.env.mjcf_path).endswith("humanoidverse/data/robots/g1_mjlab/g1_29dof.xml"))

    def test_explicit_g1_robot_config_builds_cfg(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_unit",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            robot_config="configs/robots/g1_29dof.yaml",
        )
        self.assertTrue(str(cfg.env.robot_config_path).endswith("configs/robots/g1_29dof.yaml"))

    def test_manifest_robot_config_is_used_when_cli_missing(self) -> None:
        argv = [
            "train.py",
            "--agent",
            "fb",
            "--data-manifest",
            "configs/data/example_mix.yaml",
            "--gpu-ids",
            "single",
            "--smoke",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_train_args()
        self.assertTrue(str(args.robot_config).endswith("configs/robots/g1_29dof.yaml"))

    def test_cli_manifest_robot_config_mismatch_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_robot = _write_tiny_robot_with_training(Path(tmpdir))
            with self.assertRaisesRegex(ValueError, "does not match data manifest robot_config"):
                _resolve_training_robot_config(tiny_robot, "configs/robots/g1_29dof.yaml")

    def test_tracking_manifest_robot_config_is_used_when_cli_missing(self) -> None:
        argv = [
            "tracking_inference.py",
            "--model-folder",
            "/tmp/ufo_unit_model",
            "--data-manifest",
            "configs/data/example_robot_state_auto_build.yaml",
            "--dataset",
            "g1_robot_state_sample",
            "--export-onnx",
            "false",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_tracking_args()
        self.assertTrue(str(args.robot_config).endswith("configs/robots/g1_29dof.yaml"))

    def test_tracking_cli_manifest_robot_config_mismatch_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_robot = _write_tiny_robot_with_training(Path(tmpdir))
            with self.assertRaisesRegex(ValueError, "does not match data manifest robot_config"):
                _resolve_tracking_robot_config(tiny_robot, "configs/robots/g1_29dof.yaml")

    def test_aux_rewards_require_two_contact_bodies_unconditionally(self) -> None:
        core = object.__new__(HumanoidVerseMjlabCore)
        core.reward_scales = {}
        cfg = OmegaConf.create(
            {
                "robot": {
                    "contact_bodies": ["left_foot"],
                    "left_ankle_dof_names": ["left_ankle_pitch_joint", "left_ankle_roll_joint"],
                    "right_ankle_dof_names": ["right_ankle_pitch_joint", "right_ankle_roll_joint"],
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "robot.contact_bodies.*biped foot auxiliary terms unconditionally"):
            core._validate_aux_reward_semantics(cfg)

    def test_aux_ankle_reward_requires_both_ankle_fields(self) -> None:
        core = object.__new__(HumanoidVerseMjlabCore)
        core.reward_scales = {"penalty_ankle_roll": -1.0}
        cfg = OmegaConf.create(
            {
                "robot": {
                    "contact_bodies": ["left_foot", "right_foot"],
                    "left_ankle_dof_names": ["left_ankle_pitch_joint"],
                    "right_ankle_dof_names": [],
                }
            }
        )
        with self.assertRaisesRegex(
            ValueError,
            "robot.left_ankle_dof_names, robot.right_ankle_dof_names.*penalty_ankle_roll",
        ):
            core._validate_aux_reward_semantics(cfg)


    def test_mjlab_action_input_reorders_policy_actions_to_action_term_order(self) -> None:
        core = object.__new__(HumanoidVerseMjlabCore)
        core.actions = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        core.default_dof_pos_offset = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        core.action_target_scale = torch.tensor([[1.0, 2.0, 1.0, 4.0]])
        core._action_term_dof_indices = torch.tensor([2, 0, 3, 1])

        action_input = core._mjlab_action_input()

        torch.testing.assert_close(action_input, torch.tensor([[33.0, 11.0, 41.0, 21.0]]))

    def test_yaml_actuator_missing_joint_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_robot = _write_tiny_robot_with_training(Path(tmpdir), missing_actuator_joint=True)
            with self.assertRaisesRegex(ValueError, "missing parameters for joint 'joint2'"):
                load_robot_training_spec(tiny_robot)

    def test_roban_domain_randomization_profile_is_centralized(self) -> None:
        spec = load_robot_training_spec("configs/robots/roban_s22.yaml")
        self.assertIsNotNone(spec.domain_randomization)
        assert spec.domain_randomization is not None
        self.assertTrue(spec.domain_randomization["enabled"])
        self.assertTrue(
            str(spec.domain_randomization["_profile_path"]).endswith(
                "humanoidverse/data/robots/biped_s17/config/domain_randomization.yaml"
            )
        )

    def test_roban_domain_randomization_events_and_delay(self) -> None:
        env_cfg, hv_cfg = self._compose_roban()
        events = build_profile_events(hv_cfg)
        self.assertEqual(
            set(events),
            {
                "material_friction",
                "material_restitution",
                "body_com_base",
                "body_com_waist",
                "body_com_limbs",
                "body_mass_base",
                "body_mass_waist",
                "body_mass_limbs",
                "actuator_gains",
                "joint_friction",
                "joint_armature",
                "push_robots",
            },
        )
        self.assertEqual(events["push_robots"].params["velocity_range"]["z"], (-0.1, 0.1))
        delay = actuator_delay_kwargs(hv_cfg)
        self.assertEqual(delay["delay_min_lag"], 0)
        self.assertEqual(delay["delay_max_lag"], 5)
        self.assertFalse(delay["delay_per_env_phase"])
        self.assertEqual(default_joint_position_range(hv_cfg), (-0.03, 0.03))

        mjlab_cfg = make_mjlab_ufo_env_cfg(
            hv_cfg,
            num_envs=2,
            seed=1,
            mjcf_path=env_cfg.mjcf_path,
            auto_reset=False,
            robot_training=env_cfg.robot_training,
        )
        self.assertEqual(set(mjlab_cfg.events), set(events))

    def test_disable_dr_overrides_roban_profile(self) -> None:
        env_cfg, hv_cfg = self._compose_roban(disable_dr=True)
        self.assertFalse(hv_cfg.domain_rand.profile.enabled)
        self.assertEqual(build_profile_events(hv_cfg), {})
        self.assertEqual(actuator_delay_kwargs(hv_cfg), {})
        self.assertIsNone(default_joint_position_range(hv_cfg))

        mjlab_cfg = make_mjlab_ufo_env_cfg(
            hv_cfg,
            num_envs=2,
            seed=1,
            mjcf_path=env_cfg.mjcf_path,
            auto_reset=False,
            robot_training=env_cfg.robot_training,
        )
        self.assertEqual(mjlab_cfg.events, {})
        self.assertEqual(mjlab_cfg.scene.entities["robot"].articulation.actuators[0].delay_max_lag, 0)

    def test_profile_enabled_defaults_to_false_and_bad_selectors_fail_fast(self) -> None:
        _, hv_cfg = self._compose_roban()
        del hv_cfg.domain_rand.profile["enabled"]
        self.assertEqual(build_profile_events(hv_cfg), {})

        hv_cfg.domain_rand.profile.enabled = True
        hv_cfg.domain_rand.profile.body_com.base.body_names = "missing_body"
        with self.assertRaisesRegex(ValueError, "matched no names"):
            build_profile_events(hv_cfg)

    def test_tracking_shapes_follow_num_dof(self) -> None:
        obs = {
            "ref_body_pos": torch.zeros(4, 1, 3),
            "ref_body_rots": torch.zeros(4, 1, 4),
            "ref_body_vels": torch.zeros(4, 1, 3),
            "ref_body_angular_vels": torch.zeros(4, 1, 3),
            "dof_pos": torch.zeros(4, 2),
            "ref_dof_vel": torch.ones(4, 2),
        }
        obs["ref_body_rots"][..., 3] = 1.0
        qpos = _expert_qpos_from_obs(obs, num_dof=2, dof_qpos_order_indices=torch.tensor([0, 1]).numpy())
        self.assertEqual(qpos.shape, (4, 9))
        target = _target_states_from_obs(obs, device="cpu", num_dof=2)
        self.assertEqual(tuple(target["dof_states"].shape), (1, 2, 2))


if __name__ == "__main__":
    unittest.main()
