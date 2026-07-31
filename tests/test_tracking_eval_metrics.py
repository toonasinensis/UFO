from __future__ import annotations

import unittest

import torch

from humanoidverse.agents.evaluations.humanoidverse_mjlab import _pose_state_for_metrics


class TrackingEvalMetricTests(unittest.TestCase):
    def test_pose_state_width_is_inferred_for_roban_and_g1(self) -> None:
        for num_dof in (21, 29):
            with self.subTest(num_dof=num_dof):
                observation_state = torch.zeros(4, 2 * num_dof + 6)
                target_state = torch.zeros_like(observation_state)

                # A legacy fixed-width slice would leak these velocity values
                # into the Roban metric.  The inferred pose slice must not.
                observation_state[:, num_dof : num_dof + 2] = 100.0

                actual, target = _pose_state_for_metrics(
                    {
                        "observation": {"state": observation_state},
                        "tracking_target": {"state": target_state},
                        "target_joint_pos": torch.zeros(4, num_dof),
                    }
                )

                self.assertEqual(actual.shape, (4, num_dof))
                self.assertEqual(target.shape, (4, num_dof))
                torch.testing.assert_close(actual, target)

    def test_pose_state_rejects_state_narrower_than_robot_dofs(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_dof=21"):
            _pose_state_for_metrics(
                {
                    "observation": {"state": torch.zeros(4, 20)},
                    "tracking_target": {"state": torch.zeros(4, 20)},
                    "target_joint_pos": torch.zeros(4, 21),
                }
            )


if __name__ == "__main__":
    unittest.main()
