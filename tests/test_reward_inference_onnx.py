from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from humanoidverse.reward_inference_onnx import _read_sample_block, _reward_weighted_latent


class OnnxRewardInferenceTest(unittest.TestCase):
    def test_reward_weighted_latent_has_fb_norm(self) -> None:
        encoded = np.eye(4, dtype=np.float32)
        rewards = np.array([0.0, 0.1, 0.5, 1.0], dtype=np.float32)
        latent = _reward_weighted_latent(encoded, rewards)
        self.assertEqual(latent.shape, (1, 4))
        self.assertAlmostEqual(float(np.linalg.norm(latent)), math.sqrt(4), places=5)
        self.assertGreater(float(latent[0, 3]), float(latent[0, 2]))

    def test_hdf5_sampling_uses_current_action_and_next_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            time_size, env_size = 8, 40
            with h5py.File(root / "buffer.hdf5", "w") as buffer:
                grid = np.arange(time_size * env_size, dtype=np.float32).reshape(time_size, env_size, 1)
                buffer["action"] = grid
                buffer["qpos"] = grid + 1000
                buffer["qvel"] = grid + 2000
                buffer["observation-state"] = grid + 3000
                truncated = np.zeros((time_size, env_size, 1), dtype=bool)
                truncated[2, :] = True
                buffer["truncated"] = truncated

            sample = _read_sample_block(root, 20, seed=4, required_observation_keys={"state"})
            self.assertEqual(sample["action"].shape, (20, 1))
            self.assertTrue(np.allclose(sample["qpos"] - sample["action"], 1000 + env_size))
            self.assertTrue(np.allclose(sample["qvel"] - sample["action"], 2000 + env_size))
            self.assertTrue(np.allclose(sample["observation"]["state"] - sample["action"], 3000 + env_size))

    def test_hdf5_sampling_clamps_oversized_request_to_all_valid_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            time_size, env_size = 5, 3
            with h5py.File(root / "buffer.hdf5", "w") as buffer:
                grid = np.arange(time_size * env_size, dtype=np.float32).reshape(time_size, env_size, 1)
                buffer["action"] = grid
                buffer["qpos"] = grid
                buffer["qvel"] = grid
                buffer["observation-state"] = grid
                truncated = np.zeros((time_size, env_size, 1), dtype=bool)
                truncated[1, 1] = True
                truncated[3, 2] = True
                buffer["truncated"] = truncated

            sample = _read_sample_block(root, 10**30, seed=0, required_observation_keys={"state"})

            # Four current timesteps x three envs, minus two terminal pairs.
            self.assertEqual(sample["action"].shape[0], 10)
            self.assertEqual(sample["source_block"]["sample_count"], 10)
            self.assertEqual(sample["source_block"]["max_transition_count"], 12)
            self.assertEqual(sample["source_block"]["requested_num_samples"], 10**30)


if __name__ == "__main__":
    unittest.main()
