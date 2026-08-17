"""Render two fixed BFM latents side by side without running a search."""

from __future__ import annotations

import argparse
from pathlib import Path

from humanoidverse.mjlab_inference_utils import load_mjlab_env_cfg, resolve_inference_robot_config
from humanoidverse.utils.robot_spec import load_robot_training_spec

from .search_lateral_z import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_FOLDER,
    DEFAULT_ROBOT_CONFIG,
    OnnxActor,
    load_initial_latent,
    project_latents,
    render_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True, help="Left-side latent NPY.")
    parser.add_argument("--right", type=Path, required=True, help="Right-side latent NPY.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_MODEL_FOLDER)
    parser.add_argument("--actor-onnx", type=Path, default=None)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--onnx-provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--rollout-s", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--render-size", type=int, default=480)
    args = parser.parse_args()

    model_folder = args.model_folder.expanduser().resolve()
    actor_path = (args.actor_onnx or model_folder / "exported/FBcprAuxModel.onnx").expanduser().resolve()
    robot_config = resolve_inference_robot_config(args.robot_config, None)
    actor = OnnxActor(actor_path, args.onnx_provider)
    left = project_latents(load_initial_latent(args.left.expanduser().resolve(), z_dim=256, device=args.device), 16.0)
    right = project_latents(load_initial_latent(args.right.expanduser().resolve(), z_dim=256, device=args.device), 16.0)
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
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    robot_xml = Path(load_robot_training_spec(robot_config).robot.xml_path).expanduser().resolve()
    render_comparison(
        env_cfg,
        actor,
        left,
        right,
        robot_xml=robot_xml,
        rollout_s=args.rollout_s,
        fps=args.fps,
        render_size=args.render_size,
        output_path=output,
    )
    print(f"[latent-search] comparison video={output}")


if __name__ == "__main__":
    main()
