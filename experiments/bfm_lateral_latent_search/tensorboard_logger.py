"""TensorBoard logging for fixed-latent CEM searches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.tensorboard import SummaryWriter


class LatentSearchTensorboardLogger:
    """Write and flush one complete optimization snapshot per CEM iteration."""

    def __init__(self, log_dir: Path, *, enabled: bool, config: dict[str, Any], objective_text: str):
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.writer: SummaryWriter | None = None
        self.target_forward_speed = float(config.get("target_forward_speed", 0.0))
        self.target_lateral_speed = float(config.get("target_lateral_speed", 0.5))
        if not enabled:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=1)
        serializable = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
        self.writer.add_text("run/config", f"```json\n{json.dumps(serializable, ensure_ascii=False, indent=2)}\n```", 0)
        self.writer.add_text("run/objective", objective_text, 0)
        self.writer.flush()

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def log_iteration(
        self,
        *,
        iteration: int,
        search_std: torch.Tensor,
        scores: torch.Tensor,
        candidates: torch.Tensor,
        metrics: Sequence[Any],
        best_index: int,
        global_best_score: float,
    ) -> None:
        if self.writer is None:
            return
        writer = self.writer
        best = metrics[best_index]
        baseline = metrics[0]

        self.log_aggregate_iteration(
            iteration=iteration,
            search_std_mean=float(search_std.mean().cpu()),
            search_std_min=float(search_std.min().cpu()),
            search_std_max=float(search_std.max().cpu()),
            best=best,
            baseline=baseline,
            population_mean=float(scores.mean().cpu()),
            population_std=float(scores.std().cpu()),
            global_best_score=global_best_score,
            flush=False,
        )

        metric_tensors = {
            "population/score": scores.detach().float().cpu(),
            "population/survival_s": torch.tensor([item.survival_s for item in metrics]),
            "population/cosine_to_initial": torch.tensor([item.latent_cosine_to_initial for item in metrics]),
        }
        for name in sorted(best.step_metrics):
            metric_tensors[f"population/{name}"] = torch.tensor([item.step_metrics[name] for item in metrics])
        for tag, values in metric_tensors.items():
            writer.add_histogram(tag, values, iteration)
        writer.add_histogram("latent/best_components", candidates[best_index].detach().float().cpu(), iteration)
        writer.add_histogram("search/std_per_dimension", search_std.detach().float().cpu(), iteration)
        writer.flush()

    def log_aggregate_iteration(
        self,
        *,
        iteration: int,
        search_std_mean: float,
        search_std_min: float,
        search_std_max: float,
        best: Any,
        baseline: Any,
        population_mean: float,
        population_std: float,
        global_best_score: float,
        flush: bool = True,
    ) -> None:
        """Log aggregate metrics, including backfills from an existing history.json."""
        if self.writer is None:
            return
        writer = self.writer

        writer.add_scalar("objective/score_best", best.score, iteration)
        writer.add_scalar("objective/score_baseline", baseline.score, iteration)
        writer.add_scalar(
            "objective/rollout_before_latent_penalty_best",
            best.objective_without_latent_reg,
            iteration,
        )
        writer.add_scalar(
            "objective/rollout_before_latent_penalty_baseline",
            baseline.objective_without_latent_reg,
            iteration,
        )
        writer.add_scalar(
            "objective/latent_penalty_best",
            best.objective_without_latent_reg - best.score,
            iteration,
        )
        writer.add_scalar(
            "objective/latent_penalty_baseline",
            baseline.objective_without_latent_reg - baseline.score,
            iteration,
        )
        writer.add_scalar("objective/global_best", global_best_score, iteration)
        writer.add_scalar("objective/population_mean", population_mean, iteration)
        writer.add_scalar("objective/population_std", population_std, iteration)
        writer.add_scalar("search/std_mean", search_std_mean, iteration)
        writer.add_scalar("search/std_min", search_std_min, iteration)
        writer.add_scalar("search/std_max", search_std_max, iteration)

        for name in sorted(best.step_metrics):
            writer.add_scalar(f"metrics/{name}", best.step_metrics[name], iteration)
            writer.add_scalar(f"baseline_metrics/{name}", baseline.step_metrics[name], iteration)

        reward_views = {
            "velocity": ("mean_velocity_score", "mean_velocity_weighted_contribution"),
            "yaw": ("mean_yaw_score", "mean_yaw_weighted_contribution"),
            "feet_flatness": (
                "mean_feet_flatness_score",
                "mean_feet_flatness_weighted_contribution",
            ),
        }
        for reward_name, (raw_metric, contribution_metric) in reward_views.items():
            if raw_metric in best.step_metrics:
                writer.add_scalar(f"reward_raw/{reward_name}_best", best.step_metrics[raw_metric], iteration)
                writer.add_scalar(
                    f"reward_raw/{reward_name}_baseline",
                    baseline.step_metrics[raw_metric],
                    iteration,
                )
            if contribution_metric in best.step_metrics:
                writer.add_scalar(
                    f"reward_contribution/{reward_name}_best",
                    best.step_metrics[contribution_metric],
                    iteration,
                )
                writer.add_scalar(
                    f"reward_contribution/{reward_name}_baseline",
                    baseline.step_metrics[contribution_metric],
                    iteration,
                )
        if "mean_instant_objective" in best.step_metrics:
            writer.add_scalar(
                "reward_contribution/instant_total_best",
                best.step_metrics["mean_instant_objective"],
                iteration,
            )
            writer.add_scalar(
                "reward_contribution/instant_total_baseline",
                baseline.step_metrics["mean_instant_objective"],
                iteration,
            )
        writer.add_scalar("stability/survival_s_best", best.survival_s, iteration)
        writer.add_scalar("stability/survival_s_baseline", baseline.survival_s, iteration)
        writer.add_scalar("velocity/target_vy", self.target_lateral_speed, iteration)
        writer.add_scalar("velocity/target_vx", self.target_forward_speed, iteration)
        writer.add_scalar("latent/best_cosine_to_initial", best.latent_cosine_to_initial, iteration)

        writer.add_text("iteration/best_metrics", f"```json\n{json.dumps(best.to_dict(), indent=2)}\n```", iteration)
        if flush:
            writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
