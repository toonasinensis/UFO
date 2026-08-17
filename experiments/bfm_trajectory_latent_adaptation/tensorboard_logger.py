"""Compact TensorBoard output for whole-trajectory latent adaptation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class AdaptationTensorboardLogger:
    """Write one run with only decision-relevant tracking and MPPI curves."""

    def __init__(self, log_dir: Path, config: dict[str, Any], *, enabled: bool = True):
        self.writer = SummaryWriter(str(log_dir), flush_secs=1) if enabled else None
        self.weights = {
            name: float(config[f"{name}_weight"])
            for name in (
                "root_position",
                "root_rotation",
                "body_position",
                "body_rotation",
                "joint_position",
                "body_linear_velocity",
                "body_angular_velocity",
                "joint_limit",
            )
        }
        if self.writer is not None:
            serializable = {
                key: str(value) if isinstance(value, Path) else value for key, value in config.items()
            }
            self.writer.add_text(
                "run/config",
                f"```json\n{json.dumps(serializable, indent=2, ensure_ascii=False)}\n```",
                0,
            )

    def log_baseline(self, metrics: dict[str, float]) -> None:
        # Baseline values are written on every iteration beside their comparison
        # curves, so a separate baseline card adds no useful information.
        del metrics

    def log_iteration(self, step: int, record: dict[str, Any]) -> None:
        if self.writer is None:
            return
        iteration_metrics = record["iteration_best_metrics"]
        global_metrics = record["global_best_metrics"]
        score_mode = record.get("mppi_score_mode", "objective")
        configured_blend_alpha = float(record.get("mppi_link_blend_alpha", 0.0))
        default_link_arm_eligible_ratio = float(
            record.get("mppi_eligible_ratio", 1.0)
            if score_mode == "link"
            else record.get("feasible_ratio", 1.0)
        )
        default_link_arm_active = (
            score_mode == "link" and not record.get("mppi_update_skipped", False)
        ) or (
            score_mode == "objective"
            and configured_blend_alpha > 0.0
            and default_link_arm_eligible_ratio > 0.0
        )
        curves = {
            "tracking/objective/baseline": record["baseline_objective"],
            "tracking/objective/iteration_best": record["iteration_best_objective"],
            "tracking/objective/global_best": record["global_best_objective"],
            "tracking/objective/link_best": record["link_best_objective"],
            "tracking/mpjpe_mm/baseline": 1000.0 * record["baseline_mpjpe"],
            "tracking/mpjpe_mm/iteration_best": 1000.0 * iteration_metrics["mpjpe"],
            "tracking/mpjpe_mm/global_best": 1000.0 * global_metrics["mpjpe"],
            "tracking/mpjpe_mm/link_best": 1000.0
            * record["link_best_metrics"]["mpjpe"],
            "tracking/link_relative_position_mm/baseline": 1000.0
            * record["baseline_metrics"]["mean_body_position_error"],
            "tracking/link_relative_position_mm/iteration_best": 1000.0
            * iteration_metrics["mean_body_position_error"],
            "tracking/link_relative_position_mm/global_best": 1000.0
            * global_metrics["mean_body_position_error"],
            "tracking/link_relative_position_mm/link_best": 1000.0
            * record["link_best_metrics"]["mean_body_position_error"],
            "safety/max_hard_limit_utilization/baseline": record[
                "baseline_metrics"
            ].get("max_joint_limit_utilization", 0.0),
            "safety/max_hard_limit_utilization/iteration_best": iteration_metrics.get(
                "max_joint_limit_utilization", 0.0
            ),
            "safety/max_hard_limit_utilization/global_best": global_metrics.get(
                "max_joint_limit_utilization", 0.0
            ),
            "safety/hard_contact_fraction/baseline": record["baseline_metrics"].get(
                "hard_limit_contact_fraction", 0.0
            ),
            "safety/hard_contact_fraction/global_best": global_metrics.get(
                "hard_limit_contact_fraction", 0.0
            ),
            "improvement/mpjpe_percent": record["mpjpe_improvement_percent"],
            "improvement/link_relative_position_percent": record[
                "link_relative_position_improvement_percent"
            ],
            "improvement/link_best_percent": record["link_best_improvement_percent"],
            "mppi/ess_ratio": record["mppi"]["ess_ratio"],
            "mppi/max_weight": record["mppi"]["max_weight"],
            "mppi/normalized_weight_entropy": record["mppi"][
                "normalized_weight_entropy"
            ],
            "mppi/population_objective_mean": record["population_mean"],
            "mppi/population_objective_std": record["population_std"],
            "mppi/score_p10": record["mppi"]["score_p10"],
            "mppi/score_p50": record["mppi"]["score_p50"],
            "mppi/score_p90": record["mppi"]["score_p90"],
            "mppi/noise_mean": record["noise_mean"],
            "mppi/score_mode_link": float(record.get("mppi_score_mode") == "link"),
            "mppi/guard_anchor_mean": float(
                record.get("mppi_link_guard") == "mean"
            ),
            "mppi/link_blend_configured_alpha": configured_blend_alpha,
            "mppi/link_blend_effective_alpha": record.get(
                "mppi_link_blend_effective_alpha",
                configured_blend_alpha
                if score_mode == "objective" and default_link_arm_active
                else 0.0,
            ),
            "mppi/link_arm_active": float(
                record.get("mppi_link_arm_active", default_link_arm_active)
            ),
            "mppi/link_arm_eligible_ratio": record.get(
                "mppi_link_arm_eligible_ratio",
                default_link_arm_eligible_ratio,
            ),
            "mppi/mixed_feasible_mass": record.get(
                "mppi_mixed_feasible_mass",
                record.get("mppi_raw_feasible_weight_mass", 0.0),
            ),
            "mppi/objective_arm_ess_ratio": record.get(
                "mppi_objective_arm_ess_ratio", record["mppi"]["ess_ratio"]
            ),
            "mppi/link_arm_ess_ratio": record.get(
                "mppi_link_arm_ess_ratio", 0.0
            ),
            "mppi/link_arm_feasible_mass": record.get(
                "mppi_link_arm_feasible_mass", 0.0
            ),
            "mppi/eligible_ratio": record.get("mppi_eligible_ratio", 1.0),
            "mppi/raw_feasible_weight_mass": record.get(
                "mppi_raw_feasible_weight_mass", 1.0
            ),
            "mppi/ess_ratio_eligible": record.get(
                "mppi_ess_ratio_eligible", record["mppi"]["ess_ratio"]
            ),
            "mppi/update_skipped": float(record.get("mppi_update_skipped", False)),
            "mppi/link_error_mm_p10": record.get(
                "mppi_link_error_mm_p10",
                1000.0 * iteration_metrics["mean_body_position_error"],
            ),
            "mppi/link_error_mm_p50": record.get(
                "mppi_link_error_mm_p50",
                1000.0 * iteration_metrics["mean_body_position_error"],
            ),
            "mppi/link_error_mm_p90": record.get(
                "mppi_link_error_mm_p90",
                1000.0 * iteration_metrics["mean_body_position_error"],
            ),
            "guards/objective_pass_ratio": record.get("guards", {}).get(
                "objective_pass_ratio", 1.0
            ),
            "guards/mpjpe_pass_ratio": record.get("guards", {}).get(
                "mpjpe_pass_ratio", 1.0
            ),
            "guards/link_pass_ratio": record.get("guards", {}).get(
                "link_pass_ratio", 1.0
            ),
            "guards/limit_pass_ratio": record.get("guards", {}).get(
                "limit_pass_ratio", 1.0
            ),
            "convergence/mean_update_norm": record["mean_update_norm"],
            "convergence/feasible_ratio": record["feasible_ratio"],
            "convergence/global_best_updated": float(record["global_best_updated"]),
            "convergence/link_best_updated": float(record["link_best_updated"]),
        }
        weighted_link_error_mm = record.get("mppi_weighted_link_error_mm")
        if weighted_link_error_mm is not None:
            curves["mppi/weighted_link_error_mm"] = weighted_link_error_mm
        mppi_best_metrics = record.get("mppi_best_metrics", iteration_metrics)
        if mppi_best_metrics is not None:
            curves["tracking/link_relative_position_mm/iteration_mppi_best"] = (
                1000.0 * mppi_best_metrics["mean_body_position_error"]
            )
        for tag, value in curves.items():
            self.writer.add_scalar(tag, float(value), step)
        for name, weight in self.weights.items():
            self.writer.add_scalar(
                f"objective_weighted_terms/{name}/baseline",
                weight * float(record["baseline_rewards"][name]),
                step,
            )
            self.writer.add_scalar(
                f"objective_weighted_terms/{name}/global_best",
                weight * float(record["global_best_rewards"][name]),
                step,
            )
        self.writer.flush()

    def log_validation(self, step: int, record: dict[str, Any]) -> None:
        if self.writer is None:
            return
        for variant in (
            "baseline",
            "current_mean",
            "search_objective_best",
            "search_link_best",
            "validated_objective_best",
            "validated_link_best",
        ):
            result = record["variants"][variant]
            metrics = result["metrics"]
            self.writer.add_scalar(
                f"validation/objective/{variant}", float(metrics["objective"]), step
            )
            self.writer.add_scalar(
                f"validation/mpjpe_mm/{variant}", 1000.0 * float(metrics["mpjpe"]), step
            )
            self.writer.add_scalar(
                f"validation/link_relative_position_mm/{variant}",
                1000.0 * float(metrics["mean_body_position_error"]),
                step,
            )
            self.writer.add_scalar(
                f"validation/max_hard_limit_utilization/{variant}",
                float(metrics.get("max_joint_limit_utilization", 0.0)),
                step,
            )
        self.writer.flush()

    def log_final(
        self,
        *,
        baseline_metrics: dict[str, float],
        adapted_metrics: dict[str, float],
        baseline_rewards: dict[str, float],
        adapted_rewards: dict[str, float],
        step: int,
        variant: str = "adapted",
    ) -> None:
        del baseline_rewards, adapted_rewards
        if self.writer is None:
            return
        pairs = {
            "final_validation/objective": (baseline_metrics["objective"], adapted_metrics["objective"]),
            "final_validation/mpjpe_mm": (
                1000.0 * baseline_metrics["mpjpe"],
                1000.0 * adapted_metrics["mpjpe"],
            ),
            "final_validation/link_relative_position_mm": (
                1000.0 * baseline_metrics["mean_body_position_error"],
                1000.0 * adapted_metrics["mean_body_position_error"],
            ),
            "final_validation/max_hard_limit_utilization": (
                baseline_metrics.get("max_joint_limit_utilization", 0.0),
                adapted_metrics.get("max_joint_limit_utilization", 0.0),
            ),
            "final_validation/hard_contact_fraction": (
                baseline_metrics.get("hard_limit_contact_fraction", 0.0),
                adapted_metrics.get("hard_limit_contact_fraction", 0.0),
            ),
        }
        for prefix, (baseline, adapted) in pairs.items():
            self.writer.add_scalar(f"{prefix}/baseline", float(baseline), step)
            self.writer.add_scalar(f"{prefix}/{variant}", float(adapted), step)
        self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
