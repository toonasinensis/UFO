"""Build a five-curve TensorBoard view from link-sweep history files."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=6)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    writers: dict[str, SummaryWriter] = {}
    written: dict[str, int] = {}
    try:
        while True:
            run_dirs = sorted(path for path in args.source.iterdir() if path.is_dir())
            complete = 0
            for run_dir in run_dirs:
                history_path = run_dir / "history.json"
                if not history_path.exists():
                    continue
                history = json.loads(history_path.read_text(encoding="utf-8"))
                config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
                writer = writers.setdefault(
                    run_dir.name,
                    SummaryWriter(str(args.output / run_dir.name), flush_secs=1),
                )
                start = written.get(run_dir.name, 0)
                for record in history[start:]:
                    step = int(record["iteration"])
                    writer.add_scalar(
                        "01_link_relative_error_mm",
                        1000.0 * record["global_best_metrics"]["mean_body_position_error"],
                        step,
                    )
                    writer.add_scalar(
                        "02_link_relative_improvement_percent",
                        record["link_relative_position_improvement_percent"],
                        step,
                    )
                    writer.add_scalar(
                        "03_mpjpe_mm", 1000.0 * record["global_best_mpjpe"], step
                    )
                    writer.add_scalar(
                        "04_tracking_objective/global_best",
                        record["global_best_objective"],
                        step,
                    )
                    writer.add_scalar(
                        "04_tracking_objective/iteration_best",
                        record["iteration_best_objective"],
                        step,
                    )
                    mppi = record["mppi"]
                    writer.add_scalar("05_mppi/ess_ratio", mppi["ess_ratio"], step)
                    writer.add_scalar("05_mppi/max_weight", mppi["max_weight"], step)
                    writer.add_scalar(
                        "05_mppi/population_objective_std", record["population_std"], step
                    )
                    writer.add_scalar("05_mppi/noise_mean", record["noise_mean"], step)
                    writer.add_scalar(
                        "05_mppi/score_p90_minus_p10",
                        mppi["score_p90"] - mppi["score_p10"],
                        step,
                    )
                    for term, reward in record["iteration_best_rewards"].items():
                        weight = float(config[f"{term}_weight"])
                        writer.add_scalar(
                            f"06_iteration_best_objective_weighted_terms/{term}",
                            weight * float(reward),
                            step,
                        )
                if len(history) > start:
                    writer.flush()
                    written[run_dir.name] = len(history)
                if (run_dir / "summary.json").exists():
                    complete += 1
            if len(run_dirs) >= args.expected_runs and complete >= args.expected_runs:
                break
            time.sleep(args.poll_seconds)
    finally:
        for writer in writers.values():
            writer.close()


if __name__ == "__main__":
    main()
