"""Experiments / harbor / lmpc."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from scripts.assembly.harbor import load_harbor_runtime as load_harbor_config
from scripts.cli.arguments.harbor.lmpc import parse_args
from scripts.configuration.harbor_mpc import load_harbor_mpc_config
from scripts.experiments.harbor.learning import run_distributed_harbor_lmpc
from scripts.plotting.harbor.learning import save_harbor_learning_progress
from scripts.plotting.harbor.trajectories import save_harbor_animation
from scripts.reporting.json import write_json


def main(argv: list[str] | None = None) -> None:
    """Run the controlled learning experiment and print auditable telemetry."""
    args = parse_args(argv)
    agents, simulation, communication = load_harbor_config(args.config)
    mpc_config = load_harbor_mpc_config(args.config)
    if args.iterations is not None:
        if args.iterations <= 0:
            raise ValueError("--iterations must be positive")
        mpc_config = replace(mpc_config, learning_iterations=args.iterations)
    iterations = run_distributed_harbor_lmpc(
        agents, simulation, communication, mpc_config
    )
    records = [
        {
            "label": item.label,
            "admitted": item.admitted,
            "all_goals_reached": item.result.all_goals_reached,
            "pairwise_violation_count": item.result.pairwise_violation_count,
            "min_pairwise_distance": item.result.min_pairwise_distance,
            "completion_step_sum": item.completion_step_sum,
            "first_goal_steps": item.result.first_goal_steps,
            "solver_calls": item.solver_calls,
            "solver_fallbacks": item.solver_fallbacks,
            "solve_time_seconds": item.solve_time_seconds,
            "max_collision_slack": item.max_collision_slack,
            "max_terminal_slack": item.max_terminal_slack,
            "solve_count_by_agent": item.solve_count_by_agent,
            "fallback_count_by_agent": item.fallback_count_by_agent,
            "failure_steps_by_agent": item.failure_steps_by_agent,
            "failure_status_counts": item.failure_status_counts,
            "final_residual_estimates": item.final_residual_estimates,
            "final_effectiveness_estimates": {
                name: value.tolist()
                for name, value in item.final_effectiveness_estimates.items()
            },
        }
        for item in iterations
    ]
    text = json.dumps(records, indent=2)
    print(text)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "metrics.json", records)
    save_harbor_learning_progress(
        iterations,
        agents,
        simulation,
        artifact_dir / "research_progress.png",
    )
    if not args.no_gif:
        best = min(
            (item for item in iterations if item.admitted),
            key=lambda item: item.completion_step_sum,
        )
        animation_label = (
            "Distributed MPC"
            if best.label == "distributed_mpc"
            else best.label.replace("distributed_lmpc_", "Distributed LMPC ")
        )
        save_harbor_animation(
            best.result,
            agents,
            simulation,
            artifact_dir / "harbor_lmpc.gif",
            label=animation_label,
        )
