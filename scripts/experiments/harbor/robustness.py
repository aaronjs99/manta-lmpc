"""Experiments / harbor / robustness."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.assembly.harbor import load_harbor_runtime as load_harbor_config
from scripts.cli.arguments.harbor.robustness import parse_args
from scripts.configuration.harbor_faults import load_harbor_disturbance_config
from scripts.configuration.harbor_mpc import load_harbor_mpc_config
from scripts.experiments.harbor.execution.mismatch import run_model_mismatch_study
from scripts.plotting.harbor.mismatch import save_model_mismatch_diagnostics
from scripts.plotting.harbor.trajectories import save_harbor_animation
from scripts.reporting.json import write_json


def main(argv: list[str] | None = None) -> None:
    """Run the matched robustness study and replace its curated artifacts."""
    args = parse_args(argv)
    agents, simulation, communication = load_harbor_config(args.config)
    disturbance = load_harbor_disturbance_config(args.config)
    trials = run_model_mismatch_study(
        agents,
        simulation,
        communication,
        load_harbor_mpc_config(args.config),
        disturbance,
    )
    records = [
        {
            "controller": trial.label,
            "valid": trial.valid,
            "all_goals_reached": trial.result.all_goals_reached,
            "completion_step_sum": trial.completion_step_sum,
            "first_goal_steps": trial.result.first_goal_steps,
            "final_goal_errors": trial.result.final_goal_errors,
            "final_orientation_errors": trial.result.final_orientation_errors,
            "min_pairwise_distance": trial.result.min_pairwise_distance,
            "pairwise_violation_count": trial.result.pairwise_violation_count,
            "solver_fallbacks": trial.solver_fallbacks,
            "max_collision_slack": trial.max_collision_slack,
            "final_residual_estimates": {
                name: value.tolist()
                for name, value in trial.final_residual_estimates.items()
            },
            "final_effectiveness_estimates": {
                name: value.tolist()
                for name, value in trial.final_effectiveness_estimates.items()
            },
        }
        for trial in trials
    ]
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "model_mismatch_study.json"
    write_json(data_path, records)
    figure = save_model_mismatch_diagnostics(
        trials,
        agents,
        simulation,
        disturbance,
        output / "model_mismatch_diagnostics.png",
    )
    adaptive = next(trial for trial in trials if trial.label == "Joint-adaptive LMPC")
    if not args.no_gif:
        save_harbor_animation(
            adaptive.result,
            agents,
            simulation,
            output / "robust_harbor_lmpc.gif",
            label="Joint-adaptive distributed LMPC",
        )
    print(json.dumps(records, indent=2))
    print(f"Saved robustness diagnostics: {figure}")
