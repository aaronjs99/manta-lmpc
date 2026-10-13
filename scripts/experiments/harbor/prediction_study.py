"""Experiments / harbor / prediction study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scripts.assembly.harbor import load_harbor_runtime as load_harbor_config
from scripts.cli.arguments.harbor.prediction_study import parse_args
from scripts.configuration.harbor_faults import (
    load_harbor_fault_config,
    load_harbor_fault_ensemble_config,
    load_harbor_fault_study_config,
)
from scripts.configuration.harbor_mpc import load_harbor_mpc_config
from scripts.configuration.harbor_observation import (
    load_harbor_observation_noise_config,
)
from scripts.experiments.harbor.execution.prediction_generalization import (
    run_obstacle_prediction_generalization,
)
from scripts.metrics.harbor.prediction_study import summarize_prediction_ablation
from scripts.plotting.harbor.prediction import save_prediction_ablation_plot
from scripts.reporting.json import write_json


def main(argv: list[str] | None = None) -> None:
    """Run the matched prediction ablation and replace its two artifacts."""
    args = parse_args(argv)
    agents, simulation, communication = load_harbor_config(args.config)
    ensemble = load_harbor_fault_ensemble_config(args.config)
    cases = run_obstacle_prediction_generalization(
        agents,
        simulation,
        communication,
        load_harbor_mpc_config(args.config),
        load_harbor_fault_config(args.config),
        load_harbor_fault_study_config(args.config),
        ensemble,
        load_harbor_observation_noise_config(args.config),
    )
    records = []
    for case in cases:
        truth = np.concatenate(
            [
                case.disturbance.effectiveness(agent.model, agent.name)
                for agent in agents
            ]
        )
        for trial in case.trials:
            estimate = np.concatenate(
                [trial.final_effectiveness_estimates[agent.name] for agent in agents]
            )
            sustained_cost = sum(
                (
                    trial.result.first_goal_steps[agent.name]
                    if trial.result.first_goal_steps[agent.name] is not None
                    and trial.result.final_goal_errors[agent.name]
                    <= simulation.goal_tolerance
                    and trial.result.final_orientation_errors[agent.name]
                    <= simulation.orientation_tolerance
                    else simulation.horizon + 1
                )
                for agent in agents
            )
            records.append(
                {
                    "seed": case.seed,
                    "observation_seed": case.observation_seed,
                    "controller": trial.label,
                    "all_goals_reached": trial.result.all_goals_reached,
                    "pairwise_violation_count": trial.result.pairwise_violation_count,
                    "min_pairwise_distance": trial.result.min_pairwise_distance,
                    "sustained_completion_cost": sustained_cost,
                    "solver_fallbacks": trial.solver_fallbacks,
                    "solver_fallbacks_by_agent": trial.solver_fallbacks_by_agent,
                    "solver_failure_steps_by_agent": (
                        trial.solver_failure_steps_by_agent
                    ),
                    "solver_failure_status_counts": (
                        trial.solver_failure_status_counts
                    ),
                    "max_collision_slack": trial.max_collision_slack,
                    "effectiveness_rmse": float(
                        np.sqrt(np.mean((estimate - truth) ** 2))
                    ),
                }
            )
    summary = summarize_prediction_ablation(records)
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "prediction_ablation.json"
    figure_path = output / "prediction_ablation.png"
    payload = {
        "ensemble": {
            "seeds": list(ensemble.seeds),
            "comparison": "common safe-memory, hidden fault, and observation seed",
        },
        "summary": summary,
        "trials": records,
    }
    write_json(data_path, payload)
    save_prediction_ablation_plot(records, summary, figure_path)
    print(json.dumps(summary, indent=2))
    print(f"Saved prediction-ablation data: {data_path}")
    print(f"Saved prediction-ablation plot: {figure_path}")
