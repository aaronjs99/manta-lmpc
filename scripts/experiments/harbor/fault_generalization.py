"""Experiments / harbor / fault generalization."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scripts.assembly.harbor import load_harbor_runtime as load_harbor_config
from scripts.cli.arguments.harbor.fault_generalization import parse_args
from scripts.configuration.harbor_faults import (
    load_harbor_fault_config,
    load_harbor_fault_ensemble_config,
    load_harbor_fault_study_config,
)
from scripts.configuration.harbor_mpc import load_harbor_mpc_config
from scripts.configuration.harbor_observation import (
    load_harbor_observation_noise_config,
)
from scripts.experiments.harbor.execution.actuator_faults import (
    run_actuator_fault_generalization,
)
from scripts.metrics.harbor.fault_generalization import summarize_fault_generalization
from scripts.plotting.harbor.generalization import save_fault_generalization_plot
from scripts.reporting.json import write_json


def main(argv: list[str] | None = None) -> None:
    """Run the ensemble and replace its JSON and plot artifacts."""
    args = parse_args(argv)
    agents, simulation, communication = load_harbor_config(args.config)
    ensemble_config = load_harbor_fault_ensemble_config(args.config)
    observation_noise = load_harbor_observation_noise_config(args.config)
    if args.with_observation_noise:
        observation_noise = replace(observation_noise, enabled=True)
    else:
        observation_noise = None
    cases = run_actuator_fault_generalization(
        agents,
        simulation,
        communication,
        load_harbor_mpc_config(args.config),
        load_harbor_fault_config(args.config),
        load_harbor_fault_study_config(args.config),
        ensemble_config,
        observation_noise=observation_noise,
    )
    records = []
    for case in cases:
        truth = np.concatenate(
            [
                case.disturbance.effectiveness(agent.model, agent.name)
                for agent in agents
            ]
        )
        hidden = {
            agent.name: case.disturbance.effectiveness(agent.model, agent.name).tolist()
            for agent in agents
        }
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
                    "valid": trial.valid,
                    "all_goals_reached": trial.result.all_goals_reached,
                    "first_hit_step_sum": trial.completion_step_sum,
                    "sustained_completion_cost": sustained_cost,
                    "first_goal_steps": trial.result.first_goal_steps,
                    "final_goal_errors": trial.result.final_goal_errors,
                    "final_orientation_errors": trial.result.final_orientation_errors,
                    "min_pairwise_distance": trial.result.min_pairwise_distance,
                    "pairwise_violation_count": trial.result.pairwise_violation_count,
                    "solver_fallbacks": trial.solver_fallbacks,
                    "solver_fallbacks_by_agent": trial.solver_fallbacks_by_agent,
                    "solver_failure_steps_by_agent": trial.solver_failure_steps_by_agent,
                    "solver_failure_status_counts": trial.solver_failure_status_counts,
                    "max_collision_slack": trial.max_collision_slack,
                    "effectiveness_rmse": float(
                        np.sqrt(np.mean((estimate - truth) ** 2))
                    ),
                    "probe_count": int(sum(trial.probe_count_by_agent.values())),
                    "hidden_effectiveness": hidden,
                    "final_effectiveness_estimates": {
                        name: values.tolist()
                        for name, values in trial.final_effectiveness_estimates.items()
                    },
                    "effectiveness_change_steps_by_agent": (
                        trial.effectiveness_change_steps_by_agent
                    ),
                    "probe_sequence_by_agent": trial.probe_sequence_by_agent,
                }
            )
    summary = summarize_fault_generalization(records, ensemble_config.bootstrap_samples)
    payload = {
        "ensemble": {
            "seeds": list(ensemble_config.seeds),
            "effectiveness_min": ensemble_config.effectiveness_min,
            "effectiveness_max": ensemble_config.effectiveness_max,
            "sampling": "per-channel Latin hypercube",
        },
        "observation_noise": (
            {
                "enabled": True,
                "base_seed": observation_noise.seed,
                "kind_state_std": observation_noise.kind_state_std,
                "agent_state_std": observation_noise.agent_state_std,
            }
            if observation_noise is not None
            else {"enabled": False}
        ),
        "summary": summary,
        "trials": records,
    }
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = (
        "fault_noise_generalization"
        if observation_noise is not None
        else "fault_generalization"
    )
    data_path = output / f"{stem}.json"
    write_json(data_path, payload)
    figure = save_fault_generalization_plot(
        records,
        summary,
        output / f"{stem}.png",
        noisy_observations=observation_noise is not None,
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved fault-generalization data: {data_path}")
    print(f"Saved fault-generalization plot: {figure}")
