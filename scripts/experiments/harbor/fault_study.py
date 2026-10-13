"""Experiments / harbor / fault study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scripts.assembly.harbor import load_harbor_runtime as load_harbor_config
from scripts.cli.arguments.harbor.fault_study import parse_args
from scripts.configuration.harbor_faults import (
    load_harbor_fault_config,
    load_harbor_fault_study_config,
)
from scripts.configuration.harbor_mpc import load_harbor_mpc_config
from scripts.experiments.harbor.execution.actuator_faults import (
    run_actuator_fault_study,
)
from scripts.plotting.harbor.actuator_faults import save_actuator_fault_diagnostics
from scripts.plotting.harbor.trajectories import save_harbor_animation
from scripts.reporting.json import write_json


def main(argv: list[str] | None = None) -> None:
    """Run the asymmetric fault study and replace its curated artifacts."""
    args = parse_args(argv)
    agents, simulation, communication = load_harbor_config(args.config)
    disturbance = load_harbor_fault_config(args.config)
    mpc_config = load_harbor_mpc_config(args.config)
    trials = run_actuator_fault_study(
        agents,
        simulation,
        communication,
        mpc_config,
        disturbance,
        load_harbor_fault_study_config(args.config),
    )
    records = []
    for trial in trials:
        errors = np.concatenate(
            [
                trial.final_effectiveness_estimates[agent.name]
                - disturbance.effectiveness(agent.model, agent.name)
                for agent in agents
            ]
        )
        records.append(
            {
                "controller": trial.label,
                "identification_strategy": trial.identification_strategy,
                "learning_source": trial.source_controller,
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
                "effectiveness_rmse": float(np.sqrt(np.mean(errors * errors))),
                "hidden_effectiveness": {
                    agent.name: disturbance.effectiveness(
                        agent.model, agent.name
                    ).tolist()
                    for agent in agents
                },
                "final_effectiveness_estimates": {
                    name: value.tolist()
                    for name, value in trial.final_effectiveness_estimates.items()
                },
                "probe_count_by_agent": trial.probe_count_by_agent,
                "probe_channel_counts": {
                    name: value.tolist()
                    for name, value in trial.probe_channel_counts.items()
                },
                "probe_sequence_by_agent": trial.probe_sequence_by_agent,
                "probe_rejection_counts": {
                    name: value.tolist()
                    for name, value in trial.probe_rejection_counts.items()
                },
                "final_excitation_energy": {
                    name: (
                        values[-1].tolist()
                        if len(values)
                        else np.zeros(
                            next(
                                agent.model.control_dim
                                for agent in agents
                                if agent.name == name
                            )
                        ).tolist()
                    )
                    for name, values in trial.excitation_history.items()
                },
                "final_information_std": {
                    name: (
                        values[-1].tolist()
                        if len(values)
                        else np.full(
                            next(
                                agent.model.control_dim
                                for agent in agents
                                if agent.name == name
                            ),
                            mpc_config.identification_prior_std,
                        ).tolist()
                    )
                    for name, values in trial.information_std_history.items()
                },
            }
        )
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "actuator_fault_study.json"
    write_json(data_path, records)
    figure = save_actuator_fault_diagnostics(
        trials,
        agents,
        simulation,
        disturbance,
        output / "actuator_fault_diagnostics.png",
    )
    adaptive = next(
        trial for trial in trials if trial.label == "Retained information-ID LMPC"
    )
    if not args.no_gif:
        save_harbor_animation(
            adaptive.result,
            agents,
            simulation,
            output / "fault_aware_harbor_lmpc.gif",
            label="Retained information-ID distributed LMPC",
        )
    print(json.dumps(records, indent=2))
    print(f"Saved actuator-fault diagnostics: {figure}")
