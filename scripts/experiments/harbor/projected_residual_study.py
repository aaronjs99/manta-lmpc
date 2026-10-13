"""Experiments / harbor / projected residual study."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from scripts.assembly.harbor import load_harbor_runtime as load_harbor_config
from scripts.cli.arguments.harbor.projected_residual_study import parse_args
from scripts.configuration.harbor_projected_residual_criteria import (
    load_harbor_projected_residual_criteria_config,
)
from scripts.configuration.harbor_faults import (
    load_harbor_fault_study_config,
    load_harbor_temporary_fault_ensemble_config,
    load_harbor_time_varying_fault_config,
)
from scripts.configuration.harbor_mpc import load_harbor_mpc_config
from scripts.configuration.harbor_observation import (
    load_harbor_observation_noise_config,
)
from scripts.experiments.harbor.execution.temporary_faults import (
    run_temporary_fault_generalization,
)
from scripts.metrics.harbor.projected_residual_study import (
    evaluate_projected_residual,
    summarize_projected_residual,
)
from scripts.metrics.harbor.time_varying_fault_records import (
    build_time_varying_fault_records,
    compact_time_varying_fault_records,
)
from scripts.plotting.harbor.uncertainty import save_projected_residual_plot
from scripts.reporting.json import write_json


def main(argv: list[str] | None = None) -> None:
    """Run matched full-versus-projected residual injection."""
    args = parse_args(argv)
    agents, simulation, communication = load_harbor_config(args.config)
    base_disturbance, experiment = load_harbor_time_varying_fault_config(args.config)
    section = (
        "projected_residual_confirmation"
        if args.confirmation
        else "projected_residual_development"
    )
    ensemble = load_harbor_temporary_fault_ensemble_config(args.config, section=section)
    criteria = (
        load_harbor_projected_residual_criteria_config(args.config)
        if args.confirmation
        else None
    )
    labels = (
        criteria.controller_labels
        if criteria is not None
        else (
            "Transient-offset threshold RLS",
            "Projected transient-offset RLS",
        )
    )
    mpc_config = load_harbor_mpc_config(args.config)
    cases = run_temporary_fault_generalization(
        agents,
        simulation,
        communication,
        mpc_config,
        base_disturbance,
        load_harbor_fault_study_config(args.config),
        experiment,
        ensemble,
        load_harbor_observation_noise_config(args.config),
        controller_labels=labels,
        residual_adaptation=True,
        residual_adaptation_kinds=("usv", "rov"),
    )
    records = build_time_varying_fault_records(
        cases, agents, simulation, experiment.event_detection_window_steps
    )
    summary = summarize_projected_residual(records)
    confirmation = (
        evaluate_projected_residual(summary, criteria) if criteria is not None else None
    )
    payload = {
        "ensemble": asdict(ensemble),
        "experiment": {
            "evaluation_role": section,
            "residual_adaptation_kinds": ["usv", "rov"],
            "baseline_residual_projection": "full",
            "candidate_residual_projection": "actuation_subspace",
            "controller_blind_to_hidden_fault_and_current": True,
            "candidate_status": (
                "projected residual confirmed in simulation"
                if confirmation is not None and confirmation["passed"]
                else (
                    "projected residual confirmation criteria not met"
                    if confirmation is not None
                    else "projected residual development"
                )
            ),
        },
        "projected_residual_summary": summary,
        "trials": compact_time_varying_fault_records(records),
    }
    if criteria is not None:
        payload["projected_residual_confirmation_criteria"] = asdict(criteria)
        payload["projected_residual_confirmation_result"] = confirmation
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / f"{section}.json"
    figure_path = output / f"{section}.png"
    write_json(data_path, payload)
    save_projected_residual_plot(
        records,
        summary,
        figure_path,
        study_title=(
            "Independent Actuation-Subspace Confirmation"
            if args.confirmation
            else "Actuation-Subspace Residual Development"
        ),
        confirmation_passed=(
            confirmation["passed"] if confirmation is not None else None
        ),
    )
    print(json.dumps(summary, indent=2))
    if confirmation is not None:
        print(json.dumps(confirmation, indent=2))
    print(f"Saved projected-residual data: {data_path}")
    print(f"Saved projected-residual plot: {figure_path}")
