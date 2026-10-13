"""Experiments / harbor / dynamic envelope study."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from scripts.assembly.harbor import load_harbor_runtime as load_harbor_config
from scripts.cli.arguments.harbor.dynamic_envelope_study import parse_args
from scripts.configuration.harbor_dynamic_envelope_criteria import (
    load_harbor_dynamic_envelope_criteria_config,
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
from scripts.metrics.harbor.dynamic_envelope_study import (
    evaluate_dynamic_envelope,
    summarize_dynamic_envelope,
)
from scripts.metrics.harbor.time_varying_fault_records import (
    build_time_varying_fault_records,
    compact_time_varying_fault_records,
)
from scripts.plotting.harbor.uncertainty import save_dynamic_envelope_plot
from scripts.reporting.json import write_json


def main(argv: list[str] | None = None) -> None:
    """Run matched hard-versus-elastic envelope trials and save compact evidence."""
    args = parse_args(argv)
    agents, simulation, communication = load_harbor_config(args.config)
    base_disturbance, experiment = load_harbor_time_varying_fault_config(args.config)
    section = (
        "dynamic_envelope_retry_confirmation"
        if args.confirmation
        else "dynamic_envelope_retry_development"
    )
    ensemble = load_harbor_temporary_fault_ensemble_config(args.config, section=section)
    criteria = load_harbor_dynamic_envelope_criteria_config(args.config)
    labels = criteria.controller_labels
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
    summary = summarize_dynamic_envelope(records)
    gate_result = evaluate_dynamic_envelope(summary, criteria)
    payload = {
        "ensemble": asdict(ensemble),
        "experiment": {
            "evaluation_role": section,
            "baseline_dynamic_state_slack_bound": 0.0,
            "candidate_primary_dynamic_state_slack_bound": 0.0,
            "candidate_retry_dynamic_state_slack_bound": mpc_config.dynamic_state_slack_bound,
            "dynamic_state_slack_weight": mpc_config.dynamic_state_slack_weight,
            "hard_domain_and_collision_constraints_retained": True,
            "controller_blind_to_hidden_fault_and_current": True,
        },
        "dynamic_envelope_summary": summary,
        "dynamic_envelope_criteria": asdict(criteria),
        "trials": compact_time_varying_fault_records(records),
    }
    payload[
        "dynamic_envelope_confirmation_result"
        if args.confirmation
        else "dynamic_envelope_development_screen"
    ] = gate_result
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / f"{section}.json"
    figure_path = output / f"{section}.png"
    write_json(data_path, payload)
    save_dynamic_envelope_plot(
        records,
        summary,
        figure_path,
        study_title=(
            "Independent Elastic Dynamic-Envelope Confirmation"
            if args.confirmation
            else "Elastic Dynamic-Envelope Development"
        ),
        confirmation_passed=(gate_result["passed"] if args.confirmation else None),
    )
    print(json.dumps(summary, indent=2))
    print(json.dumps(gate_result, indent=2))
    print(f"Saved dynamic-envelope data: {data_path}")
    print(f"Saved dynamic-envelope plot: {figure_path}")
