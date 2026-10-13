"""Experiments / harbor / joint uncertainty study."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from scripts.assembly.harbor import load_harbor_runtime as load_harbor_config
from scripts.cli.arguments.harbor.joint_uncertainty_study import parse_args
from scripts.configuration.harbor_joint_uncertainty_criteria import (
    load_harbor_joint_uncertainty_criteria_config,
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
from scripts.metrics.harbor.joint_uncertainty_study import (
    evaluate_joint_uncertainty,
    summarize_joint_uncertainty,
)
from scripts.metrics.harbor.temporary_fault_generalization import (
    summarize_recovery_prior,
)
from scripts.metrics.harbor.time_varying_fault_records import (
    build_time_varying_fault_records,
    compact_time_varying_fault_records,
)
from scripts.plotting.harbor.uncertainty import save_joint_uncertainty_plot
from scripts.reporting.json import write_json


def main(argv: list[str] | None = None) -> None:
    """Run matched joint-adaptive policies and retain auditable evidence."""
    args = parse_args(argv)
    agents, simulation, communication = load_harbor_config(args.config)
    base_disturbance, experiment = load_harbor_time_varying_fault_config(args.config)
    section = (
        "joint_uncertainty_confirmation"
        if args.confirmation
        else "joint_uncertainty_development"
    )
    artifact_stem = section
    ensemble = load_harbor_temporary_fault_ensemble_config(args.config, section=section)
    criteria = (
        load_harbor_joint_uncertainty_criteria_config(args.config)
        if args.confirmation
        else None
    )
    controller_labels = (
        criteria.controller_labels
        if criteria is not None
        else (
            "Innovation-threshold RLS",
            "Transient-offset threshold RLS",
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
        controller_labels=controller_labels,
        residual_adaptation=True,
        residual_adaptation_kinds=("usv", "rov"),
    )
    records = build_time_varying_fault_records(
        cases, agents, simulation, experiment.event_detection_window_steps
    )
    recovery_comparison = summarize_recovery_prior(
        records,
        ensemble.bootstrap_samples,
        candidate_label="Transient-offset threshold RLS",
    )
    summary = summarize_joint_uncertainty(records, recovery_comparison)
    confirmation = (
        evaluate_joint_uncertainty(summary, criteria) if criteria is not None else None
    )
    status = (
        "joint uncertainty confirmed in simulation"
        if confirmation is not None and confirmation["passed"]
        else (
            "joint uncertainty confirmation criteria not met"
            if confirmation is not None
            else "joint uncertainty development"
        )
    )
    payload = {
        "ensemble": asdict(ensemble),
        "experiment": {
            "evaluation_role": section,
            "residual_adaptation": True,
            "residual_adaptation_kinds": ["usv", "rov"],
            "effectiveness_adaptation": True,
            "controller_blind_to_hidden_fault_and_current": True,
            "change_threshold": experiment.change_threshold,
            "covariance_inflation": experiment.covariance_inflation,
            "recovery_prior_gain": experiment.recovery_prior_gain,
            "recovery_offset_decay": (mpc_config.effectiveness_recovery_offset_decay),
            "recovery_minimum_dwell_steps": (
                mpc_config.effectiveness_recovery_minimum_dwell_steps
            ),
            "candidate_status": status,
        },
        "joint_uncertainty_summary": summary,
        "trials": compact_time_varying_fault_records(records),
    }
    if criteria is not None:
        payload["joint_uncertainty_confirmation_criteria"] = asdict(criteria)
        payload["joint_uncertainty_confirmation_result"] = confirmation

    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / f"{artifact_stem}.json"
    figure_path = output / f"{artifact_stem}.png"
    write_json(data_path, payload)
    save_joint_uncertainty_plot(
        records,
        summary,
        figure_path,
        study_title=(
            "Independent Joint-Uncertainty Confirmation"
            if args.confirmation
            else "Joint Current and Temporary-Fault Development"
        ),
        confirmation_passed=(
            confirmation["passed"] if confirmation is not None else None
        ),
    )
    print(json.dumps(summary, indent=2))
    if confirmation is not None:
        print(json.dumps(confirmation, indent=2))
    print(f"Saved joint-uncertainty data: {data_path}")
    print(f"Saved joint-uncertainty plot: {figure_path}")
