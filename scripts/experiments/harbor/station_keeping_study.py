"""Experiments / harbor / station keeping study."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from scripts.assembly.harbor import load_harbor_runtime as load_harbor_config
from scripts.cli.arguments.harbor.station_keeping_study import parse_args
from scripts.configuration.harbor_station_keeping_criteria import (
    load_harbor_station_keeping_criteria_config,
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
from scripts.configuration.study_labels import HARD_ENVELOPE_BASELINE as BASELINE
from scripts.configuration.study_labels import KINEMATIC_EWMA, KINEMATIC_RLS
from scripts.experiments.harbor.execution.temporary_faults import (
    run_temporary_fault_generalization,
)
from scripts.metrics.harbor.station_keeping_study import (
    evaluate_station_keeping,
    summarize_station_keeping,
)
from scripts.metrics.harbor.time_varying_fault_records import (
    build_time_varying_fault_records,
    compact_time_varying_fault_records,
)
from scripts.plotting.harbor.uncertainty import save_station_keeping_plot
from scripts.reporting.json import write_json


def main(argv: list[str] | None = None) -> None:
    """Run development or untouched confirmation cases and save evidence."""
    args = parse_args(argv)
    agents, simulation, communication = load_harbor_config(args.config)
    disturbance, experiment = load_harbor_time_varying_fault_config(args.config)
    section = (
        "station_keeping_confirmation"
        if args.confirmation
        else "station_keeping_development"
    )
    ensemble = load_harbor_temporary_fault_ensemble_config(args.config, section)
    criteria = (
        load_harbor_station_keeping_criteria_config(args.config)
        if args.confirmation
        else None
    )
    labels = (
        criteria.controller_labels
        if criteria is not None
        else (BASELINE, KINEMATIC_EWMA, KINEMATIC_RLS)
    )
    mpc_config = load_harbor_mpc_config(args.config)
    cases = run_temporary_fault_generalization(
        agents,
        simulation,
        communication,
        mpc_config,
        disturbance,
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
    summary = summarize_station_keeping(records)
    confirmation = (
        evaluate_station_keeping(summary, criteria) if criteria is not None else None
    )
    payload = {
        "ensemble": asdict(ensemble),
        "experiment": {
            "evaluation_role": section,
            "candidate_status": (
                "confirmed in simulation"
                if confirmation is not None and confirmation["passed"]
                else "confirmation criteria not met"
                if confirmation is not None
                else "development"
            ),
            "actuator_independent_current_measurement": True,
            "plant_rate_observer_updates": True,
        },
        "summary": summary,
        "trials": compact_time_varying_fault_records(records),
    }
    if criteria is not None:
        payload["confirmation_criteria"] = asdict(criteria)
        payload["confirmation_result"] = confirmation
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / f"{section}.json"
    plot_path = output / f"{section}.png"
    write_json(data_path, payload)
    save_station_keeping_plot(records, summary, plot_path, confirmation)
    print(json.dumps(summary, indent=2))
    if confirmation is not None:
        print(json.dumps(confirmation, indent=2))
    print(f"Saved station-keeping data: {data_path}")
    print(f"Saved station-keeping plot: {plot_path}")
