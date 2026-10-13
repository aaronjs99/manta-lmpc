"""Experiments / harbor / time varying fault study."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.assembly.harbor import load_harbor_runtime as load_harbor_config
from scripts.cli.arguments.harbor.time_varying_fault_study import parse_args
from scripts.configuration.harbor_faults import (
    load_harbor_fault_study_config,
    load_harbor_time_varying_fault_config,
)
from scripts.configuration.harbor_mpc import load_harbor_mpc_config
from scripts.configuration.harbor_observation import (
    load_harbor_observation_noise_config,
)
from scripts.experiments.harbor.execution.temporary_faults import (
    run_time_varying_fault_study,
)
from scripts.metrics.harbor.time_varying_fault import summarize_time_varying_faults
from scripts.metrics.harbor.time_varying_fault_records import (
    build_time_varying_fault_records,
)
from scripts.plotting.harbor.recovery import save_time_varying_fault_plot
from scripts.reporting.json import write_json


def main(argv: list[str] | None = None) -> None:
    """Run the matched scheduled-fault study and replace its artifacts."""
    args = parse_args(argv)
    agents, simulation, communication = load_harbor_config(args.config)
    disturbance, experiment = load_harbor_time_varying_fault_config(args.config)
    mpc_config = load_harbor_mpc_config(args.config)
    cases = run_time_varying_fault_study(
        agents,
        simulation,
        communication,
        mpc_config,
        disturbance,
        load_harbor_fault_study_config(args.config),
        experiment,
        load_harbor_observation_noise_config(args.config),
    )
    records = build_time_varying_fault_records(
        cases, agents, simulation, experiment.event_detection_window_steps
    )
    fault_event_steps = records[0]["fault_event_steps"]
    summary = summarize_time_varying_faults(records)
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "time_varying_fault_study.json"
    figure_path = output / "time_varying_fault_study.png"
    payload = {
        "experiment": {
            "observation_seeds": list(experiment.observation_seeds),
            "change_threshold": experiment.change_threshold,
            "covariance_inflation": experiment.covariance_inflation,
            "change_persistence": mpc_config.effectiveness_rls_change_persistence,
            "change_cooldown_steps": (
                mpc_config.effectiveness_rls_change_cooldown_steps
            ),
            "cusum_drift": mpc_config.effectiveness_rls_cusum_drift,
            "cusum_threshold": mpc_config.effectiveness_rls_cusum_threshold,
            "event_detection_window_steps": experiment.event_detection_window_steps,
            "fault_event_steps": fault_event_steps,
        },
        "summary": summary,
        "trials": records,
    }
    write_json(data_path, payload)
    save_time_varying_fault_plot(records, summary, figure_path)
    print(json.dumps(summary, indent=2))
    print(f"Saved time-varying-fault data: {data_path}")
    print(f"Saved time-varying-fault plot: {figure_path}")
