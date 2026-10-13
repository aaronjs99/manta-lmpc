"""Experiments / harbor / localization study."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from scripts.assembly.harbor import load_harbor_runtime as load_harbor_config
from scripts.cli.arguments.harbor.localization_study import parse_args
from scripts.configuration.harbor_faults import load_harbor_disturbance_config
from scripts.configuration.harbor_simulation import HarborObservationNoiseConfig
from scripts.configuration.localization import load_range_aided_slam_config
from scripts.metrics.harbor.localization import build_localization_record
from scripts.plotting.harbor.localization import save_localization_diagnostics
from scripts.reporting.json import write_json
from scripts.simulation.harbor.engine import run_harbor_simulation
from scripts.simulation.harbor.localization import HarborRangeLocalization


def main(argv: list[str] | None = None) -> None:
    """Run matched localization modes and save one compact diagnostic artifact."""
    args = parse_args(argv)
    agents, simulation, communication = load_harbor_config(args.config)
    base = load_range_aided_slam_config(args.config)
    definitions = (
        ("Dead reckoning", replace(base, enabled=False, mode="known_anchor_ekf")),
        ("Known-anchor ranges", replace(base, enabled=True, mode="known_anchor_ekf")),
        ("Joint landmark SLAM", replace(base, enabled=True, mode="joint_landmark_ekf")),
        (
            "SLAM with 60% range loss",
            replace(
                base,
                enabled=True,
                mode="joint_landmark_ekf",
                dropout_probability=0.60,
            ),
        ),
    )
    records = []
    histories = {}
    for label, localization_config in definitions:
        localization = HarborRangeLocalization(localization_config)
        result = run_harbor_simulation(
            agents,
            simulation,
            communication,
            disturbance=load_harbor_disturbance_config(args.config),
            observation_noise=HarborObservationNoiseConfig(enabled=True, seed=1901),
            localization_provider=localization,
        )
        record, error_histories = build_localization_record(
            result, agents, localization, label
        )
        histories[label] = error_histories
        records.append(record)
        print(
            f"{label}: complete={result.all_goals_reached}, "
            f"ROV RMSE={records[-1]['position_rmse']['underwater_rov']:.3f} m",
            flush=True,
        )

    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "range_aided_slam_development.json"
    plot_path = output / "range_aided_slam_development.png"
    write_json(data_path, {"trials": records})

    save_localization_diagnostics(records, histories, agents, plot_path)
    print(f"Saved range-SLAM data: {data_path}")
    print(f"Saved range-SLAM plot: {plot_path}")
