"""Experiments / harbor / horizon study."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.assembly.harbor import load_harbor_runtime as load_harbor_config
from scripts.cli.arguments.harbor.horizon_study import parse_args
from scripts.configuration.harbor_mpc import load_harbor_mpc_config
from scripts.experiments.harbor.execution.network import sweep_prediction_horizons
from scripts.plotting.harbor.learning import save_horizon_efficiency
from scripts.reporting.json import write_json


def main(argv: list[str] | None = None) -> None:
    """Run the controlled horizon study and overwrite its two artifacts."""
    args = parse_args(argv)
    horizons = [
        int(value.strip()) for value in args.horizons.split(",") if value.strip()
    ]
    agents, simulation, communication = load_harbor_config(args.config)
    records = sweep_prediction_horizons(
        agents,
        simulation,
        communication,
        load_harbor_mpc_config(args.config),
        horizons=horizons,
    )
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "horizon_study.json", records)
    figure = save_horizon_efficiency(records, output / "horizon_efficiency.png")
    print(json.dumps(records, indent=2))
    print(f"Saved horizon-efficiency plot: {figure}")
