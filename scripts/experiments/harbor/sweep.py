"""Experiments / harbor / sweep."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.assembly.harbor import load_harbor_runtime as load_harbor_config
from scripts.cli.arguments.harbor.sweep import parse_args
from scripts.experiments.harbor.execution.network import sweep_network_robustness
from scripts.plotting.harbor.learning import save_network_robustness_heatmap
from scripts.reporting.json import write_json


def main(argv: list[str] | None = None) -> None:
    """Run the sweep and save compact data plus a visual robustness envelope."""
    args = parse_args(argv)
    agents, simulation, communication = load_harbor_config(args.config)
    records = sweep_network_robustness(
        agents,
        simulation,
        communication,
        delays=_parse_values(args.delays, int),
        dropout_probabilities=_parse_values(args.dropouts, float),
        seeds=_parse_values(args.seeds, int),
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "network_sweep.json", records)
    with (output / "network_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    heatmap = save_network_robustness_heatmap(
        records, output / "network_robustness.png"
    )
    print(json.dumps(records, indent=2))
    print(f"Saved network robustness heatmap: {heatmap}")


def _parse_values(text: str, value_type):
    values = [value_type(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("sweep value lists must not be empty")
    return values
