"""Experiments / manta / sweep."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

from scripts.cli.arguments.manta.sweep import parse_args
from scripts.configuration.manta import (
    list_config_scenarios,
    load_project_config,
    override_project_config,
)
from scripts.experiments.manta.runner import run_manta_lmpc
from scripts.metrics.manta.benchmark import _record_for_result
from scripts.reporting.json import write_json
from scripts.reporting.manta_benchmark import _print_table, _write_csv


def main(argv: list[str] | None = None) -> None:
    """Run the selected scenarios and save a compact benchmark table."""
    args = parse_args(argv)
    scenarios = args.scenario or list_config_scenarios(args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"results/tmp/sweep_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for scenario_name in scenarios:
        config = load_project_config(args.config, scenario_name=scenario_name)
        config = override_project_config(
            config,
            lmpc={"iterations": args.iterations, "max_steps": args.max_steps},
            apf={"max_steps": args.apf_max_steps},
        )
        lmpc = config.lmpc
        apf = config.apf
        if not args.quiet:
            print(
                f"Running {scenario_name}: "
                f"iterations={lmpc.iterations}, max_steps={lmpc.max_steps}"
            )
        started_at = perf_counter()
        result = run_manta_lmpc(
            config.scenario,
            config=lmpc,
            apf_config=apf,
            dynamics_config=config.dynamics,
            verbose=False,
        )
        elapsed_seconds = perf_counter() - started_at
        records.append(
            _record_for_result(
                config.scenario.goals,
                lmpc,
                result,
                elapsed_seconds=elapsed_seconds,
            )
        )

    _write_csv(output_dir / "benchmark_summary.csv", records)
    write_json(output_dir / "benchmark_summary.json", records, trailing_newline=False)

    _print_table(records)
    print(f"Saved benchmark sweep to: {output_dir}")
