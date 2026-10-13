"""Cli / arguments / manta / sweep."""

from __future__ import annotations

import argparse

from scripts.configuration.paths import DEFAULT_CONFIG_PATH


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse sweep options."""
    parser = argparse.ArgumentParser(description="Run manta scenario benchmarks.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH.name)
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="scenario to run; repeat for multiple scenarios",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="LMPC iterations per scenario; default 0 is APF-only",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--apf-max-steps", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory for benchmark_summary.csv/json",
    )
    return parser.parse_args(argv)
