"""Cli / arguments / manta / sanity."""

from __future__ import annotations

import argparse

from scripts.configuration.paths import DEFAULT_CONFIG_PATH


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse YAML config path and lightweight sanity-check overrides."""
    parser = argparse.ArgumentParser(description="Run manta simulation sanity checks.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH.name)
    parser.add_argument("--scenario", default="all", help="scenario name or 'all'")
    parser.add_argument("--horizon", type=int, default=None, help="simulation horizon")
    parser.add_argument("--dt", type=float, default=None, help="simulation time step")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="override output directory (default: results/sanity_<timestamp>)",
    )
    return parser.parse_args(argv)
