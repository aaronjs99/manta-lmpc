"""Cli / arguments / harbor / communication."""

from __future__ import annotations

import argparse

from scripts.configuration.paths import DEFAULT_HARBOR_CONFIG


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run communication ablations for the heterogeneous harbor scenario."
    )
    parser.add_argument("--config", default=DEFAULT_HARBOR_CONFIG.name)
    parser.add_argument(
        "--mode", choices=("compare", "communication", "independent"), default="compare"
    )
    parser.add_argument(
        "--policy",
        choices=("reciprocal", "eta_priority"),
        default=None,
        help="override the communication coordination policy",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="optional JSON path; no artifact is written when omitted",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="optional directory for a comparison PNG and coordinated GIF",
    )
    return parser.parse_args(argv)
