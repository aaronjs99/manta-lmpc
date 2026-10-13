"""Cli / arguments / harbor / dynamic envelope study."""

from __future__ import annotations

import argparse

from scripts.configuration.paths import DEFAULT_HARBOR_CONFIG


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test elastic dynamic-state envelopes under hidden current and actuator loss."
    )
    parser.add_argument("--config", default=DEFAULT_HARBOR_CONFIG.name)
    parser.add_argument("--artifact-dir", default="results/latest/harbor")
    parser.add_argument(
        "--confirmation",
        action="store_true",
        help="use the untouched dynamic-envelope confirmation ensemble",
    )
    return parser.parse_args(argv)
