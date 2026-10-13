"""Cli / arguments / harbor / sweep."""

from __future__ import annotations

import argparse

from scripts.configuration.paths import DEFAULT_HARBOR_CONFIG


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep harbor communication delay and dropout with seeded trials."
    )
    parser.add_argument("--config", default=DEFAULT_HARBOR_CONFIG.name)
    parser.add_argument("--delays", default="0,1,2,4,6,8")
    parser.add_argument("--dropouts", default="0,0.1,0.2,0.3,0.5")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--output-dir", default="results/tmp/harbor_network_sweep")
    return parser.parse_args(argv)
