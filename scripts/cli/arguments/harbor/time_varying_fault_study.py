"""Cli / arguments / harbor / time varying fault study."""

from __future__ import annotations

import argparse

from scripts.configuration.paths import DEFAULT_HARBOR_CONFIG


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate online identification of scheduled heterogeneous actuator faults."
    )
    parser.add_argument("--config", default=DEFAULT_HARBOR_CONFIG.name)
    parser.add_argument("--artifact-dir", default="results/latest/harbor")
    return parser.parse_args(argv)
