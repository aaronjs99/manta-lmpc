"""Cli / arguments / harbor / horizon study."""

from __future__ import annotations

import argparse

from scripts.configuration.paths import DEFAULT_HARBOR_CONFIG


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark distributed MPC and LMPC across matched prediction horizons."
    )
    parser.add_argument("--config", default=DEFAULT_HARBOR_CONFIG.name)
    parser.add_argument("--horizons", default="8,12,15")
    parser.add_argument("--artifact-dir", default="results/latest/harbor")
    return parser.parse_args(argv)
