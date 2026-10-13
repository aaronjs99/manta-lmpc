"""Cli / arguments / harbor / robustness."""

from __future__ import annotations

import argparse

from scripts.configuration.paths import DEFAULT_HARBOR_CONFIG


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark nominal and locally adaptive harbor control under plant mismatch."
    )
    parser.add_argument("--config", default=DEFAULT_HARBOR_CONFIG.name)
    parser.add_argument("--artifact-dir", default="results/latest/harbor")
    parser.add_argument("--no-gif", action="store_true")
    return parser.parse_args(argv)
