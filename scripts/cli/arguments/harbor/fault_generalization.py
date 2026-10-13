"""Cli / arguments / harbor / fault generalization."""

from __future__ import annotations

import argparse

from scripts.configuration.paths import DEFAULT_HARBOR_CONFIG


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate actuator-identification policies over stratified hidden faults."
    )
    parser.add_argument("--config", default=DEFAULT_HARBOR_CONFIG.name)
    parser.add_argument("--artifact-dir", default="results/latest/harbor")
    parser.add_argument("--with-observation-noise", action="store_true")
    return parser.parse_args(argv)
