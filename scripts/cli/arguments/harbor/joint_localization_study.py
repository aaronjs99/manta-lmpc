"""Cli / arguments / harbor / joint localization study."""

from __future__ import annotations

import argparse

from scripts.configuration.paths import DEFAULT_HARBOR_CONFIG
from scripts.configuration.study_labels import POLICIES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate estimated-state distributed MPC under current and actuator faults."
    )
    parser.add_argument("--config", default=DEFAULT_HARBOR_CONFIG.name)
    parser.add_argument("--artifact-dir", default="results/latest/harbor")
    parser.add_argument(
        "--case-limit", type=int, default=None, help="run only the first N YAML cases"
    )
    parser.add_argument("--case-seed", type=int, default=None)
    parser.add_argument("--policy", action="append", choices=POLICIES)
    parser.add_argument("--make-gif", action="store_true")
    parser.add_argument("--gif-only", action="store_true")
    return parser.parse_args(argv)
