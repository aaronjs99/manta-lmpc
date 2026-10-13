"""Cli / arguments / manta / lmpc."""

from __future__ import annotations

import argparse

from scripts.configuration.paths import DEFAULT_CONFIG_PATH


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI overrides for the YAML-backed manta LMPC run."""
    parser = argparse.ArgumentParser(description="Run config-driven manta APF/LMPC.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH.name)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--apf-max-steps", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--mpc-horizon", type=int, default=None)
    parser.add_argument("--k-hull", type=int, default=None)
    parser.add_argument("--goal-tolerance", type=float, default=None)
    parser.add_argument("--make-video", action="store_true", default=None)
    parser.add_argument("--no-video", action="store_false", dest="make_video")
    parser.add_argument("--quiet", action="store_true", default=None)
    parser.add_argument("--verbose", action="store_false", dest="quiet")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="override output directory for this run",
    )
    parser.add_argument(
        "--stop-file",
        default="STOP_RUN",
        help="file path that stops the run when created",
    )
    return parser.parse_args(argv)
