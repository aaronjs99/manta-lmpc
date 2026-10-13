"""Cli / arguments / harbor / temporary fault generalization."""

from __future__ import annotations

import argparse

from scripts.configuration.paths import DEFAULT_HARBOR_CONFIG


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate adaptive distributed MPC over hidden temporary actuator faults."
    )
    parser.add_argument("--config", default=DEFAULT_HARBOR_CONFIG.name)
    parser.add_argument("--artifact-dir", default="results/latest/harbor")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--holdout",
        action="store_true",
        help="use the separately seeded holdout ensemble and artifact names",
    )
    mode.add_argument(
        "--confirmation",
        action="store_true",
        help="run the frozen threshold-RLS confirmation ensemble",
    )
    mode.add_argument(
        "--recovery-development",
        action="store_true",
        help="compare the channel-selective recovery prior on new development cases",
    )
    mode.add_argument(
        "--recovery-confirmation",
        action="store_true",
        help="run the frozen recovery-prior confirmation ensemble",
    )
    mode.add_argument(
        "--transient-recovery-development",
        action="store_true",
        help="compare a decaying controller offset with ordinary threshold RLS",
    )
    mode.add_argument(
        "--transient-recovery-confirmation",
        action="store_true",
        help="run the frozen rank-gated transient recovery confirmation",
    )
    return parser.parse_args(argv)
