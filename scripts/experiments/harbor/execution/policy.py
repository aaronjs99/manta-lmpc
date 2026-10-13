"""Typed specifications for Harbor experiment policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HarborPolicySpec:
    """One controller policy and its explicit model-predictive-control overrides."""

    label: str
    adaptive_effectiveness: bool
    estimator_mode: str
    learning: bool
    active_identification: bool
    source_controller: str | None
    identification_strategy: str
    minimum_probes: int
    mpc_overrides: tuple[tuple[str, Any], ...] = ()


def policy_specs(
    definitions: tuple[HarborPolicySpec, ...],
) -> tuple[HarborPolicySpec, ...]:
    """Return a validated immutable policy sequence without reordering it."""
    labels = [policy.label for policy in definitions]
    if len(labels) != len(set(labels)):
        raise ValueError("Harbor policy labels must be unique within a trial batch")
    available: set[str] = set()
    for policy in definitions:
        if (
            policy.source_controller is not None
            and policy.source_controller not in available
        ):
            raise ValueError(
                f"source policy {policy.source_controller!r} must precede "
                f"{policy.label!r}"
            )
        available.add(policy.label)
    return definitions
