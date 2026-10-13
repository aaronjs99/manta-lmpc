"""Shared observability diagnostic data contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservabilityReport:
    """Numerical rank diagnostics for the recent nonlinear measurement geometry."""

    state_dimension: int
    rank: int
    smallest_singular_value: float
    condition_number: float
    observable: bool
