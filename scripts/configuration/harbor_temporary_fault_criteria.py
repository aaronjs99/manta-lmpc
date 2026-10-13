"""Predeclared gates for temporary-fault confirmation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.configuration.harbor_criteria_io import load_harbor_criteria
from scripts.configuration.paths import DEFAULT_HARBOR_CONFIG


@dataclass(frozen=True)
class HarborConfirmationCriteriaConfig:
    """Predeclared closed-loop gates for temporary-fault confirmation."""

    controller_labels: tuple[str, ...] = (
        "Fixed-covariance RLS",
        "Innovation-threshold RLS",
    )
    minimum_adaptive_win_rate: float = 0.8
    require_positive_bootstrap_lower_bound: bool = True
    minimum_completion_rate: float = 1.0
    minimum_safety_rate: float = 1.0
    minimum_fallback_free_rate: float = 1.0
    maximum_mean_completion_cost_delta: float = 0.0

    def __post_init__(self) -> None:
        labels = tuple(str(label) for label in self.controller_labels)
        expected = ("Fixed-covariance RLS", "Innovation-threshold RLS")
        if labels != expected:
            raise ValueError(
                "confirmation controllers must be fixed-covariance then "
                "innovation-threshold RLS"
            )
        rates = (
            self.minimum_adaptive_win_rate,
            self.minimum_completion_rate,
            self.minimum_safety_rate,
            self.minimum_fallback_free_rate,
        )
        if any(not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("confirmation rates must lie in [0, 1]")
        if not np.isfinite(self.maximum_mean_completion_cost_delta):
            raise ValueError("confirmation completion-cost bound must be finite")
        object.__setattr__(self, "controller_labels", labels)


@dataclass(frozen=True)
class HarborRecoveryConfirmationCriteriaConfig:
    """Predeclared gates for actuator-recovery confirmation."""

    controller_labels: tuple[str, ...] = (
        "Fixed-covariance RLS",
        "Innovation-threshold RLS",
        "Recovery-prior threshold RLS",
    )
    minimum_recovery_win_rate: float = 0.8
    require_positive_bootstrap_lower_bound: bool = True
    minimum_completion_rate: float = 1.0
    minimum_safety_rate: float = 1.0
    minimum_fallback_free_rate: float = 1.0
    maximum_mean_fault_interval_rmse_delta: float = 0.0
    maximum_mean_final_rmse_delta: float = 0.0
    maximum_mean_completion_cost_delta: float = 0.0
    maximum_mean_current_rmse_delta: float = 0.0
    maximum_mean_final_current_rmse_delta: float = 0.0

    def __post_init__(self) -> None:
        labels = tuple(str(label) for label in self.controller_labels)
        valid_candidates = {
            "Recovery-prior threshold RLS",
            "Transient-offset threshold RLS",
        }
        if (
            labels[:2] != ("Fixed-covariance RLS", "Innovation-threshold RLS")
            or len(labels) != 3
            or labels[2] not in valid_candidates
        ):
            raise ValueError("recovery confirmation controllers are invalid")
        rates = (
            self.minimum_recovery_win_rate,
            self.minimum_completion_rate,
            self.minimum_safety_rate,
            self.minimum_fallback_free_rate,
        )
        if any(not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("recovery confirmation rates must lie in [0, 1]")
        bounds = (
            self.maximum_mean_fault_interval_rmse_delta,
            self.maximum_mean_final_rmse_delta,
            self.maximum_mean_completion_cost_delta,
            self.maximum_mean_current_rmse_delta,
            self.maximum_mean_final_current_rmse_delta,
        )
        if any(not np.isfinite(value) for value in bounds):
            raise ValueError("recovery confirmation bounds must be finite")
        object.__setattr__(self, "controller_labels", labels)


def load_harbor_confirmation_criteria_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
) -> HarborConfirmationCriteriaConfig:
    """Load the predeclared temporary-fault confirmation gates."""
    return load_harbor_criteria(
        HarborConfirmationCriteriaConfig,
        path,
        "temporary_fault_confirmation_criteria",
    )


def load_harbor_recovery_confirmation_criteria_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
    section: str = "temporary_fault_recovery_confirmation_criteria",
) -> HarborRecoveryConfirmationCriteriaConfig:
    """Load predeclared actuator-recovery confirmation gates."""
    return load_harbor_criteria(HarborRecoveryConfirmationCriteriaConfig, path, section)
