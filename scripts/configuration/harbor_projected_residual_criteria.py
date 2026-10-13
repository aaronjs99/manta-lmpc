"""Frozen criteria for actuation-subspace disturbance injection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.configuration.harbor_criteria_io import load_harbor_criteria
from scripts.configuration.paths import DEFAULT_HARBOR_CONFIG


@dataclass(frozen=True)
class HarborProjectedResidualCriteriaConfig:
    """Frozen gates for actuation-subspace disturbance injection."""

    controller_labels: tuple[str, ...] = (
        "Transient-offset threshold RLS",
        "Projected transient-offset RLS",
    )
    minimum_candidate_completion_rate: float = 1.0
    minimum_completion_rescue_rate: float = 0.1
    maximum_completion_regression_rate: float = 0.0
    minimum_safety_rate: float = 1.0
    minimum_fallback_free_rate: float = 1.0
    maximum_mean_completion_cost_delta: float = 0.0
    maximum_mean_recovery_rmse_delta: float = 0.001
    maximum_mean_current_rmse_delta: float = 0.001
    maximum_mean_control_current_rmse_delta: float = 0.0
    maximum_mean_final_control_current_rmse_delta: float = 0.0

    def __post_init__(self) -> None:
        labels = tuple(str(label) for label in self.controller_labels)
        if labels != (
            "Transient-offset threshold RLS",
            "Projected transient-offset RLS",
        ):
            raise ValueError("projected-residual controllers are fixed")
        rates = (
            self.minimum_candidate_completion_rate,
            self.minimum_completion_rescue_rate,
            self.maximum_completion_regression_rate,
            self.minimum_safety_rate,
            self.minimum_fallback_free_rate,
        )
        if any(not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("projected-residual rates must lie in [0, 1]")
        bounds = (
            self.maximum_mean_completion_cost_delta,
            self.maximum_mean_recovery_rmse_delta,
            self.maximum_mean_current_rmse_delta,
            self.maximum_mean_control_current_rmse_delta,
            self.maximum_mean_final_control_current_rmse_delta,
        )
        if any(not np.isfinite(value) for value in bounds):
            raise ValueError("projected-residual bounds must be finite")
        object.__setattr__(self, "controller_labels", labels)


def load_harbor_projected_residual_criteria_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
) -> HarborProjectedResidualCriteriaConfig:
    """Load frozen actuation-subspace projection confirmation gates."""
    return load_harbor_criteria(
        HarborProjectedResidualCriteriaConfig,
        path,
        "projected_residual_confirmation_criteria",
    )
