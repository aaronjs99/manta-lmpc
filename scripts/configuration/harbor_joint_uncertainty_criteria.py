"""Predeclared criteria for joint hidden-current and actuator uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.configuration.harbor_criteria_io import load_harbor_criteria
from scripts.configuration.paths import DEFAULT_HARBOR_CONFIG


@dataclass(frozen=True)
class HarborJointUncertaintyCriteriaConfig:
    """Predeclared control-performance gates under joint hidden uncertainty."""

    controller_labels: tuple[str, ...] = (
        "Innovation-threshold RLS",
        "Transient-offset threshold RLS",
    )
    minimum_candidate_completion_rate: float = 1.0
    minimum_completion_rescue_rate: float = 0.1
    maximum_completion_regression_rate: float = 0.0
    minimum_safety_rate: float = 1.0
    minimum_fallback_free_rate: float = 1.0
    minimum_recovery_win_rate: float = 0.6
    maximum_mean_completion_cost_delta: float = 0.0
    maximum_mean_final_effectiveness_rmse_delta: float = 0.0
    maximum_mean_current_rmse_delta: float = 0.001
    maximum_mean_final_current_rmse_delta: float = 0.001

    def __post_init__(self) -> None:
        labels = tuple(str(label) for label in self.controller_labels)
        if labels != (
            "Innovation-threshold RLS",
            "Transient-offset threshold RLS",
        ):
            raise ValueError("joint-uncertainty controllers are fixed")
        rates = (
            self.minimum_candidate_completion_rate,
            self.minimum_completion_rescue_rate,
            self.maximum_completion_regression_rate,
            self.minimum_safety_rate,
            self.minimum_fallback_free_rate,
            self.minimum_recovery_win_rate,
        )
        if any(not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("joint-uncertainty rates must lie in [0, 1]")
        bounds = (
            self.maximum_mean_completion_cost_delta,
            self.maximum_mean_final_effectiveness_rmse_delta,
            self.maximum_mean_current_rmse_delta,
            self.maximum_mean_final_current_rmse_delta,
        )
        if any(not np.isfinite(value) for value in bounds):
            raise ValueError("joint-uncertainty bounds must be finite")
        object.__setattr__(self, "controller_labels", labels)


def load_harbor_joint_uncertainty_criteria_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
) -> HarborJointUncertaintyCriteriaConfig:
    """Load frozen joint-current and temporary-fault confirmation gates."""
    return load_harbor_criteria(
        HarborJointUncertaintyCriteriaConfig,
        path,
        "joint_uncertainty_confirmation_criteria",
    )
