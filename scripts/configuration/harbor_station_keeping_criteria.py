"""Frozen criteria for actuator-decoupled marine-current observation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.configuration.harbor_criteria_io import load_harbor_criteria
from scripts.configuration.paths import DEFAULT_HARBOR_CONFIG


@dataclass(frozen=True)
class HarborStationKeepingCriteriaConfig:
    """Frozen gates for actuator-decoupled marine current observation."""

    controller_labels: tuple[str, ...] = (
        "Hard-envelope transient-offset RLS",
        "Kinematic-current RLS transient-offset RLS",
    )
    minimum_candidate_completion_rate: float = 1.0
    minimum_usv_yaw_win_rate: float = 0.8
    minimum_safety_rate: float = 1.0
    minimum_fallback_free_rate: float = 1.0
    maximum_mean_usv_yaw_error_delta: float = -0.01
    maximum_mean_usv_position_error_delta: float = 0.03
    maximum_mean_usv_current_rmse_delta: float = -0.01
    maximum_mean_recovery_rmse_delta: float = 0.002
    maximum_mean_completion_cost_delta: float = 15.0

    def __post_init__(self) -> None:
        labels = tuple(str(label) for label in self.controller_labels)
        if labels != (
            "Hard-envelope transient-offset RLS",
            "Kinematic-current RLS transient-offset RLS",
        ):
            raise ValueError("station-keeping controllers are fixed")
        rates = (
            self.minimum_candidate_completion_rate,
            self.minimum_usv_yaw_win_rate,
            self.minimum_safety_rate,
            self.minimum_fallback_free_rate,
        )
        if any(not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("station-keeping rates must lie in [0, 1]")
        bounds = (
            self.maximum_mean_usv_yaw_error_delta,
            self.maximum_mean_usv_position_error_delta,
            self.maximum_mean_usv_current_rmse_delta,
            self.maximum_mean_recovery_rmse_delta,
            self.maximum_mean_completion_cost_delta,
        )
        if any(not np.isfinite(value) for value in bounds):
            raise ValueError("station-keeping bounds must be finite")
        object.__setattr__(self, "controller_labels", labels)


def load_harbor_station_keeping_criteria_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
) -> HarborStationKeepingCriteriaConfig:
    """Load frozen kinematic-current observer confirmation gates."""
    return load_harbor_criteria(
        HarborStationKeepingCriteriaConfig,
        path,
        "station_keeping_confirmation_criteria",
    )
