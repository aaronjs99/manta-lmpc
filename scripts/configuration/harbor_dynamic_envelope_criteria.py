"""Frozen criteria for elastic versus hard dynamic-state envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.configuration.harbor_criteria_io import load_harbor_criteria
from scripts.configuration.paths import DEFAULT_HARBOR_CONFIG


@dataclass(frozen=True)
class HarborDynamicEnvelopeCriteriaConfig:
    """Frozen gates for elastic versus hard dynamic-state envelopes."""

    controller_labels: tuple[str, ...] = (
        "Hard-envelope transient-offset RLS",
        "Retry-elastic transient-offset RLS",
    )
    minimum_candidate_completion_rate: float = 1.0
    minimum_completion_rescue_rate: float = 0.1
    maximum_completion_regression_rate: float = 0.0
    minimum_safety_rate: float = 1.0
    minimum_fallback_free_rate: float = 1.0
    maximum_mean_completion_cost_delta: float = 5.0
    maximum_mean_recovery_rmse_delta: float = 0.001
    maximum_mean_current_rmse_delta: float = 0.001
    maximum_dynamic_state_slack: float = 0.02

    def __post_init__(self) -> None:
        labels = tuple(str(label) for label in self.controller_labels)
        if labels != (
            "Hard-envelope transient-offset RLS",
            "Retry-elastic transient-offset RLS",
        ):
            raise ValueError("dynamic-envelope controllers are fixed")
        rates = (
            self.minimum_candidate_completion_rate,
            self.minimum_completion_rescue_rate,
            self.maximum_completion_regression_rate,
            self.minimum_safety_rate,
            self.minimum_fallback_free_rate,
        )
        if any(not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("dynamic-envelope rates must lie in [0, 1]")
        bounds = (
            self.maximum_mean_completion_cost_delta,
            self.maximum_mean_recovery_rmse_delta,
            self.maximum_mean_current_rmse_delta,
            self.maximum_dynamic_state_slack,
        )
        if any(not np.isfinite(value) for value in bounds):
            raise ValueError("dynamic-envelope bounds must be finite")
        if self.maximum_dynamic_state_slack < 0.0:
            raise ValueError("maximum dynamic-state slack must be nonnegative")
        object.__setattr__(self, "controller_labels", labels)


def load_harbor_dynamic_envelope_criteria_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
) -> HarborDynamicEnvelopeCriteriaConfig:
    """Load frozen elastic dynamic-envelope confirmation gates."""
    return load_harbor_criteria(
        HarborDynamicEnvelopeCriteriaConfig,
        path,
        "dynamic_envelope_confirmation_criteria",
    )
