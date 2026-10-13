"""Manta scenario and obstacle data contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrajectoryValidation:
    """Safety and learning-admission verdict for a trajectory."""

    all_goals_reached: bool
    min_pairwise_distance: float
    pairwise_violation_count: int
    min_obstacle_clearance: float
    obstacle_violation_count: int
    fallback_count: int
    safety_intervention_count: int

    @property
    def safe(self) -> bool:
        return self.pairwise_violation_count == 0 and self.obstacle_violation_count == 0

    @property
    def valid(self) -> bool:
        return self.all_goals_reached and self.safe

    @property
    def usable_for_learning(self) -> bool:
        return self.valid

    @property
    def solver_clean(self) -> bool:
        return self.fallback_count == 0 and self.safety_intervention_count == 0

    def to_dict(self) -> dict[str, bool | float | int]:
        return {
            "valid": self.valid,
            "safe": self.safe,
            "solver_clean": self.solver_clean,
            "usable_for_learning": self.usable_for_learning,
            **asdict(self),
        }


@dataclass(frozen=True)
class StaticObstacle:
    """Circular obstacle metadata used by planning, control, and plotting."""

    center: tuple[float, float]
    radius: float
    physical_radius: float | None = None

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError("obstacle center must contain two finite coordinates")
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("obstacle radius must be positive and finite")
        if self.physical_radius is not None and (
            not np.isfinite(self.physical_radius)
            or self.physical_radius <= 0.0
            or self.physical_radius > self.radius
        ):
            raise ValueError(
                "obstacle physical_radius must be positive and no larger than radius"
            )


@dataclass(frozen=True)
class Scenario:
    """Validated multi-agent manta scenario geometry and task goals."""

    name: str
    starts: np.ndarray
    goals: np.ndarray
    safety_distance: float
    obstacle: StaticObstacle

    def __post_init__(self) -> None:
        starts = np.asarray(self.starts, dtype=float)
        goals = np.asarray(self.goals, dtype=float)
        if not self.name.strip():
            raise ValueError("scenario name must not be empty")
        if starts.ndim != 2 or starts.shape[0] < 1 or starts.shape[1] != 7:
            raise ValueError(
                f"scenario starts must have shape (A>=1, 7), got {starts.shape}"
            )
        if goals.shape != starts.shape:
            raise ValueError(
                "scenario goals must match starts shape, got "
                f"{goals.shape} and {starts.shape}"
            )
        if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(goals)):
            raise ValueError("scenario starts and goals must contain finite values")
        if not np.isfinite(self.safety_distance) or self.safety_distance <= 0.0:
            raise ValueError("scenario safety_distance must be positive and finite")
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "goals", goals)


@dataclass
class MantaLMPCRunResult:
    """In-memory result from one config-driven APF/LMPC run."""

    scenario_name: str
    histories: list[dict[int, np.ndarray]]
    controls_by_iteration: list[np.ndarray]
    slack_by_iteration: list[np.ndarray]
    statuses_by_iteration: list[list[dict[int, str]]]
    success_by_iteration: list[bool]
    goal_reached_by_iteration: list[bool]
    learned_by_iteration: list[bool]
    validation_by_iteration: list[TrajectoryValidation]
    selected_iteration: int | None
    initial_controls: dict[int, np.ndarray] | None = None
    learning_settings: dict[str, Any] | None = None

    @property
    def completed_iterations(self) -> int:
        """Number of completed LMPC iterations, excluding the APF seed."""
        return len(self.histories) - 1

    @property
    def final_history(self) -> dict[int, np.ndarray]:
        """Return the selected valid history, or the latest history if none is valid."""
        if self.selected_iteration is None:
            return self.histories[-1]
        return self.histories[self.selected_iteration]

    @property
    def final_statuses(self) -> list[dict[int, str]] | None:
        """Return statuses for the selected final LMPC iteration."""
        if self.selected_iteration is None or self.selected_iteration == 0:
            return None
        return self.statuses_by_iteration[self.selected_iteration - 1]

    @property
    def final_controls(self) -> np.ndarray | None:
        """Return controls for the selected final LMPC iteration."""
        if self.selected_iteration is None or self.selected_iteration == 0:
            return None
        return self.controls_by_iteration[self.selected_iteration - 1]

    @property
    def report_histories(self) -> list[dict[int, np.ndarray]]:
        """Return histories through the selected final iteration for plots."""
        if self.selected_iteration is None:
            return self.histories
        return self.histories[: self.selected_iteration + 1]

    @property
    def report_statuses(self) -> list[list[dict[int, str]]] | None:
        """Return statuses through the selected final iteration."""
        if self.selected_iteration is None or self.selected_iteration == 0:
            return None
        return self.statuses_by_iteration[: self.selected_iteration]
