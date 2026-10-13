"""Shared Harbor domain, platform, rollout, and controller contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class HarborPlatform(Protocol):
    """Minimal physical-platform interface used by shared Harbor records."""

    kind: str
    state_dim: int
    pose_dim: int

    def position(self, state: np.ndarray) -> np.ndarray: ...

    def goal_position(self, goal: np.ndarray) -> np.ndarray: ...


class HarborDisturbance(Protocol):
    """Disturbance description consumed by matched trial records."""

    agent_control_effectiveness_schedule: dict[str, tuple[dict[str, object], ...]]

    def effectiveness(
        self,
        model: HarborPlatform,
        agent_name: str | None = None,
        step: int | None = None,
    ) -> np.ndarray: ...

    def current(self, model: HarborPlatform) -> np.ndarray: ...


@dataclass(frozen=True)
class OperatingDomain:
    """Independent workspace bounds for one platform."""

    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    z_bounds: tuple[float, float]

    def __post_init__(self) -> None:
        for name, bounds in (
            ("x", self.x_bounds),
            ("y", self.y_bounds),
            ("z", self.z_bounds),
        ):
            if len(bounds) != 2 or not bounds[0] < bounds[1]:
                raise ValueError(f"domain {name}_bounds must be increasing")

    def project(self, model: HarborPlatform, state: np.ndarray) -> np.ndarray:
        """Project only the platform's own position into its configured domain."""
        value = np.asarray(state, dtype=float).copy()
        value[0] = np.clip(value[0], *self.x_bounds)
        value[1] = np.clip(value[1], *self.y_bounds)
        if model.kind == "rov":
            value[2] = np.clip(value[2], *self.z_bounds)
        return value

    def contains(self, position: np.ndarray) -> bool:
        """Return whether a world position belongs to this platform domain."""
        value = np.asarray(position, dtype=float)
        return bool(
            self.x_bounds[0] <= value[0] <= self.x_bounds[1]
            and self.y_bounds[0] <= value[1] <= self.y_bounds[1]
            and self.z_bounds[0] <= value[2] <= self.z_bounds[1]
        )


@dataclass(frozen=True)
class HarborAgent:
    """One independently actuated platform and its task metadata."""

    name: str
    model: HarborPlatform
    start: np.ndarray
    goal: np.ndarray
    radius: float
    domain: OperatingDomain
    waypoints: np.ndarray | None = None
    profile: str | None = None
    display_name: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("harbor agent name must not be empty")
        if self.profile is not None and not self.profile.strip():
            raise ValueError("harbor agent profile must not be empty")
        if self.display_name is not None and not self.display_name.strip():
            raise ValueError("harbor agent display_name must not be empty")
        if np.asarray(self.start).shape != (self.model.state_dim,):
            raise ValueError("harbor agent start does not match model state dimension")
        if np.asarray(self.goal).shape != (self.model.pose_dim,):
            raise ValueError(
                f"harbor agent goal must have shape ({self.model.pose_dim},)"
            )
        if self.radius <= 0.0:
            raise ValueError("harbor agent radius must be positive")
        if not self.domain.contains(self.model.position(self.start)):
            raise ValueError("harbor agent start lies outside its operating domain")
        if not self.domain.contains(self.model.goal_position(self.goal)):
            raise ValueError("harbor agent goal lies outside its operating domain")
        if self.waypoints is not None:
            waypoints = np.asarray(self.waypoints, dtype=float)
            if waypoints.ndim != 2 or waypoints.shape[1] != self.model.pose_dim:
                raise ValueError(
                    f"harbor agent waypoints must have shape (N, {self.model.pose_dim})"
                )
            if not all(
                self.domain.contains(self.model.goal_position(point))
                for point in waypoints
            ):
                raise ValueError(
                    "harbor agent waypoint lies outside its operating domain"
                )

    @property
    def route(self) -> np.ndarray:
        """Return intermediate waypoints followed by the final goal."""
        intermediate = (
            np.empty((0, self.model.pose_dim))
            if self.waypoints is None
            else np.asarray(self.waypoints, dtype=float)
        )
        return np.vstack((intermediate, self.goal))


@dataclass(frozen=True)
class HarborAgentSpec:
    """Model-independent platform settings parsed from a scenario file."""

    name: str
    kind: str
    model_parameters: dict[str, float]
    start: np.ndarray
    goal: np.ndarray
    radius: float
    domain: OperatingDomain
    waypoints: np.ndarray | None = None
    profile: str | None = None
    display_name: str | None = None


@dataclass(frozen=True)
class HarborResult:
    """Trajectories and communication telemetry for one rollout."""

    states: dict[str, np.ndarray]
    observed_states: dict[str, np.ndarray]
    positions: dict[str, np.ndarray]
    controls: dict[str, np.ndarray]
    applied_controls: dict[str, np.ndarray]
    applied_effectiveness: dict[str, np.ndarray]
    first_goal_steps: dict[str, int | None]
    final_goal_errors: dict[str, float]
    final_orientation_errors: dict[str, float]
    all_goals_reached: bool
    min_pairwise_distance: float
    pairwise_violation_count: int
    messages_sent: int
    messages_delivered: int
    messages_dropped: int
    guidance_update_count: int


@dataclass(frozen=True)
class RangeMeasurement:
    """One accepted scalar range observation delivered to an estimator."""

    beacon: str
    distance: float
    capture_step: int = 0


@dataclass(frozen=True)
class AgentMessage:
    """Platform-neutral state and intent observation delivered to a peer."""

    sender: str
    sent_step: int
    position: np.ndarray
    velocity: np.ndarray
    goal: np.ndarray
    cruise_speed: float


@dataclass(frozen=True)
class HarborLearningIteration:
    """One controller rollout and its safe-set admission decision."""

    label: str
    result: HarborResult
    admitted: bool
    completion_step_sum: int
    solver_calls: int
    solver_fallbacks: int
    solve_time_seconds: float
    max_collision_slack: float
    max_terminal_slack: float
    solve_count_by_agent: dict[str, int]
    fallback_count_by_agent: dict[str, int]
    failure_steps_by_agent: dict[str, list[int]]
    failure_status_counts: dict[str, int]
    final_residual_estimates: dict[str, list[float]]
    final_effectiveness_estimates: dict[str, np.ndarray]


class HarborControlProvider(Protocol):
    """Controller boundary consumed by the Harbor simulation engine."""

    def control(
        self,
        *,
        agent: HarborAgent,
        state: np.ndarray,
        navigation_goal: np.ndarray,
        desired_velocity: np.ndarray,
        inbox: dict[str, AgentMessage],
        step: int,
        dt: float,
        belief: dict | None = None,
    ) -> np.ndarray:
        """Return one bounded platform control for the current update."""


@dataclass(frozen=True)
class HarborRobustnessTrial:
    """One matched plant-mismatch trial and its controller telemetry."""

    label: str
    identification_strategy: str
    result: HarborResult
    valid: bool
    source_controller: str | None
    completion_step_sum: int
    solver_fallbacks: int
    solver_fallbacks_by_agent: dict[str, int]
    solver_failure_steps_by_agent: dict[str, list[int]]
    solver_failure_status_counts: dict[str, int]
    max_collision_slack: float
    max_domain_buffer_slack: float
    max_dynamic_state_slack: float
    residual_history: dict[str, np.ndarray]
    control_residual_history: dict[str, np.ndarray]
    residual_projection_retry_count_by_agent: dict[str, int]
    dynamic_state_retry_count_by_agent: dict[str, int]
    residual_rejection_count_by_agent: dict[str, int]
    final_residual_estimates: dict[str, np.ndarray]
    effectiveness_history: dict[str, np.ndarray]
    raw_effectiveness_history: dict[str, np.ndarray]
    final_effectiveness_estimates: dict[str, np.ndarray]
    effectiveness_change_steps_by_agent: dict[str, list[int]]
    recovery_offset_steps_by_agent: dict[str, list[int]]
    recovery_offset_rejections_by_agent: dict[str, int]
    recovery_offset_unarmed_rejections_by_agent: dict[str, int]
    recovery_offset_episode_limit_rejections_by_agent: dict[str, int]
    recovery_offset_dwell_rejections_by_agent: dict[str, int]
    excitation_history: dict[str, np.ndarray]
    information_std_history: dict[str, np.ndarray]
    probe_count_by_agent: dict[str, int]
    probe_channel_counts: dict[str, np.ndarray]
    probe_sequence_by_agent: dict[str, list[int]]
    probe_rejection_counts: dict[str, np.ndarray]


@dataclass(frozen=True)
class HarborFaultEnsembleCase:
    """One hidden actuator-loss draw and its matched controller trials."""

    seed: int
    disturbance: HarborDisturbance
    observation_seed: int | None
    trials: tuple[HarborRobustnessTrial, ...]
