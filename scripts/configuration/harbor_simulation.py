"""Configuration / harbor simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scripts.dynamics.harbor.base import wrap_angle

if TYPE_CHECKING:
    from scripts.contracts.harbor import HarborAgent
    from scripts.dynamics.harbor.base import PlatformModel


@dataclass(frozen=True)
class HarborSimulationConfig:
    """Platform-neutral coordination and rollout settings."""

    dt: float = 0.2
    horizon: int = 160
    goal_hold_steps: int = 1
    guidance_update_interval_steps: int = 1
    goal_tolerance: float = 0.25
    orientation_tolerance: float = 0.15
    approach_speed_gain: float = 1.0
    coordination_distance: float = 2.0
    avoidance_gain: float = 1.5
    yielding_speed_scale: float = 0.15
    coordination_policy: str = "eta_priority"
    priority_response_scale: float = 0.5
    predict_delayed_messages: bool = True
    world_x_bounds: tuple[float, float] = (-5.0, 5.0)
    world_y_bounds: tuple[float, float] = (-5.0, 5.0)
    shoreline_y: float = 3.0
    seabed_z: float = -4.0

    def __post_init__(self) -> None:
        if (
            self.dt <= 0.0
            or self.horizon <= 0
            or self.goal_hold_steps <= 0
            or self.guidance_update_interval_steps <= 0
        ):
            raise ValueError("harbor dt and horizon must be positive")
        if (
            self.goal_tolerance <= 0.0
            or self.orientation_tolerance <= 0.0
            or self.approach_speed_gain <= 0.0
            or self.coordination_distance <= 0.0
        ):
            raise ValueError("harbor goal and coordination distances must be positive")
        if self.avoidance_gain < 0.0:
            raise ValueError("harbor avoidance_gain must be nonnegative")
        if not 0.0 <= self.yielding_speed_scale <= 1.0:
            raise ValueError("harbor yielding_speed_scale must be in [0, 1]")
        if self.coordination_policy not in {"reciprocal", "eta_priority"}:
            raise ValueError(
                "harbor coordination_policy must be reciprocal or eta_priority"
            )
        if not 0.0 <= self.priority_response_scale <= 1.0:
            raise ValueError("harbor priority_response_scale must be in [0, 1]")
        if not self.world_x_bounds[0] < self.world_x_bounds[1]:
            raise ValueError("harbor world_x_bounds must be increasing")
        if not self.world_y_bounds[0] < self.shoreline_y < self.world_y_bounds[1]:
            raise ValueError("harbor shoreline_y must lie inside world_y_bounds")
        if self.seabed_z >= 0.0:
            raise ValueError("harbor seabed_z must be below the water surface")


@dataclass(frozen=True)
class HarborDisturbanceConfig:
    """Unknown execution-plant effects used for robustness experiments."""

    water_current: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ugv_control_effectiveness: float | tuple[float, ...] = 1.0
    usv_control_effectiveness: float | tuple[float, ...] = 1.0
    rov_control_effectiveness: float | tuple[float, ...] = 1.0
    agent_control_effectiveness: dict[str, float | tuple[float, ...]] = field(
        default_factory=dict
    )
    agent_control_effectiveness_schedule: dict[str, tuple[dict[str, object], ...]] = (
        field(default_factory=dict)
    )
    evaluation_hold_steps: int = 12

    def __post_init__(self) -> None:
        if len(self.water_current) != 3:
            raise ValueError("water_current must contain [x, y, z] velocity")
        if not np.all(np.isfinite(self.water_current)):
            raise ValueError("water_current must be finite")
        for name in (
            "ugv_control_effectiveness",
            "usv_control_effectiveness",
            "rov_control_effectiveness",
        ):
            value = _normalize_effectiveness(getattr(self, name), name)
            object.__setattr__(self, name, value)
        normalized_agents = {
            str(name): _normalize_effectiveness(value, f"agent {name}")
            for name, value in self.agent_control_effectiveness.items()
        }
        if any(not name.strip() for name in normalized_agents):
            raise ValueError("agent effectiveness names must not be empty")
        object.__setattr__(self, "agent_control_effectiveness", normalized_agents)
        normalized_schedule = {}
        for (
            agent_name,
            configured_events,
        ) in self.agent_control_effectiveness_schedule.items():
            name = str(agent_name).strip()
            if not name:
                raise ValueError("scheduled effectiveness names must not be empty")
            events = []
            for configured in configured_events:
                if isinstance(configured, dict) and set(configured) == {
                    "step",
                    "effectiveness",
                }:
                    configured_step = configured["step"]
                    configured_effectiveness = configured["effectiveness"]
                elif isinstance(configured, (tuple, list)) and len(configured) == 2:
                    configured_step, configured_effectiveness = configured
                else:
                    raise ValueError(
                        "scheduled effectiveness events require step and effectiveness"
                    )
                step = int(configured_step)
                if step < 0 or step != configured_step:
                    raise ValueError(
                        "scheduled effectiveness steps must be nonnegative integers"
                    )
                events.append(
                    (
                        step,
                        _normalize_effectiveness(
                            configured_effectiveness, f"scheduled agent {name}"
                        ),
                    )
                )
            events.sort(key=lambda event: event[0])
            if len({step for step, _ in events}) != len(events):
                raise ValueError(
                    "scheduled effectiveness steps must be unique per agent"
                )
            normalized_schedule[name] = tuple(events)
        object.__setattr__(
            self, "agent_control_effectiveness_schedule", normalized_schedule
        )
        if self.evaluation_hold_steps <= 0:
            raise ValueError("evaluation_hold_steps must be positive")

    def effectiveness(
        self,
        model: PlatformModel,
        agent_name: str | None = None,
        step: int | None = None,
    ) -> np.ndarray:
        """Return one hidden effectiveness value per platform control channel."""
        configured = (
            self.agent_control_effectiveness[agent_name]
            if agent_name in self.agent_control_effectiveness
            else getattr(self, f"{model.kind}_control_effectiveness")
        )
        if step is not None and agent_name in self.agent_control_effectiveness_schedule:
            for event_step, event_value in self.agent_control_effectiveness_schedule[
                agent_name
            ]:
                if event_step > step:
                    break
                configured = event_value
        values = np.asarray(configured, dtype=float).reshape(-1)
        if len(values) == 1:
            return np.full(model.control_dim, values[0], dtype=float)
        if len(values) != model.control_dim:
            raise ValueError(
                f"{agent_name or model.kind} effectiveness must have "
                f"1 or {model.control_dim} entries"
            )
        return values.copy()

    def current(self, model: PlatformModel) -> np.ndarray:
        """Return the environmental world velocity acting on a platform."""
        if model.kind == "ugv":
            return np.zeros(3)
        value = np.asarray(self.water_current, dtype=float).copy()
        if model.kind == "usv":
            value[2] = 0.0
        return value


@dataclass(frozen=True)
class HarborObservationNoiseConfig:
    """Seeded onboard state-measurement noise, separate from plant truth."""

    enabled: bool = False
    seed: int = 0
    kind_state_std: dict[str, tuple[float, ...]] = field(default_factory=dict)
    agent_state_std: dict[str, tuple[float, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("observation-noise seed must be nonnegative")
        for field_name in ("kind_state_std", "agent_state_std"):
            values = {}
            for name, configured in getattr(self, field_name).items():
                key = str(name).strip()
                array = np.asarray(configured, dtype=float).reshape(-1)
                if not key or len(array) == 0 or not np.all(np.isfinite(array)):
                    raise ValueError(
                        f"{field_name} entries must be named finite vectors"
                    )
                if np.any(array < 0.0):
                    raise ValueError(
                        f"{field_name} standard deviations must be nonnegative"
                    )
                values[key] = tuple(float(value) for value in array)
            object.__setattr__(self, field_name, values)

    def state_std(self, agent: HarborAgent) -> np.ndarray:
        """Return the configured standard deviation for one local state."""
        configured = self.agent_state_std.get(
            agent.name,
            self.kind_state_std.get(agent.model.kind, (0.0,) * agent.model.state_dim),
        )
        values = np.asarray(configured, dtype=float)
        if values.shape != (agent.model.state_dim,):
            raise ValueError(
                f"{agent.name} observation noise must have "
                f"{agent.model.state_dim} entries"
            )
        return values

    def measure(
        self,
        agent: HarborAgent,
        state: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate one noisy local state while preserving angle topology."""
        truth = np.asarray(state, dtype=float)
        if not self.enabled:
            return truth.copy()
        measured = truth + rng.normal(0.0, self.state_std(agent))
        angle_indices = (2,) if agent.model.pose_dim == 3 else (3, 4, 5)
        for index in angle_indices:
            measured[index] = wrap_angle(measured[index])
        measured = agent.domain.project(agent.model, measured)
        model = agent.model
        if model.kind == "ugv":
            measured[3] = np.clip(
                measured[3], -model.max_reverse_speed, model.max_speed
            )
            if model.variant == "dynamic_skid_steer":
                measured[4] = np.clip(
                    measured[4], -model.max_yaw_rate, model.max_yaw_rate
                )
        elif model.kind == "usv":
            measured[3] = np.clip(measured[3], 0.0, model.max_speed)
            if model.variant == "marine_3dof":
                measured[4] = np.clip(
                    measured[4], -model.max_sway_speed, model.max_sway_speed
                )
                measured[5] = np.clip(
                    measured[5], -model.max_yaw_rate, model.max_yaw_rate
                )
        else:
            measured[6:8] = np.clip(
                measured[6:8], -model.max_horizontal_speed, model.max_horizontal_speed
            )
            measured[8] = np.clip(
                measured[8], -model.max_vertical_speed, model.max_vertical_speed
            )
            measured[9:12] = np.clip(
                measured[9:12], -model.max_angular_rate, model.max_angular_rate
            )
        return measured


def _normalize_effectiveness(value, name: str) -> float | tuple[float, ...]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} control effectiveness must be finite and nonempty")
    if np.any(array <= 0.0) or np.any(array > 1.5):
        raise ValueError(f"{name} control effectiveness must be in (0, 1.5]")
    if len(array) == 1:
        return float(array[0])
    return tuple(float(item) for item in array)
