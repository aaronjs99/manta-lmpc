"""Dynamics / harbor / ground."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scripts.dynamics.harbor.base import PlatformModel, wrap_angle


@dataclass(frozen=True)
class UGVModel(PlatformModel):
    """Low-speed kinematic bicycle with acceleration and steering inputs."""

    max_speed: float = 1.0
    max_reverse_speed: float = 0.5
    max_acceleration: float = 1.0
    wheelbase: float = 0.8
    max_steering_angle: float = 0.55
    position_gain: float = 1.0
    pose_heading_gain: float = 2.0
    pose_goal_heading_gain: float = -0.6
    ground_z: float = 0.0
    kind: str = field(default="ugv", init=False)
    variant: str = field(default="kinematic_bicycle", init=False)
    state_dim: int = field(default=4, init=False)
    control_dim: int = field(default=2, init=False)
    pose_dim: int = field(default=3, init=False)

    def __post_init__(self) -> None:
        positive = (
            self.max_speed,
            self.max_reverse_speed,
            self.max_acceleration,
            self.wheelbase,
            self.max_steering_angle,
            self.position_gain,
            self.pose_heading_gain,
        )
        if min(positive) <= 0.0 or self.max_steering_angle >= np.pi / 2.0:
            raise ValueError("UGV bicycle parameters must be positive and nonsingular")

    def position(self, state: np.ndarray) -> np.ndarray:
        return np.array([state[0], state[1], self.ground_z], dtype=float)

    def velocity(self, state: np.ndarray) -> np.ndarray:
        return np.array([state[3] * np.cos(state[2]), state[3] * np.sin(state[2]), 0.0])

    def goal_position(self, goal: np.ndarray) -> np.ndarray:
        return np.array([goal[0], goal[1], self.ground_z], dtype=float)

    def orientation_error(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.array([wrap_angle(goal[2] - state[2])])

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        value = np.asarray(state, dtype=float).copy()
        acceleration = np.clip(
            control[0], -self.max_acceleration, self.max_acceleration
        )
        steering = np.clip(
            control[1], -self.max_steering_angle, self.max_steering_angle
        )
        value[3] = np.clip(
            value[3] + dt * acceleration,
            -self.max_reverse_speed,
            self.max_speed,
        )
        yaw_rate = value[3] * np.tan(steering) / self.wheelbase
        value[2] = value[2] + dt * yaw_rate
        value[0] += dt * value[3] * np.cos(value[2])
        value[1] += dt * value[3] * np.sin(value[2])
        return value

    def symbolic_step(self, ca, state, control, dt: float):
        speed = state[3] + dt * control[0]
        yaw_rate = speed * ca.tan(control[1]) / self.wheelbase
        heading = state[2] + dt * yaw_rate
        return ca.vertcat(
            state[0] + dt * speed * ca.cos(heading),
            state[1] + dt * speed * ca.sin(heading),
            heading,
            speed,
        )

    def symbolic_velocity(self, ca, state):
        return state[3]

    def control_scale(self) -> np.ndarray:
        return np.array([self.max_acceleration, self.max_steering_angle])


@dataclass(frozen=True)
class SkidSteerUGVModel(PlatformModel):
    """Planar skid-steer dynamics driven by left/right side forces."""

    mass: float = 17.0
    yaw_inertia: float = 0.65
    max_speed: float = 2.0
    mission_speed: float | None = None
    max_reverse_speed: float = 1.0
    max_yaw_rate: float = 1.5
    max_force: float = 120.0
    max_yaw_moment: float = 35.0
    effective_track: float = 0.5
    drivetrain: str = "skid_steer"
    linear_drag: float = 8.0
    quadratic_drag: float = 8.0
    yaw_linear_drag: float = 1.0
    yaw_quadratic_drag: float = 0.5
    velocity_response_gain: float = 2.0
    heading_gain: float = 2.0
    yaw_response_gain: float = 2.5
    ground_z: float = 0.0
    kind: str = field(default="ugv", init=False)
    variant: str = field(default="dynamic_skid_steer", init=False)
    state_dim: int = field(default=5, init=False)
    control_dim: int = field(default=2, init=False)
    pose_dim: int = field(default=3, init=False)

    def __post_init__(self) -> None:
        positive = (
            self.mass,
            self.yaw_inertia,
            self.max_speed,
            self.max_reverse_speed,
            self.max_yaw_rate,
            self.max_force,
            self.max_yaw_moment,
            self.effective_track,
            self.velocity_response_gain,
            self.heading_gain,
            self.yaw_response_gain,
        )
        if min(positive) <= 0.0:
            raise ValueError("skid-steer UGV parameters must be positive")
        if (
            self.mission_speed is not None
            and not 0.0 < self.mission_speed <= self.max_speed
        ):
            raise ValueError("UGV mission_speed must be in (0, max_speed]")
        if (
            min(
                self.linear_drag,
                self.quadratic_drag,
                self.yaw_linear_drag,
                self.yaw_quadratic_drag,
            )
            < 0.0
        ):
            raise ValueError("skid-steer UGV damping must be nonnegative")
        if not self.drivetrain.strip():
            raise ValueError("skid-steer UGV drivetrain must be named")
        if self.max_yaw_moment > self.effective_track * self.max_side_force + 1e-12:
            raise ValueError(
                "UGV max_yaw_moment exceeds the left/right drive-side authority"
            )

    @property
    def max_side_force(self) -> float:
        """Return the symmetric force bound for either drive side."""
        return self.max_force / 2.0

    def generalized_wrench(self, control: np.ndarray) -> np.ndarray:
        """Map left/right drive-side forces to surge force and yaw moment."""
        left, right = np.clip(
            np.asarray(control, dtype=float),
            -self.max_side_force,
            self.max_side_force,
        )
        return np.array([left + right, 0.5 * self.effective_track * (right - left)])

    def position(self, state: np.ndarray) -> np.ndarray:
        return np.array([state[0], state[1], self.ground_z], dtype=float)

    def velocity(self, state: np.ndarray) -> np.ndarray:
        return np.array([state[3] * np.cos(state[2]), state[3] * np.sin(state[2]), 0.0])

    def goal_position(self, goal: np.ndarray) -> np.ndarray:
        return np.array([goal[0], goal[1], self.ground_z], dtype=float)

    def orientation_error(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.array([wrap_angle(goal[2] - state[2])])

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        value = np.asarray(state, dtype=float).copy()
        force, moment = self.generalized_wrench(control)
        speed = (
            value[3]
            + dt
            * (
                force
                - self.linear_drag * value[3]
                - self.quadratic_drag * abs(value[3]) * value[3]
            )
            / self.mass
        )
        yaw_rate = (
            value[4]
            + dt
            * (
                moment
                - self.yaw_linear_drag * value[4]
                - self.yaw_quadratic_drag * abs(value[4]) * value[4]
            )
            / self.yaw_inertia
        )
        value[3] = np.clip(speed, -self.max_reverse_speed, self.max_speed)
        value[4] = np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)
        value[2] += dt * value[4]
        value[0] += dt * value[3] * np.cos(value[2])
        value[1] += dt * value[3] * np.sin(value[2])
        return value

    def symbolic_step(self, ca, state, control, dt: float):
        force = control[0] + control[1]
        moment = 0.5 * self.effective_track * (control[1] - control[0])
        speed = (
            state[3]
            + dt
            * (
                force
                - self.linear_drag * state[3]
                - self.quadratic_drag * ca.fabs(state[3]) * state[3]
            )
            / self.mass
        )
        yaw_rate = (
            state[4]
            + dt
            * (
                moment
                - self.yaw_linear_drag * state[4]
                - self.yaw_quadratic_drag * ca.fabs(state[4]) * state[4]
            )
            / self.yaw_inertia
        )
        heading = state[2] + dt * yaw_rate
        return ca.vertcat(
            state[0] + dt * speed * ca.cos(heading),
            state[1] + dt * speed * ca.sin(heading),
            heading,
            speed,
            yaw_rate,
        )

    def symbolic_velocity(self, ca, state):
        return state[3:5]

    def control_scale(self) -> np.ndarray:
        return np.full(2, self.max_side_force)
