"""Dynamics / harbor / reduced."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scripts.dynamics.harbor.base import PlatformModel, wrap_angle


@dataclass(frozen=True)
class ReducedUGVModel(PlatformModel):
    """Acceleration-controlled planar unicycle for a harbor ground vehicle."""

    max_speed: float = 1.0
    max_acceleration: float = 1.0
    max_yaw_rate: float = 1.2
    ground_z: float = 0.0
    kind: str = field(default="ugv", init=False)
    variant: str = field(default="reduced_unicycle", init=False)
    state_dim: int = field(default=4, init=False)
    control_dim: int = field(default=2, init=False)
    pose_dim: int = field(default=3, init=False)

    def __post_init__(self) -> None:
        if min(self.max_speed, self.max_acceleration, self.max_yaw_rate) <= 0.0:
            raise ValueError(
                "UGV speed, acceleration, and yaw-rate bounds must be positive"
            )

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
        yaw_rate = np.clip(control[1], -self.max_yaw_rate, self.max_yaw_rate)
        value[3] = np.clip(value[3] + dt * acceleration, 0.0, self.max_speed)
        value[2] = wrap_angle(value[2] + dt * yaw_rate)
        value[0] += dt * value[3] * np.cos(value[2])
        value[1] += dt * value[3] * np.sin(value[2])
        return value

    def symbolic_step(self, ca, state, control, dt: float):
        speed = state[3] + dt * control[0]
        heading = state[2] + dt * control[1]
        return ca.vertcat(
            state[0] + dt * speed * ca.cos(heading),
            state[1] + dt * speed * ca.sin(heading),
            heading,
            speed,
        )

    def symbolic_velocity(self, ca, state):
        return state[3]

    def control_scale(self) -> np.ndarray:
        return np.ones(2)


@dataclass(frozen=True)
class ReducedUSVModel(PlatformModel):
    """Planar surge-yaw surface-vessel model with linear drag."""

    max_speed: float = 1.0
    max_thrust: float = 1.5
    max_yaw_rate: float = 0.8
    drag: float = 0.25
    surface_z: float = 0.0
    kind: str = field(default="usv", init=False)
    variant: str = field(default="reduced_surge_yaw", init=False)
    state_dim: int = field(default=4, init=False)
    control_dim: int = field(default=2, init=False)
    pose_dim: int = field(default=3, init=False)

    def __post_init__(self) -> None:
        if min(self.max_speed, self.max_thrust, self.max_yaw_rate) <= 0.0:
            raise ValueError("USV speed, thrust, and yaw-rate bounds must be positive")
        if self.drag < 0.0:
            raise ValueError("USV drag must be nonnegative")

    def position(self, state: np.ndarray) -> np.ndarray:
        return np.array([state[0], state[1], self.surface_z], dtype=float)

    def velocity(self, state: np.ndarray) -> np.ndarray:
        return np.array([state[3] * np.cos(state[2]), state[3] * np.sin(state[2]), 0.0])

    def goal_position(self, goal: np.ndarray) -> np.ndarray:
        return np.array([goal[0], goal[1], self.surface_z], dtype=float)

    def orientation_error(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.array([wrap_angle(goal[2] - state[2])])

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        value = np.asarray(state, dtype=float).copy()
        thrust = np.clip(control[0], -self.max_thrust, self.max_thrust)
        yaw_rate = np.clip(control[1], -self.max_yaw_rate, self.max_yaw_rate)
        value[3] = np.clip(
            value[3] + dt * (thrust - self.drag * value[3]), 0.0, self.max_speed
        )
        value[2] = wrap_angle(value[2] + dt * yaw_rate)
        value[0] += dt * value[3] * np.cos(value[2])
        value[1] += dt * value[3] * np.sin(value[2])
        return value

    def symbolic_step(self, ca, state, control, dt: float):
        speed = state[3] + dt * (control[0] - self.drag * state[3])
        heading = state[2] + dt * control[1]
        return ca.vertcat(
            state[0] + dt * speed * ca.cos(heading),
            state[1] + dt * speed * ca.sin(heading),
            heading,
            speed,
        )

    def symbolic_velocity(self, ca, state):
        return state[3]

    def control_scale(self) -> np.ndarray:
        return np.ones(2)


@dataclass(frozen=True)
class ReducedROVModel(PlatformModel):
    """Untethered damped 6-DOF ROV with world-frame wrench control."""

    max_horizontal_speed: float = 0.8
    max_vertical_speed: float = 0.5
    max_force: float = 1.2
    max_angular_rate: float = 0.7
    max_torque: float = 1.0
    linear_drag: float = 0.3
    angular_drag: float = 0.4
    velocity_response_gain: float = 1.8
    attitude_response_gain: float = 1.0
    angular_rate_response_gain: float = 2.0
    control_smoothing: float = 0.4
    kind: str = field(default="rov", init=False)
    variant: str = field(default="reduced_world_damped", init=False)
    state_dim: int = field(default=12, init=False)
    control_dim: int = field(default=6, init=False)
    pose_dim: int = field(default=6, init=False)

    def __post_init__(self) -> None:
        positive = (
            self.max_horizontal_speed,
            self.max_vertical_speed,
            self.max_force,
            self.max_angular_rate,
            self.max_torque,
            self.velocity_response_gain,
            self.attitude_response_gain,
            self.angular_rate_response_gain,
            self.control_smoothing,
        )
        if min(positive) <= 0.0:
            raise ValueError("ROV speed, force, and yaw-rate bounds must be positive")
        if self.control_smoothing > 1.0:
            raise ValueError("ROV control_smoothing must be in (0, 1]")
        if self.linear_drag < 0.0 or self.angular_drag < 0.0:
            raise ValueError("ROV drag values must be nonnegative")

    def position(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state[:3], dtype=float).copy()

    def velocity(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state[6:9], dtype=float).copy()

    def goal_position(self, goal: np.ndarray) -> np.ndarray:
        return np.asarray(goal[:3], dtype=float).copy()

    def orientation_error(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.asarray([wrap_angle(goal[i] - state[i]) for i in range(3, 6)])

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        value = np.asarray(state, dtype=float).copy()
        forces = np.clip(control[:3], -self.max_force, self.max_force)
        torques = np.clip(control[3:6], -self.max_torque, self.max_torque)
        value[6:9] += dt * (forces - self.linear_drag * value[6:9])
        horizontal_speed = np.linalg.norm(value[6:8])
        if horizontal_speed > self.max_horizontal_speed:
            value[6:8] *= self.max_horizontal_speed / horizontal_speed
        value[8] = np.clip(value[8], -self.max_vertical_speed, self.max_vertical_speed)
        value[9:12] = np.clip(
            value[9:12] + dt * (torques - self.angular_drag * value[9:12]),
            -self.max_angular_rate,
            self.max_angular_rate,
        )
        value[:3] += dt * value[6:9]
        value[3:6] = [wrap_angle(angle) for angle in value[3:6] + dt * value[9:12]]
        return value

    def symbolic_step(self, ca, state, control, dt: float):
        velocity = state[6:9] + dt * (control[:3] - self.linear_drag * state[6:9])
        angular_rate = state[9:12] + dt * (
            control[3:6] - self.angular_drag * state[9:12]
        )
        return ca.vertcat(
            state[:3] + dt * velocity,
            state[3:6] + dt * angular_rate,
            velocity,
            angular_rate,
        )

    def symbolic_velocity(self, ca, state):
        return state[6:9]

    def control_scale(self) -> np.ndarray:
        return np.ones(6)
