"""Dynamics / harbor / surface."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scripts.dynamics.harbor.base import PlatformModel, wrap_angle
from scripts.dynamics.harbor.mechanics import (
    _damping_product,
    _freeze_vectors,
    _marine_coriolis_product,
)
from scripts.dynamics.harbor.symbolic import (
    _symbolic_damping_product,
    _symbolic_marine_coriolis_product,
)


@dataclass(frozen=True)
class USVModel(PlatformModel):
    """Underactuated 3-DOF marine model with twin waterjet inputs."""

    max_speed: float = 0.9
    mission_speed: float | None = None
    max_sway_speed: float = 0.45
    max_yaw_rate: float = 0.8
    max_thrust: float = 40.0
    max_yaw_moment: float = 12.0
    waterjet_separation: float = 0.74
    mass_diagonal: tuple[float, float, float] = (35.0, 45.0, 8.0)
    linear_damping: tuple[float, float, float] = (8.0, 12.0, 4.0)
    quadratic_damping: tuple[float, float, float] = (4.0, 8.0, 2.0)
    velocity_response_gain: float = 1.6
    heading_gain: float = 1.5
    yaw_response_gain: float = 2.0
    surface_z: float = 0.0
    kind: str = field(default="usv", init=False)
    variant: str = field(default="marine_3dof", init=False)
    state_dim: int = field(default=6, init=False)
    control_dim: int = field(default=2, init=False)
    pose_dim: int = field(default=3, init=False)

    def __post_init__(self) -> None:
        _freeze_vectors(self, "mass_diagonal", "linear_damping", "quadratic_damping")
        positive = (
            self.max_speed,
            self.max_sway_speed,
            self.max_yaw_rate,
            self.max_thrust,
            self.max_yaw_moment,
            self.waterjet_separation,
            self.velocity_response_gain,
            self.heading_gain,
            self.yaw_response_gain,
            *self.mass_diagonal,
        )
        if min(positive) <= 0.0:
            raise ValueError("USV marine parameters must be positive")
        if (
            self.mission_speed is not None
            and not 0.0 < self.mission_speed <= self.max_speed
        ):
            raise ValueError("USV mission_speed must be in (0, max_speed]")
        if min(self.linear_damping) < 0.0 or min(self.quadratic_damping) < 0.0:
            raise ValueError("USV damping must be nonnegative")
        if self.max_yaw_moment > self.waterjet_separation * self.max_jet_thrust + 1e-12:
            raise ValueError("USV max_yaw_moment exceeds twin-waterjet authority")

    @property
    def max_jet_thrust(self) -> float:
        """Return the symmetric thrust bound for either waterjet."""
        return self.max_thrust / 2.0

    def generalized_wrench(self, control: np.ndarray) -> np.ndarray:
        """Map port/starboard waterjet thrust to surge force and yaw moment."""
        port, starboard = np.clip(
            np.asarray(control, dtype=float),
            -self.max_jet_thrust,
            self.max_jet_thrust,
        )
        return np.array(
            [port + starboard, 0.5 * self.waterjet_separation * (starboard - port)]
        )

    def position(self, state: np.ndarray) -> np.ndarray:
        return np.array([state[0], state[1], self.surface_z], dtype=float)

    def velocity(self, state: np.ndarray) -> np.ndarray:
        yaw = state[2]
        return np.array(
            [
                np.cos(yaw) * state[3] - np.sin(yaw) * state[4],
                np.sin(yaw) * state[3] + np.cos(yaw) * state[4],
                0.0,
            ]
        )

    def goal_position(self, goal: np.ndarray) -> np.ndarray:
        return np.array([goal[0], goal[1], self.surface_z], dtype=float)

    def orientation_error(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.array([wrap_angle(goal[2] - state[2])])

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        value = np.asarray(state, dtype=float).copy()
        nu = value[3:6]
        thrust, moment = self.generalized_wrench(control)
        tau = np.array([thrust, 0.0, moment])
        rhs = (
            tau
            - _marine_coriolis_product(nu, self.mass_diagonal)
            - _damping_product(nu, self.linear_damping, self.quadratic_damping)
        )
        nu = nu + dt * rhs / np.asarray(self.mass_diagonal)
        nu[0] = np.clip(nu[0], 0.0, self.max_speed)
        nu[1] = np.clip(nu[1], -self.max_sway_speed, self.max_sway_speed)
        nu[2] = np.clip(nu[2], -self.max_yaw_rate, self.max_yaw_rate)
        yaw = value[2]
        value[0] += dt * (np.cos(yaw) * nu[0] - np.sin(yaw) * nu[1])
        value[1] += dt * (np.sin(yaw) * nu[0] + np.cos(yaw) * nu[1])
        value[2] = yaw + dt * nu[2]
        value[3:6] = nu
        return value

    def symbolic_step(self, ca, state, control, dt: float):
        nu = state[3:6]
        mass = ca.DM(self.mass_diagonal)
        coriolis = _symbolic_marine_coriolis_product(ca, nu, mass)
        damping = _symbolic_damping_product(
            ca, nu, self.linear_damping, self.quadratic_damping
        )
        thrust = control[0] + control[1]
        moment = 0.5 * self.waterjet_separation * (control[1] - control[0])
        tau = ca.vertcat(thrust, 0.0, moment)
        next_nu = nu + dt * ca.rdivide(tau - coriolis - damping, mass)
        yaw = state[2]
        return ca.vertcat(
            state[0] + dt * (ca.cos(yaw) * next_nu[0] - ca.sin(yaw) * next_nu[1]),
            state[1] + dt * (ca.sin(yaw) * next_nu[0] + ca.cos(yaw) * next_nu[1]),
            yaw + dt * next_nu[2],
            next_nu,
        )

    def symbolic_velocity(self, ca, state):
        return state[3:6]

    def control_scale(self) -> np.ndarray:
        return np.full(2, self.max_jet_thrust)
