"""Dynamics / harbor / underwater."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scripts.dynamics.harbor.base import PlatformModel, wrap_angle
from scripts.dynamics.harbor.mechanics import (
    _damping_product,
    _euler_rate_matrix,
    _freeze_vectors,
    _marine_coriolis_product,
    _restoring_vector,
    _rotation_matrix,
)
from scripts.dynamics.harbor.symbolic import (
    _symbolic_damping_product,
    _symbolic_euler_rate_matrix,
    _symbolic_marine_coriolis_product,
    _symbolic_restoring_vector,
    _symbolic_rotation_matrix,
)


@dataclass(frozen=True)
class ROVModel(PlatformModel):
    """Body-frame 6-DOF marine craft with eight-thruster allocation."""

    max_horizontal_speed: float = 0.75
    mission_speed: float | None = None
    max_vertical_speed: float = 0.4
    max_angular_rate: float = 0.7
    max_force: float = 30.0
    max_torque: float = 8.0
    force_limits: tuple[float, float, float] | None = None
    torque_limits: tuple[float, float, float] | None = None
    thruster_limits: tuple[float, ...] | None = None
    thruster_allocation: tuple[tuple[float, ...], ...] | None = None
    mass_diagonal: tuple[float, ...] = (18.0, 18.0, 22.0, 1.2, 1.4, 1.5)
    linear_damping: tuple[float, ...] = (8.0, 10.0, 12.0, 1.2, 1.4, 1.5)
    quadratic_damping: tuple[float, ...] = (12.0, 16.0, 18.0, 0.6, 0.8, 0.8)
    weight: float = 176.58
    buoyancy: float = 176.58
    center_of_gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    center_of_buoyancy: tuple[float, float, float] = (0.0, 0.0, 0.03)
    velocity_response_gain: float = 1.8
    attitude_response_gain: float = 1.0
    angular_rate_response_gain: float = 2.0
    control_smoothing: float = 0.6
    kind: str = field(default="rov", init=False)
    variant: str = field(default="marine_6dof", init=False)
    state_dim: int = field(default=12, init=False)
    control_dim: int = field(default=8, init=False)
    pose_dim: int = field(default=6, init=False)

    def __post_init__(self) -> None:
        _freeze_vectors(
            self,
            "mass_diagonal",
            "linear_damping",
            "quadratic_damping",
            "center_of_gravity",
            "center_of_buoyancy",
        )
        if self.force_limits is not None:
            _freeze_vectors(self, "force_limits")
        if self.torque_limits is not None:
            _freeze_vectors(self, "torque_limits")
        if self.thruster_limits is not None:
            _freeze_vectors(self, "thruster_limits")
        if self.thruster_allocation is not None:
            object.__setattr__(
                self,
                "thruster_allocation",
                tuple(
                    tuple(float(value) for value in row)
                    for row in self.thruster_allocation
                ),
            )
        if not (
            len(self.mass_diagonal)
            == len(self.linear_damping)
            == len(self.quadratic_damping)
            == 6
        ):
            raise ValueError("ROV mass and damping vectors must have six entries")
        if len(self.center_of_gravity) != 3 or len(self.center_of_buoyancy) != 3:
            raise ValueError("ROV mass and buoyancy centers must have three entries")
        if len(self.force_limit_vector) != 3 or len(self.torque_limit_vector) != 3:
            raise ValueError("ROV force and torque limits must have three entries")
        if len(self.thruster_limit_vector) != 8:
            raise ValueError("ROV thruster_limits must have eight entries")
        if self.allocation_matrix.shape != (6, self.control_dim):
            raise ValueError("ROV thruster_allocation must have shape 6 x 8")
        if np.linalg.matrix_rank(self.allocation_matrix) != 6:
            raise ValueError("ROV thruster_allocation must have full row rank")
        positive = (
            self.max_horizontal_speed,
            self.max_vertical_speed,
            self.max_angular_rate,
            self.max_force,
            self.max_torque,
            self.weight,
            self.buoyancy,
            self.velocity_response_gain,
            self.attitude_response_gain,
            self.angular_rate_response_gain,
            self.control_smoothing,
            *self.mass_diagonal,
            *self.force_limit_vector,
            *self.torque_limit_vector,
            *self.thruster_limit_vector,
        )
        if min(positive) <= 0.0 or self.control_smoothing > 1.0:
            raise ValueError("ROV marine parameters must be positive and bounded")
        if (
            self.mission_speed is not None
            and not 0.0 < self.mission_speed <= self.max_horizontal_speed
        ):
            raise ValueError("ROV mission_speed must be in (0, max_horizontal_speed]")
        if min(self.linear_damping) < 0.0 or min(self.quadratic_damping) < 0.0:
            raise ValueError("ROV damping must be nonnegative")

    @property
    def force_limit_vector(self) -> np.ndarray:
        """Return axis-specific body-force limits, or the scalar fallback."""
        return np.asarray(
            self.force_limits
            if self.force_limits is not None
            else (self.max_force,) * 3,
            dtype=float,
        )

    @property
    def torque_limit_vector(self) -> np.ndarray:
        """Return axis-specific body-moment limits, or the scalar fallback."""
        return np.asarray(
            self.torque_limits
            if self.torque_limits is not None
            else (self.max_torque,) * 3,
            dtype=float,
        )

    @property
    def thruster_limit_vector(self) -> np.ndarray:
        """Return the eight symmetric physical thruster-force bounds."""
        if self.thruster_limits is not None:
            return np.asarray(self.thruster_limits, dtype=float)
        horizontal = self.force_limit_vector[0] / (2.0 * np.sqrt(2.0))
        vertical = self.force_limit_vector[2] / 4.0
        return np.array([horizontal] * 4 + [vertical] * 4)

    @property
    def allocation_matrix(self) -> np.ndarray:
        """Return the calibrated BlueROV2 Heavy 6x8 thruster allocation."""
        if self.thruster_allocation is not None:
            return np.asarray(self.thruster_allocation, dtype=float)
        horizontal_total = float(np.sum(self.thruster_limit_vector[:4]))
        vertical_total = float(np.sum(self.thruster_limit_vector[4:]))
        x_gain = self.force_limit_vector[0] / horizontal_total
        y_gain = self.force_limit_vector[1] / horizontal_total
        z_gain = self.force_limit_vector[2] / vertical_total
        roll_gain = self.torque_limit_vector[0] / vertical_total
        pitch_gain = self.torque_limit_vector[1] / vertical_total
        yaw_gain = self.torque_limit_vector[2] / horizontal_total
        return np.array(
            [
                [-x_gain, -x_gain, x_gain, x_gain, 0.0, 0.0, 0.0, 0.0],
                [y_gain, -y_gain, y_gain, -y_gain, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, z_gain, z_gain, z_gain, z_gain],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    roll_gain,
                    -roll_gain,
                    roll_gain,
                    -roll_gain,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -pitch_gain,
                    -pitch_gain,
                    pitch_gain,
                    pitch_gain,
                ],
                [yaw_gain, -yaw_gain, -yaw_gain, yaw_gain, 0.0, 0.0, 0.0, 0.0],
            ]
        )

    def generalized_wrench(self, control: np.ndarray) -> np.ndarray:
        """Map eight bounded T200 forces to body force and moment."""
        bounded = np.clip(
            np.asarray(control, dtype=float),
            -self.thruster_limit_vector,
            self.thruster_limit_vector,
        )
        return self.allocation_matrix @ bounded

    def allocate_wrench(self, wrench: np.ndarray) -> np.ndarray:
        """Allocate a desired body wrench and uniformly desaturate thrusters."""
        command = np.linalg.pinv(self.allocation_matrix) @ np.asarray(
            wrench, dtype=float
        )
        ratio = float(np.max(np.abs(command) / self.thruster_limit_vector, initial=0.0))
        if ratio > 1.0:
            command /= ratio
        return command

    def position(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state[:3], dtype=float).copy()

    def velocity(self, state: np.ndarray) -> np.ndarray:
        return _rotation_matrix(state[3:6]) @ np.asarray(state[6:9], dtype=float)

    def goal_position(self, goal: np.ndarray) -> np.ndarray:
        return np.asarray(goal[:3], dtype=float).copy()

    def orientation_error(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.asarray([wrap_angle(goal[i] - state[i]) for i in range(3, 6)])

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        value = np.asarray(state, dtype=float).copy()
        tau = self.generalized_wrench(control)
        nu = value[6:12]
        rhs = (
            tau
            - _marine_coriolis_product(nu, self.mass_diagonal)
            - _damping_product(nu, self.linear_damping, self.quadratic_damping)
            - _restoring_vector(
                value[3:6],
                self.weight,
                self.buoyancy,
                self.center_of_gravity,
                self.center_of_buoyancy,
            )
        )
        nu = nu + dt * rhs / np.asarray(self.mass_diagonal)
        horizontal_speed = np.linalg.norm(nu[:2])
        if horizontal_speed > self.max_horizontal_speed:
            nu[:2] *= self.max_horizontal_speed / horizontal_speed
        nu[2] = np.clip(nu[2], -self.max_vertical_speed, self.max_vertical_speed)
        nu[3:] = np.clip(nu[3:], -self.max_angular_rate, self.max_angular_rate)
        value[:3] += dt * (_rotation_matrix(value[3:6]) @ nu[:3])
        value[3:6] = value[3:6] + dt * (_euler_rate_matrix(value[3:6]) @ nu[3:])
        value[6:12] = nu
        return value

    def symbolic_step(self, ca, state, control, dt: float):
        nu = state[6:12]
        mass = ca.DM(self.mass_diagonal)
        coriolis = _symbolic_marine_coriolis_product(ca, nu, mass)
        damping = _symbolic_damping_product(
            ca, nu, self.linear_damping, self.quadratic_damping
        )
        restoring = _symbolic_restoring_vector(ca, state[3:6], self)
        tau = ca.DM(self.allocation_matrix) @ control
        next_nu = nu + dt * ca.rdivide(tau - coriolis - damping - restoring, mass)
        rotation = _symbolic_rotation_matrix(ca, state[3:6])
        euler_rates = _symbolic_euler_rate_matrix(ca, state[3:6]) @ next_nu[3:]
        return ca.vertcat(
            state[:3] + dt * (rotation @ next_nu[:3]),
            state[3:6] + dt * euler_rates,
            next_nu,
        )

    def symbolic_velocity(self, ca, state):
        return state[6:12]

    def control_scale(self) -> np.ndarray:
        return self.thruster_limit_vector
