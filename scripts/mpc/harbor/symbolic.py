"""Mpc / harbor / symbolic."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.dynamics.harbor.base import PlatformModel


def _symbolic_step(ca, model: PlatformModel, state, control, dt: float):
    return model.symbolic_step(ca, state, control, dt)


def _apply_symbolic_position_drift(ca, model, state, drift, dt: float):
    """Add the estimated world-velocity residual to predicted position only."""
    dimensions = 3 if model.kind == "rov" else 2
    return ca.vertcat(state[:dimensions] + dt * drift[:dimensions], state[dimensions:])


def _symbolic_position(ca, model: PlatformModel, state):
    if model.kind == "ugv":
        return ca.vertcat(state[0], state[1], model.ground_z)
    if model.kind == "usv":
        return ca.vertcat(state[0], state[1], model.surface_z)
    return state[:3]


def _symbolic_goal_position(ca, model: PlatformModel, goal):
    if model.kind == "ugv":
        return ca.vertcat(goal[0], goal[1], model.ground_z)
    if model.kind == "usv":
        return ca.vertcat(goal[0], goal[1], model.surface_z)
    return goal[:3]


def _symbolic_velocity(model: PlatformModel, state):
    return model.symbolic_velocity(None, state)


def _symbolic_orientation_cost(ca, model: PlatformModel, state, goal):
    if model.pose_dim == 3:
        return 2.0 * (1.0 - ca.cos(state[2] - goal[2]))
    return sum(
        2.0 * (1.0 - ca.cos(state[index] - goal[index])) for index in range(3, 6)
    )
