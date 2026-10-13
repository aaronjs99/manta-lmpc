"""Objective terms for the Harbor platform MPC."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scripts.mpc.harbor.symbolic import (
    _symbolic_goal_position,
    _symbolic_orientation_cost,
    _symbolic_position,
    _symbolic_velocity,
)

if TYPE_CHECKING:
    from scripts.configuration.harbor_mpc import HarborMPCConfig
    from scripts.dynamics.harbor.base import PlatformModel


def tracking_cost(
    ca: Any,
    model: PlatformModel,
    config: HarborMPCConfig,
    state: Any,
    goal: Any,
) -> Any:
    """Return position, orientation, and velocity tracking cost for one state."""
    position = _symbolic_position(ca, model, state)
    goal_position = _symbolic_goal_position(ca, model, goal)
    velocity = _symbolic_velocity(model, state)
    return (
        config.position_weight * ca.sumsqr(position - goal_position)
        + config.orientation_weight * _symbolic_orientation_cost(ca, model, state, goal)
        + config.velocity_weight * ca.sumsqr(velocity)
    )


def pose_error_cost(
    ca: Any,
    model: PlatformModel,
    config: HarborMPCConfig,
    pose: Any,
    goal: Any,
) -> Any:
    """Return terminal pose error with wrapped angular penalties."""
    linear_dim = 2 if model.pose_dim == 3 else 3
    value = ca.sumsqr(pose[:linear_dim] - goal[:linear_dim])
    for index in range(linear_dim, model.pose_dim):
        value += (
            config.terminal_orientation_weight
            * 2.0
            * (1.0 - ca.cos(pose[index] - goal[index]))
        )
    return value
