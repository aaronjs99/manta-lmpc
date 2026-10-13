"""Algorithms / manta / apf."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scripts.configuration.apf import APFConfig
from scripts.configuration.manta_dynamics import MantaDynamicsConfig
from scripts.dynamics.manta import wrap_angle

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scripts.contracts.manta import StaticObstacle


def compute_apf_control(
    current_state: np.ndarray,
    goal_state: np.ndarray,
    *,
    obstacle: StaticObstacle,
    extra_obstacles: Sequence[StaticObstacle] | None = None,
    apf_config: APFConfig = APFConfig(),
    dynamics_config: MantaDynamicsConfig = MantaDynamicsConfig(),
) -> np.ndarray:
    """Return one obstacle-aware APF control for the current manta state."""
    x, y, theta = np.asarray(current_state, dtype=float)[:3]

    goal_delta = np.asarray(goal_state[:2], dtype=float) - np.array([x, y])
    goal_dist = float(np.linalg.norm(goal_delta))
    attraction = goal_delta / (goal_dist + 1e-5)
    base_mu = np.clip(
        apf_config.base_mu_gain * goal_dist,
        apf_config.base_mu_min,
        apf_config.base_mu_max,
    )

    repulsion = np.zeros(2, dtype=float)
    for static_obstacle in (obstacle, *(extra_obstacles or ())):
        obs = np.asarray(static_obstacle.center, dtype=float)
        obs_delta = np.array([x, y]) - obs
        obs_dist = float(np.linalg.norm(obs_delta))
        if obs_dist < apf_config.influence_radius:
            surface_dist = obs_dist - (
                static_obstacle.radius + apf_config.obstacle_padding
            )
            surface_dist = max(surface_dist, 0.05)
            denominator = max(
                apf_config.influence_radius - static_obstacle.radius, 0.05
            )
            strength = apf_config.repulsion_gain * (
                1.0 / surface_dist - 1.0 / denominator
            )
            repulsion += (obs_delta / (obs_dist + 1e-9)) * strength

    desired_velocity = attraction + repulsion
    target_theta = np.arctan2(desired_velocity[1], desired_velocity[0])
    theta_error = float(wrap_angle(target_theta - theta))

    delta_mu = apf_config.heading_gain * theta_error
    return np.array(
        [
            np.clip(base_mu + delta_mu, dynamics_config.mu_min, dynamics_config.mu_max),
            np.clip(base_mu - delta_mu, dynamics_config.mu_min, dynamics_config.mu_max),
        ],
        dtype=float,
    )


backup_apf_control = compute_apf_control
