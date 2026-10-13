"""Simulation / manta autopilot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scripts.algorithms.manta.apf import compute_apf_control
from scripts.configuration.apf import APFConfig
from scripts.configuration.manta_dynamics import MantaDynamicsConfig
from scripts.contracts.manta import StaticObstacle
from scripts.dynamics.manta import rk4_step_np

if TYPE_CHECKING:
    from collections.abc import Sequence


def simulate_manta_autopilot(
    start_state: np.ndarray,
    goal_state: np.ndarray,
    *,
    dt: float = 0.2,
    obstacle: StaticObstacle = StaticObstacle(center=(3.1, 2.9), radius=0.95),
    extra_obstacles: Sequence[StaticObstacle] | None = None,
    apf_config: APFConfig = APFConfig(),
    dynamics_config: MantaDynamicsConfig = MantaDynamicsConfig(),
) -> np.ndarray:
    """Generate one APF trajectory from ``start_state`` toward ``goal_state``."""
    history, _ = simulate_manta_autopilot_with_controls(
        start_state,
        goal_state,
        dt=dt,
        obstacle=obstacle,
        extra_obstacles=extra_obstacles,
        apf_config=apf_config,
        dynamics_config=dynamics_config,
    )
    return history


def simulate_manta_autopilot_with_controls(
    start_state: np.ndarray,
    goal_state: np.ndarray,
    *,
    dt: float = 0.2,
    obstacle: StaticObstacle = StaticObstacle(center=(3.1, 2.9), radius=0.95),
    extra_obstacles: Sequence[StaticObstacle] | None = None,
    apf_config: APFConfig = APFConfig(),
    dynamics_config: MantaDynamicsConfig = MantaDynamicsConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one APF trajectory and the controls that produced it."""
    history = [np.asarray(start_state, dtype=float).copy()]
    controls: list[np.ndarray] = []
    current_state = history[0].copy()

    for _ in range(apf_config.max_steps):
        control = compute_apf_control(
            current_state=current_state,
            goal_state=goal_state,
            obstacle=obstacle,
            extra_obstacles=extra_obstacles,
            apf_config=apf_config,
            dynamics_config=dynamics_config,
        )

        controls.append(control.copy())
        current_state = rk4_step_np(current_state, control, dt, dynamics_config)
        history.append(current_state.copy())

        if (
            np.linalg.norm(current_state[:2] - goal_state[:2])
            < apf_config.goal_tolerance
        ):
            break

    return np.asarray(history, dtype=float), np.asarray(controls, dtype=float)
