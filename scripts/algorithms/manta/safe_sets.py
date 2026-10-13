"""Pure Manta safe-set helpers."""

from __future__ import annotations

import numpy as np
from scripts.configuration.manta_dynamics import MantaDynamicsConfig
from scripts.dynamics.manta import rk4_step_np


def hold_trajectory(
    state: np.ndarray,
    steps: int,
    *,
    dt: float,
    dynamics_config: MantaDynamicsConfig,
) -> np.ndarray:
    """Return a dynamically valid zero-control hold trajectory."""
    traj = [np.asarray(state, dtype=float).copy()]
    current = traj[0].copy()
    zero_u = np.zeros(2, dtype=float)

    for _ in range(steps):
        current = rk4_step_np(current, zero_u, dt, dynamics_config)
        traj.append(current.copy())

    return np.asarray(traj, dtype=float)


def hold_controls(steps: int) -> np.ndarray:
    """Return zero controls for a dynamic hold trajectory."""
    return np.zeros((steps, 2), dtype=float)


def sample_terminal_safe_set(
    safe_set: list[np.ndarray] | np.ndarray,
    target_index: int,
    k_hull: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample future terminal safe states and time-to-go costs for the LMPC hull."""
    states = np.asarray(safe_set, dtype=float)
    safe_states = np.zeros((7, k_hull), dtype=float)
    safe_costs = np.zeros((k_hull, 1), dtype=float)
    traj_len = len(states)

    for k_idx in range(k_hull):
        state_idx = min(target_index + k_idx, traj_len - 1)
        safe_states[:, k_idx] = states[state_idx]
        safe_costs[k_idx, 0] = (traj_len - 1) - state_idx

    return safe_states, safe_costs
