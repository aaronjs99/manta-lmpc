"""Compact fixed Manta routes through safe delay selection and control replay."""

from __future__ import annotations

import numpy as np

from scripts.algorithms.geometry import swept_pairwise_distances
from scripts.algorithms.manta.safe_sets import hold_controls, hold_trajectory
from scripts.configuration.manta_dynamics import MantaDynamicsConfig
from scripts.dynamics.manta import rk4_step_np


def compact_staged_safe_sets(
    starts: np.ndarray,
    staged_trajs: dict[int, np.ndarray],
    staged_controls: dict[int, np.ndarray],
    safety_distance: float,
    *,
    dt: float,
    dynamics_config: MantaDynamicsConfig,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]] | None:
    """Overlap staged routes using the shortest pairwise-safe start delays.

    The APF route geometry is unchanged. Controls are replayed after each delay,
    which preserves the full nonlinear state evolution instead of shifting only
    the plotted positions.
    """
    agents = sorted(staged_trajs)
    if agents != list(range(len(agents))) or len(agents) not in (2, 3):
        return None
    if set(staged_controls) != set(agents):
        raise ValueError("staged trajectories and controls must share agent keys")

    delays = _find_compact_start_delays(staged_trajs, safety_distance)
    if delays is None:
        return None

    total_steps = max(delays[agent] + len(staged_controls[agent]) for agent in agents)
    safe_sets: dict[int, np.ndarray] = {}
    safe_controls: dict[int, np.ndarray] = {}
    for agent in agents:
        prefix_steps = delays[agent]
        active_controls = np.asarray(staged_controls[agent], dtype=float)
        suffix_steps = total_steps - prefix_steps - len(active_controls)
        prefix = hold_trajectory(
            starts[agent],
            prefix_steps,
            dt=dt,
            dynamics_config=dynamics_config,
        )
        active = replay_controls(
            prefix[-1], active_controls, dt=dt, dynamics_config=dynamics_config
        )
        suffix = hold_trajectory(
            active[-1],
            suffix_steps,
            dt=dt,
            dynamics_config=dynamics_config,
        )
        safe_sets[agent] = np.vstack((prefix[:-1], active, suffix[1:]))
        safe_controls[agent] = np.vstack(
            (
                hold_controls(prefix_steps),
                active_controls,
                hold_controls(suffix_steps),
            )
        )
    return safe_sets, safe_controls


def _find_compact_start_delays(
    staged_trajs: dict[int, np.ndarray], safety_distance: float
) -> dict[int, int] | None:
    """Return minimum-makespan start delays for two or three fixed routes."""
    agents = sorted(staged_trajs)
    if agents != list(range(len(agents))) or len(agents) not in (2, 3):
        return None
    durations = {agent: len(staged_trajs[agent]) - 1 for agent in agents}
    max_delay = sum(durations.values())
    relative_safe = {
        (first, second): _safe_relative_delays(
            staged_trajs[first][:, :2],
            staged_trajs[second][:, :2],
            max_delay,
            safety_distance,
        )
        for first in agents
        for second in agents
        if first < second
    }

    best: tuple[tuple[int, int, tuple[int, ...]], dict[int, int]] | None = None
    for anchor in agents:
        others = [agent for agent in agents if agent != anchor]
        delay_ranges = [range(max_delay + 1) for _ in others]
        if len(others) == 1:
            delay_pairs = ((delay,) for delay in delay_ranges[0])
        else:
            delay_pairs = (
                (first_delay, second_delay)
                for first_delay in delay_ranges[0]
                for second_delay in delay_ranges[1]
            )
        for values in delay_pairs:
            delays = {anchor: 0, **dict(zip(others, values, strict=True))}
            if not all(
                relative_safe[(first, second)][
                    delays[second] - delays[first] + max_delay
                ]
                for first in agents
                for second in agents
                if first < second
            ):
                continue
            makespan = max(delays[a] + durations[a] for a in agents)
            key = (
                makespan,
                sum(delays.values()),
                tuple(delays[a] for a in agents),
            )
            if best is None or key < best[0]:
                best = (key, delays)
    return None if best is None else best[1]


def _safe_relative_delays(
    first: np.ndarray,
    second: np.ndarray,
    max_delay: int,
    safety_distance: float,
) -> np.ndarray:
    """Return safety flags indexed by relative delay plus ``max_delay``."""
    flags = np.zeros(2 * max_delay + 1, dtype=bool)
    first_steps = len(first) - 1
    second_steps = len(second) - 1
    for relative_delay in range(-max_delay, max_delay + 1):
        first_delay = max(0, -relative_delay)
        second_delay = max(0, relative_delay)
        total_steps = max(first_delay + first_steps, second_delay + second_steps)
        pair = np.stack(
            (
                _shift_positions(first, first_delay, total_steps),
                _shift_positions(second, second_delay, total_steps),
            ),
            axis=1,
        )
        flags[relative_delay + max_delay] = bool(
            np.min(swept_pairwise_distances(pair)) >= safety_distance
        )
    return flags


def _shift_positions(positions: np.ndarray, delay: int, total_steps: int) -> np.ndarray:
    """Delay a fixed path and hold its endpoints to ``total_steps``."""
    values = np.asarray(positions, dtype=float)
    steps = len(values) - 1
    shifted = np.empty((total_steps + 1, 2), dtype=float)
    shifted[: delay + 1] = values[0]
    shifted[delay : delay + steps + 1] = values
    shifted[delay + steps :] = values[-1]
    return shifted


def replay_controls(
    start: np.ndarray,
    controls: np.ndarray,
    *,
    dt: float,
    dynamics_config: MantaDynamicsConfig,
) -> np.ndarray:
    """Integrate a fixed control sequence from a supplied full state."""
    history = [np.asarray(start, dtype=float).copy()]
    for control in np.asarray(controls, dtype=float):
        history.append(rk4_step_np(history[-1], control, dt, dynamics_config))
    return np.asarray(history, dtype=float)
