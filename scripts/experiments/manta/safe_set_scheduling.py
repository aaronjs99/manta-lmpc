"""Stagger APF rollout schedules and select a validated Manta safe seed."""

from __future__ import annotations

from itertools import permutations
from typing import TYPE_CHECKING

import numpy as np

from scripts.algorithms.manta.safe_sets import hold_controls, hold_trajectory
from scripts.configuration.apf import APFConfig
from scripts.configuration.manta_dynamics import MantaDynamicsConfig
from scripts.contracts.manta import StaticObstacle
from scripts.experiments.manta.safe_set_compaction import compact_staged_safe_sets
from scripts.experiments.manta.safe_set_scoring import (
    rank_safe_set_candidates,
    score_safe_sets,
    select_safe_set_candidate,
)
from scripts.simulation.manta_autopilot import simulate_manta_autopilot_with_controls

if TYPE_CHECKING:
    from scripts.contracts.manta import Scenario


def build_staggered_safe_sets(
    scenario: Scenario,
    *,
    dt: float,
    apf_config: APFConfig = APFConfig(),
    dynamics_config: MantaDynamicsConfig = MantaDynamicsConfig(),
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Build staged APF safe sets and their corresponding controls."""
    starts = np.asarray(scenario.starts, dtype=float)
    goals = np.asarray(scenario.goals, dtype=float)
    num_agents = len(starts)
    if goals.shape[0] != num_agents:
        raise ValueError("scenario starts and goals must have the same agent count")
    if num_agents < 2:
        raise ValueError("manta LMPC requires at least two agents")

    candidates = []
    for radius_scale in apf_config.static_agent_radius_scales:
        for order in permutations(range(num_agents)):
            (
                safe_sets,
                safe_controls,
                staged_trajs,
                staged_controls,
            ) = _build_ordered_safe_sets(
                scenario,
                starts,
                goals,
                order,
                radius_scale,
                dt=dt,
                apf_config=apf_config,
                dynamics_config=dynamics_config,
            )
            score = score_safe_sets(
                safe_sets,
                goals,
                scenario.safety_distance,
                scenario.obstacle.center,
                scenario.obstacle.radius,
                apf_config.goal_tolerance,
            )
            candidates.append(
                (score, safe_sets, safe_controls, staged_trajs, staged_controls)
            )

    candidates = rank_safe_set_candidates(candidates)
    if (
        apf_config.compact_staging
        and num_agents <= apf_config.compact_staging_max_agents
    ):
        compact_candidates = []
        for candidate in candidates[: apf_config.compact_staging_candidates]:
            score, _, _, staged_trajs, staged_controls = candidate
            if not score.usable_for_learning:
                continue
            compact = compact_staged_safe_sets(
                starts,
                staged_trajs,
                staged_controls,
                scenario.safety_distance,
                dt=dt,
                dynamics_config=dynamics_config,
            )
            if compact is None:
                continue
            compact_sets, compact_controls = compact
            compact_score = score_safe_sets(
                compact_sets,
                goals,
                scenario.safety_distance,
                scenario.obstacle.center,
                scenario.obstacle.radius,
                apf_config.goal_tolerance,
            )
            compact_candidates.append(
                (
                    compact_score,
                    compact_sets,
                    compact_controls,
                    staged_trajs,
                    staged_controls,
                )
            )
        candidates.extend(compact_candidates)

    _, safe_sets, safe_controls, _, _ = select_safe_set_candidate(candidates)
    return safe_sets, safe_controls


def _build_ordered_safe_sets(
    scenario: Scenario,
    starts: np.ndarray,
    goals: np.ndarray,
    order: tuple[int, ...],
    static_agent_radius_scale: float,
    *,
    dt: float,
    apf_config: APFConfig,
    dynamics_config: MantaDynamicsConfig,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    """Generate staged APF paths while treating waiting agents as obstacles."""
    staged_trajs: dict[int, np.ndarray] = {}
    staged_controls: dict[int, np.ndarray] = {}
    start_steps: dict[int, int] = {}
    elapsed_steps = 0

    for order_index, agent in enumerate(order):
        start_steps[agent] = elapsed_steps
        active_start = hold_trajectory(
            starts[agent],
            elapsed_steps,
            dt=dt,
            dynamics_config=dynamics_config,
        )[-1]

        extra_obstacles: list[StaticObstacle] = []
        for other in order[:order_index]:
            extra_obstacles.append(
                StaticObstacle(
                    center=tuple(staged_trajs[other][-1, :2]),
                    radius=scenario.safety_distance * static_agent_radius_scale,
                )
            )
        for other in order[order_index + 1 :]:
            waiting_state = hold_trajectory(
                starts[other],
                elapsed_steps,
                dt=dt,
                dynamics_config=dynamics_config,
            )[-1]
            extra_obstacles.append(
                StaticObstacle(
                    center=tuple(waiting_state[:2]),
                    radius=scenario.safety_distance * static_agent_radius_scale,
                )
            )

        (
            staged_trajs[agent],
            staged_controls[agent],
        ) = simulate_manta_autopilot_with_controls(
            active_start,
            goals[agent],
            dt=dt,
            obstacle=scenario.obstacle,
            extra_obstacles=extra_obstacles,
            apf_config=apf_config,
            dynamics_config=dynamics_config,
        )
        elapsed_steps += len(staged_trajs[agent]) - 1

    total_steps = elapsed_steps
    safe_sets: dict[int, np.ndarray] = {}
    safe_controls: dict[int, np.ndarray] = {}
    for agent in order:
        prefix_steps = start_steps[agent]
        solo = staged_trajs[agent]
        solo_steps = len(solo) - 1
        suffix_steps = total_steps - prefix_steps - solo_steps

        prefix = hold_trajectory(
            starts[agent],
            prefix_steps,
            dt=dt,
            dynamics_config=dynamics_config,
        )
        suffix = hold_trajectory(
            solo[-1],
            suffix_steps,
            dt=dt,
            dynamics_config=dynamics_config,
        )
        safe_sets[agent] = np.vstack((prefix[:-1], solo, suffix[1:]))
        safe_controls[agent] = np.vstack(
            (
                hold_controls(prefix_steps),
                staged_controls[agent],
                hold_controls(suffix_steps),
            )
        )

    return safe_sets, safe_controls, staged_trajs, staged_controls
