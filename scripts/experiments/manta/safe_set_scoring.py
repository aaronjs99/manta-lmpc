"""Pure candidate scoring and ordering for Manta safe-set seeds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.metrics.evaluation import history_to_tensor, validate_trajectory


@dataclass(frozen=True)
class SafeSetScore:
    """Validation metrics used to choose an APF staging order."""

    all_goals_reached: bool
    usable_for_learning: bool
    pairwise_violation_count: int
    obstacle_violation_count: int
    total_first_hit_steps: int
    final_goal_error: float
    min_pairwise_distance: float
    min_obstacle_clearance: float


def score_safe_sets(
    safe_sets: dict[int, np.ndarray],
    goals: np.ndarray,
    safety_distance: float,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
    goal_tolerance: float,
) -> SafeSetScore:
    """Score a staged APF candidate with the shared admission validator."""
    agents, states = history_to_tensor(safe_sets)
    pos = states[:, :, :2]
    goal_pos = np.asarray(goals, dtype=float)[agents, :2]
    goal_errors = np.linalg.norm(pos - goal_pos[None, :, :], axis=2)
    final_goal_errors = goal_errors[-1]
    first_hit_steps = []
    for agent_idx in range(len(agents)):
        reached = np.where(goal_errors[:, agent_idx] <= goal_tolerance)[0]
        first_hit_steps.append(int(reached[0]) if len(reached) else len(states))

    validation = validate_trajectory(
        safe_sets,
        goals,
        safety_distance,
        obstacle_center,
        obstacle_radius,
        goal_tolerance,
        statuses=None,
    )

    return SafeSetScore(
        all_goals_reached=validation.all_goals_reached,
        usable_for_learning=validation.usable_for_learning,
        pairwise_violation_count=validation.pairwise_violation_count,
        obstacle_violation_count=validation.obstacle_violation_count,
        total_first_hit_steps=sum(first_hit_steps),
        final_goal_error=float(np.sum(final_goal_errors)),
        min_pairwise_distance=validation.min_pairwise_distance,
        min_obstacle_clearance=validation.min_obstacle_clearance,
    )


def safe_set_score_key(
    score: SafeSetScore,
) -> tuple[bool, bool, int, int, int, float, float, float]:
    """Prefer safe, complete, short-error APF seeds with larger margins."""
    return (
        not score.usable_for_learning,
        not score.all_goals_reached,
        score.pairwise_violation_count,
        score.obstacle_violation_count,
        score.total_first_hit_steps,
        score.final_goal_error,
        -score.min_pairwise_distance,
        -score.min_obstacle_clearance,
    )


def rank_safe_set_candidates(candidates: list[tuple]) -> list[tuple]:
    """Return candidates in the historical stable lexicographic score order."""
    return sorted(candidates, key=lambda candidate: safe_set_score_key(candidate[0]))


def select_safe_set_candidate(candidates: list[tuple]) -> tuple:
    """Select the best candidate using the same score key as candidate ranking."""
    return min(candidates, key=lambda candidate: safe_set_score_key(candidate[0]))
