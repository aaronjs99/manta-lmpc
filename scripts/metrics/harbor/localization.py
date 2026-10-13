"""Metrics / harbor / localization."""

from __future__ import annotations

import numpy as np


def _position_estimates(result, agents) -> dict[str, np.ndarray]:
    return {
        agent.name: np.asarray(
            [
                agent.model.position(state)
                for state in result.observed_states[agent.name]
            ]
        )
        for agent in agents
    }


def build_localization_record(result, agents, localization, label: str):
    """Score a localization rollout without rendering or writing artifacts."""
    estimates = _position_estimates(result, agents)
    error_histories = {
        agent.name: np.linalg.norm(
            estimates[agent.name] - result.positions[agent.name], axis=1
        )
        for agent in agents
    }
    record = {
        "mode": label,
        "all_goals_reached": result.all_goals_reached,
        "pairwise_violation_count": result.pairwise_violation_count,
        "position_rmse": {
            agent.name: float(np.sqrt(np.mean(error_histories[agent.name] ** 2)))
            for agent in agents
        },
        "final_observability": {
            name: {
                "rank": reports[-1].rank,
                "state_dimension": reports[-1].state_dimension,
                "smallest_singular_value": reports[-1].smallest_singular_value,
                "condition_number": reports[-1].condition_number,
                "observable": reports[-1].observable,
            }
            for name, reports in localization.reports.items()
        },
        "observable_fraction": {
            name: float(np.mean([report.observable for report in reports]))
            for name, reports in localization.reports.items()
        },
    }
    return record, error_histories
