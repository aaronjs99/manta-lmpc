"""Metrics / harbor / joint localization study."""

from __future__ import annotations

import numpy as np
from scripts.configuration.study_labels import POLICIES


def _position_rmse(result, agents) -> dict[str, float]:
    values = {}
    for agent in agents:
        estimates = np.asarray(
            [
                agent.model.position(state)
                for state in result.observed_states[agent.name]
            ]
        )
        error = estimates - result.positions[agent.name]
        values[agent.name] = float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
    return values


def _current_rmse(controller, agents, disturbance) -> dict[str, float]:
    values = {}
    for agent in agents:
        if agent.model.kind == "ugv":
            continue
        history = np.asarray(
            controller.telemetry.estimator.residual_history[agent.name], dtype=float
        )
        truth = disturbance.current(agent.model)
        values[agent.name] = float(
            np.sqrt(np.mean(np.sum((history - truth) ** 2, axis=1)))
        )
    return values


def _observability(localization, agents) -> dict[str, dict] | None:
    if localization is None:
        return None
    return {
        agent.name: {
            "observable_fraction": float(
                np.mean(
                    [report.observable for report in localization.reports[agent.name]]
                )
            ),
            "final_rank": localization.reports[agent.name][-1].rank,
            "final_state_dimension": localization.reports[agent.name][
                -1
            ].state_dimension,
            "pose_observable": localization.estimators[agent.name].pose_observable(),
            "deferred_landmark_updates": getattr(
                localization.estimators[agent.name], "deferred_landmark_updates", 0
            ),
            "landmark_only_updates": getattr(
                localization.estimators[agent.name], "landmark_only_updates", 0
            ),
        }
        for agent in agents
    }


def summarize(records: list[dict]) -> dict:
    """Aggregate task, safety, solver, and estimation evidence by policy."""
    summary = {}
    for policy in POLICIES:
        selected = [record for record in records if record["policy"] == policy]
        if not selected:
            continue
        summary[policy] = {
            "trials": len(selected),
            "completion_rate": float(
                np.mean([record["all_goals_reached"] for record in selected])
            ),
            "safety_rate": float(
                np.mean(
                    [record["pairwise_violation_count"] == 0 for record in selected]
                )
            ),
            "fallback_free_rate": float(
                np.mean([record["solver_fallbacks"] == 0 for record in selected])
            ),
            "total_solver_fallbacks": int(
                sum(record["solver_fallbacks"] for record in selected)
            ),
            "mean_completion_step_sum": float(
                np.mean([record["completion_step_sum"] for record in selected])
            ),
            "mean_bluerov2_position_rmse": float(
                np.mean(
                    [record["position_rmse"]["underwater_rov"] for record in selected]
                )
            ),
            "mean_heron_current_rmse": float(
                np.mean(
                    [record["current_rmse"]["surface_vessel"] for record in selected]
                )
            ),
            "total_dynamic_state_retries": int(
                sum(record["dynamic_state_retries"] for record in selected)
            ),
            "maximum_dynamic_state_slack": float(
                max(record["max_dynamic_state_slack"] for record in selected)
            ),
        }
    return summary
