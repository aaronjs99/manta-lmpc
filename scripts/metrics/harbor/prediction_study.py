"""Metrics / harbor / prediction study."""

from __future__ import annotations

import numpy as np


def summarize_prediction_ablation(records: list[dict]) -> dict:
    """Aggregate matched prediction-mode feasibility and task outcomes."""
    labels = ("Constant-velocity prediction", "Goal-bounded prediction")
    controllers = {}
    for label in labels:
        selected = [record for record in records if record["controller"] == label]
        controllers[label] = {
            "trials": len(selected),
            "total_solver_fallbacks": int(
                sum(record["solver_fallbacks"] for record in selected)
            ),
            "fallback_free_rate": float(
                np.mean([record["solver_fallbacks"] == 0 for record in selected])
            ),
            "completion_rate": float(
                np.mean([record["all_goals_reached"] for record in selected])
            ),
            "safety_rate": float(
                np.mean(
                    [
                        record["pairwise_violation_count"] == 0
                        and record.get("max_collision_slack", 0.0) <= 1e-9
                        for record in selected
                    ]
                )
            ),
            "mean_sustained_completion_cost": float(
                np.mean([record["sustained_completion_cost"] for record in selected])
            ),
            "mean_effectiveness_rmse": float(
                np.mean([record["effectiveness_rmse"] for record in selected])
            ),
        }
    by_key = {(record["seed"], record["controller"]): record for record in records}
    seeds = sorted({record["seed"] for record in records})
    legacy = [by_key[(seed, labels[0])] for seed in seeds]
    bounded = [by_key[(seed, labels[1])] for seed in seeds]
    return {
        "controllers": controllers,
        "paired_goal_bounded_vs_constant_velocity": {
            "trials": len(seeds),
            "constant_velocity_fallbacks": int(
                sum(record["solver_fallbacks"] for record in legacy)
            ),
            "goal_bounded_fallbacks": int(
                sum(record["solver_fallbacks"] for record in bounded)
            ),
            "fallback_reduction": int(
                sum(record["solver_fallbacks"] for record in legacy)
                - sum(record["solver_fallbacks"] for record in bounded)
            ),
            "complete_safe_pairs": int(
                sum(
                    left["all_goals_reached"]
                    and right["all_goals_reached"]
                    and left["pairwise_violation_count"] == 0
                    and right["pairwise_violation_count"] == 0
                    and left.get("max_collision_slack", 0.0) <= 1e-9
                    and right.get("max_collision_slack", 0.0) <= 1e-9
                    for left, right in zip(legacy, bounded, strict=True)
                )
            ),
            "mean_completion_cost_delta": float(
                np.mean(
                    [
                        right["sustained_completion_cost"]
                        - left["sustained_completion_cost"]
                        for left, right in zip(legacy, bounded, strict=True)
                    ]
                )
            ),
        },
    }
