"""Metrics / harbor / station keeping study."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scripts.configuration.study_labels import HARD_ENVELOPE_BASELINE as BASELINE
from scripts.configuration.study_labels import KINEMATIC_RLS, USV_NAME

if TYPE_CHECKING:
    from scripts.configuration.harbor_station_keeping_criteria import (
        HarborStationKeepingCriteriaConfig,
    )


def summarize_station_keeping(records: list[dict]) -> dict:
    """Aggregate paired station-keeping and observer metrics."""
    labels = tuple(dict.fromkeys(record["controller"] for record in records))
    seeds = sorted({record["seed"] for record in records})
    by_key = {(record["seed"], record["controller"]): record for record in records}
    controllers = {}
    for label in labels:
        selected = [by_key[(seed, label)] for seed in seeds]
        controllers[label] = {
            "completion_rate": float(
                np.mean([record["all_goals_reached"] for record in selected])
            ),
            "safety_rate": float(
                np.mean(
                    [
                        record["pairwise_violation_count"] == 0
                        and record["max_collision_slack"] <= 1.0e-9
                        for record in selected
                    ]
                )
            ),
            "fallback_free_rate": float(
                np.mean([record["solver_fallbacks"] == 0 for record in selected])
            ),
            "mean_usv_yaw_error": float(
                np.mean(
                    [
                        record["final_orientation_errors"][USV_NAME]
                        for record in selected
                    ]
                )
            ),
            "mean_usv_position_error": float(
                np.mean([record["final_goal_errors"][USV_NAME] for record in selected])
            ),
            "mean_usv_current_rmse": float(
                np.mean(
                    [record["platform_current_rmse"][USV_NAME] for record in selected]
                )
            ),
            "mean_usv_control_total_variation": float(
                np.mean(
                    [record["control_total_variation"][USV_NAME] for record in selected]
                )
            ),
        }
    baseline = [by_key[(seed, BASELINE)] for seed in seeds]
    candidate = [by_key[(seed, KINEMATIC_RLS)] for seed in seeds]

    def paired_delta(field: str, nested: str | None = None) -> np.ndarray:
        def value(record):
            item = record[field]
            return item[nested] if nested is not None else item

        return np.asarray([value(c) - value(b) for b, c in zip(baseline, candidate)])

    yaw_delta = paired_delta("final_orientation_errors", USV_NAME)
    return {
        "trials": len(seeds),
        "controllers": controllers,
        "usv_yaw_wins": int(np.count_nonzero(yaw_delta < 0.0)),
        "usv_yaw_win_rate": float(np.mean(yaw_delta < 0.0)),
        "mean_usv_yaw_error_delta": float(np.mean(yaw_delta)),
        "mean_usv_position_error_delta": float(
            np.mean(paired_delta("final_goal_errors", USV_NAME))
        ),
        "mean_usv_current_rmse_delta": float(
            np.mean(paired_delta("platform_current_rmse", USV_NAME))
        ),
        "mean_recovery_rmse_delta": float(np.mean(paired_delta("recovery_rmse"))),
        "mean_completion_cost_delta": float(
            np.mean(paired_delta("sustained_completion_cost"))
        ),
        "mean_usv_control_total_variation_delta": float(
            np.mean(paired_delta("control_total_variation", USV_NAME))
        ),
    }


def evaluate_station_keeping(
    summary: dict, criteria: HarborStationKeepingCriteriaConfig
) -> dict:
    """Apply predeclared station-keeping confirmation gates."""
    candidate = summary["controllers"][criteria.controller_labels[1]]
    checks = {
        "candidate_completion_rate": candidate["completion_rate"]
        >= criteria.minimum_candidate_completion_rate,
        "usv_yaw_win_rate": summary["usv_yaw_win_rate"]
        >= criteria.minimum_usv_yaw_win_rate,
        "safety_rate": candidate["safety_rate"] >= criteria.minimum_safety_rate,
        "fallback_free_rate": candidate["fallback_free_rate"]
        >= criteria.minimum_fallback_free_rate,
        "usv_yaw_error_delta": summary["mean_usv_yaw_error_delta"]
        <= criteria.maximum_mean_usv_yaw_error_delta,
        "usv_position_error_delta": summary["mean_usv_position_error_delta"]
        <= criteria.maximum_mean_usv_position_error_delta,
        "usv_current_rmse_delta": summary["mean_usv_current_rmse_delta"]
        <= criteria.maximum_mean_usv_current_rmse_delta,
        "recovery_rmse_delta": summary["mean_recovery_rmse_delta"]
        <= criteria.maximum_mean_recovery_rmse_delta,
        "completion_cost_delta": summary["mean_completion_cost_delta"]
        <= criteria.maximum_mean_completion_cost_delta,
    }
    return {"passed": all(checks.values()), "checks": checks}
