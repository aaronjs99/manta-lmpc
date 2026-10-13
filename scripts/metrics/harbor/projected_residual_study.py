"""Metrics / harbor / projected residual study."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scripts.configuration.harbor_projected_residual_criteria import (
        HarborProjectedResidualCriteriaConfig,
    )


def summarize_projected_residual(records: list[dict]) -> dict:
    """Build paired task, solver, and estimator projection metrics."""
    baseline_label = "Transient-offset threshold RLS"
    candidate_label = "Projected transient-offset RLS"
    seeds = sorted({record["seed"] for record in records})
    by_key = {(record["seed"], record["controller"]): record for record in records}
    baseline = [by_key[(seed, baseline_label)] for seed in seeds]
    candidate = [by_key[(seed, candidate_label)] for seed in seeds]
    baseline_complete = np.asarray(
        [record["all_goals_reached"] for record in baseline], dtype=bool
    )
    candidate_complete = np.asarray(
        [record["all_goals_reached"] for record in candidate], dtype=bool
    )

    def mean_delta(metric: str) -> float:
        return float(
            np.mean(
                [
                    candidate_record[metric] - baseline_record[metric]
                    for baseline_record, candidate_record in zip(
                        baseline, candidate, strict=True
                    )
                ]
            )
        )

    return {
        "trials": len(seeds),
        "completion_rescues": int(
            np.count_nonzero(candidate_complete & ~baseline_complete)
        ),
        "completion_regressions": int(
            np.count_nonzero(~candidate_complete & baseline_complete)
        ),
        "baseline_completion_rate": float(np.mean(baseline_complete)),
        "candidate_completion_rate": float(np.mean(candidate_complete)),
        "candidate_safety_rate": float(
            np.mean(
                [
                    record["pairwise_violation_count"] == 0
                    and record["max_collision_slack"] <= 1.0e-9
                    for record in candidate
                ]
            )
        ),
        "candidate_fallback_free_rate": float(
            np.mean([record["solver_fallbacks"] == 0 for record in candidate])
        ),
        "mean_completion_cost_delta": mean_delta("sustained_completion_cost"),
        "mean_recovery_rmse_delta": mean_delta("recovery_rmse"),
        "mean_current_rmse_delta": mean_delta("current_rmse"),
        "mean_control_current_rmse_delta": mean_delta("control_current_rmse"),
        "mean_final_control_current_rmse_delta": mean_delta(
            "final_control_current_rmse"
        ),
        "mean_usv_final_yaw_error_delta": float(
            np.mean(
                [
                    candidate_record["final_orientation_errors"]["surface_vessel"]
                    - baseline_record["final_orientation_errors"]["surface_vessel"]
                    for baseline_record, candidate_record in zip(
                        baseline, candidate, strict=True
                    )
                ]
            )
        ),
    }


def evaluate_projected_residual(
    summary: dict, criteria: HarborProjectedResidualCriteriaConfig
) -> dict:
    """Apply frozen closed-loop and estimator non-inferiority gates."""
    trials = summary["trials"]
    rescue_rate = summary["completion_rescues"] / trials
    regression_rate = summary["completion_regressions"] / trials
    checks = {
        "candidate_completion_rate": (
            summary["candidate_completion_rate"]
            >= criteria.minimum_candidate_completion_rate
        ),
        "completion_rescue_rate": (
            rescue_rate >= criteria.minimum_completion_rescue_rate
        ),
        "completion_regression_rate": (
            regression_rate <= criteria.maximum_completion_regression_rate
        ),
        "safety_rate": (
            summary["candidate_safety_rate"] >= criteria.minimum_safety_rate
        ),
        "fallback_free_rate": (
            summary["candidate_fallback_free_rate"]
            >= criteria.minimum_fallback_free_rate
        ),
        "completion_cost_delta": (
            summary["mean_completion_cost_delta"]
            <= criteria.maximum_mean_completion_cost_delta
        ),
        "recovery_rmse_delta": (
            summary["mean_recovery_rmse_delta"]
            <= criteria.maximum_mean_recovery_rmse_delta
        ),
        "current_rmse_delta": (
            summary["mean_current_rmse_delta"]
            <= criteria.maximum_mean_current_rmse_delta
        ),
        "control_current_rmse_delta": (
            summary["mean_control_current_rmse_delta"]
            <= criteria.maximum_mean_control_current_rmse_delta
        ),
        "final_control_current_rmse_delta": (
            summary["mean_final_control_current_rmse_delta"]
            <= criteria.maximum_mean_final_control_current_rmse_delta
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "completion_rescue_rate": rescue_rate,
            "completion_regression_rate": regression_rate,
            **{key: value for key, value in summary.items() if key != "trials"},
        },
    }
