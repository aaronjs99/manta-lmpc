"""Metrics / harbor / joint uncertainty study."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scripts.configuration.harbor_joint_uncertainty_criteria import (
        HarborJointUncertaintyCriteriaConfig,
    )


def summarize_joint_uncertainty(records: list[dict], recovery: dict) -> dict:
    """Summarize paired closed-loop resilience under both hidden effects."""
    baseline_label = "Innovation-threshold RLS"
    candidate_label = "Transient-offset threshold RLS"
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
    return {
        **recovery,
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
    }


def evaluate_joint_uncertainty(
    summary: dict, criteria: HarborJointUncertaintyCriteriaConfig
) -> dict:
    """Apply frozen task and estimator non-inferiority gates."""
    trials = summary["trials"]
    rescue_rate = summary["completion_rescues"] / trials
    regression_rate = summary["completion_regressions"] / trials
    recovery_win_rate = summary["recovery_wins"] / trials
    final_effectiveness_delta = -summary["mean_final_rmse_reduction"]
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
        "recovery_win_rate": (recovery_win_rate >= criteria.minimum_recovery_win_rate),
        "completion_cost_delta": (
            summary["mean_completion_cost_delta"]
            <= criteria.maximum_mean_completion_cost_delta
        ),
        "final_effectiveness_rmse_delta": (
            final_effectiveness_delta
            <= criteria.maximum_mean_final_effectiveness_rmse_delta
        ),
        "current_rmse_delta": (
            summary["mean_current_rmse_delta"]
            <= criteria.maximum_mean_current_rmse_delta
        ),
        "final_current_rmse_delta": (
            summary["mean_final_current_rmse_delta"]
            <= criteria.maximum_mean_final_current_rmse_delta
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "completion_rescue_rate": rescue_rate,
            "completion_regression_rate": regression_rate,
            "recovery_win_rate": recovery_win_rate,
            "final_effectiveness_rmse_delta": final_effectiveness_delta,
            **{
                key: summary[key]
                for key in (
                    "baseline_completion_rate",
                    "candidate_completion_rate",
                    "candidate_safety_rate",
                    "candidate_fallback_free_rate",
                    "mean_completion_cost_delta",
                    "mean_current_rmse_delta",
                    "mean_final_current_rmse_delta",
                )
            },
        },
    }
