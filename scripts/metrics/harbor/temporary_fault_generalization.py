"""Metrics / harbor / temporary fault generalization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scripts.configuration.harbor_temporary_fault_criteria import (
        HarborConfirmationCriteriaConfig,
        HarborRecoveryConfirmationCriteriaConfig,
    )


def evaluate_confirmation(
    summary: dict, criteria: HarborConfirmationCriteriaConfig
) -> dict:
    """Apply predeclared closed-loop acceptance gates to threshold RLS."""
    label = "Innovation-threshold RLS"
    controller = summary["controllers"][label]
    comparison = summary["paired_vs_fixed"][label]
    lower_bound = comparison["paired_mean_rmse_reduction_bootstrap_95_ci"][0]
    win_rate = comparison["adaptive_wins"] / comparison["trials"]
    checks = {
        "adaptive_win_rate": win_rate >= criteria.minimum_adaptive_win_rate,
        "positive_bootstrap_lower_bound": (
            lower_bound > 0.0
            if criteria.require_positive_bootstrap_lower_bound
            else True
        ),
        "completion_rate": (
            controller["completion_rate"] >= criteria.minimum_completion_rate
        ),
        "safety_rate": controller["safety_rate"] >= criteria.minimum_safety_rate,
        "fallback_free_rate": (
            controller["fallback_free_rate"] >= criteria.minimum_fallback_free_rate
        ),
        "completion_cost_delta": (
            comparison["mean_completion_cost_delta"]
            <= criteria.maximum_mean_completion_cost_delta
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "adaptive_win_rate": win_rate,
            "bootstrap_lower_bound": lower_bound,
            "completion_rate": controller["completion_rate"],
            "safety_rate": controller["safety_rate"],
            "fallback_free_rate": controller["fallback_free_rate"],
            "mean_completion_cost_delta": comparison["mean_completion_cost_delta"],
        },
    }


def summarize_recovery_prior(
    records: list[dict],
    bootstrap_samples: int,
    candidate_label: str = "Recovery-prior threshold RLS",
) -> dict:
    """Compare one recovery-aware threshold RLS with ordinary threshold RLS."""
    baseline_label = "Innovation-threshold RLS"
    seeds = sorted({record["seed"] for record in records})
    by_key = {(record["seed"], record["controller"]): record for record in records}

    def values(label: str, metric: str, *, default: float | None = None) -> np.ndarray:
        if default is None:
            samples = [by_key[(seed, label)][metric] for seed in seeds]
        else:
            samples = [by_key[(seed, label)].get(metric, default) for seed in seeds]
        return np.asarray(samples, dtype=float)

    baseline_recovery = values(baseline_label, "recovery_rmse")
    candidate_recovery = values(candidate_label, "recovery_rmse")
    recovery_reduction = baseline_recovery - candidate_recovery
    baseline_fault = values(baseline_label, "fault_interval_rmse")
    candidate_fault = values(candidate_label, "fault_interval_rmse")
    baseline_final = values(baseline_label, "final_effectiveness_rmse")
    candidate_final = values(candidate_label, "final_effectiveness_rmse")
    baseline_current = values(baseline_label, "current_rmse", default=0.0)
    candidate_current = values(candidate_label, "current_rmse", default=0.0)
    baseline_final_current = values(baseline_label, "final_current_rmse", default=0.0)
    candidate_final_current = values(candidate_label, "final_current_rmse", default=0.0)
    cost_delta = values(candidate_label, "sustained_completion_cost") - values(
        baseline_label, "sustained_completion_cost"
    )
    rng = np.random.default_rng(20260717)
    bootstrap = np.mean(
        rng.choice(
            recovery_reduction,
            size=(bootstrap_samples, len(recovery_reduction)),
            replace=True,
        ),
        axis=1,
    )
    candidate = [by_key[(seed, candidate_label)] for seed in seeds]
    return {
        "trials": len(seeds),
        "recovery_wins": int(np.count_nonzero(recovery_reduction > 0.0)),
        "mean_recovery_rmse_reduction": float(np.mean(recovery_reduction)),
        "mean_relative_recovery_rmse_reduction": float(
            np.mean(recovery_reduction / np.maximum(baseline_recovery, 1.0e-12))
        ),
        "paired_mean_recovery_reduction_bootstrap_95_ci": [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
        ],
        "mean_fault_interval_rmse_delta": float(
            np.mean(candidate_fault - baseline_fault)
        ),
        "mean_final_rmse_reduction": float(np.mean(baseline_final - candidate_final)),
        "mean_current_rmse_delta": float(np.mean(candidate_current - baseline_current)),
        "mean_final_current_rmse_delta": float(
            np.mean(candidate_final_current - baseline_final_current)
        ),
        "mean_completion_cost_delta": float(np.mean(cost_delta)),
        "completion_rate": float(
            np.mean([record["all_goals_reached"] for record in candidate])
        ),
        "safety_rate": float(
            np.mean(
                [
                    record["pairwise_violation_count"] == 0
                    and record["max_collision_slack"] <= 1.0e-9
                    for record in candidate
                ]
            )
        ),
        "fallback_free_rate": float(
            np.mean([record["solver_fallbacks"] == 0 for record in candidate])
        ),
    }


def evaluate_recovery_confirmation(
    comparison: dict, criteria: HarborRecoveryConfirmationCriteriaConfig
) -> dict:
    """Apply predeclared closed-loop gates to the recovery-prior comparator."""
    lower_bound = comparison["paired_mean_recovery_reduction_bootstrap_95_ci"][0]
    win_rate = comparison["recovery_wins"] / comparison["trials"]
    final_rmse_delta = -comparison["mean_final_rmse_reduction"]
    checks = {
        "recovery_win_rate": win_rate >= criteria.minimum_recovery_win_rate,
        "positive_bootstrap_lower_bound": (
            lower_bound > 0.0
            if criteria.require_positive_bootstrap_lower_bound
            else True
        ),
        "completion_rate": (
            comparison["completion_rate"] >= criteria.minimum_completion_rate
        ),
        "safety_rate": comparison["safety_rate"] >= criteria.minimum_safety_rate,
        "fallback_free_rate": (
            comparison["fallback_free_rate"] >= criteria.minimum_fallback_free_rate
        ),
        "fault_interval_rmse_delta": (
            comparison["mean_fault_interval_rmse_delta"]
            <= criteria.maximum_mean_fault_interval_rmse_delta
        ),
        "final_rmse_delta": (
            final_rmse_delta <= criteria.maximum_mean_final_rmse_delta
        ),
        "completion_cost_delta": (
            comparison["mean_completion_cost_delta"]
            <= criteria.maximum_mean_completion_cost_delta
        ),
        "current_rmse_delta": (
            comparison["mean_current_rmse_delta"]
            <= criteria.maximum_mean_current_rmse_delta
        ),
        "final_current_rmse_delta": (
            comparison["mean_final_current_rmse_delta"]
            <= criteria.maximum_mean_final_current_rmse_delta
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "recovery_win_rate": win_rate,
            "bootstrap_lower_bound": lower_bound,
            "completion_rate": comparison["completion_rate"],
            "safety_rate": comparison["safety_rate"],
            "fallback_free_rate": comparison["fallback_free_rate"],
            "mean_fault_interval_rmse_delta": comparison[
                "mean_fault_interval_rmse_delta"
            ],
            "mean_final_rmse_delta": final_rmse_delta,
            "mean_completion_cost_delta": comparison["mean_completion_cost_delta"],
            "mean_current_rmse_delta": comparison["mean_current_rmse_delta"],
            "mean_final_current_rmse_delta": comparison[
                "mean_final_current_rmse_delta"
            ],
        },
    }
