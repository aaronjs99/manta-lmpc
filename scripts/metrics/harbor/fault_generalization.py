"""Metrics / harbor / fault generalization."""

from __future__ import annotations

import numpy as np


def summarize_fault_generalization(records, bootstrap_samples: int) -> dict:
    """Return controller aggregates and a paired equal-budget comparison."""
    labels = tuple(dict.fromkeys(record["controller"] for record in records))
    controllers = {}
    for label in labels:
        selected = [record for record in records if record["controller"] == label]
        controllers[label] = {
            "trials": len(selected),
            "mean_effectiveness_rmse": float(
                np.mean([record["effectiveness_rmse"] for record in selected])
            ),
            "std_effectiveness_rmse": float(
                np.std([record["effectiveness_rmse"] for record in selected], ddof=1)
                if len(selected) > 1
                else 0.0
            ),
            "mean_first_hit_step_sum": float(
                np.mean([record["first_hit_step_sum"] for record in selected])
            ),
            "mean_sustained_completion_cost": float(
                np.mean([record["sustained_completion_cost"] for record in selected])
            ),
            "valid_rate": float(np.mean([record["valid"] for record in selected])),
            "completion_rate": float(
                np.mean([record["all_goals_reached"] for record in selected])
            ),
            "safety_rate": float(
                np.mean(
                    [record["pairwise_violation_count"] == 0 for record in selected]
                )
            ),
        }

    by_key = {(record["seed"], record["controller"]): record for record in records}
    seeds = sorted({record["seed"] for record in records})
    labels = {record["controller"] for record in records}
    one_pass_label = (
        "Recursive one-pass MPC"
        if "Recursive one-pass MPC" in labels
        else "One-pass active MPC"
    )
    information_label = (
        "Recursive information-aware MPC"
        if "Recursive information-aware MPC" in labels
        else "Information-aware MPC"
    )
    one_pass = np.asarray(
        [by_key[(seed, one_pass_label)]["effectiveness_rmse"] for seed in seeds]
    )
    information = np.asarray(
        [by_key[(seed, information_label)]["effectiveness_rmse"] for seed in seeds]
    )
    differences = one_pass - information
    rng = np.random.default_rng(20260716)
    bootstrap = np.mean(
        rng.choice(
            differences, size=(bootstrap_samples, len(differences)), replace=True
        ),
        axis=1,
    )
    cost_differences = np.asarray(
        [
            by_key[(seed, information_label)]["sustained_completion_cost"]
            - by_key[(seed, one_pass_label)]["sustained_completion_cost"]
            for seed in seeds
        ],
        dtype=float,
    )
    recursive_comparison = None
    if "Recursive diagonal MPC" in labels and "Instantaneous diagonal MPC" in labels:
        instantaneous = np.asarray(
            [
                by_key[(seed, "Instantaneous diagonal MPC")]["effectiveness_rmse"]
                for seed in seeds
            ]
        )
        recursive = np.asarray(
            [
                by_key[(seed, "Recursive diagonal MPC")]["effectiveness_rmse"]
                for seed in seeds
            ]
        )
        recursive_differences = instantaneous - recursive
        recursive_bootstrap = np.mean(
            rng.choice(
                recursive_differences,
                size=(bootstrap_samples, len(recursive_differences)),
                replace=True,
            ),
            axis=1,
        )
        recursive_comparison = {
            "trials": len(seeds),
            "recursive_wins": int(np.count_nonzero(recursive_differences > 0.0)),
            "mean_rmse_reduction": float(np.mean(recursive_differences)),
            "mean_relative_rmse_reduction": float(
                np.mean(recursive_differences / np.maximum(instantaneous, 1e-12))
            ),
            "paired_mean_reduction_bootstrap_95_ci": [
                float(value)
                for value in np.quantile(recursive_bootstrap, [0.025, 0.975])
            ],
        }
    return {
        "controllers": controllers,
        "recursive_vs_instantaneous": recursive_comparison,
        "equal_budget_information_vs_one_pass": {
            "trials": len(seeds),
            "information_wins": int(np.count_nonzero(differences > 0.0)),
            "ties": int(np.count_nonzero(np.isclose(differences, 0.0))),
            "mean_rmse_reduction": float(np.mean(differences)),
            "mean_relative_rmse_reduction": float(
                np.mean(differences / np.maximum(one_pass, 1e-12))
            ),
            "paired_mean_reduction_bootstrap_95_ci": [
                float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
            ],
            "mean_completion_cost_delta": float(np.mean(cost_differences)),
            "max_absolute_completion_cost_delta": float(
                np.max(np.abs(cost_differences))
            ),
        },
    }
