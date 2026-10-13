"""Aggregate metrics for matched scheduled temporary-fault trials."""

from __future__ import annotations

import numpy as np
from scripts.configuration.study_labels import CONTROLLER_LABELS


def summarize_time_varying_faults(
    records: list[dict], bootstrap_samples: int = 5000
) -> dict:
    """Aggregate tracking and task metrics for matched observation seeds."""
    present_labels = {record["controller"] for record in records}
    labels = tuple(label for label in CONTROLLER_LABELS if label in present_labels)
    if not labels or labels[0] != CONTROLLER_LABELS[0]:
        raise ValueError(
            "temporary-fault records require the fixed-covariance comparator"
        )
    controllers = {}
    for label in labels:
        selected = [record for record in records if record["controller"] == label]
        delays = [
            record["mean_detection_delay"]
            for record in selected
            if record["mean_detection_delay"] is not None
        ]
        controllers[label] = {
            "trials": len(selected),
            "mean_fault_interval_rmse": float(
                np.mean([record["fault_interval_rmse"] for record in selected])
            ),
            "mean_recovery_rmse": float(
                np.mean([record["recovery_rmse"] for record in selected])
            ),
            "mean_final_rmse": float(
                np.mean([record["final_effectiveness_rmse"] for record in selected])
            ),
            "mean_sustained_completion_cost": float(
                np.mean([record["sustained_completion_cost"] for record in selected])
            ),
            "fallback_free_rate": float(
                np.mean([record["solver_fallbacks"] == 0 for record in selected])
            ),
            "max_domain_buffer_slack": float(
                max(record.get("max_domain_buffer_slack", 0.0) for record in selected)
            ),
            "completion_rate": float(
                np.mean([record["all_goals_reached"] for record in selected])
            ),
            "safety_rate": float(
                np.mean(
                    [
                        record["pairwise_violation_count"] == 0
                        and record["max_collision_slack"] <= 1e-9
                        for record in selected
                    ]
                )
            ),
            "mean_change_detections": float(
                np.mean(
                    [
                        sum(
                            len(steps)
                            for steps in record["change_steps_by_agent"].values()
                        )
                        for record in selected
                    ]
                )
            ),
            "mean_event_recall": float(
                np.mean([record["event_recall"] for record in selected])
            ),
            "mean_false_inflations": float(
                np.mean([record["false_inflations"] for record in selected])
            ),
            "mean_detection_delay": float(np.mean(delays)) if delays else None,
            "mean_probe_count": float(
                np.mean([record["probe_count"] for record in selected])
            ),
        }
    by_key = {(record["seed"], record["controller"]): record for record in records}
    seeds = sorted({record["seed"] for record in records})
    fixed = np.asarray(
        [by_key[(seed, labels[0])]["fault_interval_rmse"] for seed in seeds]
    )
    fixed_final = np.asarray(
        [by_key[(seed, labels[0])]["final_effectiveness_rmse"] for seed in seeds]
    )
    fixed_recovery = np.asarray(
        [by_key[(seed, labels[0])]["recovery_rmse"] for seed in seeds]
    )
    paired = {}
    rng = np.random.default_rng(20260716)
    for label in labels[1:]:
        adaptive = np.asarray(
            [by_key[(seed, label)]["fault_interval_rmse"] for seed in seeds]
        )
        adaptive_final = np.asarray(
            [by_key[(seed, label)]["final_effectiveness_rmse"] for seed in seeds]
        )
        adaptive_recovery = np.asarray(
            [by_key[(seed, label)]["recovery_rmse"] for seed in seeds]
        )
        reduction = fixed - adaptive
        final_reduction = fixed_final - adaptive_final
        recovery_reduction = fixed_recovery - adaptive_recovery
        rmse_bootstrap = np.mean(
            rng.choice(
                reduction,
                size=(bootstrap_samples, len(reduction)),
                replace=True,
            ),
            axis=1,
        )
        recovery_bootstrap = np.mean(
            rng.choice(
                recovery_reduction,
                size=(bootstrap_samples, len(recovery_reduction)),
                replace=True,
            ),
            axis=1,
        )
        paired[label] = {
            "trials": len(seeds),
            "adaptive_wins": int(np.count_nonzero(reduction > 0.0)),
            "recovery_wins": int(np.count_nonzero(recovery_reduction > 0.0)),
            "final_wins": int(np.count_nonzero(final_reduction > 0.0)),
            "mean_rmse_reduction": float(np.mean(reduction)),
            "mean_relative_rmse_reduction": float(
                np.mean(reduction / np.maximum(fixed, 1e-12))
            ),
            "paired_mean_rmse_reduction_bootstrap_95_ci": [
                float(value) for value in np.quantile(rmse_bootstrap, [0.025, 0.975])
            ],
            "mean_recovery_rmse_reduction": float(np.mean(recovery_reduction)),
            "mean_relative_recovery_rmse_reduction": float(
                np.mean(recovery_reduction / np.maximum(fixed_recovery, 1e-12))
            ),
            "paired_mean_recovery_reduction_bootstrap_95_ci": [
                float(value)
                for value in np.quantile(recovery_bootstrap, [0.025, 0.975])
            ],
            "mean_final_rmse_reduction": float(np.mean(final_reduction)),
            "mean_relative_final_rmse_reduction": float(
                np.mean(final_reduction / np.maximum(fixed_final, 1e-12))
            ),
            "mean_completion_cost_delta": float(
                np.mean(
                    [
                        by_key[(seed, label)]["sustained_completion_cost"]
                        - by_key[(seed, labels[0])]["sustained_completion_cost"]
                        for seed in seeds
                    ]
                )
            ),
        }
    summary = {
        "controllers": controllers,
        "paired_vs_fixed": paired,
    }
    passive_label = "Chi-square CUSUM RLS"
    active_label = "CUSUM-triggered probing RLS"
    if passive_label in labels and active_label in labels:
        passive_fault = np.asarray(
            [by_key[(seed, passive_label)]["fault_interval_rmse"] for seed in seeds]
        )
        active_fault = np.asarray(
            [by_key[(seed, active_label)]["fault_interval_rmse"] for seed in seeds]
        )
        passive_recovery = np.asarray(
            [by_key[(seed, passive_label)]["recovery_rmse"] for seed in seeds]
        )
        active_recovery = np.asarray(
            [by_key[(seed, active_label)]["recovery_rmse"] for seed in seeds]
        )
        summary["triggered_probing_vs_passive_cusum"] = {
            "trials": len(seeds),
            "degraded_interval_wins": int(
                np.count_nonzero(active_fault < passive_fault)
            ),
            "mean_fault_rmse_reduction": float(np.mean(passive_fault - active_fault)),
            "recovery_wins": int(np.count_nonzero(active_recovery < passive_recovery)),
            "mean_recovery_rmse_reduction": float(
                np.mean(passive_recovery - active_recovery)
            ),
            "mean_completion_cost_delta": float(
                np.mean(
                    [
                        by_key[(seed, active_label)]["sustained_completion_cost"]
                        - by_key[(seed, passive_label)]["sustained_completion_cost"]
                        for seed in seeds
                    ]
                )
            ),
            "additional_solver_fallbacks": int(
                sum(by_key[(seed, active_label)]["solver_fallbacks"] for seed in seeds)
                - sum(
                    by_key[(seed, passive_label)]["solver_fallbacks"] for seed in seeds
                )
            ),
        }
    return summary
