"""Plotting / harbor / uncertainty."""

from __future__ import annotations

from scripts.plotting._backend import configure_matplotlib

configure_matplotlib()

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_joint_uncertainty_plot(
    records,
    comparison,
    path: str | Path,
    *,
    study_title: str,
    confirmation_passed: bool | None = None,
) -> Path:
    """Plot actuator-recovery benefit while independently auditing current fit."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    baseline = "Innovation-threshold RLS"
    candidate = "Transient-offset threshold RLS"
    seeds = sorted({record["seed"] for record in records})
    by_key = {(record["seed"], record["controller"]): record for record in records}
    case_x = np.arange(len(seeds))
    recovery_benefit = [
        by_key[(seed, baseline)]["recovery_rmse"]
        - by_key[(seed, candidate)]["recovery_rmse"]
        for seed in seeds
    ]
    current_baseline = [by_key[(seed, baseline)]["current_rmse"] for seed in seeds]
    current_candidate = [by_key[(seed, candidate)]["current_rmse"] for seed in seeds]
    cost_delta = [
        by_key[(seed, candidate)]["sustained_completion_cost"]
        - by_key[(seed, baseline)]["sustained_completion_cost"]
        for seed in seeds
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.0))
    recovery_axis, current_axis, cost_axis, platform_axis = axes.flat
    recovery_axis.plot(case_x, recovery_benefit, "o-", color="#9b6a35")
    recovery_axis.axhline(0.0, color="#555555", linewidth=1, linestyle="--")
    recovery_axis.set_title("Post-Recovery Actuator-Estimate Benefit")
    recovery_axis.set_ylabel("threshold RMSE - transient-offset RMSE")

    current_axis.plot(
        case_x, current_baseline, "o-", color="#2f9c95", label="Threshold"
    )
    current_axis.plot(
        case_x,
        current_candidate,
        "o-",
        color="#9b6a35",
        label="Threshold + transient offset",
    )
    current_axis.set_title("Independent Environmental-Current Fit")
    current_axis.set_ylabel("marine current-estimate RMSE [m/s]")
    current_axis.legend(fontsize=8)

    cost_axis.plot(case_x, cost_delta, "o-", color="#4978a8")
    cost_axis.axhline(0.0, color="#555555", linewidth=1, linestyle="--")
    cost_axis.set_title("Task-Cost Neutrality")
    cost_axis.set_ylabel("transient-offset cost - threshold cost")

    platform_names = tuple(by_key[(seeds[0], candidate)]["hidden_water_current"])
    platform_error = []
    for name in platform_names:
        errors = []
        for seed in seeds:
            record = by_key[(seed, candidate)]
            estimate = np.asarray(record["residual_estimate_history"][name][-1])
            truth = np.asarray(record["hidden_water_current"][name])
            errors.append(float(np.linalg.norm(estimate - truth)))
        platform_error.append(float(np.mean(errors)))
    platform_x = np.arange(len(platform_names))
    bars = platform_axis.bar(platform_x, platform_error, color=("#1f77b4", "#7048a8"))
    for bar, value in zip(bars, platform_error, strict=True):
        platform_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    platform_axis.set_title("Final Current-Vector Error by Marine Platform")
    platform_axis.set_ylabel("Euclidean error [m/s]")
    platform_axis.set_xticks(
        platform_x,
        ["USV" if name == "surface_vessel" else "ROV" for name in platform_names],
    )

    for axis in (recovery_axis, current_axis, cost_axis):
        axis.set_xticks(case_x, [str(seed) for seed in seeds])
        axis.set_xlabel("hidden joint-uncertainty case")
        axis.grid(True, alpha=0.25)
    platform_axis.grid(True, axis="y", alpha=0.25)
    gate = (
        ""
        if confirmation_passed is None
        else f"; overall gate {'PASSED' if confirmation_passed else 'FAILED'}"
    )
    fig.suptitle(
        f"{study_title}\n"
        f"completion rescues {comparison['completion_rescues']}/"
        f"{comparison['trials']}, regressions "
        f"{comparison['completion_regressions']}; candidate complete "
        f"{100.0 * comparison['candidate_completion_rate']:.0f}%; "
        f"current-RMSE delta {comparison['mean_current_rmse_delta']:+.2e}{gate}",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def save_projected_residual_plot(
    records,
    summary,
    path: str | Path,
    *,
    study_title: str,
    confirmation_passed: bool | None = None,
) -> Path:
    """Show task and USV-regulation effects of controllability projection."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    baseline = "Transient-offset threshold RLS"
    candidate = "Projected transient-offset RLS"
    seeds = sorted({record["seed"] for record in records})
    by_key = {(record["seed"], record["controller"]): record for record in records}
    case_x = np.arange(len(seeds))
    labels = ("Full residual", "Actuation-subspace")
    colors = ("#9b6a35", "#2f9c95")
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.0))
    completion_axis, yaw_axis, fallback_axis, current_axis = axes.flat

    width = 0.36
    for offset, label, controller, color in zip(
        (-width / 2, width / 2), labels, (baseline, candidate), colors, strict=True
    ):
        completion_axis.bar(
            case_x + offset,
            [int(by_key[(seed, controller)]["all_goals_reached"]) for seed in seeds],
            width,
            label=label,
            color=color,
        )
        yaw_axis.plot(
            case_x,
            [
                by_key[(seed, controller)]["final_orientation_errors"]["surface_vessel"]
                for seed in seeds
            ],
            "o-",
            color=color,
            label=label,
        )
        fallback_axis.plot(
            case_x,
            [by_key[(seed, controller)]["solver_fallbacks"] for seed in seeds],
            "o-",
            color=color,
            label=label,
        )
        current_axis.plot(
            case_x,
            [by_key[(seed, controller)]["control_current_rmse"] for seed in seeds],
            "o-",
            color=color,
            label=label,
        )

    completion_axis.set_title("Sustained Task Completion")
    completion_axis.set_ylabel("complete")
    completion_axis.set_yticks((0, 1), ("no", "yes"))
    yaw_axis.axhline(0.15, color="#b44a4a", linestyle="--", label="yaw tolerance")
    yaw_axis.set_title("USV Final Yaw Regulation")
    yaw_axis.set_ylabel("absolute yaw error [rad]")
    fallback_axis.set_title("NLP Fallback Actions")
    fallback_axis.set_ylabel("fallback count")
    current_axis.set_title("Control-Facing Current-Model Error")
    current_axis.set_ylabel("RMSE [m/s]")
    for axis in axes.flat:
        axis.set_xticks(case_x, [str(seed) for seed in seeds])
        axis.set_xlabel("hidden joint-uncertainty case")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    gate = (
        ""
        if confirmation_passed is None
        else f"; overall gate {'PASSED' if confirmation_passed else 'FAILED'}"
    )
    fig.suptitle(
        f"{study_title}\ncompletion rescues {summary['completion_rescues']}/"
        f"{summary['trials']}, regressions {summary['completion_regressions']}; "
        f"candidate complete {100.0 * summary['candidate_completion_rate']:.0f}%; "
        f"USV yaw delta {summary['mean_usv_final_yaw_error_delta']:+.3f} rad"
        f"{gate}",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def save_dynamic_envelope_plot(
    records,
    summary,
    path: str | Path,
    *,
    study_title: str,
    confirmation_passed: bool | None = None,
) -> Path:
    """Compare hard and elastic dynamic envelopes without hiding relaxation."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    baseline = "Hard-envelope transient-offset RLS"
    candidate = "Retry-elastic transient-offset RLS"
    seeds = sorted({record["seed"] for record in records})
    by_key = {(record["seed"], record["controller"]): record for record in records}
    case_x = np.arange(len(seeds))
    labels = ("Hard envelope", "Retry-only elastic")
    colors = ("#8f5b3e", "#267f78")
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.0))
    completion_axis, fallback_axis, slack_axis, yaw_axis = axes.flat
    width = 0.36
    for offset, label, controller, color in zip(
        (-width / 2, width / 2), labels, (baseline, candidate), colors, strict=True
    ):
        completion_axis.bar(
            case_x + offset,
            [int(by_key[(seed, controller)]["all_goals_reached"]) for seed in seeds],
            width,
            label=label,
            color=color,
        )
        fallback_axis.plot(
            case_x,
            [by_key[(seed, controller)]["solver_fallbacks"] for seed in seeds],
            "o-",
            color=color,
            label=label,
        )
        slack_axis.plot(
            case_x,
            [by_key[(seed, controller)]["max_dynamic_state_slack"] for seed in seeds],
            "o-",
            color=color,
            label=label,
        )
        yaw_axis.plot(
            case_x,
            [
                by_key[(seed, controller)]["final_orientation_errors"]["surface_vessel"]
                for seed in seeds
            ],
            "o-",
            color=color,
            label=label,
        )
    completion_axis.set_title("Sustained Task Completion")
    completion_axis.set_ylabel("complete")
    completion_axis.set_yticks((0, 1), ("no", "yes"))
    fallback_axis.set_title("NLP Fallback Actions")
    fallback_axis.set_ylabel("fallback count")
    slack_axis.set_title("Maximum Dynamic-Envelope Relaxation")
    slack_axis.set_ylabel("state-unit relaxation")
    yaw_axis.axhline(0.15, color="#b44a4a", linestyle="--", label="yaw tolerance")
    yaw_axis.set_title("USV Final Yaw Regulation")
    yaw_axis.set_ylabel("absolute yaw error [rad]")
    for axis in axes.flat:
        axis.set_xticks(case_x, [str(seed) for seed in seeds])
        axis.set_xlabel("hidden joint-uncertainty case")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    gate = (
        ""
        if confirmation_passed is None
        else f"; overall gate {'PASSED' if confirmation_passed else 'FAILED'}"
    )
    fig.suptitle(
        f"{study_title}\ncompletion rescues {summary['completion_rescues']}/"
        f"{summary['trials']}, regressions {summary['completion_regressions']}; "
        f"candidate complete {100.0 * summary['candidate_completion_rate']:.0f}%; "
        f"max relaxation {summary['maximum_dynamic_state_slack']:.4f}{gate}",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def save_station_keeping_plot(
    records, summary, path: str | Path, confirmation: dict | None = None
) -> Path:
    """Plot paired current-observer and Heron station-keeping outcomes."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = tuple(summary["controllers"])
    seeds = sorted({record["seed"] for record in records})
    by_key = {(record["seed"], record["controller"]): record for record in records}
    colors = ("#5b6472", "#2a9d8f", "#1976b9")
    short = {
        "Hard-envelope transient-offset RLS": "model residual",
        "Kinematic-current EWMA transient-offset RLS": "kinematic EWMA",
        "Kinematic-current RLS transient-offset RLS": "kinematic RLS",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharex=True)
    x = np.arange(len(seeds))
    metrics = (
        ("final_orientation_errors", "Heron final yaw error", "absolute error [rad]"),
        ("final_goal_errors", "Heron final position error", "error [m]"),
        ("platform_current_rmse", "Heron current-estimation RMSE", "RMSE [m/s]"),
        (
            "control_total_variation",
            "Heron waterjet command variation",
            "total variation",
        ),
    )
    for axis, (field, title, ylabel) in zip(axes.flat, metrics):
        for label, color in zip(labels, colors):
            axis.plot(
                x,
                [by_key[(seed, label)][field]["surface_vessel"] for seed in seeds],
                "o-",
                color=color,
                linewidth=1.8,
                markersize=5,
                label=short.get(label, label),
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[0, 0].axhline(0.15, color="#b44a4a", linestyle="--", label="yaw tolerance")
    for axis in axes[1]:
        axis.set_xticks(x, [str(seed) for seed in seeds], rotation=30)
        axis.set_xlabel("matched hidden-current and actuator-fault case")
    for axis in axes.flat:
        axis.legend(fontsize=8)
    gate = (
        "development"
        if confirmation is None
        else f"confirmation {'PASSED' if confirmation['passed'] else 'FAILED'}"
    )
    fig.suptitle(
        "Actuator-Independent Current Observation for Heron Station Keeping\n"
        f"{gate}; kinematic RLS yaw wins {summary['usv_yaw_wins']}/"
        f"{summary['trials']}; mean current-RMSE delta "
        f"{summary['mean_usv_current_rmse_delta']:+.3f} m/s",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
