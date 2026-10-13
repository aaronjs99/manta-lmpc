"""Plotting / harbor / recovery."""

from __future__ import annotations

from scripts.plotting._backend import configure_matplotlib

configure_matplotlib()

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_time_varying_fault_plot(records, summary, path: str | Path) -> Path:
    """Plot temporary-fault tracking, recovery, task cost, and event telemetry."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = (
        "Fixed-covariance RLS",
        "Innovation-threshold RLS",
        "Chi-square CUSUM RLS",
        "CUSUM-triggered probing RLS",
    )
    short_labels = ("Fixed", "Threshold", "CUSUM", "CUSUM + probes")
    colors = ("#b66a50", "#2f9c95", "#3974a8", "#7048a8")
    seeds = sorted({record["seed"] for record in records})
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.0))
    tracking_axis, phase_axis, cost_axis, event_axis = axes.flat

    for label, short, color in zip(labels, short_labels, colors, strict=True):
        histories = [
            np.asarray(record["tracking_rmse_history"], dtype=float)
            for record in records
            if record["controller"] == label
        ]
        count = min(map(len, histories))
        matrix = np.asarray([history[:count] for history in histories])
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        steps = np.arange(count)
        tracking_axis.plot(steps, mean, color=color, linewidth=2, label=short)
        tracking_axis.fill_between(
            steps, mean - std, mean + std, color=color, alpha=0.16
        )
    event_steps = sorted(
        {
            step
            for record in records
            for events in record["fault_event_steps"].values()
            for step in events
        }
    )
    for index, step in enumerate(event_steps):
        tracking_axis.axvline(
            step,
            color="#666666",
            linestyle=":",
            alpha=0.45,
            label="plant change" if index == 0 else None,
        )
    tracking_axis.set_title("Global Actuator-Effectiveness Tracking")
    tracking_axis.set_xlabel("simulation step")
    tracking_axis.set_ylabel("channel RMSE")
    tracking_axis.grid(True, alpha=0.25)
    tracking_axis.legend(fontsize=8)

    controller_x = np.arange(len(labels))
    fault_values = [
        summary["controllers"][label]["mean_fault_interval_rmse"] for label in labels
    ]
    recovery_values = [
        summary["controllers"][label]["mean_recovery_rmse"] for label in labels
    ]
    width = 0.36
    phase_axis.bar(
        controller_x - width / 2,
        fault_values,
        width,
        color="#c45a3c",
        label="degraded interval",
    )
    phase_axis.bar(
        controller_x + width / 2,
        recovery_values,
        width,
        color="#4b88b8",
        label="after recovery",
    )
    phase_axis.set_title("Tracking by Plant Phase")
    phase_axis.set_ylabel("mean channel RMSE")
    phase_axis.set_xticks(controller_x, short_labels, rotation=12)
    phase_axis.grid(True, axis="y", alpha=0.25)
    phase_axis.legend(fontsize=8)

    by_key = {(record["seed"], record["controller"]): record for record in records}
    seed_x = np.arange(len(seeds))
    for label, short, color in zip(labels, short_labels, colors, strict=True):
        values = [by_key[(seed, label)]["sustained_completion_cost"] for seed in seeds]
        cost_axis.plot(seed_x, values, "o-", color=color, linewidth=2, label=short)
    cost_axis.set_title("Sustained-Completion Cost")
    cost_axis.set_xlabel("observation seed")
    cost_axis.set_ylabel("first-hit sum with horizon penalty")
    cost_axis.set_xticks(seed_x, [str(seed) for seed in seeds])
    cost_axis.grid(True, alpha=0.25)
    cost_axis.legend(fontsize=8)

    recall = [
        100.0 * summary["controllers"][label]["mean_event_recall"] for label in labels
    ]
    false_events = [
        summary["controllers"][label]["mean_false_inflations"] for label in labels
    ]
    bars = event_axis.bar(controller_x, recall, color=colors)
    for bar, false_count in zip(bars, false_events, strict=True):
        event_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.0,
            f"false {false_count:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    event_axis.set_title("Causal Event Recall")
    event_axis.set_ylabel("scheduled changes matched within window (%)")
    event_axis.set_ylim(0.0, 112.0)
    event_axis.set_xticks(controller_x, short_labels, rotation=12)
    event_axis.grid(True, axis="y", alpha=0.25)

    comparison = summary["paired_vs_fixed"]["Chi-square CUSUM RLS"]
    fig.suptitle(
        "Online Tracking of Temporary Heterogeneous Actuator Faults\n"
        f"CUSUM wins {comparison['adaptive_wins']}/{comparison['trials']}; "
        f"mean degraded-interval RMSE reduction "
        f"{100.0 * comparison['mean_relative_rmse_reduction']:.1f}%",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def save_temporary_fault_generalization_plot(
    records,
    summary,
    path: str | Path,
    *,
    study_title: str = "Generalization Across Hidden Temporary Actuator Faults",
    headline: str | None = None,
) -> Path:
    """Plot paired adaptation benefits across hidden temporary-fault cases."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fixed_label = "Fixed-covariance RLS"
    label_metadata = {
        "Innovation-threshold RLS": ("Threshold", "#2f9c95"),
        "Recovery-prior threshold RLS": ("Threshold + recovery prior", "#d18f32"),
        "Transient-offset threshold RLS": (
            "Threshold + transient offset",
            "#9b6a35",
        ),
        "Chi-square CUSUM RLS": ("CUSUM", "#3974a8"),
        "CUSUM-triggered probing RLS": ("CUSUM + probes", "#7048a8"),
    }
    labels = tuple(label for label in label_metadata if label in summary["controllers"])
    short_labels = tuple(label_metadata[label][0] for label in labels)
    colors = tuple(label_metadata[label][1] for label in labels)
    seeds = sorted({record["seed"] for record in records})
    by_key = {(record["seed"], record["controller"]): record for record in records}
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.0))
    fault_axis, recovery_axis, cost_axis, event_axis = axes.flat
    case_x = np.arange(len(seeds))

    for label, short, color in zip(labels, short_labels, colors, strict=True):
        fault_delta = [
            by_key[(seed, fixed_label)]["fault_interval_rmse"]
            - by_key[(seed, label)]["fault_interval_rmse"]
            for seed in seeds
        ]
        recovery_delta = [
            by_key[(seed, fixed_label)]["recovery_rmse"]
            - by_key[(seed, label)]["recovery_rmse"]
            for seed in seeds
        ]
        cost_delta = [
            by_key[(seed, label)]["sustained_completion_cost"]
            - by_key[(seed, fixed_label)]["sustained_completion_cost"]
            for seed in seeds
        ]
        fault_axis.plot(case_x, fault_delta, "o-", color=color, label=short)
        recovery_axis.plot(case_x, recovery_delta, "o-", color=color, label=short)
        cost_axis.plot(case_x, cost_delta, "o-", color=color, label=short)

    for axis in (fault_axis, recovery_axis, cost_axis):
        axis.axhline(0.0, color="#555555", linewidth=1, linestyle="--")
        axis.set_xticks(case_x, [str(seed) for seed in seeds])
        axis.set_xlabel("hidden fault case")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    fault_axis.set_title("Degraded-Interval Tracking Benefit")
    fault_axis.set_ylabel("fixed RMSE - adaptive RMSE")
    recovery_axis.set_title("Post-Recovery Tracking Benefit")
    recovery_axis.set_ylabel("fixed RMSE - adaptive RMSE")
    cost_axis.set_title("Task-Cost Tradeoff")
    cost_axis.set_ylabel("adaptive cost - fixed cost")

    recovery_label = next(
        (
            label
            for label in (
                "Recovery-prior threshold RLS",
                "Transient-offset threshold RLS",
            )
            if label in labels
        ),
        None,
    )
    if recovery_label is not None:
        platform_names = tuple(
            by_key[(seeds[0], recovery_label)]["platform_recovery_rmse"]
        )
        platform_delta = [
            np.mean(
                [
                    by_key[(seed, "Innovation-threshold RLS")][
                        "platform_recovery_rmse"
                    ][name]
                    - by_key[(seed, recovery_label)]["platform_recovery_rmse"][name]
                    for seed in seeds
                ]
            )
            for name in platform_names
        ]
        platform_x = np.arange(len(platform_names))
        platform_colors = ("#4978a8", "#5f8f52", "#2f9c95", "#7048a8")
        event_axis.bar(
            platform_x,
            platform_delta,
            color=[
                platform_colors[index % len(platform_colors)]
                for index in range(len(platform_names))
            ],
        )
        event_axis.axhline(0.0, color="#555555", linewidth=1, linestyle="--")
        method_name = (
            "Transient-Offset"
            if recovery_label == "Transient-offset threshold RLS"
            else "Recovery-Prior"
        )
        event_axis.set_title(f"{method_name} Benefit by Platform")
        event_axis.set_ylabel(
            "threshold RMSE - "
            + (
                "transient-offset RMSE"
                if method_name == "Transient-Offset"
                else "recovery-prior RMSE"
            )
        )
        event_axis.set_xticks(
            platform_x,
            [
                {
                    "ground_rover_1": "UGV 1",
                    "ground_rover_2": "UGV 2",
                    "surface_vessel": "USV",
                    "underwater_rov": "ROV",
                }.get(name, name.replace("_", " "))
                for name in platform_names
            ],
        )
        event_axis.grid(True, axis="y", alpha=0.25)
    else:
        all_labels = (fixed_label, *labels)
        all_short = ("Fixed", *short_labels)
        all_colors = ("#b66a50", *colors)
        label_x = np.arange(len(all_labels))
        recall = [
            100.0 * summary["controllers"][label]["mean_event_recall"]
            for label in all_labels
        ]
        bars = event_axis.bar(label_x, recall, color=all_colors)
        for bar, label in zip(bars, all_labels, strict=True):
            false_count = summary["controllers"][label]["mean_false_inflations"]
            event_axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2.0,
                f"false {false_count:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        event_axis.set_title("Change-Event Coverage")
        event_axis.set_ylabel("events matched within causal window (%)")
        event_axis.set_ylim(0.0, 112.0)
        event_axis.set_xticks(label_x, all_short, rotation=12)
        event_axis.grid(True, axis="y", alpha=0.25)

    threshold = summary["paired_vs_fixed"]["Innovation-threshold RLS"]
    threshold_clean = summary["controllers"]["Innovation-threshold RLS"][
        "fallback_free_rate"
    ]
    fig.suptitle(
        f"{study_title}\n"
        + (
            headline
            if headline is not None
            else (
                f"threshold-RLS wins {threshold['adaptive_wins']}/"
                f"{threshold['trials']}; mean RMSE reduction "
                f"{100.0 * threshold['mean_relative_rmse_reduction']:.1f}%; "
                f"fallback-free {100.0 * threshold_clean:.0f}%"
            )
        ),
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
