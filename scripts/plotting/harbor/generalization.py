"""Plotting / harbor / faults."""

from __future__ import annotations

from scripts.plotting._backend import configure_matplotlib

configure_matplotlib()

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    pass


def save_fault_generalization_plot(
    records,
    summary,
    path: str | Path,
    *,
    noisy_observations: bool = False,
) -> Path:
    """Plot paired identification and task metrics across hidden fault draws."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    seeds = sorted({record["seed"] for record in records})
    available = {record["controller"] for record in records}
    if "Recursive diagonal MPC" in available:
        labels = (
            "Instantaneous diagonal MPC",
            "Recursive diagonal MPC",
            "Recursive one-pass MPC",
            "Recursive information-aware MPC",
        )
        short_labels = ("Instantaneous", "Recursive", "One-pass", "Information-aware")
        colors = ("#b66a50", "#58a66f", "#2f9c95", "#7048a8")
        one_pass_label = labels[2]
        information_label = labels[3]
    else:
        labels = (
            "Passive diagonal MPC",
            "One-pass active MPC",
            "Information-aware MPC",
        )
        short_labels = ("Passive", "One-pass", "Information-aware")
        colors = ("#58a66f", "#2f9c95", "#7048a8")
        one_pass_label = labels[1]
        information_label = labels[2]
    by_key = {(record["seed"], record["controller"]): record for record in records}

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.0))
    rmse_axis, paired_axis, cost_axis, coverage_axis = axes.flat
    x = np.arange(len(seeds))
    for label, short, color in zip(labels, short_labels, colors, strict=True):
        values = [by_key[(seed, label)]["effectiveness_rmse"] for seed in seeds]
        rmse_axis.plot(x, values, marker="o", linewidth=2.0, color=color, label=short)
    rmse_axis.set_title("Actuator-Effectiveness Estimation")
    rmse_axis.set_ylabel("channel RMSE")
    rmse_axis.set_xticks(x, [str(seed) for seed in seeds])
    rmse_axis.set_xlabel("fault ensemble seed")
    rmse_axis.grid(True, alpha=0.25)
    rmse_axis.legend()

    one_pass = np.asarray(
        [by_key[(seed, one_pass_label)]["effectiveness_rmse"] for seed in seeds]
    )
    information = np.asarray(
        [by_key[(seed, information_label)]["effectiveness_rmse"] for seed in seeds]
    )
    lower = min(float(np.min(one_pass)), float(np.min(information))) * 0.9
    upper = max(float(np.max(one_pass)), float(np.max(information))) * 1.1
    paired_axis.plot([lower, upper], [lower, upper], "--", color="#666666")
    paired_axis.scatter(one_pass, information, color="#7048a8", s=55)
    for seed, x_value, y_value in zip(seeds, one_pass, information, strict=True):
        paired_axis.annotate(
            str(seed),
            (x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    paired_axis.set_xlim(lower, upper)
    paired_axis.set_ylim(lower, upper)
    paired_axis.set_aspect("equal", adjustable="box")
    paired_axis.set_title("Equal-Budget Paired Comparison")
    paired_axis.set_xlabel("one-pass RMSE")
    paired_axis.set_ylabel("information-aware RMSE")
    paired_axis.grid(True, alpha=0.25)

    for label, short, color in zip(labels, short_labels, colors, strict=True):
        values = [by_key[(seed, label)]["sustained_completion_cost"] for seed in seeds]
        cost_axis.plot(x, values, marker="o", linewidth=2.0, color=color, label=short)
        invalid = [
            index
            for index, seed in enumerate(seeds)
            if not by_key[(seed, label)]["valid"]
        ]
        if invalid:
            cost_axis.scatter(
                invalid,
                [values[index] for index in invalid],
                marker="x",
                s=90,
                linewidths=2.0,
                color="#111111",
                zorder=5,
                label="solver fallback" if label == labels[0] else None,
            )
    cost_axis.set_title("Sustained-Completion Cost")
    cost_axis.set_ylabel("first-hit sum with horizon penalty")
    cost_axis.set_xticks(x, [str(seed) for seed in seeds])
    cost_axis.set_xlabel("fault ensemble seed")
    cost_axis.grid(True, alpha=0.25)
    cost_axis.legend(fontsize=8, ncol=2)

    first_hidden = by_key[(seeds[0], labels[0])]["hidden_effectiveness"]
    platform_labels = {
        "ground_rover_1": "RobEn",
        "ground_rover_2": "Inspector-Gadget",
        "surface_vessel": "Heron",
        "underwater_rov": "BlueROV2",
    }
    channel_labels = [
        f"{platform_labels.get(name, name)}:{index + 1}"
        for name, values in first_hidden.items()
        for index in range(len(values))
    ]
    hidden = np.asarray(
        [
            np.concatenate(
                [
                    np.asarray(values, dtype=float)
                    for values in by_key[(seed, labels[0])][
                        "hidden_effectiveness"
                    ].values()
                ]
            )
            for seed in seeds
        ]
    )
    image = coverage_axis.imshow(
        hidden, cmap="viridis", aspect="auto", vmin=0.5, vmax=1.0
    )
    coverage_axis.set_title("Stratified Hidden Effectiveness Coverage")
    coverage_axis.set_yticks(x, [str(seed) for seed in seeds])
    coverage_axis.set_ylabel("fault ensemble seed")
    coverage_axis.set_xticks(
        np.arange(len(channel_labels)),
        channel_labels,
        rotation=45,
        ha="right",
        fontsize=7,
    )
    fig.colorbar(image, ax=coverage_axis, fraction=0.03, pad=0.02)

    if noisy_observations:
        paired = summary["recursive_vs_instantaneous"]
        ci_low, ci_high = paired["paired_mean_reduction_bootstrap_95_ci"]
        title = (
            "Robust Recursive Identification Across Stratified Actuator Faults "
            "with Onboard Sensor Noise\n"
            f"wins {paired['recursive_wins']}/{paired['trials']}; "
            f"mean paired RMSE reduction={paired['mean_rmse_reduction']:.4f} "
            f"(bootstrap 95% CI [{ci_low:.4f}, {ci_high:.4f}])"
        )
    else:
        paired = summary["equal_budget_information_vs_one_pass"]
        ci_low, ci_high = paired["paired_mean_reduction_bootstrap_95_ci"]
        title = (
            "Information-Aware Identification Across Stratified Actuator Faults\n"
            f"wins {paired['information_wins']}/{paired['trials']}; "
            f"mean paired RMSE reduction={paired['mean_rmse_reduction']:.4f} "
            f"(bootstrap 95% CI [{ci_low:.4f}, {ci_high:.4f}])"
        )
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
