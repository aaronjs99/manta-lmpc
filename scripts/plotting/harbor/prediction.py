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


def save_prediction_ablation_plot(records, summary, path: str | Path) -> Path:
    """Plot matched constant-velocity and goal-bounded prediction outcomes."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    seeds = sorted({record["seed"] for record in records})
    labels = ("Constant-velocity prediction", "Goal-bounded prediction")
    short_labels = ("Constant velocity", "Goal bounded")
    colors = ("#b66a50", "#2f9c95")
    by_key = {(record["seed"], record["controller"]): record for record in records}
    x = np.arange(len(seeds))
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    fallback_axis, cost_axis, rmse_axis = axes
    for label, short, color in zip(labels, short_labels, colors, strict=True):
        fallbacks = [by_key[(seed, label)]["solver_fallbacks"] for seed in seeds]
        costs = [by_key[(seed, label)]["sustained_completion_cost"] for seed in seeds]
        errors = [by_key[(seed, label)]["effectiveness_rmse"] for seed in seeds]
        fallback_axis.plot(x, fallbacks, "o-", color=color, linewidth=2, label=short)
        cost_axis.plot(x, costs, "o-", color=color, linewidth=2, label=short)
        rmse_axis.plot(x, errors, "o-", color=color, linewidth=2, label=short)
    fallback_axis.set_title("Recoverable Solver Fallbacks")
    fallback_axis.set_ylabel("count")
    cost_axis.set_title("Sustained-Completion Cost")
    cost_axis.set_ylabel("first-hit sum with horizon penalty")
    rmse_axis.set_title("Actuator-Effectiveness Estimation")
    rmse_axis.set_ylabel("channel RMSE")
    for axis in axes:
        axis.set_xticks(x, [str(seed) for seed in seeds])
        axis.set_xlabel("fault ensemble seed")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    comparison = summary["paired_goal_bounded_vs_constant_velocity"]
    fig.suptitle(
        "Intent-Bounded Peer Prediction Under Noisy Distributed MPC\n"
        f"fallbacks {comparison['constant_velocity_fallbacks']} -> "
        f"{comparison['goal_bounded_fallbacks']}; "
        f"complete and collision-safe pairs "
        f"{comparison['complete_safe_pairs']}/{comparison['trials']}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
