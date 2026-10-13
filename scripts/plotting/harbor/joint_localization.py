"""Plotting / harbor / joint localization."""

from __future__ import annotations

from scripts.plotting._backend import configure_matplotlib

configure_matplotlib()

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scripts.configuration.study_labels import POLICIES

if TYPE_CHECKING:
    from pathlib import Path


def _save_plot(records: list[dict], path: Path) -> None:
    seeds = sorted({record["seed"] for record in records})
    by_key = {(record["seed"], record["policy"]): record for record in records}
    colors = ("#667085", "#c8553d", "#1976b9", "#2a9d8f", "#8e5ea2")
    x = np.arange(len(seeds))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharex=True)
    present = tuple(
        policy for policy in POLICIES if any(r["policy"] == policy for r in records)
    )
    for policy, color in zip(present, colors):
        selected = [by_key[(seed, policy)] for seed in seeds]
        axes[0, 0].plot(
            x,
            [record["all_goals_reached"] for record in selected],
            "o-",
            color=color,
            label=policy,
        )
        axes[0, 1].plot(
            x,
            [record["solver_fallbacks"] for record in selected],
            "o-",
            color=color,
            label=policy,
        )
        axes[1, 0].plot(
            x,
            [record["position_rmse"]["underwater_rov"] for record in selected],
            "o-",
            color=color,
            label=policy,
        )
        axes[1, 1].plot(
            x,
            [record["current_rmse"]["surface_vessel"] for record in selected],
            "o-",
            color=color,
            label=policy,
        )
    axes[0, 0].set_title("True Sustained Task Completion")
    axes[0, 0].set_yticks((0, 1), ("no", "yes"))
    axes[0, 1].set_title("Distributed MPC Fallbacks")
    axes[1, 0].set_title("BlueROV2 Position-Estimate RMSE")
    axes[1, 0].set_ylabel("RMSE [m]")
    axes[1, 1].set_title("Heron Current-Estimate RMSE")
    axes[1, 1].set_ylabel("RMSE [m/s]")
    for axis in axes.flat:
        axis.set_xticks(x, [str(seed) for seed in seeds])
        axis.set_xlabel("matched joint-uncertainty case")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(
        "Estimated-State Distributed MPC Under Localization, Current, and Actuator Faults",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
