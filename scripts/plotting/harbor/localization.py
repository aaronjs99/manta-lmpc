"""Range-localization diagnostic figures, separate from experiment execution."""

from pathlib import Path

import numpy as np
from scripts.plotting._backend import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt


def save_localization_diagnostics(records, histories, agents, path: Path) -> None:
    """Render the scored localization comparison without running a simulation."""
    names = [agent.name for agent in agents]
    short_names = ["UGV 1", "UGV 2", "Heron", "BlueROV2"]
    x = np.arange(len(names))
    width = 0.19
    colors = ("#667085", "#1976b9", "#2a9d8f", "#c8553d")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    for index, (record, color) in enumerate(zip(records, colors)):
        offset = (index - 1.5) * width
        axes[0, 0].bar(
            x + offset,
            [record["position_rmse"][name] for name in names],
            width,
            color=color,
            label=record["mode"],
        )
        axes[1, 0].bar(
            x + offset,
            [record["observable_fraction"][name] for name in names],
            width,
            color=color,
        )
    for label, color in zip(histories, colors):
        axes[0, 1].plot(
            histories[label]["underwater_rov"], color=color, label=label, linewidth=1.8
        )
        axes[1, 1].plot(
            histories[label]["surface_vessel"], color=color, label=label, linewidth=1.8
        )
    axes[0, 0].set_title("Position Estimate RMSE")
    axes[0, 0].set_ylabel("RMSE [m]")
    axes[1, 0].set_title("Full Observability Fraction")
    axes[1, 0].set_ylabel("fraction of simulation")
    axes[1, 0].set_ylim(0.0, 1.05)
    for axis in (axes[0, 0], axes[1, 0]):
        axis.set_xticks(x, short_names)
    axes[0, 1].set_title("BlueROV2 Position Error")
    axes[1, 1].set_title("Heron Position Error")
    for axis in (axes[0, 1], axes[1, 1]):
        axis.set_xlabel("simulation step")
        axis.set_ylabel("error [m]")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(
        "Range-Aided Harbor Localization and SLAM Development\n"
        "matched dynamics, odometry drift, range noise, and dropout",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
