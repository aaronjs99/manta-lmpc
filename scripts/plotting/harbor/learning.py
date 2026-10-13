"""Plotting / harbor / learning."""

from __future__ import annotations

from scripts.plotting._backend import configure_matplotlib

configure_matplotlib()

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scripts.plotting.harbor.common import _agent_colors, _draw_harbor_map, _draw_paths

if TYPE_CHECKING:
    from scripts.contracts.harbor import HarborAgent


def save_harbor_learning_progress(
    iterations,
    agents: list[HarborAgent],
    simulation_config,
    path: str | Path,
) -> Path:
    """Save one compact trajectory, performance, and solver-health dashboard."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = len(iterations)
    fig = plt.figure(figsize=(5.4 * count, 9.0))
    grid = fig.add_gridspec(2, count, height_ratios=(2.5, 1.0))
    colors = _agent_colors(agents)
    display_labels = {
        "guidance_iter_0": "Guidance seed",
        "distributed_mpc": "Distributed MPC",
    }
    for index, iteration in enumerate(iterations):
        axis = fig.add_subplot(grid[0, index])
        _draw_harbor_map(axis, simulation_config)
        if iteration.label.startswith("distributed_lmpc_"):
            prior = next(
                (
                    candidate
                    for candidate in reversed(iterations[:index])
                    if candidate.admitted
                ),
                None,
            )
            if prior is not None:
                for agent_index, agent in enumerate(agents):
                    safe_positions = prior.result.positions[agent.name]
                    axis.plot(
                        safe_positions[:, 0],
                        safe_positions[:, 1],
                        color=colors[agent.name],
                        linewidth=1.4,
                        linestyle=":",
                        alpha=0.55,
                        label=(
                            "prior admitted safe trajectory"
                            if agent_index == 0
                            else None
                        ),
                    )
        _draw_paths(axis, iteration.result, agents, colors)
        label = display_labels.get(
            iteration.label,
            iteration.label.replace("distributed_lmpc_", "Distributed LMPC "),
        )
        axis.set_title(
            f"{label}\nsteps={iteration.completion_step_sum}, "
            f"min={iteration.result.min_pairwise_distance:.3f} m, "
            f"{'admitted' if iteration.admitted else 'rejected'}"
        )
        if index == count - 1:
            axis.legend(loc="upper left", fontsize=7, framealpha=0.92)

    labels = [
        display_labels.get(
            item.label,
            item.label.replace("distributed_lmpc_", "LMPC "),
        )
        for item in iterations
    ]
    x = np.arange(count)
    cost_axis = fig.add_subplot(grid[1, : max(1, count // 2)])
    costs = [item.completion_step_sum for item in iterations]
    bars = cost_axis.bar(
        x,
        costs,
        color=["#377eb8" if item.admitted else "#999999" for item in iterations],
        hatch=[None if item.admitted else "//" for item in iterations],
    )
    cost_axis.bar_label(bars, padding=3)
    baseline = costs[0]
    for index, cost in enumerate(costs[1:], start=1):
        if iterations[index].admitted:
            improvement = 100.0 * (baseline - cost) / baseline
            annotation = f"{improvement:.1f}% faster"
        else:
            annotation = "rejected"
        cost_axis.text(
            index,
            cost * 0.55,
            annotation,
            ha="center",
            color="white",
        )
    cost_axis.set_xticks(x, labels=labels, rotation=12, ha="right")
    cost_axis.set_ylabel("sum of first-goal steps")
    cost_axis.set_title("Task Completion Cost (lower is better)")
    cost_axis.grid(axis="y", alpha=0.25)

    health_axis = fig.add_subplot(grid[1, max(1, count // 2) :])
    separations = [item.result.min_pairwise_distance for item in iterations]
    health_bars = health_axis.bar(
        x - 0.18,
        separations,
        width=0.36,
        color=["#4daf4a" if item.admitted else "#999999" for item in iterations],
        hatch=[None if item.admitted else "//" for item in iterations],
    )
    health_axis.bar_label(health_bars, fmt="%.3f", padding=3)
    health_axis.set_ylabel("minimum swept separation [m]")
    health_axis.set_xticks(x, labels=labels, rotation=12, ha="right")
    health_axis.set_title("Safety and Solver Reliability")
    fallback_axis = health_axis.twinx()
    fallbacks = [item.solver_fallbacks for item in iterations]
    fallback_bars = fallback_axis.bar(
        x + 0.18, fallbacks, width=0.36, color="#e41a1c", alpha=0.75
    )
    fallback_axis.bar_label(fallback_bars, padding=3)
    fallback_axis.set_ylabel("solver fallbacks")
    fallback_axis.set_ylim(0.0, max(1.0, max(fallbacks, default=0) + 0.5))
    health_axis.grid(axis="y", alpha=0.25)

    fig.suptitle("Distributed Harbor LMPC Research Progress", fontsize=17)
    fig.subplots_adjust(top=0.91, bottom=0.12, hspace=0.35, wspace=0.28)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def save_network_robustness_heatmap(
    records: list[dict[str, float | int]], path: str | Path
) -> Path:
    """Save safety-rate and completion-cost heatmaps for a network sweep."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    delays = sorted({int(record["delay_steps"]) for record in records})
    dropouts = sorted({float(record["dropout_probability"]) for record in records})
    safety = np.empty((len(dropouts), len(delays)))
    cost = np.empty_like(safety)
    completion = np.empty_like(safety)
    lookup = {
        (int(record["delay_steps"]), float(record["dropout_probability"])): record
        for record in records
    }
    for row, dropout in enumerate(dropouts):
        for column, delay in enumerate(delays):
            record = lookup[(delay, dropout)]
            safety[row, column] = float(record["safe_rate"])
            cost[row, column] = float(record["mean_completion_step_sum"])
            completion[row, column] = float(record["completion_rate"])

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8))
    images = [
        axes[0].imshow(safety, vmin=0.0, vmax=1.0, cmap="RdYlGn"),
        axes[1].imshow(cost, cmap="viridis_r"),
    ]
    titles = ["Safe Trial Rate", "Mean Completion Step Sum"]
    formats = [lambda value: f"{value:.0%}", lambda value: f"{value:.0f}"]
    for axis, values, image, title, formatter in zip(
        axes, (safety, cost), images, titles, formats, strict=True
    ):
        axis.set_title(title)
        axis.set_xticks(range(len(delays)), labels=delays)
        axis.set_yticks(range(len(dropouts)), labels=[f"{v:.0%}" for v in dropouts])
        axis.set_xlabel("communication delay [steps]")
        axis.set_ylabel("packet dropout probability")
        for row in range(len(dropouts)):
            for column in range(len(delays)):
                label = formatter(values[row, column])
                if axis is axes[1]:
                    flags = []
                    if safety[row, column] < 1.0:
                        flags.append(f"safe {safety[row, column]:.0%}")
                    if completion[row, column] < 1.0:
                        flags.append(f"done {completion[row, column]:.0%}")
                    if flags:
                        label = "\n".join((label, *flags))
                text_color = "black"
                if axis is axes[1]:
                    value_range = np.ptp(values)
                    normalized = (
                        0.0
                        if value_range <= np.finfo(float).eps
                        else (values[row, column] - np.min(values)) / value_range
                    )
                    text_color = "white" if normalized > 0.45 else "black"
                axis.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7.5 if axis is axes[1] else 8,
                )
        fig.colorbar(image, ax=axis, shrink=0.82)
    fig.suptitle("ETA-Priority Harbor Communication Robustness", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def save_horizon_efficiency(
    records: list[dict[str, float | int | str | bool]], path: str | Path
) -> Path:
    """Save task-performance and solve-time curves for matched horizons."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    horizons = sorted({int(record["prediction_horizon"]) for record in records})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    styles = {
        "MPC": ("#377eb8", "o"),
        "LMPC": ("#e41a1c", "s"),
    }
    for controller, (color, marker) in styles.items():
        selected = {
            int(record["prediction_horizon"]): record
            for record in records
            if record["controller"] == controller
        }
        costs = [
            float(selected[horizon]["completion_step_sum"]) for horizon in horizons
        ]
        times = [float(selected[horizon]["mean_solve_time_ms"]) for horizon in horizons]
        axes[0].plot(
            horizons,
            costs,
            color=color,
            marker=marker,
            linewidth=2.2,
            label=controller,
        )
        axes[1].plot(
            horizons,
            times,
            color=color,
            marker=marker,
            linewidth=2.2,
            label=controller,
        )
        for horizon, cost in zip(horizons, costs, strict=True):
            record = selected[horizon]
            suffix = "" if record["complete"] else "\nincomplete"
            axes[0].annotate(
                f"{cost:.0f}{suffix}",
                (horizon, cost),
                xytext=(0, 8 if controller == "MPC" else -16),
                textcoords="offset points",
                ha="center",
                color=color,
                fontsize=8,
            )
    axes[0].set_title("Completion Cost and Liveness")
    axes[0].set_ylabel("sum of first-goal steps")
    axes[1].set_title("Mean Per-Agent NLP Solve Latency")
    axes[1].set_ylabel("milliseconds per solve")
    for axis in axes:
        axis.set_xlabel("prediction horizon N")
        axis.set_xticks(horizons)
        axis.margins(y=0.12)
        axis.grid(True, alpha=0.28)
        axis.legend()
    fig.suptitle("Horizon-Dependent MPC and LMPC Performance", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
