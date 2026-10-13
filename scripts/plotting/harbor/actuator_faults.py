"""Plotting / harbor / faults."""

from __future__ import annotations

from scripts.plotting._backend import configure_matplotlib

configure_matplotlib()

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scripts.metrics.harbor.estimation import (
    _effectiveness_rmse_history,
    _flatten_effectiveness,
    _maximum_information_std_history,
)
from scripts.plotting.harbor.common import (
    _agent_colors,
    _control_channel_labels,
    _draw_harbor_map,
    _short_platform_name,
)

if TYPE_CHECKING:
    from scripts.configuration.harbor_simulation import HarborDisturbanceConfig
    from scripts.contracts.harbor import HarborAgent


def save_actuator_fault_diagnostics(
    trials,
    agents: list[HarborAgent],
    simulation_config,
    disturbance: HarborDisturbanceConfig,
    path: str | Path,
) -> Path:
    """Show asymmetric-fault paths, cost, convergence, and channel estimates."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(23.0, 8.4))
    grid = fig.add_gridspec(
        2, 3, width_ratios=(1.05, 0.86, 1.20), hspace=0.34, wspace=0.42
    )
    map_axis = fig.add_subplot(grid[:, 0])
    cost_axis = fig.add_subplot(grid[0, 1])
    rmse_axis = fig.add_subplot(grid[1, 1])
    matrix_axis = fig.add_subplot(grid[:, 2])
    colors = _agent_colors(agents)
    scalar = next(trial for trial in trials if trial.label == "Scalar-adaptive MPC")
    diagonal = next(trial for trial in trials if trial.label == "Diagonal-adaptive MPC")
    active_diagonal = next(
        trial for trial in trials if trial.label == "Active diagonal MPC"
    )
    information_aware = next(
        trial for trial in trials if trial.label == "Information-aware MPC"
    )
    _draw_harbor_map(map_axis, simulation_config)
    for agent in agents:
        for trial, style, alpha in (
            (diagonal, "--", 0.55),
            (information_aware, "-", 1.0),
        ):
            positions = trial.result.positions[agent.name]
            map_axis.plot(
                positions[:, 0],
                positions[:, 1],
                color=colors[agent.name],
                linestyle=style,
                linewidth=1.7 if style == "--" else 2.5,
                alpha=alpha,
            )
        map_axis.scatter(
            agent.goal[0], agent.goal[1], color=colors[agent.name], marker="*", s=115
        )
    map_axis.set_title("Executed Paths Under Identical Asymmetric Faults")
    map_axis.set_xlabel("x [m]")
    map_axis.set_ylabel("y [m]")
    map_axis.set_aspect("equal", adjustable="box")
    map_axis.grid(True, alpha=0.25)
    platform_handles = [
        Line2D(
            [0],
            [0],
            color=colors[agent.name],
            linewidth=2.5,
            label=_short_platform_name(agent),
        )
        for agent in agents
    ]
    style_handles = [
        Line2D([0], [0], color="#333333", linestyle="--", label="passive diagonal"),
        Line2D([0], [0], color="#333333", linestyle="-", label="information-aware"),
    ]
    map_axis.legend(
        handles=[*platform_handles, *style_handles],
        loc="upper left",
        fontsize=7,
        ncol=2,
    )

    labels = [
        {
            "Nominal MPC": "Nominal",
            "Scalar-adaptive MPC": "Scalar",
            "Diagonal-adaptive MPC": "Passive diag.",
            "Active diagonal MPC": "Active diag.",
            "One-pass active MPC": "One-pass",
            "Information-aware MPC": "Info-aware",
            "Retained diagonal LMPC": "Passive LMPC",
            "Retained active-ID LMPC": "Active-ID LMPC",
            "Retained information-ID LMPC": "Info-ID LMPC",
        }[trial.label]
        for trial in trials
    ]
    bars = cost_axis.bar(
        np.arange(len(trials)),
        [trial.completion_step_sum for trial in trials],
        color=[
            "#8c8c8c",
            "#d8a02b",
            "#58a66f",
            "#167a53",
            "#2f9c95",
            "#7048a8",
            "#6598c4",
            "#205f9c",
            "#463080",
        ],
    )
    for trial, bar in zip(trials, bars, strict=True):
        cost_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(trial.completion_step_sum),
            ha="center",
            va="bottom",
            fontsize=8,
        )
        if not trial.valid:
            bar.set_hatch("//")
            bar.set_edgecolor("#333333")
    cost_axis.set_title("Completion Cost")
    cost_axis.set_ylabel("sum of first-goal steps")
    cost_axis.set_xticks(np.arange(len(labels)), labels, rotation=22, ha="right")
    cost_axis.grid(True, axis="y", alpha=0.25)

    adaptive_trials = [trial for trial in trials if trial.label != "Nominal MPC"]
    for trial, color in zip(
        adaptive_trials,
        (
            "#d8a02b",
            "#58a66f",
            "#167a53",
            "#2f9c95",
            "#7048a8",
            "#6598c4",
            "#205f9c",
            "#463080",
        ),
        strict=True,
    ):
        errors = _effectiveness_rmse_history(trial, agents, disturbance)
        rmse_axis.plot(errors, color=color, linewidth=2.0, label=trial.label)
    rmse_axis.set_title("Local Effectiveness Estimation Error")
    rmse_axis.set_xlabel("control update")
    rmse_axis.set_ylabel("channel RMSE")
    rmse_axis.grid(True, alpha=0.25)
    uncertainty = _maximum_information_std_history(information_aware, agents)
    uncertainty_axis = rmse_axis.twinx()
    uncertainty_axis.plot(
        uncertainty,
        color="#111111",
        linewidth=1.5,
        linestyle=":",
        label="info-aware max posterior std",
    )
    uncertainty_axis.set_ylabel("max posterior std")
    handles, legends = rmse_axis.get_legend_handles_labels()
    extra_handles, extra_legends = uncertainty_axis.get_legend_handles_labels()
    rmse_axis.legend(handles + extra_handles, legends + extra_legends, fontsize=7)

    row_labels = []
    truth_values = []
    for agent in agents:
        truth = disturbance.effectiveness(agent.model, agent.name)
        for channel, value in zip(_control_channel_labels(agent), truth, strict=True):
            row_labels.append(f"{_short_platform_name(agent)} {channel}")
            truth_values.append(value)
    columns = [
        ("hidden", np.asarray(truth_values)),
        (
            "scalar MPC",
            _flatten_effectiveness(scalar.final_effectiveness_estimates, agents),
        ),
        (
            "diagonal MPC",
            _flatten_effectiveness(diagonal.final_effectiveness_estimates, agents),
        ),
        (
            "active MPC",
            _flatten_effectiveness(
                active_diagonal.final_effectiveness_estimates, agents
            ),
        ),
        (
            "one-pass MPC",
            _flatten_effectiveness(
                next(
                    trial for trial in trials if trial.label == "One-pass active MPC"
                ).final_effectiveness_estimates,
                agents,
            ),
        ),
        (
            "info MPC",
            _flatten_effectiveness(
                information_aware.final_effectiveness_estimates, agents
            ),
        ),
        (
            "diagonal LMPC",
            _flatten_effectiveness(
                next(
                    trial for trial in trials if trial.label == "Retained diagonal LMPC"
                ).final_effectiveness_estimates,
                agents,
            ),
        ),
        (
            "active LMPC",
            _flatten_effectiveness(
                next(
                    trial
                    for trial in trials
                    if trial.label == "Retained active-ID LMPC"
                ).final_effectiveness_estimates,
                agents,
            ),
        ),
        (
            "info LMPC",
            _flatten_effectiveness(
                next(
                    trial
                    for trial in trials
                    if trial.label == "Retained information-ID LMPC"
                ).final_effectiveness_estimates,
                agents,
            ),
        ),
    ]
    matrix = np.column_stack([values for _, values in columns])
    image = matrix_axis.imshow(
        matrix,
        cmap="viridis",
        vmin=min(0.5, float(np.min(matrix))),
        vmax=1.0,
        aspect="auto",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            matrix_axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.78 else "#111111",
                fontsize=8,
            )
    matrix_axis.set_title("Hidden and Final Channel Effectiveness")
    matrix_axis.set_xticks(
        np.arange(len(columns)), [label for label, _ in columns], rotation=12
    )
    matrix_axis.set_yticks(np.arange(len(row_labels)), row_labels)
    fig.colorbar(image, ax=matrix_axis, fraction=0.045, pad=0.03)
    fig.suptitle(
        "Passive, Round-Robin, and Information-Aware Fault Identification",
        fontsize=15,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
