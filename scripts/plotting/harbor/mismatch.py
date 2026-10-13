"""Plotting / harbor / faults."""

from __future__ import annotations

from scripts.plotting._backend import configure_matplotlib

configure_matplotlib()

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scripts.plotting.harbor.common import (
    _agent_colors,
    _display_name,
    _draw_harbor_map,
)

if TYPE_CHECKING:
    from scripts.configuration.harbor_simulation import HarborDisturbanceConfig
    from scripts.contracts.harbor import HarborAgent


def save_model_mismatch_diagnostics(
    trials,
    agents: list[HarborAgent],
    simulation_config,
    disturbance: HarborDisturbanceConfig,
    path: str | Path,
) -> Path:
    """Show robust paths, terminal error, and both local estimate histories."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18.0, 8.2))
    grid = fig.add_gridspec(
        2, 3, width_ratios=(1.12, 1.0, 0.9), hspace=0.34, wspace=0.27
    )
    map_axis = fig.add_subplot(grid[:, 0])
    error_axis = fig.add_subplot(grid[0, 1])
    residual_axis = fig.add_subplot(grid[1, 1])
    effectiveness_axis = fig.add_subplot(grid[:, 2])
    colors = _agent_colors(agents)
    nominal = next(trial for trial in trials if trial.label == "Nominal MPC")
    adaptive = next(trial for trial in trials if trial.label == "Joint-adaptive LMPC")
    _draw_harbor_map(map_axis, simulation_config)
    for agent in agents:
        color = colors[agent.name]
        for trial, style, alpha in ((nominal, "--", 0.5), (adaptive, "-", 1.0)):
            positions = trial.result.positions[agent.name]
            map_axis.plot(
                positions[:, 0],
                positions[:, 1],
                color=color,
                linestyle=style,
                linewidth=1.7 if style == "--" else 2.5,
                alpha=alpha,
                label=(
                    f"{_display_name(agent)}: {trial.label}"
                    if agent.model.kind in {"usv", "rov"}
                    else None
                ),
            )
        map_axis.scatter(agent.goal[0], agent.goal[1], color=color, marker="*", s=120)
    current = np.asarray(disturbance.water_current, dtype=float)
    map_axis.quiver(
        -4.4,
        -4.3,
        6.0 * current[0],
        6.0 * current[1],
        color="#222222",
        width=0.007,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        label="hidden water current",
    )
    map_axis.set_title("Executed Paths Under the Same Hidden Plant Mismatch")
    map_axis.set_xlabel("x [m]")
    map_axis.set_ylabel("y [m]")
    map_axis.set_aspect("equal", adjustable="box")
    map_axis.grid(True, alpha=0.25)
    map_axis.legend(loc="upper left", fontsize=8)

    marine = [agent for agent in agents if agent.model.kind in {"usv", "rov"}]
    labels = [
        trial.label.replace("-adaptive ", " ").replace("Nominal MPC", "Nominal")
        for trial in trials
    ]
    x_values = np.arange(len(labels))
    width = 0.34
    for index, agent in enumerate(marine):
        errors = [trial.result.final_goal_errors[agent.name] for trial in trials]
        bars = error_axis.bar(
            x_values + (index - 0.5) * width,
            errors,
            width,
            color=colors[agent.name],
            label=_display_name(agent),
        )
        for trial, bar in zip(trials, bars, strict=True):
            if not trial.valid:
                bar.set_hatch("//")
                bar.set_edgecolor("#333333")
    error_axis.axhline(
        simulation_config.goal_tolerance,
        color="#555555",
        linestyle=":",
        label="position tolerance",
    )
    error_axis.set_title(
        f"Terminal Error After {disturbance.evaluation_hold_steps}-Step Hold"
    )
    error_axis.set_ylabel("position error [m]")
    error_axis.set_xticks(x_values, labels, rotation=12, ha="right")
    error_axis.grid(True, axis="y", alpha=0.25)
    error_axis.legend(fontsize=8)

    component_styles = ((0, "x", "-"), (1, "y", "--"), (2, "z", ":"))
    for agent in marine:
        history = adaptive.residual_history[agent.name]
        for component, component_name, style in component_styles:
            if agent.model.kind == "usv" and component == 2:
                continue
            residual_axis.plot(
                history[:, component],
                color=colors[agent.name],
                linestyle=style,
                linewidth=1.8,
                label=f"{_display_name(agent)} {component_name}",
            )
            truth = current[component]
            residual_axis.axhline(truth, color="#444444", linestyle=style, alpha=0.3)
    residual_axis.set_title("Locally Estimated Position-Velocity Residual")
    residual_axis.set_xlabel("control update")
    residual_axis.set_ylabel("estimated residual [m/s]")
    residual_axis.grid(True, alpha=0.25)
    residual_axis.legend(ncol=2, fontsize=8)

    true_effectiveness = {
        agent.name: disturbance.effectiveness(agent.model, agent.name)
        for agent in agents
    }
    for agent in agents:
        history = adaptive.effectiveness_history[agent.name]
        effectiveness_axis.plot(
            np.mean(history, axis=1),
            color=colors[agent.name],
            linewidth=2.0,
            label=f"{_display_name(agent)} estimate",
        )
        effectiveness_axis.axhline(
            float(np.mean(true_effectiveness[agent.name])),
            color=colors[agent.name],
            linestyle=":",
            linewidth=1.3,
            alpha=0.7,
        )
    effectiveness_axis.set_title("Locally Estimated Control Effectiveness")
    effectiveness_axis.set_xlabel("control update")
    effectiveness_axis.set_ylabel("applied / commanded control")
    effectiveness_axis.set_ylim(
        min(float(np.min(value)) for value in true_effectiveness.values()) - 0.04,
        1.03,
    )
    effectiveness_axis.grid(True, alpha=0.25)
    effectiveness_axis.legend(fontsize=8)
    effectiveness_axis.text(
        0.03,
        0.03,
        "dotted lines: hidden plant values\nhatched bars: invalid trial",
        transform=effectiveness_axis.transAxes,
        fontsize=8,
        color="#444444",
    )
    fig.suptitle(
        "Distributed Harbor MPC Under Current and Actuator Mismatch", fontsize=15
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
