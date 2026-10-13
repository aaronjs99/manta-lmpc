"""Plotting / harbor / common."""

from __future__ import annotations

from scripts.plotting._backend import configure_matplotlib

configure_matplotlib()

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from scripts.contracts.harbor import HarborAgent, HarborResult


_PALETTES = {
    "ugv": ("#248f4b", "#82a832", "#146b3a"),
    "usv": ("#1f77b4", "#17a2b8"),
    "rov": ("#8e5bb7", "#6f42c1"),
}


_LABELS = {"ugv": "UGV", "usv": "USV", "rov": "ROV"}


def _control_channel_labels(agent: HarborAgent) -> tuple[str, ...]:
    if agent.model.kind == "ugv":
        return ("left drive", "right drive")
    if agent.model.kind == "usv":
        return ("port jet", "starboard jet")
    return tuple(f"T{index}" for index in range(1, 9))


def _short_platform_name(agent: HarborAgent) -> str:
    if agent.name == "ground_rover_1":
        return "RobEn"
    if agent.name == "ground_rover_2":
        return "Inspector-Gadget"
    if agent.model.kind == "usv":
        return "Heron"
    return "BlueROV2"


def _draw_paths(
    axis,
    result: HarborResult,
    agents: list[HarborAgent],
    colors: dict[str, str],
) -> None:
    accepted_safe_set = (
        result.all_goals_reached and result.pairwise_violation_count == 0
    )
    safety_label_pending = True
    safe_set_label_pending = accepted_safe_set
    for agent in agents:
        positions = result.positions[agent.name]
        color = colors[agent.name]
        linestyle = "--" if agent.model.kind == "rov" else "-"
        for point in positions[::10]:
            axis.add_patch(
                plt.Circle(
                    point[:2],
                    agent.radius,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.035,
                    label=(
                        "physical safety envelope" if safety_label_pending else None
                    ),
                    zorder=0,
                )
            )
            safety_label_pending = False
        axis.plot(
            positions[:, 0],
            positions[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=2.4,
            label=f"{_display_name(agent)} path",
        )
        if accepted_safe_set:
            samples = positions[::10]
            axis.scatter(
                samples[:, 0],
                samples[:, 1],
                facecolors="white",
                edgecolors=color,
                linewidths=0.8,
                s=18,
                alpha=0.75,
                label=(
                    "successful rollout samples" if safe_set_label_pending else None
                ),
                zorder=2,
            )
            safe_set_label_pending = False
        axis.scatter(positions[0, 0], positions[0, 1], color=color, marker="o", s=65)
        axis.scatter(agent.goal[0], agent.goal[1], color=color, marker="*", s=170)
        yaw = agent.goal[2] if agent.model.pose_dim == 3 else agent.goal[5]
        axis.quiver(
            agent.goal[0],
            agent.goal[1],
            0.45 * np.cos(yaw),
            0.45 * np.sin(yaw),
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            width=0.006,
            zorder=4,
        )
        if len(agent.route) > 1:
            waypoints = agent.route[:-1]
            axis.scatter(
                waypoints[:, 0],
                waypoints[:, 1],
                color=color,
                marker="D",
                s=42,
                label=f"{_display_name(agent)} waypoint",
            )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.3)


def _agent_colors(agents: list[HarborAgent]) -> dict[str, str]:
    """Assign stable distinct colors to agents of the same platform kind."""
    counts: dict[str, int] = {}
    colors = {}
    for agent in agents:
        index = counts.get(agent.model.kind, 0)
        palette = _PALETTES[agent.model.kind]
        colors[agent.name] = palette[index % len(palette)]
        counts[agent.model.kind] = index + 1
    return colors


def _display_name(agent: HarborAgent) -> str:
    if agent.display_name is not None:
        return agent.display_name
    suffix = agent.name.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return f"{_LABELS[agent.model.kind]} {suffix}"
    return _LABELS[agent.model.kind]


def _draw_harbor_map(axis, config) -> None:
    """Draw the configured water, shoreline/quay, and ground regions."""
    x_min, x_max = config.world_x_bounds
    y_min, y_max = config.world_y_bounds
    axis.axhspan(y_min, config.shoreline_y, color="#dceff7", zorder=-4)
    axis.axhspan(config.shoreline_y, y_max, color="#e5e8df", zorder=-4)
    axis.axhline(config.shoreline_y, color="#667078", linewidth=5, zorder=-2)
    axis.text(
        x_min + 0.25,
        config.shoreline_y + 0.18,
        "quay / ground",
        color="#394047",
        fontsize=9,
    )
    axis.text(x_min + 0.25, y_min + 0.25, "harbor water", color="#32789a", fontsize=9)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
