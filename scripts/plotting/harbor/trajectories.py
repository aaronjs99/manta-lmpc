"""Plotting / harbor / trajectories."""

from __future__ import annotations

from scripts.plotting._backend import configure_matplotlib

configure_matplotlib()

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scripts.plotting.harbor.common import (
    _agent_colors,
    _display_name,
    _draw_harbor_map,
    _draw_paths,
)

if TYPE_CHECKING:
    from scripts.contracts.harbor import HarborAgent, HarborResult


def save_harbor_comparison(
    results: dict[str, HarborResult],
    agents: list[HarborAgent],
    simulation_config,
    path: str | Path,
) -> Path:
    """Save top-down independent-versus-coordinated trajectory panels."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.4 * len(results), 8.0))
    grid = fig.add_gridspec(2, len(results), height_ratios=(3.2, 1.0))
    axes = np.asarray(
        [fig.add_subplot(grid[0, index]) for index in range(len(results))]
    )
    diagnostics = grid[1, :].subgridspec(1, 2)
    depth_axis = fig.add_subplot(diagnostics[0, 0])
    attitude_axis = fig.add_subplot(diagnostics[0, 1])
    colors = _agent_colors(agents)
    for axis, (label, result) in zip(axes, results.items(), strict=True):
        _draw_harbor_map(axis, simulation_config)
        _draw_paths(axis, result, agents, colors)
        axis.set_title(
            f"{label}\nviolations={result.pairwise_violation_count}, "
            f"min distance={result.min_pairwise_distance:.3f} m"
        )
        for agent in agents:
            if agent.model.kind != "rov":
                continue
            depth_axis.plot(
                result.positions[agent.name][:, 2],
                linewidth=2.0,
                linestyle="--" if label == "independent" else "-",
                color="#777777" if label == "independent" else colors[agent.name],
                label=f"{label} ROV depth",
            )
    depth_axis.axhline(0.0, color="#4aa3df", linewidth=1.2, label="water surface")
    depth_axis.axhline(
        simulation_config.seabed_z,
        color="#665544",
        linewidth=1.8,
        label="seabed",
    )
    depth_axis.set_title("ROV Dive Profile")
    depth_axis.set_xlabel("simulation step")
    depth_axis.set_ylabel("z [m]")
    depth_axis.set_ylim(simulation_config.seabed_z - 0.2, 0.35)
    depth_axis.grid(True, alpha=0.3)
    depth_axis.legend(loc="lower right", fontsize=8)
    reference_label, reference_result = list(results.items())[-1]
    for agent in agents:
        if agent.model.kind != "rov":
            continue
        attitude = reference_result.states[agent.name][:, 3:6]
        for index, (name, color) in enumerate(
            (("roll", "#d95f02"), ("pitch", "#1b9e77"), ("yaw", "#7570b3"))
        ):
            attitude_axis.plot(attitude[:, index], color=color, label=name)
            attitude_axis.axhline(
                agent.goal[index + 3], color=color, linestyle=":", alpha=0.55
            )
    attitude_axis.set_title(f"ROV Attitude ({reference_label})")
    attitude_axis.set_xlabel("simulation step")
    attitude_axis.set_ylabel("angle [rad]")
    attitude_axis.grid(True, alpha=0.3)
    attitude_axis.legend(loc="upper right", ncol=3, fontsize=8)
    handles = []
    labels = []
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels, strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.suptitle("Untethered Heterogeneous Harbor Coordination", fontsize=16)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=4)
    fig.subplots_adjust(top=0.82, bottom=0.08, hspace=0.38, wspace=0.25)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def save_harbor_animation(
    result: HarborResult,
    agents: list[HarborAgent],
    simulation_config,
    path: str | Path,
    *,
    fps: int = 15,
    label: str = "ETA-priority coordination",
) -> Path:
    """Save XY-yaw and ROV x-z-pitch views of a coordinated rollout."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, (map_axis, depth_axis) = plt.subplots(
        1, 2, figsize=(11.0, 5.2), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    fig.subplots_adjust(bottom=0.19, wspace=0.25)
    models = {agent.name: agent.model for agent in agents}
    by_name = {agent.name: agent for agent in agents}
    colors = _agent_colors(agents)
    radii = {agent.name: agent.radius for agent in agents}
    goals = {agent.name: agent.goal for agent in agents}
    names = [agent.name for agent in agents]
    rov_agent = next(agent for agent in agents if agent.model.kind == "rov")
    frames = max(len(result.positions[name]) for name in names)
    accepted_safe_set = (
        result.all_goals_reached and result.pairwise_violation_count == 0
    )

    def update(frame: int):
        map_axis.clear()
        depth_axis.clear()
        _draw_harbor_map(map_axis, simulation_config)
        for name in names:
            kind = models[name].kind
            color = colors[name]
            positions = result.positions[name]
            index = min(frame, len(positions) - 1)
            shown = positions[: index + 1]
            if accepted_safe_set:
                samples = positions[::10]
                map_axis.scatter(
                    samples[:, 0],
                    samples[:, 1],
                    facecolors="white",
                    edgecolors=color,
                    linewidths=0.7,
                    s=15,
                    alpha=0.45,
                )
            map_axis.plot(shown[:, 0], shown[:, 1], color=color, linewidth=2)
            map_axis.scatter(
                shown[-1, 0],
                shown[-1, 1],
                color=color,
                s=90,
                label=_display_name(by_name[name]),
            )
            map_axis.scatter(
                goals[name][0], goals[name][1], color=color, marker="*", s=150
            )
            footprint = plt.Circle(
                shown[-1, :2], radii[name], facecolor=color, alpha=0.12, edgecolor=color
            )
            map_axis.add_patch(footprint)
            state = result.states[name][index]
            yaw = state[5] if kind == "rov" else state[2]
            map_axis.quiver(
                shown[-1, 0],
                shown[-1, 1],
                0.5 * np.cos(yaw),
                0.5 * np.sin(yaw),
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=color,
                width=0.008,
                zorder=5,
            )
            for waypoint in by_name[name].route[:-1]:
                map_axis.scatter(
                    waypoint[0],
                    waypoint[1],
                    color=color,
                    marker="D",
                    s=38,
                    alpha=0.8,
                )
        map_axis.set_title(f"{label}, step {frame}")
        map_axis.set_xlabel("x [m]")
        map_axis.set_ylabel("y [m]")
        map_axis.grid(True, alpha=0.3)
        map_axis.set_aspect("equal", adjustable="box")
        map_axis.set_xlim(*simulation_config.world_x_bounds)
        map_axis.set_ylim(*simulation_config.world_y_bounds)
        map_axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=4,
            fontsize=8,
            framealpha=0.95,
        )
        rov_positions = result.positions[rov_agent.name]
        rov_index = min(frame, len(rov_positions) - 1)
        rov_shown = rov_positions[: rov_index + 1]
        rov_color = colors[rov_agent.name]
        depth_axis.plot(
            rov_shown[:, 0], rov_shown[:, 2], color=rov_color, linewidth=2.4
        )
        depth_axis.scatter(rov_shown[-1, 0], rov_shown[-1, 2], color=rov_color, s=90)
        depth_axis.scatter(
            rov_agent.goal[0],
            rov_agent.goal[2],
            color=rov_color,
            marker="*",
            s=150,
        )
        for waypoint in rov_agent.route[:-1]:
            depth_axis.scatter(
                waypoint[0], waypoint[2], color=rov_color, marker="D", s=42
            )
        pitch = result.states[rov_agent.name][rov_index, 4]
        depth_axis.quiver(
            rov_shown[-1, 0],
            rov_shown[-1, 2],
            0.65 * np.cos(pitch),
            0.65 * np.sin(pitch),
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=rov_color,
            width=0.012,
            zorder=5,
        )
        depth_axis.axhline(0.0, color="#4aa3df", linewidth=1.2, alpha=0.7)
        depth_axis.axhline(
            simulation_config.seabed_z,
            color="#665544",
            linewidth=2.0,
            alpha=0.8,
        )
        depth_axis.set_title(f"ROV Side Elevation, pitch={pitch:+.2f} rad")
        depth_axis.set_xlabel("x [m]")
        depth_axis.set_ylabel("z [m]")
        depth_axis.set_xlim(*simulation_config.world_x_bounds)
        depth_axis.set_ylim(simulation_config.seabed_z - 0.2, 0.35)
        depth_axis.grid(True, alpha=0.3)
        return ()

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output
