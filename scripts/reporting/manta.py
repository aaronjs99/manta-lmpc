"""Reporting / manta."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scripts.metrics.manta.report import MantaRunReport, prepare_manta_report
from scripts.reporting.json import write_json
from scripts.plotting.manta import (
    plot_cost_decrease,
    plot_learning_progression,
    save_manta_animation,
)
from scripts.plotting.trajectories import plot_pairwise_distances, plot_trajectories

if TYPE_CHECKING:
    from scripts.configuration.manta import ProjectConfig
    from scripts.contracts.manta import MantaLMPCRunResult


def save_manta_run_report(
    output_dir: str | Path,
    result: MantaLMPCRunResult,
    project_config: ProjectConfig,
    *,
    config_path: str | Path,
) -> MantaRunReport:
    """Save JSON, CSV, plots, and the optional GIF for a completed run."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    report = prepare_manta_report(result, project_config, config_path=config_path)
    scenario = project_config.scenario
    lmpc = project_config.lmpc
    apf = project_config.apf

    _save_histories_csv(root / "states_by_iteration.csv", result.histories, lmpc.dt)
    plot_learning_progression(
        report.report_histories,
        scenario.goals,
        scenario.obstacle,
        str(root / "learning_progression.png"),
        goal_tolerance=lmpc.goal_tolerance,
        obstacle_padding=apf.obstacle_padding,
        safety_distance=scenario.safety_distance,
        statuses_by_iteration=result.report_statuses,
    )
    plot_cost_decrease(
        report.report_histories,
        scenario.goals,
        str(root / "cost_decrease.png"),
        goal_tolerance=project_config.plots.cost_goal_tolerance,
    )
    plot_trajectories(
        states=report.final_states,
        goals=scenario.goals,
        title=f"{scenario.name}: final manta LMPC iteration",
        out_path=str(root / "final_trajectories.png"),
        goal_tolerance=lmpc.goal_tolerance,
        obstacle=scenario.obstacle,
        obstacle_padding=apf.obstacle_padding,
        safety_distance=scenario.safety_distance,
        statuses=result.final_statuses,
    )
    plot_pairwise_distances(
        distances=report.pairwise_distances,
        dt=lmpc.dt,
        safety_distance=scenario.safety_distance,
        title=f"{scenario.name}: pairwise distances",
        out_path=str(root / "pairwise_distances.png"),
    )
    if project_config.make_video:
        save_manta_animation(
            result.final_history,
            scenario.goals,
            scenario.obstacle,
            lmpc.dt,
            str(root / "final_iteration.gif"),
            fps=project_config.plots.animation_fps,
            goal_tolerance=lmpc.goal_tolerance,
            obstacle_padding=apf.obstacle_padding,
            safety_distance=scenario.safety_distance,
            statuses=result.final_statuses,
        )
    report.summary["run_status"] = "completed"
    report.summary["artifacts"] = {
        "state_history": "states_by_iteration.csv",
        "learning_plot": "learning_progression.png",
        "cost_plot": "cost_decrease.png",
        "trajectory_plot": "final_trajectories.png",
        "separation_plot": "pairwise_distances.png",
        "animation": "final_iteration.gif" if project_config.make_video else None,
    }
    write_json(root / "summary.json", report.summary, atomic=True)
    return report


def save_manta_progress_summary(
    output_dir: str | Path,
    result: MantaLMPCRunResult,
    project_config: ProjectConfig,
    *,
    config_path: str | Path,
    run_status: str,
) -> None:
    """Save inspectable progress without rendering plots on every iteration."""
    report = prepare_manta_report(result, project_config, config_path=config_path)
    report.summary["run_status"] = run_status
    report.summary["artifacts"] = {}
    write_json(Path(output_dir) / "summary.json", report.summary, atomic=True)


def _save_histories_csv(
    path: Path,
    histories: list[dict[int, np.ndarray]],
    dt: float,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            (
                "iteration",
                "step",
                "time",
                "agent",
                "x",
                "y",
                "theta",
                "p_L",
                "q_L",
                "p_R",
                "q_R",
            )
        )
        for iteration, history in enumerate(histories):
            for agent in sorted(history):
                trajectory = np.asarray(history[agent], dtype=float)
                for step, state in enumerate(trajectory):
                    writer.writerow(
                        (iteration, step, step * dt, agent, *state.tolist())
                    )
