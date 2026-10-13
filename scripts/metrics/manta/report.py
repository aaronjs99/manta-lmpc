"""Manta rollout metric preparation for reports and visualizations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scripts.metrics.core import compute_rollout_metrics, pairwise_distances
from scripts.metrics.evaluation import cost_by_iteration, history_to_tensor
from scripts.metrics.optimizer import (
    summarize_optimizer_slack,
    summarize_optimizer_slack_by_agent,
)

if TYPE_CHECKING:
    from typing import Any

    from scripts.configuration.manta import ProjectConfig
    from scripts.contracts.manta import MantaLMPCRunResult


@dataclass(frozen=True)
class MantaRunReport:
    """Prepared numerical data shared by serialization and plotting."""

    summary: dict[str, Any]
    final_states: np.ndarray
    pairwise_distances: np.ndarray
    report_histories: list[dict[int, np.ndarray]]


def prepare_manta_report(
    result: MantaLMPCRunResult,
    project_config: ProjectConfig,
    *,
    config_path: str | Path,
) -> MantaRunReport:
    """Compute the stable summary contract and plot-ready numerical arrays."""
    scenario = project_config.scenario
    lmpc = project_config.lmpc
    _, final_states = history_to_tensor(result.final_history)
    metrics = compute_rollout_metrics(
        states=final_states,
        goals=scenario.goals,
        safety_distance=scenario.safety_distance,
        dt=lmpc.dt,
        controls=result.final_controls,
        goal_tolerance=lmpc.goal_tolerance,
    )
    report_histories = result.report_histories
    costs = cost_by_iteration(
        result.histories,
        scenario.goals,
        goal_tolerance=lmpc.goal_tolerance,
    )
    report_costs = cost_by_iteration(
        report_histories,
        scenario.goals,
        goal_tolerance=project_config.plots.cost_goal_tolerance,
    )

    summary = {
        "scenario": scenario.name,
        "config": str(Path(config_path).resolve()),
        "iterations": lmpc.iterations,
        "dt": lmpc.dt,
        "prediction_horizon": lmpc.prediction_horizon,
        "k_hull": lmpc.k_hull,
        "optimizer_relaxations": {
            "terminal_slack_bound": lmpc.terminal_slack_bound,
            "terminal_slack_weight": lmpc.terminal_slack_weight,
            "static_slack_bound": lmpc.static_slack_bound,
            "static_slack_weight": lmpc.static_slack_weight,
            "hyperplane_slack_bound": lmpc.hyperplane_slack_bound,
            "hyperplane_slack_weight": lmpc.hyperplane_slack_weight,
        },
        "max_steps": lmpc.max_steps,
        "apf_max_steps": project_config.apf.max_steps,
        "goal_tolerance": lmpc.goal_tolerance,
        "cost_goal_tolerance": project_config.plots.cost_goal_tolerance,
        "cost_by_iteration_goal_tolerance": lmpc.goal_tolerance,
        "plot_cost_goal_tolerance": project_config.plots.cost_goal_tolerance,
        "selected_iteration": result.selected_iteration,
        "success_by_iteration": result.success_by_iteration,
        "goal_reached_by_iteration": result.goal_reached_by_iteration,
        "learned_by_iteration": result.learned_by_iteration,
        "validation_by_iteration": [
            validation.to_dict() for validation in result.validation_by_iteration
        ],
        "status_counts_by_iteration": _status_counts_by_iteration(
            result.statuses_by_iteration
        ),
        "optimizer_slack_by_iteration": [
            summarize_optimizer_slack(np.zeros((0, len(scenario.starts), 3)))
        ]
        + [summarize_optimizer_slack(values) for values in result.slack_by_iteration],
        "optimizer_slack_by_agent_by_iteration": [
            summarize_optimizer_slack_by_agent(np.zeros((0, len(scenario.starts), 3)))
        ]
        + [
            summarize_optimizer_slack_by_agent(values)
            for values in result.slack_by_iteration
        ],
        "cost_by_iteration": {str(agent): values for agent, values in costs.items()},
        "report_cost_by_iteration": {
            str(agent): values for agent, values in report_costs.items()
        },
        "final_goal_error_by_agent": {
            str(agent): error
            for agent, error in enumerate(
                _final_goal_errors(final_states, scenario.goals)
            )
        },
        "metrics_final_iteration": metrics.to_dict(),
    }
    return MantaRunReport(
        summary=summary,
        final_states=final_states,
        pairwise_distances=pairwise_distances(final_states),
        report_histories=report_histories,
    )


def _status_counts_by_iteration(
    statuses_by_iteration: list[list[dict[int, str]]],
) -> list[dict[str, int]]:
    counts_by_iteration: list[dict[str, int]] = [{}]
    for iteration_statuses in statuses_by_iteration:
        counter: Counter[str] = Counter()
        for step_statuses in iteration_statuses:
            counter.update(step_statuses.values())
        counts_by_iteration.append(dict(sorted(counter.items())))
    return counts_by_iteration


def _final_goal_errors(states: np.ndarray, goals: np.ndarray) -> list[float]:
    goal_positions = np.asarray(goals, dtype=float)[:, :2]
    final_positions = np.asarray(states, dtype=float)[-1, :, :2]
    return np.linalg.norm(final_positions - goal_positions, axis=1).tolist()
