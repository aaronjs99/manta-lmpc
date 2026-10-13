"""Metrics / manta / benchmark."""

from __future__ import annotations

import numpy as np
from scripts.metrics.evaluation import cost_by_iteration
from scripts.metrics.optimizer import (
    summarize_optimizer_slack,
    summarize_optimizer_slack_by_agent,
)


def _record_for_result(
    goals,
    lmpc,
    result,
    *,
    elapsed_seconds: float,
) -> dict[str, object]:
    selected = result.selected_iteration
    selected_validation = (
        result.validation_by_iteration[selected] if selected is not None else None
    )
    latest_validation = result.validation_by_iteration[-1]
    costs = cost_by_iteration(
        result.histories,
        goals,
        goal_tolerance=lmpc.goal_tolerance,
    )
    selected_costs = {
        str(agent): values[selected] if selected is not None else values[-1]
        for agent, values in costs.items()
    }
    selected_slack = (
        summarize_optimizer_slack(result.slack_by_iteration[selected - 1])
        if selected is not None and selected > 0
        else summarize_optimizer_slack(np.zeros((0, len(goals), 3)))
    )
    latest_slack = (
        summarize_optimizer_slack(result.slack_by_iteration[-1])
        if result.slack_by_iteration
        else summarize_optimizer_slack(np.zeros((0, len(goals), 3)))
    )
    empty_slack = np.zeros((0, len(goals), 3))
    selected_slack_by_agent = (
        summarize_optimizer_slack_by_agent(result.slack_by_iteration[selected - 1])
        if selected is not None and selected > 0
        else summarize_optimizer_slack_by_agent(empty_slack)
    )
    latest_slack_by_agent = (
        summarize_optimizer_slack_by_agent(result.slack_by_iteration[-1])
        if result.slack_by_iteration
        else summarize_optimizer_slack_by_agent(empty_slack)
    )

    return {
        "scenario": result.scenario_name,
        "iterations": lmpc.iterations,
        "max_steps": lmpc.max_steps,
        "terminal_slack_weight": lmpc.terminal_slack_weight,
        "elapsed_seconds": elapsed_seconds,
        "selected_iteration": selected,
        "selected_valid": selected_validation.valid if selected_validation else False,
        "selected_safe": selected_validation.safe if selected_validation else False,
        "selected_solver_clean": (
            selected_validation.solver_clean if selected_validation else False
        ),
        "selected_min_pairwise": (
            selected_validation.min_pairwise_distance if selected_validation else None
        ),
        "selected_min_obstacle_clearance": (
            selected_validation.min_obstacle_clearance if selected_validation else None
        ),
        "selected_fallback_count": (
            selected_validation.fallback_count if selected_validation else None
        ),
        "selected_safety_interventions": (
            selected_validation.safety_intervention_count
            if selected_validation
            else None
        ),
        "selected_max_static_slack": selected_slack["max_static_slack"],
        "selected_max_hyperplane_slack": selected_slack["max_hyperplane_slack"],
        "selected_max_terminal_slack": selected_slack["max_terminal_slack"],
        "selected_nonzero_static_slack_steps": selected_slack[
            "nonzero_static_slack_steps"
        ],
        "selected_nonzero_hyperplane_slack_steps": selected_slack[
            "nonzero_hyperplane_slack_steps"
        ],
        "selected_nonzero_terminal_slack_steps": selected_slack[
            "nonzero_terminal_slack_steps"
        ],
        "latest_max_static_slack": latest_slack["max_static_slack"],
        "latest_max_hyperplane_slack": latest_slack["max_hyperplane_slack"],
        "latest_max_terminal_slack": latest_slack["max_terminal_slack"],
        "latest_nonzero_static_slack_steps": latest_slack["nonzero_static_slack_steps"],
        "latest_nonzero_hyperplane_slack_steps": latest_slack[
            "nonzero_hyperplane_slack_steps"
        ],
        "latest_nonzero_terminal_slack_steps": latest_slack[
            "nonzero_terminal_slack_steps"
        ],
        "selected_slack_by_agent": selected_slack_by_agent,
        "latest_slack_by_agent": latest_slack_by_agent,
        "latest_valid": latest_validation.valid,
        "latest_safe": latest_validation.safe,
        "latest_solver_clean": latest_validation.solver_clean,
        "latest_fallback_count": latest_validation.fallback_count,
        "latest_safety_interventions": latest_validation.safety_intervention_count,
        "cost_by_iteration": {str(agent): values for agent, values in costs.items()},
        "selected_costs": selected_costs,
    }
