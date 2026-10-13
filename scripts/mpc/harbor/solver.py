"""Numeric parameterization and execution of a Harbor MPC problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from scripts.mpc.harbor.problem import HarborProblemVariables
from scripts.mpc.harbor.result import HarborMPCStep

if TYPE_CHECKING:
    from scripts.configuration.harbor_mpc import HarborMPCConfig
    from scripts.contracts.harbor import HarborAgent


@dataclass(frozen=True)
class HarborSolveRequest:
    """Values supplied to one optimization solve."""

    state: np.ndarray
    goal: np.ndarray
    obstacle_predictions: dict[str, np.ndarray]
    safe_states: np.ndarray
    safe_costs: np.ndarray
    warm_states: np.ndarray
    warm_controls: np.ndarray
    previous_control: np.ndarray
    position_drift: np.ndarray
    control_effectiveness: np.ndarray
    identification_probe: np.ndarray
    identification_probe_mask: np.ndarray
    belief_collision_margin: float = 0.0


def solve_numeric_problem(
    *,
    problem: HarborProblemVariables,
    request: HarborSolveRequest,
    agent: HarborAgent,
    other_agents: tuple[HarborAgent, ...],
    config: HarborMPCConfig,
    learning: bool,
) -> tuple[HarborMPCStep, str]:
    """Set numerical values, invoke IPOPT, and translate its solution."""
    opti = problem.opti
    opti.set_value(problem.p_initial, request.state)
    opti.set_value(problem.p_goal, request.goal)
    opti.set_value(problem.p_previous_control, request.previous_control)
    opti.set_value(problem.p_position_drift, request.position_drift)
    opti.set_value(problem.p_control_effectiveness, request.control_effectiveness)
    opti.set_value(
        problem.p_belief_collision_margin,
        float(max(request.belief_collision_margin, 0.0)),
    )
    if problem.p_identification_probe is not None:
        opti.set_value(problem.p_identification_probe, request.identification_probe)
        opti.set_value(
            problem.p_identification_probe_mask, request.identification_probe_mask
        )
    for index, other in enumerate(other_agents):
        opti.set_value(
            problem.p_obstacles[index], request.obstacle_predictions[other.name].T
        )
    if learning:
        opti.set_value(
            problem.p_safe_pose, _terminal_rows(agent, config, request.safe_states)
        )
        opti.set_value(problem.p_safe_costs, request.safe_costs.reshape(-1, 1))
        opti.set_initial(problem.lambdas, 1.0 / config.terminal_samples)
        opti.set_initial(problem.terminal_slack, 0.0)
    opti.set_initial(problem.states, request.warm_states.T)
    opti.set_initial(problem.controls, request.warm_controls.T)
    opti.set_initial(problem.collision_slack, 0.0)
    opti.set_initial(problem.domain_buffer_slack, 0.0)
    opti.set_initial(problem.dynamic_state_slack, 0.0)
    solution = opti.solve()
    status = str(opti.stats().get("return_status", "unknown"))
    states = np.asarray(solution.value(problem.states), dtype=float).T
    collision = np.asarray(
        solution.value(problem.collision_slack), dtype=float
    ).reshape(-1)
    domain_buffer = np.asarray(
        solution.value(problem.domain_buffer_slack), dtype=float
    ).reshape(-1)
    dynamic_state = np.asarray(
        solution.value(problem.dynamic_state_slack), dtype=float
    ).reshape(-1)
    terminal = (
        np.asarray(solution.value(problem.terminal_slack), dtype=float).reshape(-1)
        if learning
        else np.zeros(1)
    )
    step = HarborMPCStep(
        control=np.asarray(solution.value(problem.controls[:, 0]), dtype=float).reshape(
            agent.model.control_dim
        ),
        predicted_states=states,
        max_collision_slack=float(np.max(collision, initial=0.0)),
        max_domain_buffer_slack=float(np.max(domain_buffer, initial=0.0)),
        max_dynamic_state_slack=float(np.max(dynamic_state, initial=0.0)),
        max_terminal_slack=float(np.max(np.abs(terminal), initial=0.0)),
    )
    return step, status


def _terminal_rows(
    agent: HarborAgent, config: HarborMPCConfig, states: np.ndarray
) -> np.ndarray:
    terminal_dim = (
        (2 if agent.model.pose_dim == 3 else 3)
        if config.terminal_position_only
        else agent.model.pose_dim
    )
    return np.asarray(states, dtype=float)[:, :terminal_dim]
