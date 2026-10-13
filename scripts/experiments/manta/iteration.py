"""Execute one LMPC iteration against the simulation plant boundary."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scripts.algorithms.manta.hyperplanes import get_symmetric_hyperplanes_spatial
from scripts.algorithms.manta.policies import priority_margins, warm_start_from_safe_set
from scripts.algorithms.manta.recovery import (
    repair_incomplete_with_apf,
    safe_fallback_apf_step,
)
from scripts.algorithms.manta.safe_sets import sample_terminal_safe_set
from scripts.algorithms.manta.safety import filter_unsafe_transitions
from scripts.configuration.apf import APFConfig
from scripts.configuration.manta_dynamics import MantaDynamicsConfig
from scripts.configuration.manta_mpc import MantaLMPCConfig
from scripts.contracts.manta import Scenario
from scripts.mpc.manta_lmpc import MantaAgentOptimizer
from scripts.simulation.environment import MantaEnvConfig, MultiMantaRayEnv

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class MantaIterationRequest:
    """Explicit inputs for one policy iteration and its safety/recovery stages."""

    iteration: int
    starts: np.ndarray
    goals: np.ndarray
    scenario: Scenario
    config: MantaLMPCConfig
    apf_config: APFConfig
    dynamics_config: MantaDynamicsConfig
    optimizers: dict[int, MantaAgentOptimizer]
    safe_sets: dict[int, list[np.ndarray] | np.ndarray]
    safe_controls: dict[int, np.ndarray]
    should_stop: Callable[[], bool] | None = None
    verbose: bool = True


@dataclass(frozen=True)
class MantaIterationResult:
    """Per-iteration trajectory, controls, constraints, and step statuses."""

    history: dict[int, list[np.ndarray]]
    controls: np.ndarray
    slacks: np.ndarray
    statuses: list[dict[int, str]]
    all_reached: bool


def execute_manta_iteration(request: MantaIterationRequest) -> MantaIterationResult:
    """Run the current LMPC iteration without performing admission decisions."""
    config = request.config
    scenario = request.scenario
    starts = request.starts
    goals = request.goals
    num_agents = len(starts)
    plant = MultiMantaRayEnv(
        starts,
        goals,
        MantaEnvConfig(
            dt=config.dt, horizon=config.max_steps, dynamics=request.dynamics_config
        ),
    )
    current_states = {agent: starts[agent].copy() for agent in range(num_agents)}
    history = {agent: [current_states[agent].copy()] for agent in range(num_agents)}
    iteration_controls = np.zeros((config.max_steps, num_agents, 2), dtype=float)
    iteration_statuses: list[dict[int, str]] = []
    iteration_slacks: list[np.ndarray] = []
    all_reached = False
    reached_agents: set[int] = set()

    for step in range(config.max_steps):
        _raise_if_stop_requested(request.should_stop)
        step_start = time.time()
        for agent in range(num_agents):
            if (
                np.linalg.norm(current_states[agent][:2] - goals[agent][:2])
                <= config.goal_tolerance
            ):
                reached_agents.add(agent)

        if len(reached_agents) == num_agents:
            all_reached = True
            if request.verbose:
                print(
                    f"  success: all agents reached the goal tolerance at step {step}."
                )
            break

        reference_sets = _live_reference_sets(
            request.safe_sets, current_states, reached_agents, step, config
        )
        hyperplanes = _build_pairwise_hyperplanes(
            reference_sets,
            step,
            config,
            current_states=current_states,
            goals=goals,
        )
        next_states: dict[int, np.ndarray] = {}
        step_controls: dict[int, np.ndarray] = {}
        step_statuses: dict[int, str] = {}
        step_slacks = np.full((num_agents, 3), np.nan, dtype=float)

        for agent in range(num_agents):
            _raise_if_stop_requested(request.should_stop)
            if agent in reached_agents:
                step_controls[agent] = np.zeros(2, dtype=float)
                next_states[agent] = current_states[agent].copy()
                step_statuses[agent] = "hold"
                continue

            target_idx = step + config.prediction_horizon
            safe_states, safe_costs = sample_terminal_safe_set(
                request.safe_sets[agent], target_idx, config.k_hull
            )
            warm_states, warm_controls = warm_start_from_safe_set(
                request.safe_sets[agent], request.safe_controls[agent], step, config
            )
            agent_hyperplanes = [
                hyperplanes[agent][other]
                for other in range(num_agents)
                if other != agent
            ]

            try:
                solution = request.optimizers[agent].solve_step(
                    current_state=current_states[agent],
                    goal_state=goals[agent],
                    hyperplanes=agent_hyperplanes,
                    safe_states=safe_states,
                    safe_costs=safe_costs,
                    warm_states=warm_states,
                    warm_controls=warm_controls,
                )
                control = solution.control
                next_state = solution.next_state
                step_slacks[agent] = (
                    solution.max_static_slack,
                    solution.max_hyperplane_slack,
                    solution.max_terminal_slack,
                )
                step_statuses[agent] = "ok"
            except RuntimeError as exc:
                if _is_interrupted_solve(exc):
                    raise KeyboardInterrupt from exc
                control, next_state = safe_fallback_apf_step(
                    current_state=current_states[agent],
                    goal_state=goals[agent],
                    obstacle=scenario.obstacle,
                    apf_config=request.apf_config,
                    dt=config.dt,
                    config=config,
                    dynamics_config=request.dynamics_config,
                )
                step_statuses[agent] = "fallback_apf"

            step_controls[agent] = control
            next_states[agent] = next_state

        filtered = filter_unsafe_transitions(
            current_states=current_states,
            proposed_controls=step_controls,
            proposed_next_states=next_states,
            goals=goals,
            scenario=scenario,
            config=config,
            apf_config=request.apf_config,
            dynamics_config=request.dynamics_config,
            protected_agents=reached_agents,
        )
        step_controls = filtered.controls
        next_states = filtered.next_states
        step_statuses.update(filtered.statuses)
        for agent in range(num_agents):
            iteration_controls[step, agent] = step_controls[agent]

        accepted_states = plant.advance_with_transition(
            np.asarray([next_states[agent] for agent in range(num_agents)], dtype=float)
        )
        for agent in range(num_agents):
            current_states[agent] = accepted_states[agent]
            history[agent].append(current_states[agent].copy())

        iteration_statuses.append(step_statuses)
        iteration_slacks.append(step_slacks)
        for agent in range(num_agents):
            if (
                np.linalg.norm(current_states[agent][:2] - goals[agent][:2])
                <= config.goal_tolerance
            ):
                reached_agents.add(agent)
        all_reached = len(reached_agents) == num_agents
        if request.verbose and step % config.log_interval == 0:
            elapsed = time.time() - step_start
            status_text = " ".join(
                f"A{agent}:{step_statuses.get(agent, '?')}"
                for agent in range(num_agents)
            )
            print(
                f"  step {step:03d}/{config.max_steps}: {elapsed:.2f}s | {status_text}"
            )

        if all_reached:
            if request.verbose:
                print(
                    "  success: all agents reached the goal tolerance "
                    f"after step {step}."
                )
            break

    if request.verbose and not all_reached:
        print(
            f"  warning: iteration {request.iteration} ended at "
            f"{config.max_steps} steps without all goals reached."
        )

    main_controls = iteration_controls[: len(iteration_statuses)]
    stored_controls = main_controls
    if not all_reached and config.repair_incomplete_with_apf:
        recovery = repair_incomplete_with_apf(
            history=history,
            current_states=current_states,
            safe_sets=request.safe_sets,
            reached_agents=reached_agents,
            goals=goals,
            scenario=scenario,
            config=config,
            apf_config=request.apf_config,
            dynamics_config=request.dynamics_config,
            should_stop=request.should_stop,
            verbose=request.verbose,
        )
        if recovery.statuses:
            iteration_statuses.extend(recovery.statuses)
            iteration_slacks.extend(
                np.full((num_agents, 3), np.nan, dtype=float) for _ in recovery.statuses
            )
            stored_controls = np.vstack((main_controls, recovery.controls))
            all_reached = len(reached_agents) == num_agents

    return MantaIterationResult(
        history=history,
        controls=stored_controls,
        slacks=np.asarray(iteration_slacks, dtype=float).reshape(-1, num_agents, 3),
        statuses=iteration_statuses,
        all_reached=all_reached,
    )


def _live_reference_sets(
    safe_sets: dict[int, list[np.ndarray] | np.ndarray],
    current_states: dict[int, np.ndarray],
    reached_agents: set[int],
    step: int,
    config: MantaLMPCConfig,
) -> dict[int, list[np.ndarray] | np.ndarray]:
    """Use held current states as references after individual goal arrival."""
    if not reached_agents:
        return safe_sets
    reference_sets: dict[int, list[np.ndarray] | np.ndarray] = dict(safe_sets)
    reference_len = step + config.prediction_horizon + 1
    for agent in reached_agents:
        reference_sets[agent] = np.tile(current_states[agent], (reference_len, 1))
    return reference_sets


def _build_pairwise_hyperplanes(
    safe_sets: dict[int, list[np.ndarray] | np.ndarray],
    step: int,
    config: MantaLMPCConfig,
    *,
    current_states: dict[int, np.ndarray],
    goals: np.ndarray,
) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
    agents = sorted(safe_sets)
    hyperplanes: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        agent: {} for agent in agents
    }
    for index, i in enumerate(agents):
        for j in agents[index + 1 :]:
            margin_i, margin_j = priority_margins(
                i, j, safe_sets, step, current_states, goals, config
            )
            H_i, h_i, H_j, h_j = get_symmetric_hyperplanes_spatial(
                step,
                step,
                config.prediction_horizon,
                safe_sets[i],
                safe_sets[j],
                safety_margin=config.hyperplane_safety_margin,
                safety_margin_i=margin_i,
                safety_margin_j=margin_j,
                ignore_distance=config.hyperplane_ignore_distance,
            )
            hyperplanes[i][j] = (H_i, h_i)
            hyperplanes[j][i] = (H_j, h_j)
    return hyperplanes


def _is_interrupted_solve(exc: RuntimeError) -> bool:
    """Return true when CasADi/IPOPT converted Ctrl+C into RuntimeError."""
    message = str(exc)
    return "KeyboardInterrupt" in message or "KeyboardInterruptException" in message


def _raise_if_stop_requested(should_stop: Callable[[], bool] | None) -> None:
    if should_stop is not None and should_stop():
        raise KeyboardInterrupt
