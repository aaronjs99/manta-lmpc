"""Coordinate safe-set learning iterations for the Manta LMPC experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scripts.configuration.apf import APFConfig
from scripts.configuration.manta_dynamics import MantaDynamicsConfig
from scripts.configuration.manta_mpc import MantaLMPCConfig
from scripts.contracts.manta import MantaLMPCRunResult, Scenario
from scripts.experiments.manta.iteration import (
    MantaIterationRequest,
    execute_manta_iteration,
)
from scripts.experiments.manta.safe_set_scheduling import build_staggered_safe_sets
from scripts.metrics.evaluation import validate_trajectory
from scripts.mpc.manta_lmpc import MantaAgentOptimizer

if TYPE_CHECKING:
    from collections.abc import Callable


def run_manta_lmpc(
    scenario: Scenario,
    *,
    config: MantaLMPCConfig = MantaLMPCConfig(),
    apf_config: APFConfig = APFConfig(),
    dynamics_config: MantaDynamicsConfig = MantaDynamicsConfig(),
    should_stop: Callable[[], bool] | None = None,
    verbose: bool = True,
) -> MantaLMPCRunResult:
    """Run APF iteration 0 followed by decentralized manta LMPC iterations."""
    starts = np.asarray(scenario.starts, dtype=float)
    goals = np.asarray(scenario.goals, dtype=float)
    num_agents = len(starts)
    if goals.shape[0] != num_agents:
        raise ValueError("scenario starts and goals must have the same agent count")
    if num_agents < 2:
        raise ValueError("manta LMPC requires at least two agents")

    if verbose:
        print("Generating staggered APF trajectories (iteration 0)...")
    safe_sets, safe_controls = build_staggered_safe_sets(
        scenario,
        dt=config.dt,
        apf_config=apf_config,
        dynamics_config=dynamics_config,
    )

    histories = [_snapshot_safe_sets(safe_sets)]
    controls_by_iteration: list[np.ndarray] = []
    slack_by_iteration: list[np.ndarray] = []
    statuses_by_iteration: list[list[dict[int, str]]] = []
    initial_validation = validate_trajectory(
        safe_sets,
        goals,
        scenario.safety_distance,
        scenario.obstacle.center,
        scenario.obstacle.radius,
        config.goal_tolerance,
        statuses=None,
    )
    validation_by_iteration = [initial_validation]
    success_by_iteration: list[bool] = [initial_validation.valid]
    goal_reached_by_iteration: list[bool] = [initial_validation.all_goals_reached]
    learned_by_iteration: list[bool] = [initial_validation.usable_for_learning]
    selected_iteration: int | None = 0 if initial_validation.valid else None

    if config.iterations <= 0:
        return MantaLMPCRunResult(
            scenario_name=scenario.name,
            histories=histories,
            controls_by_iteration=controls_by_iteration,
            slack_by_iteration=slack_by_iteration,
            statuses_by_iteration=statuses_by_iteration,
            success_by_iteration=success_by_iteration,
            goal_reached_by_iteration=goal_reached_by_iteration,
            learned_by_iteration=learned_by_iteration,
            validation_by_iteration=validation_by_iteration,
            selected_iteration=selected_iteration,
        )

    optimizers = {
        agent: MantaAgentOptimizer(
            config=config,
            num_obstacles=num_agents - 1,
            obstacle=scenario.obstacle,
            dynamics_config=dynamics_config,
        )
        for agent in range(num_agents)
    }

    for iteration in range(1, config.iterations + 1):
        _raise_if_stop_requested(should_stop)
        if verbose:
            print(f"Starting LMPC iteration {iteration}...")

        execution = execute_manta_iteration(
            MantaIterationRequest(
                iteration=iteration,
                starts=starts,
                goals=goals,
                scenario=scenario,
                config=config,
                apf_config=apf_config,
                dynamics_config=dynamics_config,
                optimizers=optimizers,
                safe_sets=safe_sets,
                safe_controls=safe_controls,
                should_stop=should_stop,
                verbose=verbose,
            )
        )
        histories.append(_snapshot_safe_sets(execution.history))
        controls_by_iteration.append(execution.controls)
        slack_by_iteration.append(execution.slacks)
        statuses_by_iteration.append(execution.statuses)
        validation = validate_trajectory(
            execution.history,
            goals,
            scenario.safety_distance,
            scenario.obstacle.center,
            scenario.obstacle.radius,
            config.goal_tolerance,
            statuses=execution.statuses,
        )
        validation_by_iteration.append(validation)
        goal_reached_by_iteration.append(validation.all_goals_reached)
        success_by_iteration.append(validation.valid)
        learned_by_iteration.append(validation.usable_for_learning)
        if validation.usable_for_learning:
            safe_sets = {agent: execution.history[agent] for agent in range(num_agents)}
            safe_controls = _controls_by_agent(execution.controls, num_agents)
            histories[-1] = _snapshot_safe_sets(safe_sets)
            if verbose:
                print(
                    "  learned: complete safe trajectory kept for the next iteration."
                )
        elif validation.safe and verbose:
            print(
                "  skipped learning: trajectory was safe but did not reach all goals."
            )
        if validation.valid:
            selected_iteration = len(histories) - 1
        elif verbose:
            print(
                "  invalid: keeping previous safe set "
                f"(pair violations={validation.pairwise_violation_count}, "
                f"obstacle violations={validation.obstacle_violation_count}, "
                f"fallbacks={validation.fallback_count}, "
                f"safety interventions={validation.safety_intervention_count})."
            )

    return MantaLMPCRunResult(
        scenario_name=scenario.name,
        histories=histories,
        controls_by_iteration=controls_by_iteration,
        slack_by_iteration=slack_by_iteration,
        statuses_by_iteration=statuses_by_iteration,
        success_by_iteration=success_by_iteration,
        goal_reached_by_iteration=goal_reached_by_iteration,
        learned_by_iteration=learned_by_iteration,
        validation_by_iteration=validation_by_iteration,
        selected_iteration=selected_iteration,
    )


def _snapshot_safe_sets(
    safe_sets: dict[int, list[np.ndarray]],
) -> dict[int, np.ndarray]:
    return {
        agent: np.asarray(states, dtype=float).copy()
        for agent, states in safe_sets.items()
    }


def _controls_by_agent(controls: np.ndarray, num_agents: int) -> dict[int, np.ndarray]:
    """Convert iteration controls from ``(T, A, 2)`` to per-agent histories."""
    control_array = np.asarray(controls, dtype=float)
    if control_array.ndim != 3 or control_array.shape[1:] != (num_agents, 2):
        raise ValueError(
            f"controls must have shape (T, {num_agents}, 2), got {control_array.shape}"
        )
    return {agent: control_array[:, agent, :].copy() for agent in range(num_agents)}


def _raise_if_stop_requested(should_stop: Callable[[], bool] | None) -> None:
    if should_stop is not None and should_stop():
        raise KeyboardInterrupt
