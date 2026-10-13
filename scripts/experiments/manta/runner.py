"""Coordinate safe-set learning iterations for the Manta LMPC experiment."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from scripts.configuration.apf import APFConfig
from scripts.configuration.manta_dynamics import MantaDynamicsConfig
from scripts.configuration.manta_learning import manta_learning_settings
from scripts.configuration.manta_mpc import MantaLMPCConfig
from scripts.contracts.manta import MantaLMPCRunResult, Scenario
from scripts.experiments.manta.iteration import (
    MantaIterationRequest,
    execute_manta_iteration,
)
from scripts.experiments.manta.safe_set_scheduling import build_staggered_safe_sets
from scripts.metrics.evaluation import validate_trajectory
from scripts.metrics.manta.learning import revalidate_result
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
    resume_result: MantaLMPCRunResult | None = None,
    on_iteration_completed: Callable[[MantaLMPCRunResult], None] | None = None,
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

    _raise_if_stop_requested(should_stop)
    settings = manta_learning_settings(scenario, config, apf_config, dynamics_config)
    if resume_result is None:
        if verbose:
            print("Generating staggered APF trajectories (iteration 0)...")
        safe_sets, safe_controls = build_staggered_safe_sets(
            scenario,
            dt=config.dt,
            apf_config=apf_config,
            dynamics_config=dynamics_config,
            should_stop=should_stop,
        )
        initial_validation = validate_trajectory(
            safe_sets,
            goals,
            scenario.safety_distance,
            scenario.obstacle.center,
            scenario.obstacle.radius,
            config.goal_tolerance,
            statuses=None,
        )
        result = MantaLMPCRunResult(
            scenario_name=scenario.name,
            histories=[_snapshot_safe_sets(safe_sets)],
            controls_by_iteration=[],
            slack_by_iteration=[],
            statuses_by_iteration=[],
            success_by_iteration=[initial_validation.valid],
            goal_reached_by_iteration=[initial_validation.all_goals_reached],
            learned_by_iteration=[initial_validation.usable_for_learning],
            validation_by_iteration=[initial_validation],
            selected_iteration=0 if initial_validation.valid else None,
            initial_controls={a: c.copy() for a, c in safe_controls.items()},
            learning_settings=settings,
        )
        if on_iteration_completed is not None:
            on_iteration_completed(deepcopy(result))
    else:
        if resume_result.learning_settings != settings:
            raise ValueError(
                "resume numerical settings do not match the saved result; "
                "reuse the original scenario, APF, dynamics, and LMPC settings"
            )
        result = deepcopy(resume_result)
        revalidate_result(result, scenario, config)
        if result.completed_iterations == config.iterations:
            return result
        admitted = [
            i
            for i, v in enumerate(result.validation_by_iteration)
            if v.usable_for_learning
        ]
        if not admitted:
            raise ValueError("checkpoint has no complete, collision-free learning seed")
        seed = admitted[-1]
        safe_sets = _snapshot_safe_sets(result.histories[seed])
        safe_controls = (
            {a: c.copy() for a, c in result.initial_controls.items()}
            if seed == 0
            else _controls_by_agent(result.controls_by_iteration[seed - 1], num_agents)
        )
        if verbose:
            print(
                f"Resuming after LMPC iteration {result.completed_iterations} "
                f"using learning seed {seed}..."
            )

    if result.completed_iterations >= config.iterations:
        return result

    if not any(v.usable_for_learning for v in result.validation_by_iteration):
        raise ValueError(
            "APF seed is incomplete or unsafe; increase APF max_steps or adjust "
            "the scenario in a new run before starting LMPC learning"
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

    for iteration in range(result.completed_iterations + 1, config.iterations + 1):
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
        result.histories.append(_snapshot_safe_sets(execution.history))
        result.controls_by_iteration.append(execution.controls)
        result.slack_by_iteration.append(execution.slacks)
        result.statuses_by_iteration.append(execution.statuses)
        validation = validate_trajectory(
            execution.history,
            goals,
            scenario.safety_distance,
            scenario.obstacle.center,
            scenario.obstacle.radius,
            config.goal_tolerance,
            statuses=execution.statuses,
        )
        result.validation_by_iteration.append(validation)
        result.goal_reached_by_iteration.append(validation.all_goals_reached)
        result.success_by_iteration.append(validation.valid)
        result.learned_by_iteration.append(validation.usable_for_learning)
        if validation.usable_for_learning:
            safe_sets = {agent: execution.history[agent] for agent in range(num_agents)}
            safe_controls = _controls_by_agent(execution.controls, num_agents)
            result.histories[-1] = _snapshot_safe_sets(safe_sets)
            if verbose:
                print(
                    "  learned: complete safe trajectory kept for the next iteration."
                )
        elif validation.safe and verbose:
            print(
                "  skipped learning: trajectory was safe but did not reach all goals."
            )
        if validation.valid:
            result.selected_iteration = len(result.histories) - 1
        elif verbose:
            print(
                "  invalid: keeping previous safe set "
                f"(pair violations={validation.pairwise_violation_count}, "
                f"obstacle violations={validation.obstacle_violation_count}, "
                f"fallbacks={validation.fallback_count}, "
                f"safety interventions={validation.safety_intervention_count})."
            )
        if on_iteration_completed is not None:
            on_iteration_completed(deepcopy(result))

    return result


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
