"""Validate retained learning data and derive admission decisions from states."""

from __future__ import annotations

import numpy as np

from scripts.configuration.manta_mpc import MantaLMPCConfig
from scripts.contracts.manta import MantaLMPCRunResult, Scenario
from scripts.metrics.evaluation import validate_trajectory


def revalidate_result(
    result: MantaLMPCRunResult,
    scenario: Scenario,
    config: MantaLMPCConfig,
) -> None:
    """Check aligned histories/telemetry and recompute every learning verdict.

    Archive flags are never an authority for admitting a trajectory. NaN slack
    telemetry is permitted because held/fallback steps have no solver output.
    """
    count = len(result.histories)
    if result.scenario_name != scenario.name:
        raise ValueError("checkpoint scenario does not match the requested scenario")
    if count < 1 or count - 1 > config.iterations:
        raise ValueError(
            "checkpoint completed iterations exceed the requested iterations"
        )
    for values in (
        result.controls_by_iteration,
        result.slack_by_iteration,
        result.statuses_by_iteration,
    ):
        if len(values) != count - 1:
            raise ValueError("checkpoint iteration histories and telemetry must align")
    agents = set(range(len(scenario.starts)))
    if len(agents) < 2:
        raise ValueError("manta LMPC requires at least two agents")
    if result.initial_controls is None or set(result.initial_controls) != agents:
        raise ValueError("checkpoint must retain APF controls for every agent")

    validations = []
    for iteration, history in enumerate(result.histories):
        if not isinstance(history, dict) or set(history) != agents:
            raise ValueError("checkpoint history must contain every scenario agent")
        lengths = set()
        for agent, values in history.items():
            states = np.asarray(values)
            if states.ndim != 2 or states.shape[1] != 7 or len(states) < 1:
                raise ValueError("checkpoint states must have shape (T>=1, 7)")
            if not np.all(np.isfinite(states)):
                raise ValueError("checkpoint states must be finite")
            if not np.allclose(states[0], scenario.starts[agent], atol=1e-12, rtol=0):
                raise ValueError("checkpoint initial states do not match the scenario")
            lengths.add(len(states))
        if len(lengths) != 1:
            raise ValueError("checkpoint agent history lengths must align")
        steps = lengths.pop() - 1
        statuses = None
        if iteration == 0:
            for agent in agents:
                _check_controls(result.initial_controls[agent], (steps, 2))
        else:
            _check_controls(
                result.controls_by_iteration[iteration - 1], (steps, len(agents), 2)
            )
            slacks = np.asarray(result.slack_by_iteration[iteration - 1])
            if slacks.shape != (steps, len(agents), 3) or np.any(np.isinf(slacks)):
                raise ValueError(
                    "checkpoint slack telemetry has invalid shape or infinity"
                )
            if np.any(slacks < 0):
                raise ValueError(
                    "checkpoint slack telemetry must be nonnegative or NaN"
                )
            statuses = result.statuses_by_iteration[iteration - 1]
            if len(statuses) != steps or any(
                not isinstance(row, dict)
                or set(row) != agents
                or any(
                    not isinstance(value, str) or not value for value in row.values()
                )
                for row in statuses
            ):
                raise ValueError(
                    "checkpoint statuses must align with every step and agent"
                )
        validations.append(
            validate_trajectory(
                history,
                scenario.goals,
                scenario.safety_distance,
                scenario.obstacle.center,
                scenario.obstacle.radius,
                config.goal_tolerance,
                statuses=statuses,
            )
        )
    result.validation_by_iteration = validations
    result.success_by_iteration = [v.valid for v in validations]
    result.goal_reached_by_iteration = [v.all_goals_reached for v in validations]
    result.learned_by_iteration = [v.usable_for_learning for v in validations]
    result.selected_iteration = next(
        (i for i in reversed(range(count)) if validations[i].valid),
        None,
    )


def _check_controls(values: np.ndarray, shape: tuple[int, ...]) -> None:
    controls = np.asarray(values)
    if controls.shape != shape or not np.all(np.isfinite(controls)):
        raise ValueError(f"checkpoint controls must be finite and have shape {shape}")
