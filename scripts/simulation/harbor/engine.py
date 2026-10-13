"""Simulation / harbor / engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scripts.algorithms.harbor.geometry import pairwise_clearance_metrics
from scripts.algorithms.harbor.guidance import (
    _coordinate_velocity,
    _goal_reached,
    _intermediate_waypoint_complete,
    _platform_speed,
)
from scripts.algorithms.harbor.guidance_control import guidance_control
from scripts.communication.harbor import CommunicationNetwork
from scripts.configuration.harbor_simulation import (
    HarborDisturbanceConfig,
    HarborObservationNoiseConfig,
)
from scripts.contracts.harbor import HarborResult

if TYPE_CHECKING:
    from scripts.communication.harbor import LinkConfig
    from scripts.configuration.harbor_simulation import HarborSimulationConfig
    from scripts.contracts.harbor import HarborAgent, HarborControlProvider
    from scripts.dynamics.harbor.base import PlatformModel


def run_harbor_simulation(
    agents: list[HarborAgent],
    config: HarborSimulationConfig,
    communication: LinkConfig,
    control_provider: HarborControlProvider | None = None,
    disturbance: HarborDisturbanceConfig | None = None,
    observation_noise: HarborObservationNoiseConfig | None = None,
    localization_provider=None,
) -> HarborResult:
    """Simulate independent platforms coordinated only by received messages."""
    if len(agents) < 2:
        raise ValueError("harbor simulation requires at least two platforms")
    names = [agent.name for agent in agents]
    if len(set(names)) != len(names):
        raise ValueError("harbor platform names must be unique")

    by_name = {agent.name: agent for agent in agents}
    current = {
        agent.name: np.asarray(agent.start, dtype=float).copy() for agent in agents
    }
    histories = {name: [state.copy()] for name, state in current.items()}
    sensor = observation_noise or HarborObservationNoiseConfig()
    seed_sequence = np.random.SeedSequence(sensor.seed)
    sensor_rngs = {
        agent.name: np.random.default_rng(child)
        for agent, child in zip(agents, seed_sequence.spawn(len(agents)), strict=True)
    }
    observed_histories = {}
    for agent in agents:
        onboard = sensor.measure(agent, current[agent.name], sensor_rngs[agent.name])
        if localization_provider is not None:
            onboard = localization_provider.estimate(
                agent=agent,
                truth=current[agent.name],
                onboard=onboard,
                step=0,
            )
        observed_histories[agent.name] = [onboard]
    controls = {name: [] for name in names}
    applied_controls = {name: [] for name in names}
    applied_effectiveness = {name: [] for name in names}
    plant = disturbance or HarborDisturbanceConfig()
    last_controls: dict[str, np.ndarray] = {}
    guidance_update_count = 0
    first_goal_steps: dict[str, int | None] = {name: None for name in names}
    route_indices = {name: 0 for name in names}
    stopped_at_goal: set[str] = set()
    goal_hold_counts = {
        name: int(
            _goal_reached(by_name[name], current[name], by_name[name].goal, config)
        )
        for name in names
    }
    network = CommunicationNetwork(names, communication)

    for step in range(config.horizon):
        measured = {name: observed_histories[name][-1] for name in names}
        observations = {
            name: (
                agent.model.position(measured[name]),
                agent.model.velocity(measured[name]) + plant.current(agent.model),
                agent.model.goal_position(agent.goal),
                _platform_speed(agent.model),
            )
            for name, agent in by_name.items()
        }
        if all(count >= config.goal_hold_steps for count in goal_hold_counts.values()):
            for name in names:
                if first_goal_steps[name] is None:
                    first_goal_steps[name] = step
            break
        inboxes = network.exchange(step, observations)
        next_states = {}
        for name, agent in by_name.items():
            observe = (
                getattr(control_provider, "observe", None)
                if control_provider is not None
                else None
            )
            if observe is not None:
                observe(
                    agent=agent,
                    state=measured[name],
                    step=step,
                    dt=config.dt,
                )
            position = observations[name][0]
            route = agent.route
            while route_indices[name] < len(
                route
            ) - 1 and _intermediate_waypoint_complete(
                agent,
                measured[name],
                route,
                route_indices[name],
                config,
            ):
                route_indices[name] += 1
            navigation_goal = route[route_indices[name]]
            if localization_provider is not None and hasattr(
                localization_provider, "information_goal"
            ):
                navigation_goal = localization_provider.information_goal(
                    agent, measured[name], navigation_goal
                )
            navigation_position = agent.model.goal_position(navigation_goal)
            goal_delta = navigation_position - position
            goal_distance = float(np.linalg.norm(goal_delta))
            at_final_goal = route_indices[name] == len(route) - 1
            reached_navigation_goal = _goal_reached(
                agent, measured[name], navigation_goal, config
            )
            if (
                at_final_goal
                and first_goal_steps[name] is None
                and _goal_reached(agent, current[name], agent.goal, config)
            ):
                first_goal_steps[name] = step
            if at_final_goal and reached_navigation_goal:
                desired_velocity = np.zeros(3)
            else:
                desired_velocity = np.zeros(3)
                if goal_distance > config.goal_tolerance:
                    desired_velocity = goal_delta / max(goal_distance, 1e-9)
                    desired_velocity *= min(
                        _platform_speed(agent.model),
                        config.approach_speed_gain * goal_distance,
                    )
                    desired_velocity = _coordinate_velocity(
                        name,
                        position,
                        navigation_position,
                        _platform_speed(agent.model),
                        desired_velocity,
                        inboxes[name],
                        config,
                        step,
                    )
            at_goal = at_final_goal and reached_navigation_goal
            left_goal = name in stopped_at_goal and not at_goal
            if left_goal:
                stopped_at_goal.remove(name)
            continuous_station_keeping = control_provider is not None or (
                agent.model.variant in {"marine_3dof", "marine_6dof"}
            )
            update_guidance = name not in last_controls or (
                at_goal and name not in stopped_at_goal
            )
            if at_goal and continuous_station_keeping:
                update_guidance = True
            if left_goal:
                update_guidance = True
            if not at_goal and step % config.guidance_update_interval_steps == 0:
                update_guidance = True
            if update_guidance:
                if control_provider is None:
                    proposed_control = guidance_control(
                        agent.model,
                        measured[name],
                        desired_velocity,
                        config.dt,
                        desired_pose=navigation_goal,
                    )
                    smoothing = float(getattr(agent.model, "control_smoothing", 1.0))
                else:
                    proposed_control = control_provider.control(
                        agent=agent,
                        state=measured[name],
                        navigation_goal=navigation_goal,
                        desired_velocity=desired_velocity,
                        inbox=inboxes[name],
                        step=step,
                        dt=config.dt,
                        belief=(
                            localization_provider.belief(name)
                            if localization_provider is not None
                            and hasattr(localization_provider, "belief")
                            else None
                        ),
                    )
                    smoothing = 1.0
                control = (
                    proposed_control
                    if name not in last_controls
                    else smoothing * proposed_control
                    + (1.0 - smoothing) * last_controls[name]
                )
                last_controls[name] = control
                guidance_update_count += 1
                if at_goal:
                    stopped_at_goal.add(name)
            else:
                control = last_controls[name]
            effectiveness = plant.effectiveness(agent.model, agent.name, step)
            applied_control = effectiveness * control
            next_state = agent.model.step(current[name], applied_control, config.dt)
            current_velocity = plant.current(agent.model)
            next_state = _advect_position(
                agent.model, next_state, current_velocity, config.dt
            )
            next_states[name] = agent.domain.project(agent.model, next_state)
            controls[name].append(control)
            applied_controls[name].append(applied_control)
            applied_effectiveness[name].append(effectiveness)

        current = next_states
        for name in names:
            histories[name].append(current[name].copy())
            onboard = sensor.measure(by_name[name], current[name], sensor_rngs[name])
            if localization_provider is not None:
                onboard = localization_provider.estimate(
                    agent=by_name[name],
                    truth=current[name],
                    onboard=onboard,
                    step=step + 1,
                )
            observed_histories[name].append(onboard)
            goal_hold_counts[name] = (
                goal_hold_counts[name] + 1
                if _goal_reached(
                    by_name[name], current[name], by_name[name].goal, config
                )
                else 0
            )

    for name, agent in by_name.items():
        if first_goal_steps[name] is None:
            if _goal_reached(agent, current[name], agent.goal, config):
                first_goal_steps[name] = config.horizon

    state_arrays = {name: np.asarray(values) for name, values in histories.items()}
    position_arrays = {
        name: np.asarray([by_name[name].model.position(state) for state in values])
        for name, values in state_arrays.items()
    }
    final_goal_errors = {
        name: float(
            np.linalg.norm(
                position_arrays[name][-1]
                - by_name[name].model.goal_position(by_name[name].goal)
            )
        )
        for name in names
    }
    final_orientation_errors = {
        name: float(
            np.linalg.norm(
                by_name[name].model.orientation_error(
                    state_arrays[name][-1], by_name[name].goal
                )
            )
        )
        for name in names
    }
    min_distance, violation_count = pairwise_clearance_metrics(position_arrays, by_name)
    return HarborResult(
        states=state_arrays,
        observed_states={
            name: np.asarray(values) for name, values in observed_histories.items()
        },
        positions=position_arrays,
        controls={name: np.asarray(values) for name, values in controls.items()},
        applied_controls={
            name: np.asarray(values) for name, values in applied_controls.items()
        },
        applied_effectiveness={
            name: np.asarray(values) for name, values in applied_effectiveness.items()
        },
        first_goal_steps=first_goal_steps,
        final_goal_errors=final_goal_errors,
        final_orientation_errors=final_orientation_errors,
        all_goals_reached=all(
            final_goal_errors[name] <= config.goal_tolerance
            and final_orientation_errors[name] <= config.orientation_tolerance
            for name in names
        ),
        min_pairwise_distance=min_distance,
        pairwise_violation_count=violation_count,
        messages_sent=network.sent_count,
        messages_delivered=network.delivered_count,
        messages_dropped=network.dropped_count,
        guidance_update_count=guidance_update_count,
    )


def _advect_position(
    model: PlatformModel, state: np.ndarray, current: np.ndarray, dt: float
) -> np.ndarray:
    """Apply an unmodeled world-frame current to marine position states."""
    value = np.asarray(state, dtype=float).copy()
    dimensions = 3 if model.kind == "rov" else 2
    value[:dimensions] += dt * np.asarray(current, dtype=float)[:dimensions]
    return value
