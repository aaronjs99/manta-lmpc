"""Algorithms / harbor / guidance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scripts.dynamics.harbor.base import PlatformModel


def _coordinate_velocity(
    name, position, goal, cruise_speed, desired, inbox, config, current_step
):
    adjusted = np.asarray(desired, dtype=float).copy()
    for other_name, message in inbox.items():
        message_position = message.position
        if config.predict_delayed_messages:
            message_age = max(0, current_step - message.sent_step)
            message_position = (
                message.position + message_age * config.dt * message.velocity
            )
        relative = position - message_position
        distance = float(np.linalg.norm(relative))
        if distance >= config.coordination_distance or distance < 1e-9:
            continue
        closing = float(np.dot(desired - message.velocity, -relative / distance))
        if closing <= 0.0:
            continue
        should_yield = name > other_name
        if config.coordination_policy == "eta_priority":
            own_eta = np.linalg.norm(goal - position) / cruise_speed
            other_eta = (
                np.linalg.norm(message.goal - message_position) / message.cruise_speed
            )
            should_yield = (own_eta, name) > (other_eta, other_name)
        if should_yield:
            adjusted *= config.yielding_speed_scale
            planar = relative[:2] / distance
            adjusted[:2] += config.avoidance_gain * np.array([-planar[1], planar[0]])
        else:
            response_scale = (
                1.0
                if config.coordination_policy == "reciprocal"
                else config.priority_response_scale
            )
            adjusted += response_scale * config.avoidance_gain * relative / distance
    return adjusted


def _platform_speed(model: PlatformModel) -> float:
    mission_speed = getattr(model, "mission_speed", None)
    if mission_speed is not None:
        return float(mission_speed)
    if model.kind == "rov":
        return float(model.max_horizontal_speed)
    return float(model.max_speed)


def _goal_reached(agent, state, goal, config) -> bool:
    position_error = np.linalg.norm(
        agent.model.position(state) - agent.model.goal_position(goal)
    )
    orientation_error = np.linalg.norm(agent.model.orientation_error(state, goal))
    return bool(
        position_error <= config.goal_tolerance
        and orientation_error <= config.orientation_tolerance
    )


def _intermediate_waypoint_complete(agent, state, route, index, config) -> bool:
    """Advance a waypoint when reached or crossed toward the next route pose."""
    position = agent.model.position(state)
    waypoint = agent.model.goal_position(route[index])
    if np.linalg.norm(position - waypoint) <= config.goal_tolerance:
        return True
    next_position = agent.model.goal_position(route[index + 1])
    outgoing = next_position - waypoint
    length_sq = float(np.dot(outgoing, outgoing))
    if length_sq <= np.finfo(float).eps:
        return False
    progress = float(np.dot(position - waypoint, outgoing) / length_sq)
    projection = waypoint + np.clip(progress, 0.0, 1.0) * outgoing
    cross_track = float(np.linalg.norm(position - projection))
    return progress >= 0.0 and cross_track <= 2.0 * config.goal_tolerance
