"""Algorithms / harbor / prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scripts.contracts.harbor import AgentMessage
    from scripts.dynamics.harbor.base import PlatformModel


def _goal_bounded_velocity_prediction(
    message: AgentMessage,
    times: np.ndarray,
    alignment_threshold: float,
) -> np.ndarray:
    """Extrapolate velocity without carrying an aligned agent past its intent."""
    position = np.asarray(message.position, dtype=float).reshape(3)
    velocity = np.asarray(message.velocity, dtype=float).reshape(3)
    prediction_times = np.asarray(times, dtype=float).reshape(-1, 1)
    speed = float(np.linalg.norm(velocity))
    goal_delta = np.asarray(message.goal, dtype=float).reshape(3) - position
    goal_distance = float(np.linalg.norm(goal_delta))
    if speed <= 1e-9 or goal_distance <= 1e-9:
        return np.repeat(position[None, :], len(prediction_times), axis=0)
    direction = velocity / speed
    alignment = float(np.dot(direction, goal_delta / goal_distance))
    if alignment < alignment_threshold:
        return position + prediction_times * velocity
    along_track_remaining = max(float(np.dot(goal_delta, direction)), 0.0)
    travel = np.minimum(speed * prediction_times, along_track_remaining)
    return position + travel * direction


def _approach_pose_goal(
    model: PlatformModel,
    navigation_goal: np.ndarray,
    desired_velocity: np.ndarray,
) -> np.ndarray:
    """Use line-of-sight yaw while a planar underactuated agent is moving."""
    goal = np.asarray(navigation_goal, dtype=float).copy()
    desired = np.asarray(desired_velocity, dtype=float)
    if model.kind in {"ugv", "usv"} and np.linalg.norm(desired[:2]) > 1e-6:
        goal[2] = np.arctan2(desired[1], desired[0])
    return goal


def _control_position_drift(
    model: PlatformModel,
    state: np.ndarray,
    estimate: np.ndarray,
    projection: str,
    desired_velocity: np.ndarray | None = None,
) -> np.ndarray:
    """Project USV drift onto its surge-controllable local subspace."""
    drift = np.asarray(estimate, dtype=float).copy()
    moving = (
        desired_velocity is not None
        and np.linalg.norm(np.asarray(desired_velocity, dtype=float)[:2]) > 1.0e-6
    )
    if (
        projection == "full"
        or model.kind != "usv"
        or (projection == "station_keeping_subspace" and moving)
    ):
        return drift
    yaw = float(np.asarray(state, dtype=float)[2])
    surge = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    return float(np.dot(drift, surge)) * surge


def _nearest_reference_index(
    model: PlatformModel, states: np.ndarray, current_state: np.ndarray
) -> int:
    positions = np.asarray([model.position(state) for state in states])
    current = model.position(current_state)
    return int(np.argmin(np.linalg.norm(positions - current, axis=1)))


def _sample_safe_states(
    states: np.ndarray, target_index: int, sample_count: int
) -> tuple[np.ndarray, np.ndarray]:
    trajectory = np.asarray(states, dtype=float)
    indices = np.minimum(
        np.arange(target_index, target_index + sample_count), len(trajectory) - 1
    )
    costs = (len(trajectory) - 1 - indices).astype(float)
    return trajectory[indices], costs


def _warm_reference(
    states: np.ndarray,
    controls: np.ndarray,
    start_index: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    state_indices = np.minimum(
        np.arange(start_index, start_index + horizon + 1), len(states) - 1
    )
    if len(controls):
        control_indices = np.minimum(
            np.arange(start_index, start_index + horizon), len(controls) - 1
        )
        warm_controls = np.asarray(controls, dtype=float)[control_indices]
    else:
        warm_controls = np.zeros((horizon, 0))
    return np.asarray(states, dtype=float)[state_indices], warm_controls
