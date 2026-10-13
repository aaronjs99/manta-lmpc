"""Platform-specific guidance policies kept outside the physical models."""

from __future__ import annotations

import numpy as np
from scripts.dynamics.harbor.base import PlatformModel, wrap_angle
from scripts.dynamics.harbor.ground import SkidSteerUGVModel, UGVModel
from scripts.dynamics.harbor.reduced import (
    ReducedROVModel,
    ReducedUGVModel,
    ReducedUSVModel,
)
from scripts.dynamics.harbor.surface import (
    USVModel,
    _damping_product,
    _marine_coriolis_product,
)
from scripts.dynamics.harbor.underwater import (
    ROVModel,
    _restoring_vector,
    _rotation_matrix,
)
from scripts.dynamics.harbor.underwater import (
    _damping_product as _rov_damping_product,
)
from scripts.dynamics.harbor.underwater import (
    _marine_coriolis_product as _rov_marine_coriolis_product,
)


def guidance_control(
    model: PlatformModel,
    state: np.ndarray,
    desired_velocity: np.ndarray,
    dt: float,
    desired_pose: np.ndarray | None = None,
) -> np.ndarray:
    """Map a guidance velocity/pose request into the model's physical controls."""
    if isinstance(model, UGVModel):
        return _ugv_guidance(model, state, desired_velocity, dt, desired_pose)
    if isinstance(model, SkidSteerUGVModel):
        return _skid_steer_guidance(model, state, desired_velocity, desired_pose)
    if isinstance(model, ReducedUGVModel):
        return _reduced_ugv_guidance(model, state, desired_velocity, dt, desired_pose)
    if isinstance(model, ReducedUSVModel):
        return _reduced_usv_guidance(model, state, desired_velocity, dt, desired_pose)
    if isinstance(model, ReducedROVModel):
        return _reduced_rov_guidance(model, state, desired_velocity, desired_pose)
    if isinstance(model, USVModel):
        return _usv_guidance(model, state, desired_velocity, dt, desired_pose)
    if isinstance(model, ROVModel):
        return _rov_guidance(model, state, desired_velocity, desired_pose)
    raise TypeError(f"unsupported Harbor guidance model: {type(model).__name__}")


def _ugv_guidance(model, state, desired_velocity, dt, desired_pose):
    desired = np.asarray(desired_velocity, dtype=float)[:2]
    if desired_pose is None:
        target = np.asarray(state[:3], dtype=float)
        target[:2] += desired * dt
    else:
        target = np.asarray(desired_pose, dtype=float)
    delta = target[:2] - state[:2]
    distance = float(np.linalg.norm(delta))
    requested_speed = float(np.linalg.norm(desired))
    heading_to_goal = (
        float(np.arctan2(desired[1], desired[0]))
        if requested_speed > 1e-9
        else state[2]
        if distance < 1e-9
        else float(np.arctan2(delta[1], delta[0]))
    )
    alpha = wrap_angle(heading_to_goal - state[2])
    beta = wrap_angle(target[2] - state[2] - alpha)
    speed = model.position_gain * distance * np.cos(alpha)
    speed = np.clip(speed, -model.max_reverse_speed, model.max_speed)
    orientation_error = abs(wrap_angle(target[2] - state[2]))
    speed_cap = max(
        requested_speed, min(0.2, distance) if orientation_error > 0.05 else 0.0
    )
    speed = np.clip(speed, -speed_cap, speed_cap)
    desired_yaw_rate = (
        model.pose_heading_gain * alpha + model.pose_goal_heading_gain * beta
    )
    reference_speed = np.copysign(max(abs(speed), 0.08), speed or 1.0)
    steering = np.arctan(model.wheelbase * desired_yaw_rate / reference_speed)
    return np.array(
        [
            np.clip(
                (speed - state[3]) / dt, -model.max_acceleration, model.max_acceleration
            ),
            np.clip(steering, -model.max_steering_angle, model.max_steering_angle),
        ]
    )


def _skid_steer_guidance(model, state, desired_velocity, desired_pose):
    desired = np.asarray(desired_velocity, dtype=float)[:2]
    requested_speed = min(float(np.linalg.norm(desired)), model.max_speed)
    if requested_speed > 1e-9:
        desired_heading = float(np.arctan2(desired[1], desired[0]))
    elif desired_pose is not None:
        desired_heading = float(desired_pose[2])
    else:
        desired_heading = float(state[2])
    heading_error = wrap_angle(desired_heading - state[2])
    signed_speed = requested_speed * max(0.0, np.cos(heading_error))
    desired_yaw_rate = np.clip(
        model.heading_gain * heading_error, -model.max_yaw_rate, model.max_yaw_rate
    )
    force = (
        model.mass * model.velocity_response_gain * (signed_speed - state[3])
        + model.linear_drag * state[3]
        + model.quadratic_drag * abs(state[3]) * state[3]
    )
    moment = (
        model.yaw_inertia * model.yaw_response_gain * (desired_yaw_rate - state[4])
        + model.yaw_linear_drag * state[4]
        + model.yaw_quadratic_drag * abs(state[4]) * state[4]
    )
    force = np.clip(force, -model.max_force, model.max_force)
    moment = np.clip(moment, -model.max_yaw_moment, model.max_yaw_moment)
    left = 0.5 * force - moment / model.effective_track
    right = 0.5 * force + moment / model.effective_track
    return np.clip(np.array([left, right]), -model.max_side_force, model.max_side_force)


def _reduced_ugv_guidance(model, state, desired_velocity, dt, desired_pose):
    desired = np.asarray(desired_velocity, dtype=float)[:2]
    speed = min(float(np.linalg.norm(desired)), model.max_speed)
    heading = (
        float(desired_pose[2])
        if speed < 1e-9 and desired_pose is not None
        else state[2]
        if speed < 1e-9
        else float(np.arctan2(desired[1], desired[0]))
    )
    return np.array(
        [
            np.clip(
                (speed - state[3]) / dt, -model.max_acceleration, model.max_acceleration
            ),
            np.clip(
                wrap_angle(heading - state[2]) / dt,
                -model.max_yaw_rate,
                model.max_yaw_rate,
            ),
        ]
    )


def _reduced_usv_guidance(model, state, desired_velocity, dt, desired_pose):
    desired = np.asarray(desired_velocity, dtype=float)[:2]
    speed = min(float(np.linalg.norm(desired)), model.max_speed)
    heading = (
        float(desired_pose[2])
        if speed < 1e-9 and desired_pose is not None
        else state[2]
        if speed < 1e-9
        else float(np.arctan2(desired[1], desired[0]))
    )
    thrust = (speed - state[3]) / dt + model.drag * state[3]
    return np.array(
        [
            np.clip(thrust, -model.max_thrust, model.max_thrust),
            np.clip(
                wrap_angle(heading - state[2]) / dt,
                -model.max_yaw_rate,
                model.max_yaw_rate,
            ),
        ]
    )


def _reduced_rov_guidance(model, state, desired_velocity, desired_pose):
    desired = np.asarray(desired_velocity, dtype=float)
    planar_speed = np.linalg.norm(desired[:2])
    if planar_speed > model.max_horizontal_speed:
        desired[:2] *= model.max_horizontal_speed / planar_speed
    desired[2] = np.clip(
        desired[2], -model.max_vertical_speed, model.max_vertical_speed
    )
    angular_error = (
        np.zeros(3)
        if desired_pose is None
        else model.orientation_error(state, desired_pose)
    )
    desired_rates = np.clip(
        model.attitude_response_gain * angular_error,
        -model.max_angular_rate,
        model.max_angular_rate,
    )
    forces = np.clip(
        model.velocity_response_gain * (desired - state[6:9])
        + model.linear_drag * state[6:9],
        -model.max_force,
        model.max_force,
    )
    torques = np.clip(
        model.angular_rate_response_gain * (desired_rates - state[9:12])
        + model.angular_drag * state[9:12],
        -model.max_torque,
        model.max_torque,
    )
    return np.concatenate((forces, torques))


def _usv_guidance(model, state, desired_velocity, dt, desired_pose):
    desired = np.asarray(desired_velocity, dtype=float)[:2]
    speed = min(float(np.linalg.norm(desired)), model.max_speed)
    heading = (
        float(desired_pose[2])
        if speed < 1e-9 and desired_pose is not None
        else state[2]
        if speed < 1e-9
        else float(np.arctan2(desired[1], desired[0]))
    )
    desired_yaw_rate = np.clip(
        model.heading_gain * wrap_angle(heading - state[2]),
        -model.max_yaw_rate,
        model.max_yaw_rate,
    )
    nu = np.asarray(state[3:6], dtype=float)
    coriolis = _marine_coriolis_product(nu, model.mass_diagonal)
    damping = _damping_product(nu, model.linear_damping, model.quadratic_damping)
    thrust = (
        model.mass_diagonal[0] * model.velocity_response_gain * (speed - nu[0])
        + coriolis[0]
        + damping[0]
    )
    moment = (
        model.mass_diagonal[2] * model.yaw_response_gain * (desired_yaw_rate - nu[2])
        + coriolis[2]
        + damping[2]
    )
    thrust = np.clip(thrust, -model.max_thrust, model.max_thrust)
    moment = np.clip(moment, -model.max_yaw_moment, model.max_yaw_moment)
    port = 0.5 * thrust - moment / model.waterjet_separation
    starboard = 0.5 * thrust + moment / model.waterjet_separation
    return np.clip(
        np.array([port, starboard]), -model.max_jet_thrust, model.max_jet_thrust
    )


def _rov_guidance(model, state, desired_velocity, desired_pose):
    desired_world = np.asarray(desired_velocity, dtype=float).copy()
    planar_speed = np.linalg.norm(desired_world[:2])
    if planar_speed > model.max_horizontal_speed:
        desired_world[:2] *= model.max_horizontal_speed / planar_speed
    desired_world[2] = np.clip(
        desired_world[2], -model.max_vertical_speed, model.max_vertical_speed
    )
    rotation = _rotation_matrix(state[3:6])
    desired_body = rotation.T @ desired_world
    angular_error = (
        np.zeros(3)
        if desired_pose is None
        else model.orientation_error(state, desired_pose)
    )
    desired_rates = np.clip(
        model.attitude_response_gain * angular_error,
        -model.max_angular_rate,
        model.max_angular_rate,
    )
    nu = np.asarray(state[6:12], dtype=float)
    desired_nu = np.concatenate((desired_body, desired_rates))
    acceleration = np.concatenate(
        (
            model.velocity_response_gain * (desired_nu[:3] - nu[:3]),
            model.angular_rate_response_gain * (desired_nu[3:] - nu[3:]),
        )
    )
    tau = (
        np.asarray(model.mass_diagonal) * acceleration
        + _rov_marine_coriolis_product(nu, model.mass_diagonal)
        + _rov_damping_product(nu, model.linear_damping, model.quadratic_damping)
        + _restoring_vector(
            state[3:6],
            model.weight,
            model.buoyancy,
            model.center_of_gravity,
            model.center_of_buoyancy,
        )
    )
    tau[:3] = np.clip(tau[:3], -model.force_limit_vector, model.force_limit_vector)
    tau[3:] = np.clip(tau[3:], -model.torque_limit_vector, model.torque_limit_vector)
    return model.allocate_wrench(tau)
