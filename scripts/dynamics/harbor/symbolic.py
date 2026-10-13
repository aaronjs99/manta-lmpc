"""Dynamics / harbor / symbolic."""

from __future__ import annotations


def _symbolic_marine_coriolis_product(ca, velocity, mass):
    if int(velocity.numel()) == 3:
        return ca.vertcat(
            -mass[1] * velocity[1] * velocity[2],
            mass[0] * velocity[0] * velocity[2],
            (mass[1] - mass[0]) * velocity[0] * velocity[1],
        )
    linear, angular = velocity[:3], velocity[3:]
    linear_momentum = ca.times(mass[:3], linear)
    angular_momentum = ca.times(mass[3:], angular)
    return ca.vertcat(
        ca.cross(angular, linear_momentum),
        ca.cross(linear, linear_momentum) + ca.cross(angular, angular_momentum),
    )


def _symbolic_damping_product(ca, velocity, linear, quadratic):
    return ca.times(ca.DM(linear), velocity) + ca.times(
        ca.DM(quadratic), ca.times(ca.fabs(velocity), velocity)
    )


def _symbolic_rotation_matrix(ca, angles):
    roll, pitch, yaw = angles[0], angles[1], angles[2]
    cr, sr = ca.cos(roll), ca.sin(roll)
    cp, sp = ca.cos(pitch), ca.sin(pitch)
    cy, sy = ca.cos(yaw), ca.sin(yaw)
    return ca.vertcat(
        ca.horzcat(cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        ca.horzcat(sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        ca.horzcat(-sp, cp * sr, cp * cr),
    )


def _symbolic_euler_rate_matrix(ca, angles):
    roll, pitch = angles[0], angles[1]
    cr, sr = ca.cos(roll), ca.sin(roll)
    cp = ca.cos(pitch)
    tp = ca.tan(pitch)
    return ca.vertcat(
        ca.horzcat(1.0, sr * tp, cr * tp),
        ca.horzcat(0.0, cr, -sr),
        ca.horzcat(0.0, sr / cp, cr / cp),
    )


def _symbolic_restoring_vector(ca, angles, model):
    rotation = _symbolic_rotation_matrix(ca, angles)
    gravity = rotation.T @ ca.DM([0.0, 0.0, -model.weight])
    lift = rotation.T @ ca.DM([0.0, 0.0, model.buoyancy])
    moment = ca.cross(ca.DM(model.center_of_gravity), gravity) + ca.cross(
        ca.DM(model.center_of_buoyancy), lift
    )
    return -ca.vertcat(gravity + lift, moment)
