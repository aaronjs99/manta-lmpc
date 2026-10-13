"""Dynamics / harbor / mechanics."""

from __future__ import annotations

import numpy as np


def _freeze_vectors(instance, *names: str) -> None:
    for name in names:
        object.__setattr__(
            instance,
            name,
            tuple(float(value) for value in getattr(instance, name)),
        )


def _rotation_matrix(angles: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(angles, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _euler_rate_matrix(angles: np.ndarray) -> np.ndarray:
    roll, pitch, _ = np.asarray(angles, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp = np.cos(pitch)
    cp = np.copysign(max(abs(cp), 1e-6), cp)
    tp = np.sin(pitch) / cp
    return np.array([[1.0, sr * tp, cr * tp], [0.0, cr, -sr], [0.0, sr / cp, cr / cp]])


def _marine_coriolis_product(
    velocity: np.ndarray, mass_diagonal: tuple[float, ...]
) -> np.ndarray:
    nu = np.asarray(velocity, dtype=float)
    mass = np.asarray(mass_diagonal, dtype=float)
    if len(nu) == 3:
        surge, sway, yaw_rate = nu
        return np.array(
            [
                -mass[1] * sway * yaw_rate,
                mass[0] * surge * yaw_rate,
                (mass[1] - mass[0]) * surge * sway,
            ]
        )
    linear, angular = nu[:3], nu[3:]
    linear_momentum = mass[:3] * linear
    angular_momentum = mass[3:] * angular
    return np.concatenate(
        (
            np.cross(angular, linear_momentum),
            np.cross(linear, linear_momentum) + np.cross(angular, angular_momentum),
        )
    )


def _damping_product(
    velocity: np.ndarray,
    linear: tuple[float, ...],
    quadratic: tuple[float, ...],
) -> np.ndarray:
    nu = np.asarray(velocity, dtype=float)
    return np.asarray(linear) * nu + np.asarray(quadratic) * np.abs(nu) * nu


def _restoring_vector(
    angles: np.ndarray,
    weight: float,
    buoyancy: float,
    center_of_gravity: tuple[float, float, float],
    center_of_buoyancy: tuple[float, float, float],
) -> np.ndarray:
    rotation = _rotation_matrix(angles)
    gravity = rotation.T @ np.array([0.0, 0.0, -weight])
    lift = rotation.T @ np.array([0.0, 0.0, buoyancy])
    external = np.concatenate(
        (
            gravity + lift,
            np.cross(center_of_gravity, gravity) + np.cross(center_of_buoyancy, lift),
        )
    )
    return -external
