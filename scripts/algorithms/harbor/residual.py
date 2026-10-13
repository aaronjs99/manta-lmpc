"""Pure local motion-residual and kinematic-current estimators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scripts.configuration.harbor_mpc import HarborMPCConfig
    from scripts.dynamics.harbor.base import PlatformModel


def _constant_bias_residual_update(
    prior: np.ndarray,
    variance: np.ndarray,
    measurement: np.ndarray,
    config: HarborMPCConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Update independent constant-current states with innovation gating."""
    estimate = np.asarray(prior, dtype=float).copy()
    predicted_variance = (
        np.asarray(variance, dtype=float) + config.residual_rls_process_noise
    )
    innovation = np.asarray(measurement, dtype=float) - estimate
    innovation_variance = predicted_variance + config.residual_rls_measurement_noise**2
    normalized = np.abs(innovation) / np.sqrt(innovation_variance)
    accepted = normalized <= config.residual_rls_innovation_gate
    gain = predicted_variance / innovation_variance
    estimate[accepted] += gain[accepted] * innovation[accepted]
    posterior_variance = predicted_variance.copy()
    posterior_variance[accepted] *= 1.0 - gain[accepted]
    return estimate, posterior_variance, int(np.count_nonzero(~accepted))


def _kinematic_position_drift_measurement(
    model: PlatformModel,
    previous_state: np.ndarray,
    current_state: np.ndarray,
    dt: float,
    elapsed_steps: int,
) -> np.ndarray:
    """Infer world-frame drift without using controls or actuator estimates."""
    prior = np.asarray(previous_state, dtype=float)
    current = np.asarray(current_state, dtype=float)
    ground_velocity = (model.position(current) - model.position(prior)) / dt
    if elapsed_steps == 1:
        velocity_state = current.copy()
        if model.kind == "usv":
            velocity_state[2] = prior[2]
        elif model.kind == "rov":
            velocity_state[3:6] = prior[3:6]
        through_water_velocity = model.velocity(velocity_state)
    else:
        through_water_velocity = 0.5 * (model.velocity(prior) + model.velocity(current))
    return ground_velocity - through_water_velocity
