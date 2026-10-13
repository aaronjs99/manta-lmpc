"""Pure information and excitation-based identification helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scripts.configuration.harbor_mpc import HarborMPCConfig
    from scripts.dynamics.harbor.base import PlatformModel


from scripts.algorithms.harbor.effectiveness import _effectiveness_sensitivity_matrix


def _posterior_effectiveness_std(
    information_matrix: np.ndarray, config: HarborMPCConfig
) -> np.ndarray:
    """Approximate local actuator-gain posterior standard deviations."""
    information = np.asarray(information_matrix, dtype=float)
    precision = information + np.eye(len(information)) / (
        config.identification_prior_std**2
    )
    covariance = np.linalg.pinv(precision, hermitian=True)
    return np.sqrt(np.maximum(np.diag(covariance), 0.0))


def _candidate_probe_information(
    model: PlatformModel,
    state: np.ndarray,
    prior: np.ndarray,
    dt: float,
    config: HarborMPCConfig,
) -> np.ndarray:
    """Predict Fisher-information increments for isolated channel probes."""
    increments = np.zeros(
        (model.control_dim, model.control_dim, model.control_dim), dtype=float
    )
    for channel in range(model.control_dim):
        command = np.zeros(model.control_dim, dtype=float)
        command[channel] = (
            config.identification_probe_fraction * model.control_scale()[channel]
        )
        sensitivity = _effectiveness_sensitivity_matrix(
            model,
            state,
            command,
            dt,
            prior,
            config,
            normalize_dynamic_state=True,
        )
        jacobian = sensitivity / config.identification_measurement_noise
        increments[channel] = jacobian.T @ jacobian
    return increments


def _information_identification_channel(
    information_matrix: np.ndarray,
    candidate_increments: np.ndarray,
    probe_counts: np.ndarray,
    rejection_counts: np.ndarray,
    config: HarborMPCConfig,
    effectiveness_estimate: np.ndarray | None = None,
) -> int | None:
    """Select a safe probe by fault-focused expected information gain."""
    information = np.asarray(information_matrix, dtype=float)
    dimension = len(information)
    increments = np.asarray(candidate_increments, dtype=float)
    if information.shape != (dimension, dimension) or increments.shape != (
        dimension,
        dimension,
        dimension,
    ):
        raise ValueError("actuator information matrices have inconsistent shapes")
    counts = np.asarray(probe_counts, dtype=int).reshape(dimension)
    rejections = np.asarray(rejection_counts, dtype=int).reshape(dimension)
    available = rejections < config.identification_max_rejections
    posterior_std = _posterior_effectiveness_std(information, config)
    estimate = (
        np.ones(dimension, dtype=float)
        if effectiveness_estimate is None
        else np.asarray(effectiveness_estimate, dtype=float).reshape(dimension)
    )
    fault_evidence = np.abs(1.0 - estimate) / config.identification_prior_std
    calibration = np.flatnonzero(
        available & (counts < config.identification_min_probes_per_channel)
    )
    if len(calibration):
        minimum_count = int(np.min(counts[calibration]))
        candidates = calibration[counts[calibration] == minimum_count]
    else:
        candidates = np.flatnonzero(
            available & (posterior_std > config.identification_target_std)
        )
    if len(candidates) == 0:
        return None
    precision = information + np.eye(dimension) / (config.identification_prior_std**2)
    base_sign, base_logdet = np.linalg.slogdet(precision)
    if base_sign <= 0.0:
        raise ValueError("actuator information precision must be positive definite")
    scores = []
    for channel in candidates:
        sign, logdet = np.linalg.slogdet(precision + increments[channel])
        gain = -np.inf if sign <= 0.0 else logdet - base_logdet
        focus = 1.0 + config.identification_fault_focus_weight * fault_evidence[channel]
        scores.append(gain * posterior_std[channel] * focus)
    return int(candidates[int(np.argmax(scores))])


def _least_excited_channel(
    excitation_energy: np.ndarray, target_energy: float
) -> int | None:
    """Return the least-observed channel below a common information target."""
    energy = np.asarray(excitation_energy, dtype=float).reshape(-1)
    candidates = np.flatnonzero(energy < target_energy)
    if len(candidates) == 0:
        return None
    return int(candidates[np.argmin(energy[candidates])])


def _identification_channel(
    excitation_energy: np.ndarray,
    probe_counts: np.ndarray,
    rejection_counts: np.ndarray,
    config: HarborMPCConfig,
) -> int | None:
    """Prioritize calibration quotas, then remaining low-energy channels."""
    energy = np.asarray(excitation_energy, dtype=float).reshape(-1)
    counts = np.asarray(probe_counts, dtype=int).reshape(energy.shape)
    rejections = np.asarray(rejection_counts, dtype=int).reshape(energy.shape)
    available = rejections < config.identification_max_rejections
    calibration = np.flatnonzero(
        available & (counts < config.identification_min_probes_per_channel)
    )
    if len(calibration):
        ordering = np.lexsort((energy[calibration], counts[calibration]))
        return int(calibration[ordering[0]])
    eligible_energy = energy.copy()
    eligible_energy[~available] = config.identification_target_energy
    return _least_excited_channel(
        eligible_energy,
        config.identification_target_energy,
    )
