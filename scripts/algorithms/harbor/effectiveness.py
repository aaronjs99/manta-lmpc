"""Local actuator-effectiveness estimation and recovery mathematics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scripts.configuration.harbor_mpc import HarborMPCConfig
    from scripts.dynamics.harbor.base import PlatformModel


def _estimate_control_effectiveness(
    model: PlatformModel,
    previous_state: np.ndarray,
    previous_control: np.ndarray,
    current_state: np.ndarray,
    dt: float,
    prior: float,
    config: HarborMPCConfig,
) -> float:
    """Fit scalar actuator effectiveness from locally measured dynamic states."""
    command = np.asarray(previous_control, dtype=float)
    normalized_excitation = float(np.linalg.norm(command / model.control_scale()))
    if normalized_excitation < config.effectiveness_excitation_threshold:
        return prior
    epsilon = 0.05
    lower = max(config.effectiveness_min, prior - epsilon)
    upper = min(config.effectiveness_max, prior + epsilon)
    if upper - lower <= np.finfo(float).eps:
        return prior
    dynamic = slice(model.pose_dim, model.state_dim)
    low_state = model.step(previous_state, lower * command, dt)[dynamic]
    high_state = model.step(previous_state, upper * command, dt)[dynamic]
    sensitivity = (high_state - low_state) / (upper - lower)
    denominator = float(np.dot(sensitivity, sensitivity))
    if denominator <= 1e-12:
        return prior
    predicted = model.step(previous_state, prior * command, dt)[dynamic]
    measured = np.asarray(current_state, dtype=float)[dynamic]
    instantaneous = (
        prior + float(np.dot(sensitivity, measured - predicted)) / denominator
    )
    instantaneous = float(
        np.clip(
            instantaneous,
            config.effectiveness_min,
            config.effectiveness_max,
        )
    )
    gain = config.effectiveness_estimator_gain
    return float((1.0 - gain) * prior + gain * instantaneous)


def _effectiveness_change_is_loss(
    previous: np.ndarray, updated: np.ndarray, tolerance: float = 1.0e-6
) -> bool:
    """Classify a detected aggregate actuator change without plant truth."""
    previous = np.asarray(previous, dtype=float)
    updated = np.asarray(updated, dtype=float)
    return bool(np.mean(updated - previous) < -tolerance)


def _effectiveness_change_is_recovery(
    previous: np.ndarray, updated: np.ndarray, tolerance: float = 1.0e-6
) -> bool:
    """Classify an aggregate positive actuator change without plant truth."""
    previous = np.asarray(previous, dtype=float)
    updated = np.asarray(updated, dtype=float)
    return bool(np.mean(updated - previous) > tolerance)


def _apply_nominal_recovery_prior(
    previous: np.ndarray,
    updated: np.ndarray,
    gain: float,
    direction_tolerance: float,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Pull only positively changing actuator channels toward nominal health."""
    previous = np.asarray(previous, dtype=float)
    result = np.asarray(updated, dtype=float).copy()
    result += _channel_selective_recovery_offset(
        previous, result, gain, direction_tolerance
    )
    return np.clip(result, lower, upper)


def _channel_selective_recovery_offset(
    previous: np.ndarray,
    updated: np.ndarray,
    gain: float,
    direction_tolerance: float,
) -> np.ndarray:
    """Return a nominal-health offset only for positively changing channels."""
    previous = np.asarray(previous, dtype=float)
    updated = np.asarray(updated, dtype=float)
    offset = np.zeros_like(updated)
    recovering = updated - previous > direction_tolerance
    offset[recovering] = gain * (1.0 - updated[recovering])
    return offset


def _has_full_column_rank(matrix: np.ndarray, tolerance: float) -> bool:
    """Return whether local measurements identify every actuator direction."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return False
    return bool(np.linalg.matrix_rank(matrix, tol=tolerance) == matrix.shape[1])


def _dynamic_state_scale(model: PlatformModel) -> np.ndarray:
    """Return characteristic velocity/rate scales for local information."""
    if model.kind == "ugv":
        values = [model.max_speed]
        if model.state_dim - model.pose_dim > 1:
            values.append(model.max_yaw_rate)
    elif model.kind == "usv":
        values = [model.max_speed]
        if model.state_dim - model.pose_dim > 1:
            values.extend((model.max_sway_speed, model.max_yaw_rate))
    else:
        values = [
            model.max_horizontal_speed,
            model.max_horizontal_speed,
            model.max_vertical_speed,
            model.max_angular_rate,
            model.max_angular_rate,
            model.max_angular_rate,
        ]
    dynamic_dim = model.state_dim - model.pose_dim
    return np.maximum(np.asarray(values[:dynamic_dim], dtype=float), 1e-9)


def _effectiveness_sensitivity_matrix(
    model: PlatformModel,
    previous_state: np.ndarray,
    previous_control: np.ndarray,
    dt: float,
    prior: np.ndarray,
    config: HarborMPCConfig,
    *,
    normalize_dynamic_state: bool,
) -> np.ndarray:
    """Linearize one-step dynamic response with respect to actuator gains."""
    command = np.asarray(previous_control, dtype=float).reshape(model.control_dim)
    prior = np.asarray(prior, dtype=float).reshape(model.control_dim)
    dynamic = slice(model.pose_dim, model.state_dim)
    sensitivity = np.zeros(
        (model.state_dim - model.pose_dim, model.control_dim), dtype=float
    )
    epsilon = 0.05
    for channel in np.flatnonzero(np.abs(command) > 1e-12):
        lower = prior.copy()
        upper = prior.copy()
        lower[channel] = max(config.effectiveness_min, prior[channel] - epsilon)
        upper[channel] = min(config.effectiveness_max, prior[channel] + epsilon)
        span = upper[channel] - lower[channel]
        if span <= np.finfo(float).eps:
            continue
        low_state = model.step(previous_state, lower * command, dt)[dynamic]
        high_state = model.step(previous_state, upper * command, dt)[dynamic]
        sensitivity[:, channel] = (high_state - low_state) / span
    if normalize_dynamic_state:
        sensitivity /= _dynamic_state_scale(model)[:, None]
    return sensitivity


def _recursive_diagonal_effectiveness_update(
    model: PlatformModel,
    previous_state: np.ndarray,
    previous_control: np.ndarray,
    current_state: np.ndarray,
    dt: float,
    prior: np.ndarray,
    covariance: np.ndarray,
    config: HarborMPCConfig,
    *,
    change_evidence: float = 0.0,
    allow_change: bool = True,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Update per-actuator gains with robust covariance-form recursive LS."""
    prior = np.asarray(prior, dtype=float).reshape(model.control_dim)
    covariance = np.asarray(covariance, dtype=float).reshape(
        model.control_dim, model.control_dim
    )
    dynamic = slice(model.pose_dim, model.state_dim)
    scale = _dynamic_state_scale(model)
    predicted = model.step(
        previous_state, prior * np.asarray(previous_control, dtype=float), dt
    )[dynamic]
    innovation = (np.asarray(current_state, dtype=float)[dynamic] - predicted) / scale
    sensitivity = _effectiveness_sensitivity_matrix(
        model,
        previous_state,
        previous_control,
        dt,
        prior,
        config,
        normalize_dynamic_state=True,
    )

    identity = np.eye(model.control_dim)
    predicted_covariance = (
        covariance / config.effectiveness_rls_forgetting_factor
        + config.effectiveness_rls_process_noise * identity
    )
    measurement_covariance = config.effectiveness_rls_measurement_noise**2 * np.eye(
        model.state_dim - model.pose_dim
    )

    def innovation_statistics(candidate_covariance: np.ndarray):
        innovation_covariance = (
            sensitivity @ candidate_covariance @ sensitivity.T + measurement_covariance
        )
        inverse = np.linalg.pinv(innovation_covariance, hermitian=True)
        distance = float(np.sqrt(max(float(innovation.T @ inverse @ innovation), 0.0)))
        return inverse, distance

    inverse_innovation, mahalanobis = innovation_statistics(predicted_covariance)
    excited = bool(
        np.linalg.norm(previous_control / model.control_scale())
        >= config.effectiveness_excitation_threshold
    )
    change_detected = False
    if not config.effectiveness_rls_adaptive_covariance:
        change_evidence = 0.0
    elif config.effectiveness_rls_change_detector == "threshold":
        change_candidate = bool(
            excited and mahalanobis > config.effectiveness_rls_change_threshold
        )
        change_evidence = change_evidence + 1.0 if change_candidate else 0.0
        change_detected = bool(
            change_candidate
            and change_evidence >= config.effectiveness_rls_change_persistence
            and allow_change
        )
    elif allow_change:
        normalized_innovation_squared = mahalanobis**2 / len(innovation)
        change_evidence = (
            max(
                0.0,
                change_evidence
                + normalized_innovation_squared
                - config.effectiveness_rls_cusum_drift,
            )
            if excited
            else 0.0
        )
        change_detected = bool(
            change_evidence >= config.effectiveness_rls_cusum_threshold
        )
    else:
        change_evidence = 0.0
    if change_detected:
        change_evidence = 0.0
        predicted_covariance *= config.effectiveness_rls_covariance_inflation
        inverse_innovation, mahalanobis = innovation_statistics(predicted_covariance)
    if mahalanobis > config.effectiveness_rls_innovation_gate:
        innovation *= config.effectiveness_rls_innovation_gate / mahalanobis

    gain = predicted_covariance @ sensitivity.T @ inverse_innovation
    estimate = np.clip(
        prior + gain @ innovation,
        config.effectiveness_min,
        config.effectiveness_max,
    )
    correction = identity - gain @ sensitivity
    updated_covariance = (
        correction @ predicted_covariance @ correction.T
        + gain @ measurement_covariance @ gain.T
    )
    updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
    return estimate, updated_covariance, change_detected, change_evidence


def _estimate_effectiveness_vector(
    model: PlatformModel,
    previous_state: np.ndarray,
    previous_control: np.ndarray,
    current_state: np.ndarray,
    dt: float,
    prior: np.ndarray,
    config: HarborMPCConfig,
) -> np.ndarray:
    """Estimate scalar-shared or diagonal local actuator effectiveness."""
    prior_vector = np.asarray(prior, dtype=float).reshape(model.control_dim)
    if config.effectiveness_estimator_mode == "scalar":
        estimate = _estimate_control_effectiveness(
            model,
            previous_state,
            previous_control,
            current_state,
            dt,
            float(np.mean(prior_vector)),
            config,
        )
        return np.full(model.control_dim, estimate, dtype=float)
    return _estimate_diagonal_control_effectiveness(
        model,
        previous_state,
        previous_control,
        current_state,
        dt,
        prior_vector,
        config,
    )


def _estimate_diagonal_control_effectiveness(
    model: PlatformModel,
    previous_state: np.ndarray,
    previous_control: np.ndarray,
    current_state: np.ndarray,
    dt: float,
    prior: np.ndarray,
    config: HarborMPCConfig,
) -> np.ndarray:
    """Fit independent channel gains from one-step local dynamic response."""
    command = np.asarray(previous_control, dtype=float).reshape(model.control_dim)
    scales = np.asarray(model.control_scale(), dtype=float)
    active = np.abs(command / scales) >= config.effectiveness_excitation_threshold
    if not np.any(active):
        return np.asarray(prior, dtype=float).copy()
    prior = np.asarray(prior, dtype=float).reshape(model.control_dim)
    dynamic = slice(model.pose_dim, model.state_dim)
    predicted = model.step(previous_state, prior * command, dt)[dynamic]
    measured = np.asarray(current_state, dtype=float)[dynamic]
    active_indices = np.flatnonzero(active)
    sensitivity = np.zeros((len(predicted), len(active_indices)), dtype=float)
    epsilon = 0.05
    for column, channel in enumerate(active_indices):
        lower = prior.copy()
        upper = prior.copy()
        lower[channel] = max(config.effectiveness_min, prior[channel] - epsilon)
        upper[channel] = min(config.effectiveness_max, prior[channel] + epsilon)
        span = upper[channel] - lower[channel]
        if span <= np.finfo(float).eps:
            continue
        low_state = model.step(previous_state, lower * command, dt)[dynamic]
        high_state = model.step(previous_state, upper * command, dt)[dynamic]
        sensitivity[:, column] = (high_state - low_state) / span
    if float(np.linalg.norm(sensitivity)) <= 1e-12:
        return prior.copy()
    correction = np.linalg.lstsq(
        sensitivity,
        measured - predicted,
        rcond=1e-8,
    )[0]
    instantaneous = prior.copy()
    instantaneous[active_indices] += correction
    instantaneous = np.clip(
        instantaneous,
        config.effectiveness_min,
        config.effectiveness_max,
    )
    gain = config.effectiveness_estimator_gain
    return (1.0 - gain) * prior + gain * instantaneous
