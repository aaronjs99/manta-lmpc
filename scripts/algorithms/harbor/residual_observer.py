"""Stateful observer for local motion residuals and current estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.algorithms.harbor.residual import (
    _constant_bias_residual_update,
    _kinematic_position_drift_measurement,
)


@dataclass(frozen=True)
class ResidualTelemetry:
    position_drift_estimates: dict
    residual_history: dict
    control_residual_history: dict
    residual_rejection_count_by_agent: dict


class ResidualEstimator:
    """Own per-agent local residual estimates, uncertainty, and histories."""

    def __init__(self, agents, config) -> None:
        self.config = config
        self.position_drift_estimates = {agent.name: np.zeros(3) for agent in agents}
        self.position_drift_variances = {
            agent.name: np.full(3, config.residual_rls_initial_std**2)
            for agent in agents
        }
        self.residual_rejection_count_by_agent = {agent.name: 0 for agent in agents}
        self.residual_history = {agent.name: [] for agent in agents}
        self.control_residual_history = {agent.name: [] for agent in agents}

    def update(
        self,
        *,
        agent,
        previous_state: np.ndarray,
        current_state: np.ndarray,
        previous_control: np.ndarray,
        elapsed_dt: float,
        elapsed_steps: int,
        current_step: int,
        effectiveness: np.ndarray,
        last_effectiveness_change_step: int,
    ) -> None:
        """Apply the configured residual update after effectiveness estimation."""
        if (
            not self.config.residual_adaptation
            or agent.model.kind not in self.config.residual_adaptation_kinds
        ):
            return
        if self.config.residual_measurement_source == "kinematic_velocity":
            measured_velocity_error = _kinematic_position_drift_measurement(
                agent.model,
                previous_state,
                current_state,
                elapsed_dt,
                elapsed_steps,
            )
        else:
            predicted = agent.model.step(
                previous_state,
                effectiveness * previous_control,
                elapsed_dt,
            )
            measured_velocity_error = (
                agent.model.position(current_state) - agent.model.position(predicted)
            ) / elapsed_dt
        name = agent.name
        if self.config.residual_estimator_mode == "constant_bias_rls":
            elapsed_since_change = current_step - last_effectiveness_change_step
            if 0 <= elapsed_since_change < self.config.residual_change_holdoff_steps:
                self.position_drift_variances[name] += (
                    self.config.residual_rls_process_noise
                )
                return
            estimate, variance, rejected = _constant_bias_residual_update(
                self.position_drift_estimates[name],
                self.position_drift_variances[name],
                measured_velocity_error,
                self.config,
            )
            self.position_drift_variances[name] = variance
            self.residual_rejection_count_by_agent[name] += rejected
        else:
            gain = self.config.residual_estimator_gain
            estimate = (1.0 - gain) * self.position_drift_estimates[name] + (
                gain * measured_velocity_error
            )
        norm = float(np.linalg.norm(estimate))
        if norm > self.config.residual_max_speed:
            estimate *= self.config.residual_max_speed / norm
        self.position_drift_estimates[name] = estimate

    def record_sample(self, agent_name: str) -> None:
        self.residual_history[agent_name].append(
            self.position_drift_estimates[agent_name].copy()
        )

    def record_control_drift(self, agent_name: str, value: np.ndarray) -> None:
        self.control_residual_history[agent_name].append(value.copy())

    def telemetry(self) -> ResidualTelemetry:
        return ResidualTelemetry(
            position_drift_estimates=self.position_drift_estimates,
            residual_history=self.residual_history,
            control_residual_history=self.control_residual_history,
            residual_rejection_count_by_agent=self.residual_rejection_count_by_agent,
        )
