"""Coordinator and stateful effectiveness observer for Harbor estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.algorithms.harbor.effectiveness import (
    _effectiveness_change_is_loss,
    _effectiveness_change_is_recovery,
    _estimate_effectiveness_vector,
    _recursive_diagonal_effectiveness_update,
)
from scripts.algorithms.harbor.probing import IdentificationProbe
from scripts.algorithms.harbor.recovery_state import EffectivenessRecoveryState
from scripts.algorithms.harbor.residual_observer import ResidualEstimator


@dataclass(frozen=True)
class EstimatorTelemetry:
    position_drift_estimates: dict
    residual_history: dict
    control_residual_history: dict
    residual_rejection_count_by_agent: dict
    control_effectiveness_estimates: dict
    raw_effectiveness_estimates: dict
    recovery_effectiveness_offsets: dict
    effectiveness_history: dict
    raw_effectiveness_history: dict
    effectiveness_covariances: dict
    effectiveness_change_steps_by_agent: dict
    recovery_offset_steps_by_agent: dict
    recovery_offset_rejections_by_agent: dict
    recovery_offset_unarmed_rejections_by_agent: dict
    recovery_offset_episode_limit_rejections_by_agent: dict
    recovery_offset_dwell_rejections_by_agent: dict
    recovery_offset_armed_by_agent: dict
    recovery_offset_armed_step_by_agent: dict


class EffectivenessEstimator:
    """Own effectiveness estimates, recursive covariance, and sample histories."""

    def __init__(self, agents, config, recovery, initial_estimates=None) -> None:
        self.config = config
        self.recovery = recovery
        self.control_effectiveness_estimates = {
            agent.name: np.ones(agent.model.control_dim, dtype=float)
            for agent in agents
        }
        if initial_estimates is not None:
            names = {agent.name for agent in agents}
            unknown = set(initial_estimates) - names
            if unknown:
                raise ValueError(
                    "initial effectiveness contains unknown agent(s): "
                    + ", ".join(sorted(unknown))
                )
            for agent in agents:
                if agent.name not in initial_estimates:
                    continue
                estimate = np.asarray(
                    initial_estimates[agent.name], dtype=float
                ).reshape(-1)
                if estimate.shape != (agent.model.control_dim,):
                    raise ValueError(
                        f"{agent.name} initial effectiveness must have "
                        f"{agent.model.control_dim} entries"
                    )
                if np.any(estimate < config.effectiveness_min) or np.any(
                    estimate > config.effectiveness_max
                ):
                    raise ValueError(
                        f"{agent.name} initial effectiveness is outside estimator bounds"
                    )
                self.control_effectiveness_estimates[agent.name] = estimate.copy()
        self.raw_effectiveness_estimates = {
            name: estimate.copy()
            for name, estimate in self.control_effectiveness_estimates.items()
        }
        self.effectiveness_history = {agent.name: [] for agent in agents}
        self.raw_effectiveness_history = {agent.name: [] for agent in agents}
        self.effectiveness_covariances = {
            agent.name: np.eye(agent.model.control_dim)
            * config.identification_prior_std**2
            for agent in agents
        }
        self.effectiveness_change_steps_by_agent = {agent.name: [] for agent in agents}
        self.effectiveness_change_evidence_by_agent = {
            agent.name: 0.0 for agent in agents
        }
        self.effectiveness_last_change_step_by_agent = {
            agent.name: -config.effectiveness_rls_change_cooldown_steps
            for agent in agents
        }

    def update_transition(
        self,
        *,
        agent,
        previous_state: np.ndarray,
        previous_control: np.ndarray,
        current_state: np.ndarray,
        elapsed_dt: float,
        step: int,
        normalized_sensitivity: np.ndarray,
        identification: IdentificationProbe,
    ) -> None:
        """Update actuator estimates before residual-current estimation."""
        if not self.config.control_effectiveness_adaptation:
            return
        name = agent.name
        recursive_prior = (
            self.raw_effectiveness_estimates[name]
            if self.config.effectiveness_recovery_prior_mode == "transient"
            else self.control_effectiveness_estimates[name]
        )
        if self.config.effectiveness_estimator_mode == "recursive_diagonal":
            previous_effectiveness = recursive_prior.copy()
            allow_change = (
                step >= self.config.effectiveness_rls_change_warmup_steps
                and step - self.effectiveness_last_change_step_by_agent[name]
                >= self.config.effectiveness_rls_change_cooldown_steps
            )
            estimate, covariance, change_detected, evidence = (
                _recursive_diagonal_effectiveness_update(
                    agent.model,
                    previous_state,
                    previous_control,
                    np.asarray(current_state, dtype=float),
                    elapsed_dt,
                    recursive_prior,
                    self.effectiveness_covariances[name],
                    self.config,
                    change_evidence=self.effectiveness_change_evidence_by_agent[name],
                    allow_change=allow_change,
                )
            )
            self.effectiveness_change_evidence_by_agent[name] = evidence
            loss_detected = False
            if change_detected:
                self.effectiveness_change_steps_by_agent[name].append(step)
                self.effectiveness_last_change_step_by_agent[name] = step
                loss_detected = _effectiveness_change_is_loss(
                    previous_effectiveness, estimate
                )
                recovery_detected = _effectiveness_change_is_recovery(
                    previous_effectiveness,
                    estimate,
                    self.config.effectiveness_recovery_direction_tolerance,
                )
                estimate = self.recovery.apply_change(
                    agent_name=name,
                    step=step,
                    previous_effectiveness=previous_effectiveness,
                    estimate=estimate,
                    loss_detected=loss_detected,
                    recovery_detected=recovery_detected,
                    normalized_sensitivity=normalized_sensitivity,
                )
                permit_identification = (
                    not self.config.identification_arm_on_loss_only or loss_detected
                )
                identification.notify_change(
                    name, permit_identification=permit_identification
                )
            if self.config.effectiveness_recovery_prior_mode == "transient":
                self.raw_effectiveness_estimates[name] = estimate
                offset = self.recovery.recovery_effectiveness_offsets[name]
                self.control_effectiveness_estimates[name] = np.clip(
                    estimate + offset,
                    self.config.effectiveness_min,
                    self.config.effectiveness_max,
                )
                self.recovery.recovery_effectiveness_offsets[name] = (
                    self.config.effectiveness_recovery_offset_decay * offset
                )
            else:
                self.control_effectiveness_estimates[name] = estimate
                self.raw_effectiveness_estimates[name] = estimate.copy()
            self.effectiveness_covariances[name] = covariance
        else:
            self.control_effectiveness_estimates[name] = _estimate_effectiveness_vector(
                agent.model,
                previous_state,
                previous_control,
                np.asarray(current_state, dtype=float),
                elapsed_dt,
                self.control_effectiveness_estimates[name],
                self.config,
            )
            self.raw_effectiveness_estimates[name] = (
                self.control_effectiveness_estimates[name].copy()
            )

    def record_sample(self, agent_name: str) -> None:
        self.effectiveness_history[agent_name].append(
            self.control_effectiveness_estimates[agent_name].copy()
        )
        self.raw_effectiveness_history[agent_name].append(
            self.raw_effectiveness_estimates[agent_name].copy()
        )


class HarborStateObserver:
    """Sequence actuator, recovery, residual, and identification updates per sample."""

    def __init__(
        self, agents, config, dt: float, identification, initial_estimates=None
    ) -> None:
        self.config = config
        self.dt = dt
        self.identification = identification
        self.previous_states: dict[str, np.ndarray] = {}
        self.previous_update_steps: dict[str, int] = {}
        self.recovery = EffectivenessRecoveryState(agents, config)
        self.effectiveness = EffectivenessEstimator(
            agents, config, self.recovery, initial_estimates
        )
        self.residual_estimator = ResidualEstimator(agents, config)

    def observe(
        self, *, agent, state: np.ndarray, step: int, dt: float, previous_controls
    ) -> None:
        """Update once at plant rate; keep the established component ordering."""
        if not np.isclose(dt, self.dt):
            raise ValueError("observer and MPC sample times must match")
        name = agent.name
        if self.previous_update_steps.get(name) == step:
            return
        if name in self.previous_states:
            previous_state = self.previous_states[name]
            previous_control = previous_controls[name]
            elapsed_steps = step - self.previous_update_steps[name]
            if elapsed_steps <= 0:
                raise ValueError("MPC estimator updates must advance in time")
            elapsed_dt = elapsed_steps * self.dt
            recursive_prior = (
                self.effectiveness.raw_effectiveness_estimates[name]
                if self.config.effectiveness_recovery_prior_mode == "transient"
                else self.effectiveness.control_effectiveness_estimates[name]
            )
            sensitivity = self.identification.observe_transition(
                agent,
                previous_state,
                previous_control,
                elapsed_dt,
                recursive_prior,
                self.config,
            )
            self.effectiveness.update_transition(
                agent=agent,
                previous_state=previous_state,
                previous_control=previous_control,
                current_state=state,
                elapsed_dt=elapsed_dt,
                step=step,
                normalized_sensitivity=sensitivity,
                identification=self.identification,
            )
            effectiveness = (
                self.effectiveness.raw_effectiveness_estimates[name]
                if self.config.effectiveness_recovery_prior_mode == "transient"
                else self.effectiveness.control_effectiveness_estimates[name]
            )
            self.residual_estimator.update(
                agent=agent,
                previous_state=previous_state,
                current_state=state,
                previous_control=previous_control,
                elapsed_dt=elapsed_dt,
                elapsed_steps=elapsed_steps,
                current_step=step,
                effectiveness=effectiveness,
                last_effectiveness_change_step=(
                    self.effectiveness.effectiveness_last_change_step_by_agent[name]
                ),
            )
        self.previous_states[name] = np.asarray(state, dtype=float).copy()
        self.previous_update_steps[name] = step
        self.residual_estimator.record_sample(name)
        self.effectiveness.record_sample(name)
        self.identification.observe_sample(
            agent, self.effectiveness.control_effectiveness_estimates[name]
        )

    def telemetry(self) -> EstimatorTelemetry:
        """Compose component views without taking ownership of their state."""
        residual = self.residual_estimator.telemetry()
        effectiveness = self.effectiveness
        recovery = self.recovery.telemetry()
        return EstimatorTelemetry(
            position_drift_estimates=residual.position_drift_estimates,
            residual_history=residual.residual_history,
            control_residual_history=residual.control_residual_history,
            residual_rejection_count_by_agent=residual.residual_rejection_count_by_agent,
            control_effectiveness_estimates=effectiveness.control_effectiveness_estimates,
            raw_effectiveness_estimates=effectiveness.raw_effectiveness_estimates,
            recovery_effectiveness_offsets=recovery.recovery_effectiveness_offsets,
            effectiveness_history=effectiveness.effectiveness_history,
            raw_effectiveness_history=effectiveness.raw_effectiveness_history,
            effectiveness_covariances=effectiveness.effectiveness_covariances,
            effectiveness_change_steps_by_agent=effectiveness.effectiveness_change_steps_by_agent,
            recovery_offset_steps_by_agent=recovery.recovery_offset_steps_by_agent,
            recovery_offset_rejections_by_agent=recovery.recovery_offset_rejections_by_agent,
            recovery_offset_unarmed_rejections_by_agent=recovery.recovery_offset_unarmed_rejections_by_agent,
            recovery_offset_episode_limit_rejections_by_agent=recovery.recovery_offset_episode_limit_rejections_by_agent,
            recovery_offset_dwell_rejections_by_agent=recovery.recovery_offset_dwell_rejections_by_agent,
            recovery_offset_armed_by_agent=recovery.recovery_offset_armed_by_agent,
            recovery_offset_armed_step_by_agent=recovery.recovery_offset_armed_step_by_agent,
        )
