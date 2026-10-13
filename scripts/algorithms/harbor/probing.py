"""Stateful local actuator-identification probing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.algorithms.harbor.effectiveness import _effectiveness_sensitivity_matrix
from scripts.algorithms.harbor.identification import (
    _candidate_probe_information,
    _identification_channel,
    _information_identification_channel,
    _posterior_effectiveness_std,
)


@dataclass(frozen=True)
class IdentificationTelemetry:
    excitation_energy: dict
    excitation_history: dict
    information_matrices: dict
    information_std_history: dict
    probe_count_by_agent: dict
    probe_channel_counts: dict
    probe_quota_counts: dict
    probe_sequence_by_agent: dict
    probe_rejection_counts: dict
    change_armed_by_agent: dict


class IdentificationProbe:
    """Own excitation, information, quota, and accepted/rejected probe state."""

    def __init__(self, agents, config) -> None:
        self.config = config
        self.excitation_energy = {
            agent.name: np.zeros(agent.model.control_dim, dtype=float)
            for agent in agents
        }
        self.excitation_history = {agent.name: [] for agent in agents}
        self.information_matrices = {
            agent.name: np.zeros(
                (agent.model.control_dim, agent.model.control_dim), dtype=float
            )
            for agent in agents
        }
        self.information_std_history = {agent.name: [] for agent in agents}
        self.probe_count_by_agent = {agent.name: 0 for agent in agents}
        self.probe_channel_counts = {
            agent.name: np.zeros(agent.model.control_dim, dtype=int) for agent in agents
        }
        self.probe_quota_counts = {
            agent.name: np.zeros(agent.model.control_dim, dtype=int) for agent in agents
        }
        self.probe_sequence_by_agent = {agent.name: [] for agent in agents}
        self.probe_rejection_counts = {
            agent.name: np.zeros(agent.model.control_dim, dtype=int) for agent in agents
        }
        self.change_armed_by_agent = {
            agent.name: not config.identification_arm_on_change for agent in agents
        }

    def observe_transition(
        self, agent, previous_state, previous_control, elapsed_dt, prior, config
    ) -> np.ndarray:
        name = agent.name
        normalized = previous_control / agent.model.control_scale()
        self.excitation_energy[name] += normalized * normalized
        sensitivity = _effectiveness_sensitivity_matrix(
            agent.model,
            previous_state,
            previous_control,
            elapsed_dt,
            prior,
            config,
            normalize_dynamic_state=True,
        )
        jacobian = sensitivity / config.identification_measurement_noise
        self.information_matrices[name] += jacobian.T @ jacobian
        return sensitivity

    def observe_sample(self, agent, effectiveness_estimate) -> None:
        name = agent.name
        self.excitation_history[name].append(self.excitation_energy[name].copy())
        self.information_std_history[name].append(
            _posterior_effectiveness_std(self.information_matrices[name], self.config)
        )

    def notify_change(self, agent_name: str, *, permit_identification: bool) -> None:
        if permit_identification and self.config.identification_reset_on_change:
            self.excitation_energy[agent_name].fill(0.0)
            self.information_matrices[agent_name].fill(0.0)
            self.probe_quota_counts[agent_name].fill(0)
            self.probe_rejection_counts[agent_name].fill(0)
        if permit_identification and self.config.identification_arm_on_change:
            self.change_armed_by_agent[agent_name] = True

    def select(
        self,
        *,
        agent,
        state: np.ndarray,
        desired_velocity: np.ndarray,
        inbox,
        agents,
        effectiveness_estimate: np.ndarray,
        dt: float,
        step: int,
    ) -> tuple[np.ndarray, np.ndarray, int | None]:
        """Select one direct, constraint-aware local identification pulse."""
        model = agent.model
        empty = np.zeros(model.control_dim, dtype=float)
        if (
            not self.config.active_identification
            or not self.config.control_effectiveness_adaptation
            or not self.change_armed_by_agent[agent.name]
            or self.config.effectiveness_estimator_mode
            not in {"diagonal", "recursive_diagonal"}
            or np.linalg.norm(desired_velocity) <= 1e-6
            or step % self.config.identification_probe_interval_steps != 0
        ):
            return empty, empty.copy(), None
        own_position = model.position(np.asarray(state, dtype=float))
        for other_name, message in inbox.items():
            other = agents.get(other_name)
            if other is None:
                continue
            clearance = float(np.linalg.norm(own_position - message.position))
            required = (
                agent.radius
                + other.radius
                + self.config.collision_buffer
                + self.config.identification_clearance_buffer
            )
            if clearance <= required:
                return empty, empty.copy(), None
        if self.config.identification_strategy == "information":
            increments = _candidate_probe_information(
                model,
                np.asarray(state, dtype=float),
                effectiveness_estimate,
                dt,
                self.config,
            )
            channel = _information_identification_channel(
                self.information_matrices[agent.name],
                increments,
                self.probe_quota_counts[agent.name],
                self.probe_rejection_counts[agent.name],
                self.config,
                effectiveness_estimate=effectiveness_estimate,
            )
        else:
            channel = _identification_channel(
                self.excitation_energy[agent.name],
                self.probe_quota_counts[agent.name],
                self.probe_rejection_counts[agent.name],
                self.config,
            )
        if channel is None:
            return empty, empty.copy(), None
        count = self.probe_channel_counts[agent.name][channel]
        sign = 1.0 if count % 2 == 0 else -1.0
        probe = empty.copy()
        probe[channel] = (
            sign
            * self.config.identification_probe_fraction
            * model.control_scale()[channel]
        )
        mask = empty.copy()
        mask[channel] = 1.0
        return probe, mask, channel

    def reject(self, agent_name: str, channel: int) -> None:
        self.probe_rejection_counts[agent_name][channel] += 1

    def accept(self, agent_name: str, channel: int) -> None:
        self.probe_count_by_agent[agent_name] += 1
        self.probe_channel_counts[agent_name][channel] += 1
        self.probe_quota_counts[agent_name][channel] += 1
        self.probe_sequence_by_agent[agent_name].append(channel)

    def telemetry(self) -> IdentificationTelemetry:
        return IdentificationTelemetry(
            excitation_energy=self.excitation_energy,
            excitation_history=self.excitation_history,
            information_matrices=self.information_matrices,
            information_std_history=self.information_std_history,
            probe_count_by_agent=self.probe_count_by_agent,
            probe_channel_counts=self.probe_channel_counts,
            probe_quota_counts=self.probe_quota_counts,
            probe_sequence_by_agent=self.probe_sequence_by_agent,
            probe_rejection_counts=self.probe_rejection_counts,
            change_armed_by_agent=self.change_armed_by_agent,
        )
