"""Stateful loss/recovery episode tracking for actuator effectiveness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.algorithms.harbor.effectiveness import (
    _apply_nominal_recovery_prior,
    _channel_selective_recovery_offset,
    _has_full_column_rank,
)


@dataclass(frozen=True)
class RecoveryTelemetry:
    recovery_effectiveness_offsets: dict
    recovery_offset_steps_by_agent: dict
    recovery_offset_rejections_by_agent: dict
    recovery_offset_unarmed_rejections_by_agent: dict
    recovery_offset_episode_limit_rejections_by_agent: dict
    recovery_offset_dwell_rejections_by_agent: dict
    recovery_offset_armed_by_agent: dict
    recovery_offset_armed_step_by_agent: dict


class EffectivenessRecoveryState:
    """Own transient recovery offsets, arming, episode budgets, and diagnostics."""

    def __init__(self, agents, config) -> None:
        self.config = config
        self.recovery_effectiveness_offsets = {
            agent.name: np.zeros(agent.model.control_dim, dtype=float)
            for agent in agents
        }
        self.recovery_offset_steps_by_agent = {agent.name: [] for agent in agents}
        self.recovery_offset_rejections_by_agent = {agent.name: 0 for agent in agents}
        self.recovery_offset_unarmed_rejections_by_agent = {
            agent.name: 0 for agent in agents
        }
        self.recovery_offset_episode_limit_rejections_by_agent = {
            agent.name: 0 for agent in agents
        }
        self.recovery_offset_dwell_rejections_by_agent = {
            agent.name: 0 for agent in agents
        }
        self.recovery_offset_armed_by_agent = {agent.name: False for agent in agents}
        self.recovery_offset_armed_step_by_agent: dict[str, int | None] = {
            agent.name: None for agent in agents
        }

    def apply_change(
        self,
        *,
        agent_name: str,
        step: int,
        previous_effectiveness: np.ndarray,
        estimate: np.ndarray,
        loss_detected: bool,
        recovery_detected: bool,
        normalized_sensitivity: np.ndarray,
    ) -> np.ndarray:
        """Apply configured embedded prior and update transient recovery state."""
        if (
            recovery_detected
            and self.config.effectiveness_recovery_prior_mode == "embedded"
        ):
            estimate = _apply_nominal_recovery_prior(
                previous_effectiveness,
                estimate,
                self.config.effectiveness_recovery_prior_gain,
                self.config.effectiveness_recovery_direction_tolerance,
                self.config.effectiveness_min,
                self.config.effectiveness_max,
            )
        if self.config.effectiveness_recovery_prior_mode == "transient":
            self._update_transient(
                agent_name,
                step,
                previous_effectiveness,
                estimate,
                loss_detected,
                recovery_detected,
                normalized_sensitivity,
            )
        return estimate

    def _update_transient(
        self,
        name,
        step,
        previous_effectiveness,
        estimate,
        loss_detected,
        recovery_detected,
        normalized_sensitivity,
    ) -> None:
        if loss_detected:
            self.recovery_effectiveness_offsets[name].fill(0.0)
            episode_count = len(self.recovery_offset_steps_by_agent[name])
            episode_limit = self.config.effectiveness_recovery_max_episodes_per_agent
            within_budget = episode_limit == 0 or episode_count < episode_limit
            if within_budget and not self.recovery_offset_armed_by_agent[name]:
                self.recovery_offset_armed_by_agent[name] = True
                self.recovery_offset_armed_step_by_agent[name] = step
            if not within_budget:
                self.recovery_offset_armed_by_agent[name] = False
                self.recovery_offset_armed_step_by_agent[name] = None
                self.recovery_offset_episode_limit_rejections_by_agent[name] += 1
        elif recovery_detected:
            armed = self.recovery_offset_armed_by_agent[name]
            armed_step = self.recovery_offset_armed_step_by_agent[name]
            if not armed:
                self.recovery_offset_unarmed_rejections_by_agent[name] += 1
            elif (
                armed_step is not None
                and step - armed_step
                < self.config.effectiveness_recovery_minimum_dwell_steps
            ):
                self.recovery_offset_dwell_rejections_by_agent[name] += 1
            else:
                identifiable = (
                    not self.config.effectiveness_recovery_require_full_rank
                    or _has_full_column_rank(
                        normalized_sensitivity,
                        self.config.effectiveness_recovery_rank_tolerance,
                    )
                )
                self.recovery_offset_armed_by_agent[name] = False
                self.recovery_offset_armed_step_by_agent[name] = None
                if identifiable:
                    self.recovery_effectiveness_offsets[name] = (
                        _channel_selective_recovery_offset(
                            previous_effectiveness,
                            estimate,
                            self.config.effectiveness_recovery_prior_gain,
                            self.config.effectiveness_recovery_direction_tolerance,
                        )
                    )
                    self.recovery_offset_steps_by_agent[name].append(step)
                else:
                    self.recovery_effectiveness_offsets[name].fill(0.0)
                    self.recovery_offset_rejections_by_agent[name] += 1

    def telemetry(self) -> RecoveryTelemetry:
        return RecoveryTelemetry(
            recovery_effectiveness_offsets=self.recovery_effectiveness_offsets,
            recovery_offset_steps_by_agent=self.recovery_offset_steps_by_agent,
            recovery_offset_rejections_by_agent=self.recovery_offset_rejections_by_agent,
            recovery_offset_unarmed_rejections_by_agent=self.recovery_offset_unarmed_rejections_by_agent,
            recovery_offset_episode_limit_rejections_by_agent=self.recovery_offset_episode_limit_rejections_by_agent,
            recovery_offset_dwell_rejections_by_agent=self.recovery_offset_dwell_rejections_by_agent,
            recovery_offset_armed_by_agent=self.recovery_offset_armed_by_agent,
            recovery_offset_armed_step_by_agent=self.recovery_offset_armed_step_by_agent,
        )
