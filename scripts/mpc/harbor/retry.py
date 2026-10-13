"""Ordered Harbor MPC solve retries and fallback diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scripts.algorithms.harbor.prediction import _control_position_drift


@dataclass(frozen=True)
class RetryOutcome:
    solution: Any | None
    control_position_drift: np.ndarray
    used_dynamic_retry: bool
    probe_channel: int | None


@dataclass(frozen=True)
class RetryTelemetry:
    residual_projection_retry_count_by_agent: dict[str, int]
    dynamic_state_retry_count_by_agent: dict[str, int]
    fallback_count: int
    fallback_count_by_agent: dict[str, int]
    failure_steps_by_agent: dict[str, list[int]]
    failure_status_counts: dict[str, int]
    failure_diagnostics_by_agent: dict[str, list[dict[str, Any]]]


class HarborSolveRetryPolicy:
    """Apply the established probe, elastic, projection, and orientation retries."""

    def __init__(self, agents, config) -> None:
        self.config = config
        self.residual_projection_retry_count_by_agent = {
            agent.name: 0 for agent in agents
        }
        self.dynamic_state_retry_count_by_agent = {agent.name: 0 for agent in agents}
        self.fallback_count = 0
        self.fallback_count_by_agent = {agent.name: 0 for agent in agents}
        self.failure_steps_by_agent = {agent.name: [] for agent in agents}
        self.failure_status_counts: dict[str, int] = {}
        self.failure_diagnostics_by_agent: dict[str, list[dict[str, Any]]] = {
            agent.name: [] for agent in agents
        }

    def solve(
        self,
        *,
        agent,
        state: np.ndarray,
        navigation_goal: np.ndarray,
        desired_velocity: np.ndarray,
        step: int,
        primary_optimizer,
        dynamic_retry_optimizer,
        solve_arguments: dict,
        probe: np.ndarray,
        probe_mask: np.ndarray,
        probe_channel: int | None,
        control_effectiveness: np.ndarray,
        previous_control: np.ndarray,
        estimated_position_drift: np.ndarray,
        identification,
    ) -> RetryOutcome:
        """Solve using the exact current fallback ordering and diagnostics."""
        used_dynamic_retry = False
        original_drift = solve_arguments["position_drift"]
        try:
            solution = primary_optimizer.solve(**solve_arguments)
        except RuntimeError:
            solution = None
            if probe_channel is not None:
                identification.reject(agent.name, probe_channel)
                solve_arguments["identification_probe"] = np.zeros_like(probe)
                solve_arguments["identification_probe_mask"] = np.zeros_like(probe_mask)
                try:
                    solution = primary_optimizer.solve(**solve_arguments)
                except RuntimeError:
                    pass
            if solution is None and dynamic_retry_optimizer is not None:
                try:
                    solution = dynamic_retry_optimizer.solve(**solve_arguments)
                except RuntimeError:
                    pass
                else:
                    self.dynamic_state_retry_count_by_agent[agent.name] += 1
                    used_dynamic_retry = True
            control_position_drift = original_drift
            if solution is None:
                retry_projection = _control_position_drift(
                    agent.model,
                    state,
                    estimated_position_drift,
                    "actuation_subspace",
                    desired_velocity,
                )
                can_retry_projection = (
                    self.config.residual_control_projection
                    == "station_keeping_subspace"
                    and agent.model.kind == "usv"
                    and not np.allclose(retry_projection, original_drift, atol=1.0e-12)
                )
                if can_retry_projection:
                    solve_arguments["position_drift"] = retry_projection
                    try:
                        solution = primary_optimizer.solve(**solve_arguments)
                    except RuntimeError:
                        pass
                    else:
                        control_position_drift = retry_projection
                        self.residual_projection_retry_count_by_agent[agent.name] += 1
            if solution is None:
                self._record_failure(
                    agent=agent,
                    state=state,
                    step=step,
                    optimizer=primary_optimizer,
                    control_effectiveness=control_effectiveness,
                    previous_control=previous_control,
                )
            else:
                probe_channel = None
        else:
            control_position_drift = original_drift

        if solution is not None and not used_dynamic_retry:
            yaw_index = agent.model.pose_dim - 1
            measured_yaw_error = abs(
                np.arctan2(
                    np.sin(state[yaw_index] - navigation_goal[yaw_index]),
                    np.cos(state[yaw_index] - navigation_goal[yaw_index]),
                )
            )
            should_compare_retry = (
                dynamic_retry_optimizer is not None
                and agent.model.kind == "usv"
                and self.config.dynamic_state_slack_retry_orientation_error > 0.0
                and np.linalg.norm(state[:2] - navigation_goal[:2])
                <= self.config.dynamic_state_slack_retry_goal_radius
                and measured_yaw_error
                > self.config.dynamic_state_slack_retry_orientation_error
            )
            if should_compare_retry:
                try:
                    retry_solution = dynamic_retry_optimizer.solve(**solve_arguments)
                except RuntimeError:
                    pass
                else:
                    solution = retry_solution
                    self.dynamic_state_retry_count_by_agent[agent.name] += 1

        return RetryOutcome(
            solution=solution,
            control_position_drift=control_position_drift,
            used_dynamic_retry=used_dynamic_retry,
            probe_channel=probe_channel,
        )

    def _record_failure(
        self, *, agent, state, step, optimizer, control_effectiveness, previous_control
    ) -> None:
        name = agent.name
        self.fallback_count += 1
        self.fallback_count_by_agent[name] += 1
        self.failure_steps_by_agent[name].append(step)
        status = optimizer.last_status
        self.failure_status_counts[status] = (
            self.failure_status_counts.get(status, 0) + 1
        )
        if not self.failure_diagnostics_by_agent[name]:
            diagnostic = {
                "step": int(step),
                "state": np.asarray(state, dtype=float).tolist(),
                "previous_control": previous_control.tolist(),
                "control_effectiveness": control_effectiveness.tolist(),
                **optimizer.last_failure_diagnostics,
            }
            self.failure_diagnostics_by_agent[name].append(diagnostic)
            print(f"MPC failure diagnostic for {name}: {diagnostic}", flush=True)

    def telemetry(self) -> RetryTelemetry:
        return RetryTelemetry(
            residual_projection_retry_count_by_agent=self.residual_projection_retry_count_by_agent,
            dynamic_state_retry_count_by_agent=self.dynamic_state_retry_count_by_agent,
            fallback_count=self.fallback_count,
            fallback_count_by_agent=self.fallback_count_by_agent,
            failure_steps_by_agent=self.failure_steps_by_agent,
            failure_status_counts=self.failure_status_counts,
            failure_diagnostics_by_agent=self.failure_diagnostics_by_agent,
        )
