"""Mpc / harbor / controller."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from scripts.algorithms.harbor.observer import (
    EstimatorTelemetry,
    HarborStateObserver,
)
from scripts.algorithms.harbor.probing import (
    IdentificationProbe,
    IdentificationTelemetry,
)
from scripts.algorithms.harbor.guidance_control import guidance_control
from scripts.algorithms.harbor.prediction import (
    _approach_pose_goal,
    _control_position_drift,
    _goal_bounded_velocity_prediction,
    _nearest_reference_index,
    _sample_safe_states,
    _warm_reference,
)
from scripts.mpc.harbor.optimizer import HarborAgentOptimizer
from scripts.mpc.harbor.retry import HarborSolveRetryPolicy, RetryTelemetry

if TYPE_CHECKING:
    from scripts.configuration.harbor_mpc import HarborMPCConfig
    from scripts.contracts.harbor import AgentMessage, HarborAgent


@dataclass(frozen=True)
class HarborControllerTelemetry:
    estimator: EstimatorTelemetry
    identification: IdentificationTelemetry
    retry: RetryTelemetry
    solve_count: int
    solve_time_seconds: float
    solve_count_by_agent: dict[str, int]
    max_collision_slack: float
    max_domain_buffer_slack: float
    max_dynamic_state_slack: float
    max_terminal_slack: float


class DistributedHarborMPC:
    """Per-agent MPC using local state, received messages, and optional safe data."""

    def __init__(
        self,
        *,
        agents: list[HarborAgent],
        config: HarborMPCConfig,
        dt: float,
        safe_states: dict[str, np.ndarray],
        safe_controls: dict[str, np.ndarray],
        learning: bool,
        initial_effectiveness_estimates: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.agents = {agent.name: agent for agent in agents}
        self.config = config
        self.dt = dt
        self.safe_states = safe_states
        self.safe_controls = safe_controls
        self.learning = learning
        self.optimizers = {
            agent.name: HarborAgentOptimizer(
                agent=agent,
                other_agents=[other for other in agents if other.name != agent.name],
                config=config,
                dt=dt,
                learning=learning,
            )
            for agent in agents
        }
        self.dynamic_state_retry_optimizers = (
            {
                agent.name: HarborAgentOptimizer(
                    agent=agent,
                    other_agents=[
                        other for other in agents if other.name != agent.name
                    ],
                    config=replace(
                        config,
                        dynamic_state_slack_bound=(
                            config.dynamic_state_slack_retry_bound
                        ),
                    ),
                    dt=dt,
                    learning=learning,
                )
                for agent in agents
            }
            if config.dynamic_state_slack_retry_bound > 0.0
            else {}
        )
        self.previous_controls = {
            agent.name: np.zeros(agent.model.control_dim) for agent in agents
        }
        self.identification = IdentificationProbe(agents, config)
        self.observer = HarborStateObserver(
            agents,
            config,
            dt,
            self.identification,
            initial_estimates=initial_effectiveness_estimates,
        )
        self.retry_policy = HarborSolveRetryPolicy(agents, config)
        self.solve_count = 0
        self.solve_time_seconds = 0.0
        self.max_collision_slack = 0.0
        self.max_domain_buffer_slack = 0.0
        self.max_dynamic_state_slack = 0.0
        self.max_terminal_slack = 0.0
        self.solve_count_by_agent = {agent.name: 0 for agent in agents}

    @property
    def telemetry(self) -> HarborControllerTelemetry:
        """Read-only view of state owned by observer, probe, and retry components."""
        return HarborControllerTelemetry(
            estimator=self.observer.telemetry(),
            identification=self.identification.telemetry(),
            retry=self.retry_policy.telemetry(),
            solve_count=self.solve_count,
            solve_time_seconds=self.solve_time_seconds,
            solve_count_by_agent=self.solve_count_by_agent,
            max_collision_slack=self.max_collision_slack,
            max_domain_buffer_slack=self.max_domain_buffer_slack,
            max_dynamic_state_slack=self.max_dynamic_state_slack,
            max_terminal_slack=self.max_terminal_slack,
        )

    def control(
        self,
        *,
        agent: HarborAgent,
        state: np.ndarray,
        navigation_goal: np.ndarray,
        desired_velocity: np.ndarray,
        inbox: dict[str, AgentMessage],
        step: int,
        dt: float,
        belief: dict | None = None,
    ) -> np.ndarray:
        self.observe(agent=agent, state=state, step=step, dt=dt)
        safe_states = self.safe_states[agent.name]
        safe_controls = self.safe_controls[agent.name]
        nearest = _nearest_reference_index(agent.model, safe_states, state)
        reference_index = min(max(step, nearest), len(safe_states) - 1)
        safe_sample_states, safe_costs = _sample_safe_states(
            safe_states,
            reference_index + self.config.prediction_horizon,
            self.config.terminal_samples,
        )
        warm_states, warm_controls = _warm_reference(
            safe_states,
            safe_controls,
            reference_index,
            self.config.prediction_horizon,
        )
        predictions = self._obstacle_predictions(agent, inbox, step)
        observer = self.observer
        identification = self.identification
        probe, probe_mask, probe_channel = identification.select(
            agent=agent,
            state=state,
            desired_velocity=desired_velocity,
            inbox=inbox,
            agents=self.agents,
            effectiveness_estimate=observer.effectiveness.control_effectiveness_estimates[
                agent.name
            ],
            dt=self.dt,
            step=step,
        )
        started = perf_counter()
        control_position_drift = _control_position_drift(
            agent.model,
            state,
            observer.residual_estimator.position_drift_estimates[agent.name],
            self.config.residual_control_projection,
            desired_velocity,
        )
        belief_margin = 0.0
        if self.config.belief_safety_enabled and belief is not None:
            covariance = np.asarray(belief.get("position_covariance", []), dtype=float)
            if covariance.ndim == 2 and covariance.size:
                belief_margin = min(
                    self.config.belief_safety_max_margin,
                    self.config.belief_safety_quantile
                    * float(np.sqrt(max(np.trace(covariance), 0.0))),
                )
        solve_arguments = dict(
            state=np.asarray(state, dtype=float),
            goal=_approach_pose_goal(
                agent.model,
                navigation_goal,
                desired_velocity,
            ),
            obstacle_predictions=predictions,
            safe_states=safe_sample_states,
            safe_costs=safe_costs,
            warm_states=warm_states,
            warm_controls=warm_controls,
            previous_control=self.previous_controls[agent.name],
            position_drift=control_position_drift,
            control_effectiveness=observer.effectiveness.control_effectiveness_estimates[
                agent.name
            ],
            identification_probe=probe,
            identification_probe_mask=probe_mask,
            belief_collision_margin=belief_margin,
        )
        retry_outcome = self.retry_policy.solve(
            agent=agent,
            state=state,
            navigation_goal=navigation_goal,
            desired_velocity=desired_velocity,
            step=step,
            primary_optimizer=self.optimizers[agent.name],
            dynamic_retry_optimizer=self.dynamic_state_retry_optimizers.get(agent.name),
            solve_arguments=solve_arguments,
            probe=probe,
            probe_mask=probe_mask,
            probe_channel=probe_channel,
            control_effectiveness=observer.effectiveness.control_effectiveness_estimates[
                agent.name
            ],
            previous_control=self.previous_controls[agent.name],
            estimated_position_drift=observer.residual_estimator.position_drift_estimates[
                agent.name
            ],
            identification=identification,
        )
        solution = retry_outcome.solution
        control_position_drift = retry_outcome.control_position_drift
        probe_channel = retry_outcome.probe_channel
        control = (
            solution.control
            if solution is not None
            else guidance_control(
                agent.model,
                state,
                desired_velocity,
                dt,
                desired_pose=navigation_goal,
            )
        )
        if solution is not None:
            self.solve_count += 1
            self.solve_count_by_agent[agent.name] += 1
            self.max_collision_slack = max(
                self.max_collision_slack, solution.max_collision_slack
            )
            self.max_domain_buffer_slack = max(
                self.max_domain_buffer_slack,
                solution.max_domain_buffer_slack,
            )
            self.max_dynamic_state_slack = max(
                self.max_dynamic_state_slack,
                solution.max_dynamic_state_slack,
            )
            self.max_terminal_slack = max(
                self.max_terminal_slack, solution.max_terminal_slack
            )
            if probe_channel is not None:
                identification.accept(agent.name, probe_channel)
        self.solve_time_seconds += perf_counter() - started
        self.previous_controls[agent.name] = np.asarray(control, dtype=float)
        observer.residual_estimator.record_control_drift(
            agent.name, control_position_drift
        )
        return self.previous_controls[agent.name]

    def observe(
        self,
        *,
        agent: HarborAgent,
        state: np.ndarray,
        step: int,
        dt: float,
    ) -> None:
        """Update local estimators at plant rate without forcing an NLP solve."""
        self.observer.observe(
            agent=agent,
            state=state,
            step=step,
            dt=dt,
            previous_controls=self.previous_controls,
        )

    def _obstacle_predictions(
        self,
        agent: HarborAgent,
        inbox: dict[str, AgentMessage],
        current_step: int,
    ) -> dict[str, np.ndarray]:
        horizon = self.config.prediction_horizon
        predictions = {}
        for other in self.optimizers[agent.name].other_agents:
            message = inbox.get(other.name)
            if message is None:
                predictions[other.name] = np.full((horizon, 3), 1.0e3)
                continue
            age = max(0, current_step - message.sent_step)
            times = (age + np.arange(1, horizon + 1))[:, None] * self.dt
            if self.config.obstacle_prediction_mode == "goal_bounded_velocity":
                predictions[other.name] = _goal_bounded_velocity_prediction(
                    message,
                    times,
                    self.config.obstacle_prediction_alignment_threshold,
                )
            else:
                predictions[other.name] = message.position + times * message.velocity
        return predictions
