"""Assembly of recorded Harbor trial results from simulation telemetry."""

from __future__ import annotations

import numpy as np

from scripts.contracts.harbor import HarborResult, HarborRobustnessTrial
from scripts.mpc.harbor.controller import DistributedHarborMPC


def assemble_trial_result(
    *,
    label: str,
    source_controller: str | None,
    result: HarborResult,
    controller: DistributedHarborMPC,
    horizon: int,
) -> HarborRobustnessTrial:
    """Snapshot trial quality and controller telemetry using existing copy rules."""
    telemetry = controller.telemetry
    estimator = telemetry.estimator
    identification = telemetry.identification
    retry = telemetry.retry
    completion_cost = sum(
        step if step is not None else horizon + 1
        for step in result.first_goal_steps.values()
    )
    return HarborRobustnessTrial(
        label=label,
        identification_strategy=controller.config.identification_strategy,
        result=result,
        valid=(
            result.all_goals_reached
            and result.pairwise_violation_count == 0
            and retry.fallback_count == 0
            and telemetry.max_collision_slack <= 1e-9
        ),
        source_controller=source_controller,
        completion_step_sum=completion_cost,
        solver_fallbacks=retry.fallback_count,
        solver_fallbacks_by_agent=retry.fallback_count_by_agent.copy(),
        solver_failure_steps_by_agent={
            name: steps.copy() for name, steps in retry.failure_steps_by_agent.items()
        },
        solver_failure_status_counts=retry.failure_status_counts.copy(),
        max_collision_slack=telemetry.max_collision_slack,
        max_domain_buffer_slack=telemetry.max_domain_buffer_slack,
        max_dynamic_state_slack=telemetry.max_dynamic_state_slack,
        residual_history={
            name: np.asarray(values, dtype=float)
            for name, values in estimator.residual_history.items()
        },
        control_residual_history={
            name: np.asarray(values, dtype=float)
            for name, values in estimator.control_residual_history.items()
        },
        residual_projection_retry_count_by_agent=dict(
            retry.residual_projection_retry_count_by_agent
        ),
        dynamic_state_retry_count_by_agent=dict(
            retry.dynamic_state_retry_count_by_agent
        ),
        residual_rejection_count_by_agent=dict(
            estimator.residual_rejection_count_by_agent
        ),
        final_residual_estimates={
            name: value.copy()
            for name, value in estimator.position_drift_estimates.items()
        },
        effectiveness_history={
            name: np.asarray(values, dtype=float)
            for name, values in estimator.effectiveness_history.items()
        },
        raw_effectiveness_history={
            name: np.asarray(values, dtype=float)
            for name, values in estimator.raw_effectiveness_history.items()
        },
        final_effectiveness_estimates={
            name: value.copy()
            for name, value in estimator.control_effectiveness_estimates.items()
        },
        effectiveness_change_steps_by_agent={
            name: steps.copy()
            for name, steps in estimator.effectiveness_change_steps_by_agent.items()
        },
        recovery_offset_steps_by_agent={
            name: steps.copy()
            for name, steps in estimator.recovery_offset_steps_by_agent.items()
        },
        recovery_offset_rejections_by_agent=dict(
            estimator.recovery_offset_rejections_by_agent
        ),
        recovery_offset_unarmed_rejections_by_agent=dict(
            estimator.recovery_offset_unarmed_rejections_by_agent
        ),
        recovery_offset_episode_limit_rejections_by_agent=dict(
            estimator.recovery_offset_episode_limit_rejections_by_agent
        ),
        recovery_offset_dwell_rejections_by_agent=dict(
            estimator.recovery_offset_dwell_rejections_by_agent
        ),
        excitation_history={
            name: np.asarray(values, dtype=float)
            for name, values in identification.excitation_history.items()
        },
        information_std_history={
            name: np.asarray(values, dtype=float)
            for name, values in identification.information_std_history.items()
        },
        probe_count_by_agent=dict(identification.probe_count_by_agent),
        probe_channel_counts={
            name: value.copy()
            for name, value in identification.probe_channel_counts.items()
        },
        probe_sequence_by_agent={
            name: list(value)
            for name, value in identification.probe_sequence_by_agent.items()
        },
        probe_rejection_counts={
            name: value.copy()
            for name, value in identification.probe_rejection_counts.items()
        },
    )
