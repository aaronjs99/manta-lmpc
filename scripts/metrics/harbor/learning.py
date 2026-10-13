"""Metrics / harbor / learning."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.contracts.harbor import HarborLearningIteration

if TYPE_CHECKING:
    from scripts.contracts.harbor import HarborResult


def _controller_record(
    label,
    result,
    controller,
    *,
    admitted,
    simulation,
) -> HarborLearningIteration:
    telemetry = controller.telemetry
    retry = telemetry.retry
    estimator = telemetry.estimator
    return _record(
        label,
        result,
        admitted=admitted,
        simulation=simulation,
        solver_calls=controller.solve_count,
        solver_fallbacks=retry.fallback_count,
        solve_time_seconds=controller.solve_time_seconds,
        max_collision_slack=controller.max_collision_slack,
        max_terminal_slack=controller.max_terminal_slack,
        solve_count_by_agent=controller.solve_count_by_agent,
        fallback_count_by_agent=retry.fallback_count_by_agent,
        failure_steps_by_agent=retry.failure_steps_by_agent,
        failure_status_counts=retry.failure_status_counts,
        final_residual_estimates={
            name: values.tolist()
            for name, values in estimator.position_drift_estimates.items()
        },
        final_effectiveness_estimates={
            name: value.copy()
            for name, value in estimator.control_effectiveness_estimates.items()
        },
    )


def _record(
    label,
    result,
    *,
    admitted,
    simulation,
    solver_calls=0,
    solver_fallbacks=0,
    solve_time_seconds=0.0,
    max_collision_slack=0.0,
    max_terminal_slack=0.0,
    solve_count_by_agent=None,
    fallback_count_by_agent=None,
    failure_steps_by_agent=None,
    failure_status_counts=None,
    final_residual_estimates=None,
    final_effectiveness_estimates=None,
) -> HarborLearningIteration:
    return HarborLearningIteration(
        label=label,
        result=result,
        admitted=admitted,
        completion_step_sum=_completion_cost(result, simulation.horizon),
        solver_calls=solver_calls,
        solver_fallbacks=solver_fallbacks,
        solve_time_seconds=solve_time_seconds,
        max_collision_slack=max_collision_slack,
        max_terminal_slack=max_terminal_slack,
        solve_count_by_agent=solve_count_by_agent or {},
        fallback_count_by_agent=fallback_count_by_agent or {},
        failure_steps_by_agent=failure_steps_by_agent or {},
        failure_status_counts=failure_status_counts or {},
        final_residual_estimates=final_residual_estimates or {},
        final_effectiveness_estimates=final_effectiveness_estimates or {},
    )


def _completion_cost(result: HarborResult, horizon: int) -> int:
    return sum(
        step if step is not None else horizon + 1
        for step in result.first_goal_steps.values()
    )
