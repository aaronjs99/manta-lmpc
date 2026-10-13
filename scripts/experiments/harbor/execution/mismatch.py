"""Experiments / harbor / execution / mismatch."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from scripts.contracts.harbor import HarborRobustnessTrial
from scripts.experiments.harbor.learning import run_distributed_harbor_lmpc
from scripts.mpc.harbor.controller import DistributedHarborMPC
from scripts.simulation.harbor.engine import run_harbor_simulation

if TYPE_CHECKING:
    from scripts.communication.harbor import LinkConfig
    from scripts.configuration.harbor_mpc import HarborMPCConfig
    from scripts.configuration.harbor_simulation import (
        HarborDisturbanceConfig,
        HarborSimulationConfig,
    )
    from scripts.contracts.harbor import HarborAgent


def run_model_mismatch_study(
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    communication: LinkConfig,
    mpc_config: HarborMPCConfig,
    disturbance: HarborDisturbanceConfig,
) -> list[HarborRobustnessTrial]:
    """Compare nominal and residual-adaptive controllers on one hidden plant."""
    seed_iterations = run_distributed_harbor_lmpc(
        agents,
        simulation,
        communication,
        replace(
            mpc_config,
            learning_iterations=1,
            residual_adaptation=False,
            control_effectiveness_adaptation=False,
        ),
    )
    seed = min(
        (record for record in seed_iterations if record.admitted),
        key=lambda record: record.completion_step_sum,
    )
    evaluation = replace(
        simulation,
        guidance_update_interval_steps=mpc_config.replan_interval_steps,
        goal_hold_steps=disturbance.evaluation_hold_steps,
    )
    definitions = (
        ("Nominal MPC", False, False, False),
        ("Residual-adaptive MPC", True, False, False),
        ("Joint-adaptive MPC", True, True, False),
        ("Joint-adaptive LMPC", True, True, True),
    )
    trials = []
    for label, residual_adaptive, effectiveness_adaptive, learning in definitions:
        controller = DistributedHarborMPC(
            agents=agents,
            config=replace(
                mpc_config,
                residual_adaptation=residual_adaptive,
                control_effectiveness_adaptation=effectiveness_adaptive,
            ),
            dt=simulation.dt,
            safe_states=seed.result.states,
            safe_controls=seed.result.controls,
            learning=learning,
        )
        result = run_harbor_simulation(
            agents,
            evaluation,
            communication,
            control_provider=controller,
            disturbance=disturbance,
        )
        completion_cost = sum(
            step if step is not None else simulation.horizon + 1
            for step in result.first_goal_steps.values()
        )
        telemetry = controller.telemetry
        estimator = telemetry.estimator
        identification = telemetry.identification
        retry = telemetry.retry
        valid = (
            result.all_goals_reached
            and result.pairwise_violation_count == 0
            and retry.fallback_count == 0
            and controller.max_collision_slack <= 1e-9
        )
        trials.append(
            HarborRobustnessTrial(
                label=label,
                identification_strategy=controller.config.identification_strategy,
                result=result,
                valid=valid,
                source_controller=None,
                completion_step_sum=completion_cost,
                solver_fallbacks=retry.fallback_count,
                solver_fallbacks_by_agent=retry.fallback_count_by_agent.copy(),
                solver_failure_steps_by_agent={
                    name: steps.copy()
                    for name, steps in retry.failure_steps_by_agent.items()
                },
                solver_failure_status_counts=retry.failure_status_counts.copy(),
                max_collision_slack=controller.max_collision_slack,
                max_domain_buffer_slack=controller.max_domain_buffer_slack,
                max_dynamic_state_slack=controller.max_dynamic_state_slack,
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
        )
        print(
            f"{label}: complete={result.all_goals_reached}, "
            f"safe={result.pairwise_violation_count == 0}, "
            f"cost={completion_cost}, fallbacks={controller.telemetry.retry.fallback_count}",
            flush=True,
        )
    return trials
