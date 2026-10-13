"""Policy definitions and orchestration for matched Harbor trial batches."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from scripts.configuration.harbor_faults import HarborTimeVaryingFaultConfig
from scripts.configuration.harbor_mpc import HarborMPCConfig
from scripts.configuration.harbor_simulation import HarborObservationNoiseConfig
from scripts.contracts.harbor import HarborRobustnessTrial
from scripts.experiments.harbor.execution.policy import (
    HarborPolicySpec,
    policy_specs,
)
from scripts.experiments.harbor.execution.scenario import (
    prepare_trial_scenario,
)
from scripts.experiments.harbor.execution.trial_runner import execute_policy_batch

if TYPE_CHECKING:
    from scripts.communication.harbor import LinkConfig
    from scripts.configuration.harbor_simulation import (
        HarborDisturbanceConfig,
        HarborSimulationConfig,
    )
    from scripts.contracts.harbor import HarborAgent, HarborResult


def temporary_fault_policies(
    experiment_config: HarborTimeVaryingFaultConfig,
    study_mpc: HarborMPCConfig,
    controller_labels: tuple[str, ...] | None = None,
) -> tuple[HarborPolicySpec, ...]:
    """Define the temporary-fault comparisons in their established order."""
    recovery_gain = experiment_config.recovery_prior_gain
    threshold = experiment_config.change_threshold
    inflation = experiment_config.covariance_inflation

    def policy(
        label: str,
        *,
        adaptive: bool,
        detector: str = "threshold",
        recovery: float = 0.0,
        recovery_mode: str = "embedded",
        projection: str = "full",
        slack: float | None = None,
        retry_slack: float | None = None,
        retry_orientation: float | None = None,
        retry_goal_radius: float | None = None,
        residual_mode: str | None = None,
        holdoff: int | None = None,
        measurement_source: str | None = None,
    ) -> HarborPolicySpec:
        overrides: dict[str, object] = {
            "effectiveness_rls_adaptive_covariance": adaptive,
            "effectiveness_rls_change_detector": detector,
            "effectiveness_rls_change_threshold": threshold,
            "effectiveness_rls_covariance_inflation": inflation,
            "effectiveness_recovery_prior_gain": recovery,
            "effectiveness_recovery_prior_mode": recovery_mode,
            "residual_control_projection": projection,
        }
        optional = {
            "dynamic_state_slack_bound": slack,
            "dynamic_state_slack_retry_bound": retry_slack,
            "dynamic_state_slack_retry_orientation_error": retry_orientation,
            "dynamic_state_slack_retry_goal_radius": retry_goal_radius,
            "residual_estimator_mode": residual_mode,
            "residual_change_holdoff_steps": holdoff,
            "residual_measurement_source": measurement_source,
        }
        overrides.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return HarborPolicySpec(
            label=label,
            adaptive_effectiveness=adaptive,
            estimator_mode="recursive_diagonal",
            learning=False,
            active_identification=False,
            source_controller=None,
            identification_strategy="energy",
            minimum_probes=1,
            mpc_overrides=tuple(overrides.items()),
        )

    policies = (
        policy("Fixed-covariance RLS", adaptive=False),
        policy("Innovation-threshold RLS", adaptive=True),
        policy("Recovery-prior threshold RLS", adaptive=True, recovery=recovery_gain),
        policy(
            "Transient-offset threshold RLS",
            adaptive=True,
            recovery=recovery_gain,
            recovery_mode="transient",
        ),
        policy(
            "Projected transient-offset RLS",
            adaptive=True,
            recovery=recovery_gain,
            recovery_mode="transient",
            projection="actuation_subspace",
        ),
        policy(
            "Event-projected transient-offset RLS",
            adaptive=True,
            recovery=recovery_gain,
            recovery_mode="transient",
            projection="station_keeping_subspace",
        ),
        policy(
            "Hard-envelope transient-offset RLS",
            adaptive=True,
            recovery=recovery_gain,
            recovery_mode="transient",
            slack=0.0,
            retry_slack=0.0,
        ),
        policy(
            "Elastic-envelope transient-offset RLS",
            adaptive=True,
            recovery=recovery_gain,
            recovery_mode="transient",
        ),
        policy(
            "Retry-elastic transient-offset RLS",
            adaptive=True,
            recovery=recovery_gain,
            recovery_mode="transient",
            slack=0.0,
            retry_slack=study_mpc.dynamic_state_slack_bound,
            retry_orientation=0.12,
            retry_goal_radius=0.5,
        ),
        policy(
            "Constant-current transient-offset RLS",
            adaptive=True,
            recovery=recovery_gain,
            recovery_mode="transient",
            slack=0.0,
            retry_slack=0.0,
            residual_mode="constant_bias_rls",
            holdoff=0,
        ),
        policy(
            "Event-isolated current transient-offset RLS",
            adaptive=True,
            recovery=recovery_gain,
            recovery_mode="transient",
            slack=0.0,
            retry_slack=0.0,
            residual_mode="constant_bias_rls",
            holdoff=5,
        ),
        policy(
            "Kinematic-current EWMA transient-offset RLS",
            adaptive=True,
            recovery=recovery_gain,
            recovery_mode="transient",
            slack=0.0,
            retry_slack=0.0,
            measurement_source="kinematic_velocity",
        ),
        policy(
            "Kinematic-current RLS transient-offset RLS",
            adaptive=True,
            recovery=recovery_gain,
            recovery_mode="transient",
            slack=0.0,
            retry_slack=0.0,
            residual_mode="constant_bias_rls",
            measurement_source="kinematic_velocity",
        ),
        policy("Chi-square CUSUM RLS", adaptive=True, detector="cusum"),
        HarborPolicySpec(
            label="CUSUM-triggered probing RLS",
            adaptive_effectiveness=True,
            estimator_mode="recursive_diagonal",
            learning=False,
            active_identification=True,
            source_controller=None,
            identification_strategy="information",
            minimum_probes=1,
            mpc_overrides=(
                ("effectiveness_rls_adaptive_covariance", True),
                ("effectiveness_rls_change_detector", "cusum"),
                ("effectiveness_rls_change_threshold", threshold),
                ("effectiveness_rls_covariance_inflation", inflation),
                ("identification_reset_on_change", True),
                ("identification_arm_on_change", True),
                ("identification_arm_on_loss_only", True),
            ),
        ),
    )
    available = {spec.label for spec in policies}
    default = available - {
        "Recovery-prior threshold RLS",
        "Transient-offset threshold RLS",
        "Projected transient-offset RLS",
        "Event-projected transient-offset RLS",
        "Hard-envelope transient-offset RLS",
        "Elastic-envelope transient-offset RLS",
        "Retry-elastic transient-offset RLS",
        "Constant-current transient-offset RLS",
        "Event-isolated current transient-offset RLS",
        "Kinematic-current EWMA transient-offset RLS",
        "Kinematic-current RLS transient-offset RLS",
    }
    selected = default if controller_labels is None else set(controller_labels)
    if not selected or not selected <= available:
        raise ValueError("temporary-fault controller labels must be known and nonempty")
    return policy_specs(tuple(item for item in policies if item.label in selected))


def run_temporary_fault_trials(
    *,
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    evaluation: HarborSimulationConfig,
    communication: LinkConfig,
    study_mpc: HarborMPCConfig,
    disturbance: HarborDisturbanceConfig,
    seed_result: HarborResult,
    experiment_config: HarborTimeVaryingFaultConfig,
    observation_noise: HarborObservationNoiseConfig | None,
    controller_labels: tuple[str, ...] | None = None,
    residual_adaptation: bool = False,
    residual_adaptation_kinds: tuple[str, ...] | None = None,
) -> list[HarborRobustnessTrial]:
    """Prepare scenario once and execute the selected temporary-fault policies."""
    configured_mpc = replace(
        study_mpc,
        effectiveness_rls_change_warmup_steps=experiment_config.change_warmup_steps,
        effectiveness_rls_change_cooldown_steps=experiment_config.change_cooldown_steps,
    )
    scenario = prepare_trial_scenario(
        agents=agents,
        simulation=simulation,
        communication=communication,
        mpc=configured_mpc,
        disturbance=disturbance,
        seed_result=seed_result,
        evaluation=evaluation,
        observation_noise=observation_noise,
        residual_adaptation=residual_adaptation,
        residual_adaptation_kinds=residual_adaptation_kinds,
    )
    policies = temporary_fault_policies(
        experiment_config, configured_mpc, controller_labels
    )
    return execute_policy_batch(scenario, policies)


def run_policy_trials(
    *,
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    evaluation: HarborSimulationConfig,
    communication: LinkConfig,
    study_mpc: HarborMPCConfig,
    disturbance: HarborDisturbanceConfig,
    seed_result: HarborResult,
    policies: tuple[HarborPolicySpec, ...],
    observation_noise: HarborObservationNoiseConfig | None = None,
    residual_adaptation: bool = False,
    residual_adaptation_kinds: tuple[str, ...] | None = None,
) -> list[HarborRobustnessTrial]:
    """Prepare a scenario and execute policy specifications in their given order."""
    scenario = prepare_trial_scenario(
        agents=agents,
        simulation=simulation,
        communication=communication,
        mpc=study_mpc,
        disturbance=disturbance,
        seed_result=seed_result,
        evaluation=evaluation,
        observation_noise=observation_noise,
        residual_adaptation=residual_adaptation,
        residual_adaptation_kinds=residual_adaptation_kinds,
    )
    return execute_policy_batch(scenario, policy_specs(policies))
