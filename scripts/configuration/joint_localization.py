"""Configuration / joint localization."""

from __future__ import annotations

from dataclasses import replace

from scripts.configuration.harbor_faults import (
    load_harbor_fault_study_config,
    load_harbor_time_varying_fault_config,
)
from scripts.configuration.harbor_mpc import load_harbor_mpc_config


def _adaptive_config(path: str):
    base = load_harbor_mpc_config(path)
    fault = load_harbor_fault_study_config(path)
    _, experiment = load_harbor_time_varying_fault_config(path)
    return replace(
        base,
        prediction_horizon=fault.prediction_horizon,
        terminal_goal_weight=fault.terminal_goal_weight,
        terminal_slack_bound=fault.terminal_slack_bound,
        terminal_slack_weight=fault.terminal_slack_weight,
        residual_adaptation=True,
        residual_adaptation_kinds=("usv", "rov"),
        residual_measurement_source="kinematic_velocity",
        residual_estimator_mode="constant_bias_rls",
        control_effectiveness_adaptation=True,
        effectiveness_estimator_mode="recursive_diagonal",
        effectiveness_rls_adaptive_covariance=True,
        effectiveness_rls_change_detector="threshold",
        effectiveness_rls_change_threshold=experiment.change_threshold,
        effectiveness_rls_covariance_inflation=experiment.covariance_inflation,
        effectiveness_rls_change_warmup_steps=experiment.change_warmup_steps,
        effectiveness_rls_change_cooldown_steps=experiment.change_cooldown_steps,
        effectiveness_recovery_prior_gain=experiment.recovery_prior_gain,
        effectiveness_recovery_prior_mode="transient",
        dynamic_state_slack_bound=0.0,
        dynamic_state_slack_retry_bound=0.0,
    )


def _policy_controller_config(config, policy: str):
    if policy != "Joint SLAM + belief retry":
        return config
    return replace(config, dynamic_state_slack_retry_bound=0.02)
