"""Configuration / harbor mpc."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml
from scripts.configuration.paths import DEFAULT_HARBOR_CONFIG, resolve_config_path


@dataclass(frozen=True)
class HarborMPCConfig:
    """Shared optimization settings for MPC and safe-set LMPC modes."""

    prediction_horizon: int = 8
    replan_interval_steps: int = 1
    learning_iterations: int = 2
    terminal_samples: int = 8
    terminal_position_only: bool = True
    seed_learning_from_mpc: bool = True
    position_weight: float = 4.0
    orientation_weight: float = 0.8
    velocity_weight: float = 0.15
    control_weight: float = 0.04
    control_rate_weight: float = 0.25
    terminal_goal_weight: float = 20.0
    terminal_orientation_weight: float = 2.5
    terminal_safe_cost_weight: float = 1.0
    terminal_slack_weight: float = 5000.0
    terminal_slack_bound: float = 1.0
    collision_buffer: float = 0.1
    belief_safety_enabled: bool = False
    belief_safety_quantile: float = 2.0
    belief_safety_max_margin: float = 0.5
    collision_slack_weight: float = 100000.0
    collision_slack_bound: float = 0.0
    domain_boundary_margin: float = 0.0
    domain_boundary_slack_weight: float = 100000.0
    dynamic_state_slack_bound: float = 0.10
    dynamic_state_slack_weight: float = 100000.0
    dynamic_state_slack_retry_bound: float = 0.0
    dynamic_state_slack_retry_orientation_error: float = 0.0
    dynamic_state_slack_retry_goal_radius: float = 0.0
    residual_adaptation: bool = False
    residual_adaptation_kinds: tuple[str, ...] = ("ugv", "usv", "rov")
    residual_control_projection: str = "full"
    residual_estimator_gain: float = 0.35
    residual_max_speed: float = 0.35
    residual_estimator_mode: str = "ewma"
    residual_measurement_source: str = "model_prediction"
    residual_rls_initial_std: float = 0.20
    residual_rls_measurement_noise: float = 0.25
    residual_rls_process_noise: float = 1.0e-4
    residual_rls_innovation_gate: float = 3.0
    residual_change_holdoff_steps: int = 0
    control_effectiveness_adaptation: bool = False
    effectiveness_estimator_mode: str = "scalar"
    effectiveness_estimator_gain: float = 0.25
    effectiveness_rls_forgetting_factor: float = 0.995
    effectiveness_rls_measurement_noise: float = 0.02
    effectiveness_rls_process_noise: float = 1.0e-5
    effectiveness_rls_innovation_gate: float = 3.0
    effectiveness_rls_adaptive_covariance: bool = False
    effectiveness_rls_change_detector: str = "threshold"
    effectiveness_rls_change_threshold: float = 3.0
    effectiveness_rls_covariance_inflation: float = 8.0
    effectiveness_rls_change_persistence: int = 1
    effectiveness_rls_change_cooldown_steps: int = 20
    effectiveness_rls_change_warmup_steps: int = 0
    effectiveness_recovery_prior_gain: float = 0.0
    effectiveness_recovery_direction_tolerance: float = 1.0e-6
    effectiveness_recovery_prior_mode: str = "embedded"
    effectiveness_recovery_offset_decay: float = 0.90
    effectiveness_recovery_require_full_rank: bool = True
    effectiveness_recovery_rank_tolerance: float = 1.0e-8
    effectiveness_recovery_max_episodes_per_agent: int = 1
    effectiveness_recovery_minimum_dwell_steps: int = 0
    effectiveness_rls_cusum_drift: float = 1.5
    effectiveness_rls_cusum_threshold: float = 4.0
    effectiveness_min: float = 0.5
    effectiveness_max: float = 1.2
    effectiveness_excitation_threshold: float = 0.05
    active_identification: bool = False
    identification_strategy: str = "information"
    identification_probe_fraction: float = 0.12
    identification_target_energy: float = 0.06
    identification_prior_std: float = 0.25
    identification_measurement_noise: float = 0.005
    identification_target_std: float = 0.15
    identification_fault_focus_weight: float = 2.0
    identification_reset_on_change: bool = False
    identification_arm_on_change: bool = False
    identification_arm_on_loss_only: bool = False
    identification_probe_interval_steps: int = 1
    identification_min_probes_per_channel: int = 2
    identification_max_rejections: int = 2
    identification_clearance_buffer: float = 0.4
    obstacle_prediction_mode: str = "goal_bounded_velocity"
    obstacle_prediction_alignment_threshold: float = 0.5
    ipopt_max_iter: int = 120
    ipopt_print_level: int = 0

    def __post_init__(self) -> None:
        residual_kinds = tuple(str(kind) for kind in self.residual_adaptation_kinds)
        if len(set(residual_kinds)) != len(residual_kinds) or not set(
            residual_kinds
        ) <= {"ugv", "usv", "rov"}:
            raise ValueError("residual_adaptation_kinds must be unique platform kinds")
        object.__setattr__(self, "residual_adaptation_kinds", residual_kinds)
        if self.residual_control_projection not in {
            "full",
            "actuation_subspace",
            "station_keeping_subspace",
        }:
            raise ValueError(
                "residual_control_projection must be full, actuation_subspace, "
                "or station_keeping_subspace"
            )
        if self.residual_estimator_mode not in {"ewma", "constant_bias_rls"}:
            raise ValueError(
                "residual_estimator_mode must be ewma or constant_bias_rls"
            )
        if self.residual_measurement_source not in {
            "model_prediction",
            "kinematic_velocity",
        }:
            raise ValueError(
                "residual_measurement_source must be model_prediction or "
                "kinematic_velocity"
            )
        integer_values = {
            "prediction_horizon": self.prediction_horizon,
            "replan_interval_steps": self.replan_interval_steps,
            "learning_iterations": self.learning_iterations,
            "terminal_samples": self.terminal_samples,
            "identification_probe_interval_steps": self.identification_probe_interval_steps,
            "identification_min_probes_per_channel": self.identification_min_probes_per_channel,
            "identification_max_rejections": self.identification_max_rejections,
            "effectiveness_rls_change_persistence": self.effectiveness_rls_change_persistence,
            "effectiveness_rls_change_cooldown_steps": self.effectiveness_rls_change_cooldown_steps,
            "ipopt_max_iter": self.ipopt_max_iter,
        }
        if any(value <= 0 for value in integer_values.values()):
            raise ValueError("harbor_mpc integer settings must be positive")
        if int(self.effectiveness_rls_change_warmup_steps) != (
            self.effectiveness_rls_change_warmup_steps
        ):
            raise ValueError("effectiveness RLS change warmup must be an integer")
        if self.effectiveness_rls_change_warmup_steps < 0:
            raise ValueError("effectiveness RLS change warmup must be nonnegative")
        if (
            int(self.residual_change_holdoff_steps)
            != self.residual_change_holdoff_steps
        ):
            raise ValueError("residual change holdoff must be an integer")
        if self.residual_change_holdoff_steps < 0:
            raise ValueError("residual change holdoff must be nonnegative")
        nonnegative = (
            self.position_weight,
            self.orientation_weight,
            self.velocity_weight,
            self.control_weight,
            self.control_rate_weight,
            self.terminal_goal_weight,
            self.terminal_orientation_weight,
            self.terminal_safe_cost_weight,
            self.terminal_slack_weight,
            self.terminal_slack_bound,
            self.collision_buffer,
            self.belief_safety_quantile,
            self.belief_safety_max_margin,
            self.collision_slack_weight,
            self.collision_slack_bound,
            self.domain_boundary_margin,
            self.domain_boundary_slack_weight,
            self.dynamic_state_slack_bound,
            self.dynamic_state_slack_weight,
            self.dynamic_state_slack_retry_bound,
            self.dynamic_state_slack_retry_orientation_error,
            self.dynamic_state_slack_retry_goal_radius,
            self.residual_estimator_gain,
            self.residual_max_speed,
            self.residual_rls_initial_std,
            self.residual_rls_measurement_noise,
            self.residual_rls_process_noise,
            self.residual_rls_innovation_gate,
            self.effectiveness_estimator_gain,
            self.effectiveness_rls_measurement_noise,
            self.effectiveness_rls_process_noise,
            self.effectiveness_rls_innovation_gate,
            self.effectiveness_rls_change_threshold,
            self.effectiveness_rls_covariance_inflation,
            self.effectiveness_rls_cusum_drift,
            self.effectiveness_rls_cusum_threshold,
            self.effectiveness_recovery_prior_gain,
            self.effectiveness_recovery_direction_tolerance,
            self.effectiveness_recovery_rank_tolerance,
            self.effectiveness_min,
            self.effectiveness_max,
            self.effectiveness_excitation_threshold,
            self.identification_probe_fraction,
            self.identification_target_energy,
            self.identification_prior_std,
            self.identification_measurement_noise,
            self.identification_target_std,
            self.identification_fault_focus_weight,
            self.identification_clearance_buffer,
            self.obstacle_prediction_alignment_threshold,
        )
        if any(value < 0.0 for value in nonnegative):
            raise ValueError("harbor_mpc weights and slack bounds must be nonnegative")
        if self.residual_estimator_gain > 1.0:
            raise ValueError("residual_estimator_gain must be in [0, 1]")
        if (
            min(
                self.residual_rls_initial_std,
                self.residual_rls_measurement_noise,
                self.residual_rls_innovation_gate,
            )
            <= 0.0
        ):
            raise ValueError("residual RLS uncertainty and gate must be positive")
        if self.effectiveness_estimator_gain > 1.0:
            raise ValueError("effectiveness_estimator_gain must be in [0, 1]")
        if not 0.0 < self.effectiveness_rls_forgetting_factor <= 1.0:
            raise ValueError("effectiveness_rls_forgetting_factor must be in (0, 1]")
        if (
            min(
                self.effectiveness_rls_measurement_noise,
                self.effectiveness_rls_innovation_gate,
                self.effectiveness_rls_change_threshold,
                self.effectiveness_rls_covariance_inflation,
            )
            <= 0.0
        ):
            raise ValueError(
                "RLS noise, gates, and covariance inflation must be positive"
            )
        if self.effectiveness_rls_covariance_inflation < 1.0:
            raise ValueError(
                "effectiveness_rls_covariance_inflation must be at least 1"
            )
        if self.effectiveness_recovery_prior_gain > 1.0:
            raise ValueError("effectiveness_recovery_prior_gain must be in [0, 1]")
        if self.effectiveness_recovery_prior_mode not in {"embedded", "transient"}:
            raise ValueError(
                "effectiveness_recovery_prior_mode must be embedded or transient"
            )
        if not 0.0 <= self.effectiveness_recovery_offset_decay < 1.0:
            raise ValueError("effectiveness_recovery_offset_decay must be in [0, 1)")
        if (
            int(self.effectiveness_recovery_max_episodes_per_agent)
            != (self.effectiveness_recovery_max_episodes_per_agent)
            or self.effectiveness_recovery_max_episodes_per_agent < 0
        ):
            raise ValueError(
                "effectiveness_recovery_max_episodes_per_agent must be a "
                "nonnegative integer"
            )
        if (
            int(self.effectiveness_recovery_minimum_dwell_steps)
            != (self.effectiveness_recovery_minimum_dwell_steps)
            or self.effectiveness_recovery_minimum_dwell_steps < 0
        ):
            raise ValueError(
                "effectiveness_recovery_minimum_dwell_steps must be a "
                "nonnegative integer"
            )
        if self.effectiveness_rls_change_detector not in {"threshold", "cusum"}:
            raise ValueError(
                "effectiveness_rls_change_detector must be threshold or cusum"
            )
        if not self.effectiveness_min <= 1.0 <= self.effectiveness_max:
            raise ValueError("effectiveness bounds must contain the nominal value 1")
        if self.effectiveness_estimator_mode not in {
            "scalar",
            "diagonal",
            "recursive_diagonal",
        }:
            raise ValueError(
                "effectiveness_estimator_mode must be scalar, diagonal, or "
                "recursive_diagonal"
            )
        if self.identification_probe_fraction > 1.0:
            raise ValueError("identification_probe_fraction must be in [0, 1]")
        if self.active_identification and self.identification_probe_fraction == 0.0:
            raise ValueError(
                "identification_probe_fraction must be positive when active"
            )
        if self.identification_strategy not in {"energy", "information"}:
            raise ValueError("identification_strategy must be energy or information")
        if self.obstacle_prediction_mode not in {
            "constant_velocity",
            "goal_bounded_velocity",
        }:
            raise ValueError(
                "obstacle_prediction_mode must be constant_velocity or "
                "goal_bounded_velocity"
            )
        if self.obstacle_prediction_alignment_threshold > 1.0:
            raise ValueError(
                "obstacle_prediction_alignment_threshold must be in [0, 1]"
            )
        if (
            min(
                self.identification_prior_std,
                self.identification_measurement_noise,
                self.identification_target_std,
            )
            <= 0.0
        ):
            raise ValueError("identification uncertainty settings must be positive")


def load_harbor_mpc_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
) -> HarborMPCConfig:
    """Load the strict ``harbor_mpc`` YAML section."""
    config_path = resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    data = raw.get("harbor_mpc", {})
    known = {field.name for field in fields(HarborMPCConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown harbor_mpc field(s): {', '.join(unknown)}")
    return HarborMPCConfig(**data)
