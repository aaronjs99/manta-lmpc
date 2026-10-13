"""Configuration / manta mpc."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MantaLMPCConfig:
    """Runtime LMPC settings loaded from ``config/manta.yaml``."""

    prediction_horizon: int = 20
    k_hull: int = 10
    dt: float = 0.2
    safety_constraint_substeps: int = 2
    safety_filter_buffer: float = 0.01
    max_steps: int = 300
    iterations: int = 4
    goal_tolerance: float = 0.65
    hyperplane_safety_margin: float = 0.3
    hyperplane_slack_bound: float = 2.0
    static_slack_bound: float = 2.0
    terminal_slack_bound: float = 4.0
    terminal_position_only: bool = True
    terminal_slack_weight: float = 10000.0
    hyperplane_slack_weight: float = 10000.0
    static_slack_weight: float = 1000000.0
    safe_cost_weight: float = 10.0
    state_cost_weights: tuple[float, ...] = (1.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0)
    control_cost_weights: tuple[float, ...] = (0.5, 0.5)
    hyperplane_ignore_distance: float = 4.0
    priority_hyperplanes: bool = True
    priority_metric: str = "goal_distance"
    priority_margin_scale: float = 0.35
    repair_incomplete_with_apf: bool = False
    repair_max_steps: int = 400
    repair_static_agent_scale: float = 1.2
    repair_waypoint_lookahead: int = 30
    fallback_control_levels: tuple[float, ...] = (0.25, 0.75, 1.5)
    fallback_diagonal_levels: tuple[float, ...] = (0.25, 0.75)
    fallback_clearance_weight: float = 0.25
    fallback_extra_clearance_weight: float = 0.25
    fallback_control_weight: float = 0.01
    warm_start_control: float = 1.0
    warm_start_control_blend: float = 0.25
    ipopt_max_iter: int = 200
    ipopt_print_level: int = 0
    log_interval: int = 5

    def __post_init__(self) -> None:
        """Reject invalid settings before building a CasADi problem."""
        positive_integers = {
            "prediction_horizon": self.prediction_horizon,
            "k_hull": self.k_hull,
            "safety_constraint_substeps": self.safety_constraint_substeps,
            "max_steps": self.max_steps,
            "ipopt_max_iter": self.ipopt_max_iter,
            "log_interval": self.log_interval,
        }
        for name, value in positive_integers.items():
            if value <= 0:
                raise ValueError(f"lmpc.{name} must be positive")
        if self.iterations < 0:
            raise ValueError("lmpc.iterations must be nonnegative")
        if self.dt <= 0.0:
            raise ValueError("lmpc.dt must be positive")
        if self.goal_tolerance <= 0.0:
            raise ValueError("lmpc.goal_tolerance must be positive")
        nonnegative_values = {
            "hyperplane_safety_margin": self.hyperplane_safety_margin,
            "hyperplane_slack_bound": self.hyperplane_slack_bound,
            "static_slack_bound": self.static_slack_bound,
            "safety_filter_buffer": self.safety_filter_buffer,
            "terminal_slack_bound": self.terminal_slack_bound,
            "terminal_slack_weight": self.terminal_slack_weight,
            "hyperplane_slack_weight": self.hyperplane_slack_weight,
            "static_slack_weight": self.static_slack_weight,
            "safe_cost_weight": self.safe_cost_weight,
            "fallback_clearance_weight": self.fallback_clearance_weight,
            "fallback_extra_clearance_weight": self.fallback_extra_clearance_weight,
            "fallback_control_weight": self.fallback_control_weight,
        }
        for name, value in nonnegative_values.items():
            if value < 0.0:
                raise ValueError(f"lmpc.{name} must be nonnegative")
        if len(self.state_cost_weights) != 7:
            raise ValueError("lmpc.state_cost_weights must contain 7 values")
        if len(self.control_cost_weights) != 2:
            raise ValueError("lmpc.control_cost_weights must contain 2 values")
        if any(weight < 0.0 for weight in self.state_cost_weights):
            raise ValueError("lmpc.state_cost_weights must be nonnegative")
        if any(weight < 0.0 for weight in self.control_cost_weights):
            raise ValueError("lmpc.control_cost_weights must be nonnegative")
        if self.hyperplane_ignore_distance <= 0.0:
            raise ValueError("lmpc.hyperplane_ignore_distance must be positive")
        if self.repair_max_steps <= 0:
            raise ValueError("lmpc.repair_max_steps must be positive")
        if self.repair_static_agent_scale <= 0.0:
            raise ValueError("lmpc.repair_static_agent_scale must be positive")
        if self.repair_waypoint_lookahead <= 0:
            raise ValueError("lmpc.repair_waypoint_lookahead must be positive")
        if not self.fallback_control_levels or any(
            level < 0.0 for level in self.fallback_control_levels
        ):
            raise ValueError("lmpc.fallback_control_levels must be nonnegative")
        if any(level < 0.0 for level in self.fallback_diagonal_levels):
            raise ValueError("lmpc.fallback_diagonal_levels must be nonnegative")
        if self.priority_metric not in {"goal_distance", "remaining_safe_time"}:
            raise ValueError(
                "lmpc.priority_metric must be 'goal_distance' or 'remaining_safe_time'"
            )
        if not 0.0 <= self.priority_margin_scale < 1.0:
            raise ValueError("lmpc.priority_margin_scale must be in [0, 1)")
        if not 0.0 <= self.warm_start_control_blend <= 1.0:
            raise ValueError("lmpc.warm_start_control_blend must be in [0, 1]")
