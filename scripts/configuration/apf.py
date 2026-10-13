"""Configuration / apf."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class APFConfig:
    """APF tuning loaded from ``config/manta.yaml``."""

    heading_gain: float = 1.5
    influence_radius: float = 4.0
    obstacle_padding: float = 0.2
    repulsion_gain: float = 1.5
    goal_tolerance: float = 0.35
    max_steps: int = 400
    base_mu_gain: float = 0.8
    base_mu_min: float = 0.5
    base_mu_max: float = 1.5
    static_agent_radius_scales: tuple[float, ...] = (1.5, 1.2, 1.0, 0.85, 0.7)
    compact_staging: bool = True
    compact_staging_candidates: int = 3
    compact_staging_max_agents: int = 3

    def __post_init__(self) -> None:
        """Validate APF tuning independently of a simulation run."""
        if self.max_steps <= 0:
            raise ValueError("apf.max_steps must be positive")
        if self.influence_radius <= 0.0:
            raise ValueError("apf.influence_radius must be positive")
        if self.goal_tolerance <= 0.0:
            raise ValueError("apf.goal_tolerance must be positive")
        if self.heading_gain < 0.0 or self.repulsion_gain < 0.0:
            raise ValueError("apf heading and repulsion gains must be nonnegative")
        if self.obstacle_padding < 0.0:
            raise ValueError("apf.obstacle_padding must be nonnegative")
        if self.base_mu_gain <= 0.0:
            raise ValueError("apf.base_mu_gain must be positive")
        if not 0.0 <= self.base_mu_min <= self.base_mu_max:
            raise ValueError("apf base_mu bounds must satisfy 0 <= min <= max")
        if not self.static_agent_radius_scales or any(
            scale <= 0.0 for scale in self.static_agent_radius_scales
        ):
            raise ValueError("apf.static_agent_radius_scales must be positive")
        if self.compact_staging_candidates <= 0:
            raise ValueError("apf.compact_staging_candidates must be positive")
        if self.compact_staging_max_agents < 2:
            raise ValueError("apf.compact_staging_max_agents must be at least 2")
