"""Configuration / factor graph."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorGraphConfig:
    """Numerical settings for the fixed-lag smoother."""

    lag: int = 8
    iterations: int = 4
    huber_delta: float = 2.5
    range_std: float = 0.08
    odometry_std: float = 0.12
    attitude_std: float = 0.08
    depth_std: float = 0.06
    prior_std: float = 1.0
    damping: float = 1.0e-5
    rank_tolerance: float = 1.0e-7

    def __post_init__(self) -> None:
        if self.lag < 2 or self.iterations < 1:
            raise ValueError("fixed-lag windows require lag >= 2 and iterations >= 1")
        if (
            min(
                self.huber_delta,
                self.range_std,
                self.odometry_std,
                self.attitude_std,
                self.depth_std,
                self.prior_std,
                self.damping,
            )
            <= 0.0
        ):
            raise ValueError("factor-graph scales must be positive")
