"""Configuration / manta dynamics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MantaDynamicsConfig:
    """Constants for the 7-state manta/CPG model loaded from YAML.

    State is ``[x, y, theta, p_L, q_L, p_R, q_R]`` and control is
    ``[mu_L, mu_R]``.
    """

    c_v: float = 0.3
    c_tau: float = 0.2
    wing_span: float = 1.0
    oscillator_frequency: float = 4.0
    rk4_substeps: int = 4
    mu_min: float = 0.0
    mu_max: float = 2.5
    oscillator_bound: float = 6.0

    def __post_init__(self) -> None:
        """Validate physical and integration bounds at config load time."""
        if self.c_v < 0.0 or self.c_tau < 0.0:
            raise ValueError("dynamics c_v and c_tau must be nonnegative")
        if self.wing_span <= 0.0:
            raise ValueError("dynamics.wing_span must be positive")
        if self.oscillator_frequency <= 0.0:
            raise ValueError("dynamics.oscillator_frequency must be positive")
        if self.rk4_substeps <= 0:
            raise ValueError("dynamics.rk4_substeps must be positive")
        if not 0.0 <= self.mu_min <= self.mu_max:
            raise ValueError("dynamics mu bounds must satisfy 0 <= min <= max")
        if self.oscillator_bound <= 0.0:
            raise ValueError("dynamics.oscillator_bound must be positive")
