"""Dynamics / harbor / base."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi]``."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class PlatformModel(ABC):
    """Minimal dynamics contract required by heterogeneous coordination."""

    kind: str
    state_dim: int
    control_dim: int
    pose_dim: int
    variant: str

    @abstractmethod
    def position(self, state: np.ndarray) -> np.ndarray:
        """Return world-frame ``[x, y, z]`` position."""

    @abstractmethod
    def velocity(self, state: np.ndarray) -> np.ndarray:
        """Return world-frame ``[vx, vy, vz]`` velocity."""

    @abstractmethod
    def goal_position(self, goal: np.ndarray) -> np.ndarray:
        """Map a platform pose goal to world-frame ``[x, y, z]``."""

    @abstractmethod
    def orientation_error(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Return wrapped angular pose error in the platform's DOF space."""

    @abstractmethod
    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """Advance one platform state."""

    @abstractmethod
    def symbolic_step(self, ca, state, control, dt: float):
        """Return the matching CasADi state transition."""

    @abstractmethod
    def symbolic_velocity(self, ca, state):
        """Return velocity coordinates used by the MPC tracking cost."""

    @abstractmethod
    def control_scale(self) -> np.ndarray:
        """Return characteristic control magnitudes for cost normalization."""
