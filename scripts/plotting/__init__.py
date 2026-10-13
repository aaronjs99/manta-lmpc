"""Plotting."""

from __future__ import annotations

from scripts.plotting.manta import (
    plot_cost_decrease,
    plot_learning_progression,
    save_manta_animation,
)
from scripts.plotting.trajectories import plot_pairwise_distances, plot_trajectories

__all__ = [
    "plot_cost_decrease",
    "plot_learning_progression",
    "plot_pairwise_distances",
    "plot_trajectories",
    "save_manta_animation",
]
