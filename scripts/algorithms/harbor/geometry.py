"""Geometry calculations shared by Harbor simulation and evaluation."""

from __future__ import annotations

import numpy as np


def pairwise_clearance_metrics(position_arrays, agents):
    """Measure minimum swept clearance and collision sample count."""
    names = sorted(position_arrays)
    minimum = float("inf")
    violations = 0
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            relative = position_arrays[first] - position_arrays[second]
            starts = relative[:-1]
            deltas = relative[1:] - starts
            speed_sq = np.sum(deltas * deltas, axis=1)
            alpha = np.zeros(len(starts))
            moving = speed_sq > np.finfo(float).eps
            alpha[moving] = (
                -np.sum(starts[moving] * deltas[moving], axis=1) / speed_sq[moving]
            )
            closest = starts + np.clip(alpha, 0.0, 1.0)[:, None] * deltas
            distances = np.linalg.norm(closest, axis=1)
            required = agents[first].radius + agents[second].radius
            minimum = min(minimum, float(np.min(distances)))
            violations += int(np.sum(distances < required))
    return minimum, violations
