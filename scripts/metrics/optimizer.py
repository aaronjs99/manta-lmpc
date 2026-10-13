"""Metrics / optimizer."""

from __future__ import annotations

import numpy as np


def summarize_optimizer_slack(values: np.ndarray) -> dict[str, float | int | None]:
    """Summarize per-step static, hyperplane, and terminal slack telemetry."""
    telemetry = np.asarray(values, dtype=float)
    if telemetry.ndim != 3 or telemetry.shape[2] != 3:
        raise ValueError(
            f"slack telemetry must have shape (steps, agents, 3), got {telemetry.shape}"
        )

    finite = np.isfinite(telemetry)
    solved_agent_steps = int(np.sum(np.all(finite, axis=2)))
    summary: dict[str, float | int | None] = {
        "solved_agent_steps": solved_agent_steps,
    }
    for index, name in enumerate(("static", "hyperplane", "terminal")):
        channel = telemetry[:, :, index]
        valid = channel[np.isfinite(channel)]
        summary[f"max_{name}_slack"] = float(np.max(valid)) if len(valid) else None
        summary[f"nonzero_{name}_slack_steps"] = int(np.sum(valid > 1e-8))
    return summary


def summarize_optimizer_slack_by_agent(
    values: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    """Summarize optimizer slack independently for each agent column."""
    telemetry = np.asarray(values, dtype=float)
    summarize_optimizer_slack(telemetry)
    return {
        str(agent): summarize_optimizer_slack(telemetry[:, agent : agent + 1, :])
        for agent in range(telemetry.shape[1])
    }
