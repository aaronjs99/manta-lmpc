"""Metrics / harbor / estimation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scripts.contracts.harbor import HarborAgent


def _effectiveness_rmse_history(trial, agents, disturbance) -> np.ndarray:
    lengths = [len(trial.effectiveness_history[agent.name]) for agent in agents]
    count = min(lengths, default=0)
    values = []
    for index in range(count):
        error = np.concatenate(
            [
                trial.effectiveness_history[agent.name][index]
                - disturbance.effectiveness(agent.model, agent.name)
                for agent in agents
            ]
        )
        values.append(float(np.sqrt(np.mean(error * error))))
    return np.asarray(values, dtype=float)


def _maximum_information_std_history(trial, agents) -> np.ndarray:
    lengths = [len(trial.information_std_history[agent.name]) for agent in agents]
    count = min(lengths, default=0)
    return np.asarray(
        [
            max(
                float(np.max(trial.information_std_history[agent.name][index]))
                for agent in agents
            )
            for index in range(count)
        ],
        dtype=float,
    )


def _flatten_effectiveness(estimates, agents: list[HarborAgent]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(estimates[agent.name], dtype=float) for agent in agents]
    )
