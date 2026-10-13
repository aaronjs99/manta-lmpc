"""Experiments / harbor / execution / ensembles."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scripts.configuration.harbor_faults import (
        HarborFaultEnsembleConfig,
        HarborTemporaryFaultEnsembleConfig,
    )
    from scripts.configuration.harbor_simulation import HarborDisturbanceConfig
    from scripts.contracts.harbor import HarborAgent


def generate_fault_ensemble(
    agents: list[HarborAgent],
    base_disturbance: HarborDisturbanceConfig,
    config: HarborFaultEnsembleConfig,
) -> list[tuple[int, HarborDisturbanceConfig]]:
    """Generate deterministic Latin-hypercube actuator losses by channel."""
    channel_count = sum(agent.model.control_dim for agent in agents)
    case_count = len(config.seeds)
    rng = np.random.default_rng(np.random.SeedSequence(config.seeds))
    unit_samples = np.empty((case_count, channel_count), dtype=float)
    for channel in range(channel_count):
        strata = rng.permutation(case_count)
        unit_samples[:, channel] = (strata + rng.random(case_count)) / case_count
    samples = config.effectiveness_min + unit_samples * (
        config.effectiveness_max - config.effectiveness_min
    )

    cases = []
    for row, seed_value in zip(samples, config.seeds, strict=True):
        offset = 0
        effectiveness = {}
        for agent in agents:
            dimension = agent.model.control_dim
            effectiveness[agent.name] = tuple(row[offset : offset + dimension])
            offset += dimension
        cases.append(
            (
                seed_value,
                replace(
                    base_disturbance,
                    agent_control_effectiveness=effectiveness,
                ),
            )
        )
    return cases


def generate_temporary_fault_ensemble(
    agents: list[HarborAgent],
    base_disturbance: HarborDisturbanceConfig,
    config: HarborTemporaryFaultEnsembleConfig,
) -> list[tuple[int, int, HarborDisturbanceConfig]]:
    """Generate matched temporary losses with stratified timing and severity."""
    case_count = len(config.seeds)
    channel_count = sum(agent.model.control_dim for agent in agents)
    timing_count = 2 * len(agents)
    rng = np.random.default_rng(np.random.SeedSequence(config.seeds))
    unit_samples = np.empty((case_count, channel_count + timing_count), dtype=float)
    for dimension in range(unit_samples.shape[1]):
        strata = rng.permutation(case_count)
        unit_samples[:, dimension] = (strata + rng.random(case_count)) / case_count

    effectiveness = config.effectiveness_min + unit_samples[:, :channel_count] * (
        config.effectiveness_max - config.effectiveness_min
    )
    cases = []
    for case_index, seed_value in enumerate(config.seeds):
        schedule = {}
        channel_offset = 0
        for agent_index, agent in enumerate(agents):
            dimension = agent.model.control_dim
            degraded = tuple(
                effectiveness[case_index, channel_offset : channel_offset + dimension]
            )
            onset_unit = unit_samples[case_index, channel_count + 2 * agent_index]
            duration_unit = unit_samples[
                case_index, channel_count + 2 * agent_index + 1
            ]
            onset = config.onset_step_min + int(
                onset_unit * (config.onset_step_max - config.onset_step_min + 1)
            )
            duration = config.duration_step_min + int(
                duration_unit
                * (config.duration_step_max - config.duration_step_min + 1)
            )
            schedule[agent.name] = (
                (onset, degraded),
                (onset + duration, tuple(np.ones(dimension))),
            )
            channel_offset += dimension
        water_current = base_disturbance.water_current
        if config.water_current_min is not None:
            current_rng = np.random.default_rng(
                np.random.SeedSequence([seed_value, 20260718])
            )
            water_current = tuple(
                current_rng.uniform(
                    np.asarray(config.water_current_min, dtype=float),
                    np.asarray(config.water_current_max, dtype=float),
                )
            )
        cases.append(
            (
                seed_value,
                config.observation_seed_offset + seed_value,
                replace(
                    base_disturbance,
                    water_current=water_current,
                    agent_control_effectiveness={},
                    agent_control_effectiveness_schedule=schedule,
                ),
            )
        )
    return cases
