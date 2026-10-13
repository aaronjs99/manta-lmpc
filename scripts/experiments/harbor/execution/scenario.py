"""Prepared immutable inputs for one matched Harbor trial batch."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from scripts.configuration.harbor_simulation import (
    HarborDisturbanceConfig,
    HarborObservationNoiseConfig,
    HarborSimulationConfig,
)
from scripts.configuration.harbor_mpc import HarborMPCConfig
from scripts.contracts.harbor import HarborResult

if TYPE_CHECKING:
    from scripts.communication.harbor import LinkConfig
    from scripts.contracts.harbor import HarborAgent


@dataclass(frozen=True)
class HarborTrialScenario:
    """Shared scenario and seed trajectory consumed by a matched policy batch."""

    agents: list[HarborAgent]
    simulation: HarborSimulationConfig
    evaluation: HarborSimulationConfig
    communication: LinkConfig
    mpc: HarborMPCConfig
    disturbance: HarborDisturbanceConfig
    seed_result: HarborResult
    observation_noise: HarborObservationNoiseConfig | None = None
    residual_adaptation: bool = False
    residual_adaptation_kinds: tuple[str, ...] | None = None


def prepare_trial_scenario(
    *,
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    communication: LinkConfig,
    mpc: HarborMPCConfig,
    disturbance: HarborDisturbanceConfig,
    seed_result: HarborResult,
    evaluation: HarborSimulationConfig | None = None,
    observation_noise: HarborObservationNoiseConfig | None = None,
    residual_adaptation: bool = False,
    residual_adaptation_kinds: tuple[str, ...] | None = None,
) -> HarborTrialScenario:
    """Derive evaluation timing from the study config once, preserving inputs."""
    if evaluation is None:
        evaluation = replace(
            simulation,
            guidance_update_interval_steps=mpc.replan_interval_steps,
            goal_hold_steps=disturbance.evaluation_hold_steps,
        )
    return HarborTrialScenario(
        agents=agents,
        simulation=simulation,
        evaluation=evaluation,
        communication=communication,
        mpc=mpc,
        disturbance=disturbance,
        seed_result=seed_result,
        observation_noise=observation_noise,
        residual_adaptation=residual_adaptation,
        residual_adaptation_kinds=residual_adaptation_kinds,
    )
