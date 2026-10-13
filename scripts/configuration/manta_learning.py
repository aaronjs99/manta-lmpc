"""Numerical settings that must remain identical when learning is resumed."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts.configuration.apf import APFConfig
    from scripts.configuration.manta_dynamics import MantaDynamicsConfig
    from scripts.configuration.manta_mpc import MantaLMPCConfig
    from scripts.contracts.manta import Scenario


def manta_learning_settings(
    scenario: Scenario,
    config: MantaLMPCConfig,
    apf_config: APFConfig,
    dynamics_config: MantaDynamicsConfig,
) -> dict[str, Any]:
    """Capture provenance without output settings or the extendable target."""
    lmpc = asdict(config)
    del lmpc["iterations"]
    return {
        "scenario": {
            "name": scenario.name,
            "starts": scenario.starts.tolist(),
            "goals": scenario.goals.tolist(),
            "safety_distance": scenario.safety_distance,
            "obstacle": asdict(scenario.obstacle),
        },
        "dynamics": asdict(dynamics_config),
        "apf": asdict(apf_config),
        "lmpc": lmpc,
    }
