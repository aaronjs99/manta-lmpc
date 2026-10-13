"""Configuration / harbor."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from scripts.communication.harbor import LinkConfig
from scripts.configuration._validation import bounds as _bounds
from scripts.configuration._validation import (
    dataclass_from_mapping as _dataclass_from_mapping,
)
from scripts.configuration.harbor_simulation import (
    HarborSimulationConfig,
)
from scripts.configuration.paths import DEFAULT_HARBOR_CONFIG, resolve_config_path
from scripts.contracts.harbor import HarborAgentSpec, OperatingDomain

if TYPE_CHECKING:
    pass


def load_harbor_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
) -> tuple[list[HarborAgentSpec], HarborSimulationConfig, LinkConfig]:
    """Parse platform settings without constructing physical models."""
    config_path = resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    simulation = _dataclass_from_mapping(
        HarborSimulationConfig, raw.get("simulation", {}), "simulation"
    )
    communication = _dataclass_from_mapping(
        LinkConfig, raw.get("communication", {}), "communication"
    )
    model_parameters = raw.get("platform_models", {})
    platform_profiles = raw.get("platform_profiles", {})
    agents: list[HarborAgentSpec] = []
    for entry in raw.get("agents", []):
        profile_name = entry.get("profile")
        if profile_name is not None:
            if profile_name not in platform_profiles:
                raise ValueError(
                    f"agent {entry['name']} references unknown profile {profile_name}"
                )
            profile = platform_profiles[profile_name]
            kind = str(profile["kind"]).lower()
            if "kind" in entry and str(entry["kind"]).lower() != kind:
                raise ValueError(
                    f"agent {entry['name']} kind disagrees with profile {profile_name}"
                )
            parameters = dict(profile.get("parameters", {}))
            display_name = str(profile.get("label", profile_name))
        else:
            kind = str(entry["kind"]).lower()
            parameters = dict(model_parameters.get(kind, {}))
            display_name = entry.get("display_name")
        start = np.asarray(entry["start"], dtype=float)
        goal = np.asarray(entry["goal"], dtype=float)
        domain_data = entry["domain"]
        domain = OperatingDomain(
            x_bounds=_bounds(domain_data["x"], f"{entry['name']}.domain.x"),
            y_bounds=_bounds(domain_data["y"], f"{entry['name']}.domain.y"),
            z_bounds=_bounds(domain_data["z"], f"{entry['name']}.domain.z"),
        )
        radius = float(entry["radius"])
        if radius <= 0.0:
            raise ValueError(f"agent {entry['name']} radius must be positive")
        agents.append(
            HarborAgentSpec(
                name=str(entry["name"]),
                kind=kind,
                model_parameters=parameters,
                start=start,
                goal=goal,
                radius=radius,
                domain=domain,
                waypoints=(
                    np.asarray(entry["waypoints"], dtype=float)
                    if "waypoints" in entry
                    else None
                ),
                profile=(str(profile_name) if profile_name is not None else None),
                display_name=(
                    str(entry.get("display_name", display_name))
                    if entry.get("display_name", display_name) is not None
                    else None
                ),
            )
        )
    if len(agents) < 2:
        raise ValueError("harbor config requires at least two agents")
    return agents, simulation, communication
