"""Construct Harbor runtime platforms from parsed, model-independent settings."""

from __future__ import annotations

from pathlib import Path

from scripts.configuration.harbor import load_harbor_config
from scripts.contracts.harbor import HarborAgent, HarborAgentSpec
from scripts.dynamics.harbor.factory import make_platform_model


def assemble_harbor_agents(specs: list[HarborAgentSpec]) -> list[HarborAgent]:
    """Build and validate physical platforms from parsed configuration specs."""
    agents = []
    for spec in specs:
        model = make_platform_model(spec.kind, spec.model_parameters)
        agents.append(
            HarborAgent(
                name=spec.name,
                model=model,
                start=spec.start,
                goal=spec.goal,
                radius=spec.radius,
                domain=spec.domain,
                waypoints=spec.waypoints,
                profile=spec.profile,
                display_name=spec.display_name,
            )
        )
    return agents


def load_harbor_runtime(path: str | Path):
    """Load validated settings and assemble their runtime Harbor platform models."""
    specs, simulation, communication = load_harbor_config(path)
    agents = assemble_harbor_agents(specs)
    for agent in agents:
        if (
            agent.model.kind == "ugv"
            and agent.domain.y_bounds[0] < simulation.shoreline_y
        ):
            raise ValueError("UGV domain must remain on land above the shoreline")
        if (
            agent.model.kind in {"usv", "rov"}
            and agent.domain.y_bounds[1] > simulation.shoreline_y
        ):
            raise ValueError("USV/ROV domains must remain in harbor water")
        if agent.model.kind == "rov" and agent.domain.z_bounds[1] >= 0.0:
            raise ValueError("ROV domain must remain below the water surface")
    return agents, simulation, communication
