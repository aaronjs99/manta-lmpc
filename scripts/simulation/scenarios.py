"""Simulation / scenarios."""

from __future__ import annotations

from scripts.contracts.manta import Scenario


def get_scenario(name: str, config_path: str | None = None) -> Scenario:
    """Load one scenario from the YAML config.

    New code should usually call ``scripts.config.load_project_config`` so all
    runtime settings come from one place. This helper remains for small scripts
    or notebooks that only need scenario geometry.
    """
    from scripts.configuration.manta import load_project_config
    from scripts.configuration.paths import DEFAULT_CONFIG_PATH

    return load_project_config(
        config_path or DEFAULT_CONFIG_PATH, scenario_name=name
    ).scenario


def list_scenarios(config_path: str | None = None) -> list[str]:
    """List scenario names from the YAML config."""
    from scripts.configuration.manta import list_config_scenarios
    from scripts.configuration.paths import DEFAULT_CONFIG_PATH

    return list_config_scenarios(config_path or DEFAULT_CONFIG_PATH)
