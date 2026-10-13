"""Shared YAML loading for Harbor experiment-specific criteria contracts."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml

from scripts.configuration._validation import dataclass_from_mapping
from scripts.configuration.paths import resolve_config_path

Criteria = TypeVar("Criteria")


def load_harbor_criteria(
    criteria_type: type[Criteria],
    path: str | Path,
    section: str,
) -> Criteria:
    """Load one criteria dataclass from its exact configured YAML section."""
    config_path = resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return dataclass_from_mapping(criteria_type, raw.get(section, {}), section)
