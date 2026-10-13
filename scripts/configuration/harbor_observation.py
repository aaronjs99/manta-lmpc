"""Harbor observation-noise settings loader."""

from __future__ import annotations

from pathlib import Path

import yaml
from scripts.configuration._validation import (
    dataclass_from_mapping as _dataclass_from_mapping,
)
from scripts.configuration.harbor_simulation import HarborObservationNoiseConfig
from scripts.configuration.paths import DEFAULT_HARBOR_CONFIG, resolve_config_path


def load_harbor_observation_noise_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
) -> HarborObservationNoiseConfig:
    """Load strict platform-aware onboard observation noise settings."""
    config_path = resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return _dataclass_from_mapping(
        HarborObservationNoiseConfig,
        raw.get("observation_noise", {}),
        "observation_noise",
    )
