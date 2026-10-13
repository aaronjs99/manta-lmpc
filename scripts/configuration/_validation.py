"""Small reusable validators for configuration mapping loaders."""

from __future__ import annotations

from dataclasses import fields

import numpy as np


def dataclass_from_mapping(cls, data: dict, section: str):
    """Reject unknown keys before constructing a validated settings record."""
    known = {field.name for field in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown {section} field(s): {', '.join(unknown)}")
    return cls(**data)


def bounds(values, name: str) -> tuple[float, float]:
    """Parse a strictly increasing two-value interval."""
    array = np.asarray(values, dtype=float)
    if array.shape != (2,) or not array[0] < array[1]:
        raise ValueError(f"{name} must contain increasing [min, max] values")
    return float(array[0]), float(array[1])
