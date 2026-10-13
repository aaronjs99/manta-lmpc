"""Dynamics / harbor / factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.dynamics.harbor.ground import SkidSteerUGVModel, UGVModel
from scripts.dynamics.harbor.reduced import (
    ReducedROVModel,
    ReducedUGVModel,
    ReducedUSVModel,
)
from scripts.dynamics.harbor.surface import USVModel
from scripts.dynamics.harbor.underwater import ROVModel

if TYPE_CHECKING:
    from scripts.dynamics.harbor.base import PlatformModel


def make_platform_model(kind: str, parameters: dict[str, float]) -> PlatformModel:
    """Construct one platform model from YAML parameters."""
    parameters = dict(parameters)
    requested = str(parameters.pop("model", "physical")).lower()
    models = {
        ("ugv", "physical"): UGVModel,
        ("ugv", "kinematic_bicycle"): UGVModel,
        ("ugv", "dynamic_skid_steer"): SkidSteerUGVModel,
        ("ugv", "reduced"): ReducedUGVModel,
        ("ugv", "reduced_unicycle"): ReducedUGVModel,
        ("usv", "physical"): USVModel,
        ("usv", "marine_3dof"): USVModel,
        ("usv", "reduced"): ReducedUSVModel,
        ("usv", "reduced_surge_yaw"): ReducedUSVModel,
        ("rov", "physical"): ROVModel,
        ("rov", "marine_6dof"): ROVModel,
        ("rov", "reduced"): ReducedROVModel,
        ("rov", "reduced_world_damped"): ReducedROVModel,
    }
    try:
        return models[(kind.lower(), requested)](**parameters)
    except KeyError as exc:
        raise ValueError(
            f"unsupported harbor platform model: {kind}/{requested}"
        ) from exc
