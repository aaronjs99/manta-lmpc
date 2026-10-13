"""Configuration / localization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scripts.configuration.paths import resolve_config_path


@dataclass(frozen=True)
class RangeBeacon:
    """A simulated harbor transponder and its estimator prior."""

    name: str
    true_position: tuple[float, ...]
    initial_position: tuple[float, ...]
    fixed: bool = False

    def __post_init__(self) -> None:
        truth = tuple(float(value) for value in self.true_position)
        initial = tuple(float(value) for value in self.initial_position)
        if (
            not self.name.strip()
            or len(truth) not in {2, 3}
            or len(initial) != len(truth)
        ):
            raise ValueError(
                "range beacons require a name and matching 2D or 3D positions"
            )
        if not np.all(np.isfinite(truth + initial)):
            raise ValueError("range beacon positions must be finite")
        object.__setattr__(self, "true_position", truth)
        object.__setattr__(self, "initial_position", initial)


@dataclass(frozen=True)
class RangeAidedSLAMConfig:
    """Sensor, process, and observability settings for range-aided estimation."""

    enabled: bool = False
    mode: str = "joint_landmark_ekf"
    fixed_lag: int = 8
    factor_iterations: int = 4
    robust_range_loss: str = "huber"
    range_std: float = 0.08
    range_bias: float = 0.0
    dropout_probability: float = 0.0
    nlos_probability: float = 0.0
    nlos_bias: float = 0.0
    outlier_probability: float = 0.0
    outlier_std: float = 1.0
    burst_dropout_probability: float = 0.0
    burst_dropout_steps: int = 0
    measurement_delay_steps: int = 0
    stale_measurement_probability: float = 0.0
    active_observability: bool = False
    information_weight: float = 0.25
    information_max_excursion: float = 0.5
    rov_depth_std: float = 0.06
    rov_depth_bias: float = 0.0
    maximum_range: float = 20.0
    update_interval_steps: int = 1
    motion_std: tuple[float, ...] = (0.03, 0.03, 0.02)
    odometry_bias: tuple[float, ...] = (0.002, -0.001, 0.001)
    simulate_odometry_noise: bool = True
    initial_pose_std: float = 0.5
    initial_landmark_std: float = 2.0
    observability_window: int = 20
    observability_rank_tolerance: float = 1.0e-6
    defer_landmarks_until_pose_observable: bool = True
    landmark_pose_coupling_std: float = 0.5
    seed: int = 801
    beacons: tuple[RangeBeacon, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {
            "known_anchor_ekf",
            "joint_landmark_ekf",
            "fixed_lag_slam",
        }:
            raise ValueError(
                "range-aided mode must be known_anchor_ekf, joint_landmark_ekf, or fixed_lag_slam"
            )
        if self.fixed_lag < 2 or self.factor_iterations < 1:
            raise ValueError("fixed-lag settings must be positive")
        if self.robust_range_loss not in {"huber", "none"}:
            raise ValueError("robust_range_loss must be huber or none")
        if self.range_std <= 0.0 or self.maximum_range <= 0.0:
            raise ValueError("range noise and maximum range must be positive")
        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError("range dropout probability must lie in [0, 1]")
        probabilities = (
            self.nlos_probability,
            self.outlier_probability,
            self.burst_dropout_probability,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("range fault probabilities must lie in [0, 1]")
        if self.nlos_bias < 0.0 or self.outlier_std <= 0.0:
            raise ValueError("NLOS bias must be nonnegative and outlier_std positive")
        if not 0.0 <= self.stale_measurement_probability <= 1.0:
            raise ValueError("stale_measurement_probability must lie in [0, 1]")
        if self.burst_dropout_steps < 0 or self.measurement_delay_steps < 0:
            raise ValueError("range delay settings must be nonnegative")
        if self.information_weight < 0.0 or self.information_max_excursion <= 0.0:
            raise ValueError("information settings must be nonnegative and bounded")
        if self.rov_depth_std <= 0.0:
            raise ValueError("rov_depth_std must be positive")
        if self.update_interval_steps <= 0 or self.observability_window <= 0:
            raise ValueError(
                "range update interval and observability window must be positive"
            )
        if self.initial_pose_std <= 0.0 or self.initial_landmark_std <= 0.0:
            raise ValueError("range-aided prior standard deviations must be positive")
        if self.landmark_pose_coupling_std <= 0.0:
            raise ValueError(
                "landmark pose-coupling standard deviation must be positive"
            )
        dimensions = {len(beacon.true_position) for beacon in self.beacons}
        if len(dimensions) > 1 or len({beacon.name for beacon in self.beacons}) != len(
            self.beacons
        ):
            raise ValueError(
                "range beacons must have unique names and one spatial dimension"
            )
        if len(self.motion_std) not in {2, 3} or len(self.odometry_bias) not in {2, 3}:
            raise ValueError("motion_std and odometry_bias must cover 2D or 3D motion")
        if np.any(np.asarray(self.motion_std) < 0.0):
            raise ValueError("motion_std must be nonnegative")


def load_range_aided_slam_config(path: str | Path) -> RangeAidedSLAMConfig:
    """Load nested range-SLAM settings while rejecting unknown fields."""
    config_path = resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        section = (yaml.safe_load(stream) or {}).get("range_aided_slam", {})
    beacon_fields = {"name", "true_position", "initial_position", "fixed"}
    beacons = []
    for raw in section.get("beacons", []):
        unknown = set(raw) - beacon_fields
        if unknown:
            raise ValueError(
                f"unknown range beacon field(s): {', '.join(sorted(unknown))}"
            )
        beacons.append(RangeBeacon(**raw))
    values = {key: value for key, value in section.items() if key != "beacons"}
    known = set(RangeAidedSLAMConfig.__dataclass_fields__) - {"beacons"}
    unknown = set(values) - known
    if unknown:
        raise ValueError(
            f"unknown range_aided_slam field(s): {', '.join(sorted(unknown))}"
        )
    return RangeAidedSLAMConfig(**values, beacons=tuple(beacons))
