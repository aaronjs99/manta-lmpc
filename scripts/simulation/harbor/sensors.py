"""Simulation / harbor / sensors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scripts.configuration.localization import RangeAidedSLAMConfig

from scripts.contracts.harbor import RangeMeasurement


class RangeSensor:
    """Generate configurable ranges without exposing truth to the estimator."""

    def __init__(self, config: RangeAidedSLAMConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._dropout_remaining = 0
        self._pending: list[tuple[int, RangeMeasurement]] = []
        self.telemetry: list[dict] = []

    def measure(self, position: np.ndarray, step: int) -> tuple[RangeMeasurement, ...]:
        delivered = [measurement for due, measurement in self._pending if due <= step]
        self._pending = [
            (due, measurement) for due, measurement in self._pending if due > step
        ]
        if not self.config.enabled or step % self.config.update_interval_steps:
            return tuple(delivered)
        if self._dropout_remaining:
            self._dropout_remaining -= 1
            self.telemetry.append({"step": step, "fault": "burst_dropout", "count": 0})
            return tuple(delivered)
        if self._rng.random() < self.config.burst_dropout_probability:
            self._dropout_remaining = self.config.burst_dropout_steps
            self.telemetry.append({"step": step, "fault": "burst_dropout", "count": 0})
            return tuple(delivered)
        position = np.asarray(position, dtype=float)
        measurements = []
        for beacon in self.config.beacons:
            truth = np.asarray(beacon.true_position, dtype=float)
            if position.size == 2 and truth.size == 3 and truth[2] == 0.0:
                truth = truth[:2]
            if truth.shape != position.shape:
                continue
            distance = float(np.linalg.norm(position - truth))
            if distance > self.config.maximum_range:
                continue
            if self._rng.random() < self.config.dropout_probability:
                self.telemetry.append(
                    {"step": step, "beacon": beacon.name, "fault": "dropout"}
                )
                continue
            fault = "none"
            bias = self.config.range_bias
            if self._rng.random() < self.config.nlos_probability:
                bias += self.config.nlos_bias
                fault = "nlos"
            noise_std = self.config.range_std
            if self._rng.random() < self.config.outlier_probability:
                noise_std = self.config.outlier_std
                fault = "outlier"
            noisy = distance + bias + self._rng.normal(0.0, noise_std)
            measurement = RangeMeasurement(beacon.name, max(0.0, float(noisy)), step)
            delay = self.config.measurement_delay_steps
            if self._rng.random() < self.config.stale_measurement_probability:
                delay = max(delay, 1)
                fault = "stale" if fault == "none" else f"{fault}_stale"
            if delay:
                self._pending.append((step + delay, measurement))
            else:
                measurements.append(measurement)
            self.telemetry.append(
                {
                    "step": step,
                    "beacon": beacon.name,
                    "fault": fault,
                    "delivery_step": step + delay,
                }
            )
        return tuple(delivered + measurements)
