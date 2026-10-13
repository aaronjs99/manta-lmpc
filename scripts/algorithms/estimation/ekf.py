"""Algorithms / estimation / ekf."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scripts.contracts.observability import ObservabilityReport

if TYPE_CHECKING:
    from scripts.configuration.localization import RangeAidedSLAMConfig
    from scripts.contracts.harbor import RangeMeasurement


class RangeAidedEKF:
    """Estimate platform position and, optionally, unknown beacon positions."""

    def __init__(self, config: RangeAidedSLAMConfig, initial_position: np.ndarray):
        self.config = config
        self.dimension = int(np.asarray(initial_position).size)
        if self.dimension not in {2, 3}:
            raise ValueError("range-aided EKF position must be 2D or 3D")
        compatible = [
            beacon
            for beacon in config.beacons
            if len(beacon.true_position) == self.dimension
            or (
                self.dimension == 2
                and len(beacon.true_position) == 3
                and beacon.true_position[2] == 0.0
            )
        ]
        self._beacons = {beacon.name: beacon for beacon in compatible}
        self._landmark_slices: dict[str, slice] = {}
        state = list(np.asarray(initial_position, dtype=float))
        for beacon in compatible:
            estimated = config.mode == "joint_landmark_ekf" and not beacon.fixed
            if estimated:
                start = len(state)
                self._landmark_slices[beacon.name] = slice(
                    start, start + self.dimension
                )
                state.extend(beacon.initial_position[: self.dimension])
        self.state = np.asarray(state, dtype=float)
        variances = np.full(self.state.size, config.initial_pose_std**2)
        for landmark_slice in self._landmark_slices.values():
            variances[landmark_slice] = config.initial_landmark_std**2
        self.covariance = np.diag(variances)
        self._jacobian_window: list[np.ndarray] = []
        self._anchor_jacobian_window: list[np.ndarray] = []
        self.deferred_landmark_updates = 0
        self.landmark_only_updates = 0

    @property
    def position(self) -> np.ndarray:
        return self.state[: self.dimension].copy()

    @property
    def landmark_estimates(self) -> dict[str, np.ndarray]:
        return {
            name: self.state[index].copy()
            for name, index in self._landmark_slices.items()
        }

    def predict(self, odometry_displacement: np.ndarray) -> None:
        displacement = np.asarray(odometry_displacement, dtype=float)
        if displacement.shape != (self.dimension,):
            raise ValueError("odometry displacement has the wrong dimension")
        self.state[: self.dimension] += displacement
        configured = np.asarray(self.config.motion_std, dtype=float)
        motion_std = configured[: self.dimension]
        if motion_std.size != self.dimension:
            raise ValueError("motion_std must cover the estimator dimension")
        self.covariance[: self.dimension, : self.dimension] += np.diag(
            motion_std * motion_std
        )

    def update(self, measurements: tuple[RangeMeasurement, ...]) -> int:
        accepted = 0
        for measurement in measurements:
            beacon = self._beacons.get(measurement.beacon)
            if beacon is None:
                continue
            landmark_slice = self._landmark_slices.get(measurement.beacon)
            if (
                landmark_slice is not None
                and self.config.defer_landmarks_until_pose_observable
                and not self.pose_observable()
            ):
                self.deferred_landmark_updates += 1
                continue
            landmark = (
                self.state[landmark_slice]
                if landmark_slice is not None
                else np.asarray(beacon.true_position[: self.dimension], dtype=float)
            )
            delta = self.state[: self.dimension] - landmark
            predicted = float(np.linalg.norm(delta))
            if predicted <= 1.0e-9:
                continue
            direction = delta / predicted
            jacobian = np.zeros((1, self.state.size))
            jacobian[0, : self.dimension] = direction
            if landmark_slice is not None:
                jacobian[0, landmark_slice] = -direction
            innovation_variance = float(
                (
                    jacobian @ self.covariance @ jacobian.T + self.config.range_std**2
                ).item()
            )
            gain = self.covariance @ jacobian.T / innovation_variance
            landmark_only = False
            if landmark_slice is not None:
                landmark_std = np.sqrt(np.diag(self.covariance)[landmark_slice])
                if float(np.max(landmark_std)) > self.config.landmark_pose_coupling_std:
                    gain[: self.dimension] = 0.0
                    self.landmark_only_updates += 1
                    landmark_only = True
            navigation_covariance = self.covariance[: self.dimension, :].copy()
            innovation = measurement.distance - self.config.range_bias - predicted
            self.state += gain[:, 0] * innovation
            identity = np.eye(self.state.size)
            residual = identity - gain @ jacobian
            noise = self.config.range_std**2
            self.covariance = (
                residual @ self.covariance @ residual.T + noise * gain @ gain.T
            )
            if landmark_only:
                self.covariance[: self.dimension, :] = navigation_covariance
                self.covariance[:, : self.dimension] = navigation_covariance.T
            self._jacobian_window.append(jacobian)
            self._jacobian_window = self._jacobian_window[
                -self.config.observability_window :
            ]
            if landmark_slice is None:
                self._anchor_jacobian_window.append(direction.reshape(1, -1))
                self._anchor_jacobian_window = self._anchor_jacobian_window[
                    -self.config.observability_window :
                ]
            accepted += 1
        return accepted

    def pose_observable(self) -> bool:
        """Return whether recent fixed-anchor geometry spans platform position."""
        if not self._anchor_jacobian_window:
            return False
        matrix = np.vstack(self._anchor_jacobian_window)
        singular = np.linalg.svd(matrix, compute_uv=False)
        rank = int(
            np.count_nonzero(singular > self.config.observability_rank_tolerance)
        )
        return rank == self.dimension

    def observability(self) -> ObservabilityReport:
        if not self._jacobian_window:
            return ObservabilityReport(self.state.size, 0, 0.0, float("inf"), False)
        matrix = np.vstack(self._jacobian_window) / self.config.range_std
        singular = np.linalg.svd(matrix, compute_uv=False)
        tolerance = self.config.observability_rank_tolerance
        rank = int(np.count_nonzero(singular > tolerance))
        smallest = float(singular[-1]) if singular.size else 0.0
        condition = (
            float(singular[0] / smallest) if smallest > tolerance else float("inf")
        )
        return ObservabilityReport(
            self.state.size,
            rank,
            smallest,
            condition,
            rank == self.state.size,
        )
