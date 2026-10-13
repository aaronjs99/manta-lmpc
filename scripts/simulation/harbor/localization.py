"""Simulation / harbor / localization."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from scripts.algorithms.estimation.ekf import RangeAidedEKF
from scripts.algorithms.estimation.factor_graph import FixedLagRangeSLAM, GraphRange
from scripts.algorithms.estimation.information import choose_information_waypoint
from scripts.configuration.factor_graph import FactorGraphConfig
from scripts.contracts.observability import ObservabilityReport
from scripts.simulation.harbor.sensors import RangeSensor

if TYPE_CHECKING:
    from scripts.configuration.localization import RangeAidedSLAMConfig


class HarborRangeLocalization:
    """Apply independent range-aided estimators behind one simulator boundary."""

    def __init__(self, config: RangeAidedSLAMConfig):
        self.config = config
        self.estimators: dict[str, RangeAidedEKF | FixedLagRangeSLAM] = {}
        self.sensors: dict[str, RangeSensor] = {}
        self._last_onboard_position: dict[str, np.ndarray] = {}
        self._odometry_rngs: dict[str, np.random.Generator] = {}
        self.reports: dict[str, list[ObservabilityReport]] = {}

    def estimate(
        self, *, agent, truth: np.ndarray, onboard: np.ndarray, step: int
    ) -> np.ndarray:
        """Fuse one platform's odometry increment and ranges into its state estimate."""
        dimension = 3 if agent.model.kind == "rov" else 2
        truth_position = np.asarray(agent.model.position(truth), dtype=float)[
            :dimension
        ]
        onboard_position = np.asarray(agent.model.position(onboard), dtype=float)[
            :dimension
        ]
        if agent.name not in self.estimators:
            if self.config.mode == "fixed_lag_slam":
                dimension = 3 if agent.model.kind == "rov" else 2
                compatible = {
                    beacon.name: np.asarray(
                        beacon.initial_position[:dimension], dtype=float
                    )
                    for beacon in self.config.beacons
                    if len(beacon.initial_position) >= dimension
                }
                fixed = {beacon.name for beacon in self.config.beacons if beacon.fixed}
                initial_pose = (
                    np.asarray(onboard, dtype=float)[:6]
                    if agent.model.kind == "rov"
                    else onboard_position
                )
                self.estimators[agent.name] = FixedLagRangeSLAM(
                    FactorGraphConfig(
                        lag=self.config.fixed_lag,
                        iterations=self.config.factor_iterations,
                        range_std=self.config.range_std,
                        odometry_std=max(
                            float(np.mean(self.config.motion_std[:dimension])), 1.0e-3
                        ),
                        depth_std=self.config.rov_depth_std,
                    ),
                    initial_pose,
                    compatible,
                    fixed,
                )
            else:
                self.estimators[agent.name] = RangeAidedEKF(
                    self.config, onboard_position
                )
            seed_offset = sum(
                (index + 1) * ord(char) for index, char in enumerate(agent.name)
            )
            self.sensors[agent.name] = RangeSensor(
                replace(self.config, seed=self.config.seed + seed_offset)
            )
            self._last_onboard_position[agent.name] = onboard_position.copy()
            self._odometry_rngs[agent.name] = np.random.default_rng(
                self.config.seed + seed_offset + 100000
            )
            self.reports[agent.name] = []
        estimator = self.estimators[agent.name]
        displacement = onboard_position - self._last_onboard_position[agent.name]
        if self.config.simulate_odometry_noise:
            std = np.asarray(self.config.motion_std[:dimension], dtype=float)
            bias = np.asarray(self.config.odometry_bias[:dimension], dtype=float)
            displacement = (
                displacement + bias + self._odometry_rngs[agent.name].normal(0.0, std)
            )
        if isinstance(estimator, FixedLagRangeSLAM):
            graph_measurements = tuple(
                GraphRange(
                    m.beacon, m.distance, robust=self.config.robust_range_loss != "none"
                )
                for m in self.sensors[agent.name].measure(truth_position, step)
            )
            attitude = (
                np.asarray(onboard, dtype=float)[3:6]
                if agent.model.kind == "rov"
                else None
            )
            depth = (
                float(onboard[2]) + self.config.rov_depth_bias
                if agent.model.kind == "rov"
                else None
            )
            estimator.update(
                displacement, graph_measurements, attitude=attitude, depth=depth
            )
            self._last_onboard_position[agent.name] = onboard_position.copy()
            self.reports[agent.name].append(
                ObservabilityReport(
                    estimator.pose.size,
                    estimator.last_report.rank,
                    estimator.last_report.sigma_min,
                    estimator.last_report.condition_number,
                    estimator.last_report.mode == "fully-observable",
                )
            )
            estimated = np.asarray(onboard, dtype=float).copy()
            if agent.model.kind == "rov":
                estimated[:6] = estimator.pose
            else:
                estimated[:dimension] = estimator.position
            return agent.domain.project(agent.model, estimated)
        estimator.predict(displacement)
        measurements = self.sensors[agent.name].measure(truth_position, step)
        estimator.update(measurements)
        self._last_onboard_position[agent.name] = onboard_position.copy()
        self.reports[agent.name].append(estimator.observability())
        estimated = np.asarray(onboard, dtype=float).copy()
        estimated[:dimension] = estimator.position
        return agent.domain.project(agent.model, estimated)

    def belief(self, agent_name: str) -> dict:
        """Expose estimator uncertainty without exposing plant truth."""
        estimator = self.estimators.get(agent_name)
        if estimator is None:
            return {"position_covariance": np.eye(3), "observability": None}
        if isinstance(estimator, FixedLagRangeSLAM):
            covariance = estimator.position_covariance.copy()
            report = estimator.last_report
        else:
            covariance = estimator.covariance[
                : estimator.dimension, : estimator.dimension
            ].copy()
            report = estimator.observability()
        return {"position_covariance": covariance, "observability": report}

    def information_goal(
        self, agent, state: np.ndarray, goal: np.ndarray
    ) -> np.ndarray:
        """Return a bounded sensing detour only during weak observability."""
        if not self.config.active_observability or agent.name not in self.estimators:
            return np.asarray(goal, dtype=float)
        report = self.belief(agent.name)["observability"]
        if report is None or getattr(report, "observable", False):
            return np.asarray(goal, dtype=float)
        dimension = 3 if agent.model.kind == "rov" else 2
        position = np.asarray(agent.model.position(state), dtype=float)[:dimension]
        beacons = [
            np.asarray(beacon.true_position, dtype=float)[:dimension]
            for beacon in self.config.beacons
            if len(beacon.true_position) >= dimension
        ]
        if len(beacons) < dimension:
            return np.asarray(goal, dtype=float)
        step = self.config.information_max_excursion
        candidates = np.vstack(
            [position]
            + [
                position + direction * step
                for axis in range(dimension)
                for direction in (-np.eye(dimension)[axis], np.eye(dimension)[axis])
            ]
        )
        candidate, gain = choose_information_waypoint(
            position,
            candidates,
            np.asarray(beacons),
            information_weight=self.config.information_weight,
            max_step=step,
        )
        if gain <= 1.0e-8:
            return np.asarray(goal, dtype=float)
        detour = np.asarray(goal, dtype=float).copy()
        detour[:dimension] = candidate
        return detour
