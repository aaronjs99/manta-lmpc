"""Simulation / harbor / localization modes."""

from __future__ import annotations

from dataclasses import replace

from scripts.simulation.harbor.localization import HarborRangeLocalization


def _localization(policy: str, config, seed: int):
    configured = replace(config, seed=seed)
    if policy == "Direct position sensor":
        return None
    if policy == "Dead reckoning":
        return HarborRangeLocalization(
            replace(configured, enabled=False, mode="known_anchor_ekf")
        )
    if policy == "Known harbor map":
        return HarborRangeLocalization(
            replace(configured, enabled=True, mode="known_anchor_ekf")
        )
    if policy == "Robust fixed-lag SLAM":
        return HarborRangeLocalization(
            replace(configured, enabled=True, mode="fixed_lag_slam")
        )
    return HarborRangeLocalization(
        replace(configured, enabled=True, mode="joint_landmark_ekf")
    )
