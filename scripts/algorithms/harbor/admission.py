"""Algorithms / harbor / admission."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.contracts.harbor import HarborResult


def _admissible(result: HarborResult) -> bool:
    return result.all_goals_reached and result.pairwise_violation_count == 0


def _controller_admissible(result: HarborResult, controller) -> bool:
    """Reject physically invalid or solver-contaminated learning data."""
    return (
        _admissible(result)
        and controller.telemetry.retry.fallback_count == 0
        and controller.max_collision_slack <= 1e-9
    )
