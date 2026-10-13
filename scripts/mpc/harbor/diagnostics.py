"""Failure diagnostics for Harbor nonlinear optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any


def solver_status(opti: Any) -> str:
    """Return the most recent solver status, with a stable fallback value."""
    return str(opti.stats().get("return_status", "unknown"))


def collect_failure_diagnostics(
    opti: Any, status: str, limit: int = 6
) -> dict[str, Any]:
    """Capture the largest constraint residuals at IPOPT's final iterate."""
    try:
        values = np.asarray(opti.debug.value(opti.g), dtype=float).reshape(-1)
        lower = np.asarray(opti.debug.value(opti.lbg), dtype=float).reshape(-1)
        upper = np.asarray(opti.debug.value(opti.ubg), dtype=float).reshape(-1)
    except Exception:
        return {"status": status, "constraints": []}
    violations = np.maximum(lower - values, values - upper)
    indices = np.argsort(violations)[::-1]
    constraints = []
    for index in indices[:limit]:
        violation = float(max(violations[index], 0.0))
        if violation <= 1.0e-7:
            break
        constraints.append(
            {
                "index": int(index),
                "violation": violation,
                "value": float(values[index]),
                "lower": float(lower[index]),
                "upper": float(upper[index]),
                "description": str(opti.debug.g_describe(int(index))),
            }
        )
    return {"status": status, "constraints": constraints}
