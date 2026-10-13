"""Public result type for one Harbor optimizer invocation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HarborMPCStep:
    """One optimizer result with prediction and relaxation telemetry."""

    control: np.ndarray
    predicted_states: np.ndarray
    max_collision_slack: float
    max_domain_buffer_slack: float
    max_dynamic_state_slack: float
    max_terminal_slack: float
