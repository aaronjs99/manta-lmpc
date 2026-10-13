"""Harbor optimizer facade coordinating construction, execution, and diagnosis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scripts.mpc.harbor.diagnostics import collect_failure_diagnostics, solver_status
from scripts.mpc.harbor.problem import HarborProblemBuilder, HarborProblemVariables
from scripts.mpc.harbor.result import HarborMPCStep
from scripts.mpc.harbor.solver import HarborSolveRequest, solve_numeric_problem

if TYPE_CHECKING:
    from typing import Any

    from scripts.configuration.harbor_mpc import HarborMPCConfig
    from scripts.contracts.harbor import HarborAgent


class HarborAgentOptimizer:
    """One CasADi NLP for a single platform and communicated obstacle tracks."""

    def __init__(
        self,
        *,
        agent: HarborAgent,
        other_agents: list[HarborAgent],
        config: HarborMPCConfig,
        dt: float,
        learning: bool,
    ) -> None:
        try:
            import casadi as ca
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CasADi is required for harbor MPC. Install requirements.txt."
            ) from exc
        self.agent = agent
        self.other_agents = tuple(other_agents)
        self.config = config
        self.dt = dt
        self.learning = learning
        self.last_status = "not_run"
        self.last_failure_diagnostics: dict[str, Any] = {}
        self.problem: HarborProblemVariables = HarborProblemBuilder(
            agent=agent,
            other_agents=self.other_agents,
            config=config,
            dt=dt,
            learning=learning,
            ca=ca,
        ).build()

    def solve(
        self,
        *,
        state: np.ndarray,
        goal: np.ndarray,
        obstacle_predictions: dict[str, np.ndarray],
        safe_states: np.ndarray,
        safe_costs: np.ndarray,
        warm_states: np.ndarray,
        warm_controls: np.ndarray,
        previous_control: np.ndarray,
        position_drift: np.ndarray,
        control_effectiveness: np.ndarray,
        identification_probe: np.ndarray,
        identification_probe_mask: np.ndarray,
        belief_collision_margin: float = 0.0,
    ) -> HarborMPCStep:
        """Solve from one explicit numeric request, retaining status telemetry."""
        request = HarborSolveRequest(
            state=state,
            goal=goal,
            obstacle_predictions=obstacle_predictions,
            safe_states=safe_states,
            safe_costs=safe_costs,
            warm_states=warm_states,
            warm_controls=warm_controls,
            previous_control=previous_control,
            position_drift=position_drift,
            control_effectiveness=control_effectiveness,
            identification_probe=identification_probe,
            identification_probe_mask=identification_probe_mask,
            belief_collision_margin=belief_collision_margin,
        )
        try:
            result, self.last_status = solve_numeric_problem(
                problem=self.problem,
                request=request,
                agent=self.agent,
                other_agents=self.other_agents,
                config=self.config,
                learning=self.learning,
            )
        except RuntimeError:
            self.last_status = solver_status(self.problem.opti)
            self.last_failure_diagnostics = collect_failure_diagnostics(
                self.problem.opti, self.last_status
            )
            raise
        self.last_failure_diagnostics = {}
        return result


__all__ = ["HarborAgentOptimizer", "HarborMPCStep"]
