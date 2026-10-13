"""Harbor MPC and LMPC experiment orchestration."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from scripts.algorithms.harbor.admission import _admissible, _controller_admissible
from scripts.metrics.harbor.learning import (
    _completion_cost,
    _controller_record,
    _record,
)
from scripts.mpc.harbor.controller import DistributedHarborMPC
from scripts.simulation.harbor.engine import run_harbor_simulation

if TYPE_CHECKING:
    from scripts.communication.harbor import LinkConfig
    from scripts.configuration.harbor_mpc import HarborMPCConfig
    from scripts.configuration.harbor_simulation import (
        HarborDisturbanceConfig,
        HarborSimulationConfig,
    )
    from scripts.contracts.harbor import HarborAgent, HarborLearningIteration


def run_distributed_harbor_lmpc(
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    communication: LinkConfig,
    mpc_config: HarborMPCConfig,
    disturbance: HarborDisturbanceConfig | None = None,
) -> list[HarborLearningIteration]:
    """Run guidance, plain MPC, and repeated safe-set LMPC experiments."""
    optimized_simulation = replace(
        simulation,
        guidance_update_interval_steps=mpc_config.replan_interval_steps,
    )
    guidance = run_harbor_simulation(
        agents, simulation, communication, disturbance=disturbance
    )
    if not _admissible(guidance):
        raise RuntimeError(
            "harbor iteration-0 guidance rollout is not safe and complete"
        )
    iterations = [
        _record(
            "guidance_iter_0",
            guidance,
            admitted=True,
            simulation=simulation,
        )
    ]
    safe_states = guidance.states
    safe_controls = guidance.controls
    best_cost = _completion_cost(guidance, simulation.horizon)

    mpc_controller = DistributedHarborMPC(
        agents=agents,
        config=mpc_config,
        dt=simulation.dt,
        safe_states=safe_states,
        safe_controls=safe_controls,
        learning=False,
    )
    mpc_result = run_harbor_simulation(
        agents,
        optimized_simulation,
        communication,
        control_provider=mpc_controller,
        disturbance=disturbance,
    )
    mpc_admitted = _controller_admissible(mpc_result, mpc_controller)
    mpc_cost = _completion_cost(mpc_result, simulation.horizon)
    iterations.append(
        _controller_record(
            "distributed_mpc",
            mpc_result,
            mpc_controller,
            admitted=mpc_admitted,
            simulation=simulation,
        )
    )
    if mpc_config.seed_learning_from_mpc and mpc_admitted and mpc_cost <= best_cost:
        safe_states = mpc_result.states
        safe_controls = mpc_result.controls
        best_cost = mpc_cost

    for iteration in range(1, mpc_config.learning_iterations + 1):
        controller = DistributedHarborMPC(
            agents=agents,
            config=mpc_config,
            dt=simulation.dt,
            safe_states=safe_states,
            safe_controls=safe_controls,
            learning=True,
        )
        result = run_harbor_simulation(
            agents,
            optimized_simulation,
            communication,
            control_provider=controller,
            disturbance=disturbance,
        )
        cost = _completion_cost(result, simulation.horizon)
        admitted = _controller_admissible(result, controller) and cost <= best_cost
        iterations.append(
            _controller_record(
                f"distributed_lmpc_{iteration}",
                result,
                controller,
                admitted=admitted,
                simulation=simulation,
            )
        )
        if admitted:
            safe_states = result.states
            safe_controls = result.controls
            best_cost = cost
    return iterations
