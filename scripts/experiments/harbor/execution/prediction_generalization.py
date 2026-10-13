"""Peer-motion prediction generalization study execution."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from scripts.contracts.harbor import HarborFaultEnsembleCase
from scripts.experiments.harbor.execution.ensembles import generate_fault_ensemble
from scripts.experiments.harbor.execution.policy import HarborPolicySpec
from scripts.experiments.harbor.execution.trials import run_policy_trials
from scripts.experiments.harbor.learning import run_distributed_harbor_lmpc

if TYPE_CHECKING:
    from scripts.communication.harbor import LinkConfig
    from scripts.configuration.harbor_faults import (
        HarborFaultEnsembleConfig,
        HarborFaultStudyConfig,
    )
    from scripts.configuration.harbor_mpc import HarborMPCConfig
    from scripts.configuration.harbor_simulation import (
        HarborDisturbanceConfig,
        HarborObservationNoiseConfig,
        HarborSimulationConfig,
    )
    from scripts.contracts.harbor import HarborAgent


def run_obstacle_prediction_generalization(
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    communication: LinkConfig,
    mpc_config: HarborMPCConfig,
    base_disturbance: HarborDisturbanceConfig,
    study_config: HarborFaultStudyConfig,
    ensemble_config: HarborFaultEnsembleConfig,
    observation_noise: HarborObservationNoiseConfig,
) -> list[HarborFaultEnsembleCase]:
    """Compare peer-motion predictors from one common clean trajectory seed."""
    study_mpc = replace(
        mpc_config,
        prediction_horizon=study_config.prediction_horizon,
        terminal_goal_weight=study_config.terminal_goal_weight,
        terminal_slack_bound=study_config.terminal_slack_bound,
        terminal_slack_weight=study_config.terminal_slack_weight,
    )
    seed_iterations = run_distributed_harbor_lmpc(
        agents,
        simulation,
        communication,
        replace(
            study_mpc,
            obstacle_prediction_mode="constant_velocity",
            learning_iterations=1,
            residual_adaptation=False,
            control_effectiveness_adaptation=False,
        ),
    )
    seed = min(
        (record for record in seed_iterations if record.admitted),
        key=lambda record: record.completion_step_sum,
    )
    modes = (
        ("Constant-velocity prediction", "constant_velocity"),
        ("Goal-bounded prediction", "goal_bounded_velocity"),
    )
    cases = []
    for seed_value, disturbance in generate_fault_ensemble(
        agents, base_disturbance, ensemble_config
    ):
        case_noise = replace(
            observation_noise,
            enabled=True,
            seed=observation_noise.seed + seed_value,
        )
        evaluation = replace(
            simulation,
            guidance_update_interval_steps=study_mpc.replan_interval_steps,
            goal_hold_steps=disturbance.evaluation_hold_steps,
        )
        trials = []
        print(f"Prediction ensemble seed {seed_value}", flush=True)
        for label, mode in modes:
            trials.extend(
                run_policy_trials(
                    agents=agents,
                    simulation=simulation,
                    evaluation=evaluation,
                    communication=communication,
                    study_mpc=replace(study_mpc, obstacle_prediction_mode=mode),
                    disturbance=disturbance,
                    seed_result=seed.result,
                    policies=(
                        HarborPolicySpec(
                            label,
                            True,
                            "recursive_diagonal",
                            False,
                            False,
                            None,
                            "energy",
                            1,
                        ),
                    ),
                    observation_noise=case_noise,
                )
            )
        cases.append(
            HarborFaultEnsembleCase(
                seed=seed_value,
                disturbance=disturbance,
                observation_seed=case_noise.seed,
                trials=tuple(trials),
            )
        )
    return cases
