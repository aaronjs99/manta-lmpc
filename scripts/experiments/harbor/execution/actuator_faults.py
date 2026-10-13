"""Per-channel actuator-fault studies and hidden-fault generalization."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from scripts.contracts.harbor import HarborFaultEnsembleCase, HarborRobustnessTrial
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


def run_actuator_fault_study(
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    communication: LinkConfig,
    mpc_config: HarborMPCConfig,
    disturbance: HarborDisturbanceConfig,
    study_config: HarborFaultStudyConfig,
) -> list[HarborRobustnessTrial]:
    """Compare scalar and diagonal local estimators under asymmetric faults."""
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
            learning_iterations=1,
            residual_adaptation=False,
            control_effectiveness_adaptation=False,
        ),
    )
    seed = min(
        (record for record in seed_iterations if record.admitted),
        key=lambda record: record.completion_step_sum,
    )
    evaluation = replace(
        simulation,
        guidance_update_interval_steps=study_mpc.replan_interval_steps,
        goal_hold_steps=disturbance.evaluation_hold_steps,
    )
    policies = (
        HarborPolicySpec(
            "Nominal MPC", False, "scalar", False, False, None, "energy", 2
        ),
        HarborPolicySpec(
            "Scalar-adaptive MPC", True, "scalar", False, False, None, "energy", 2
        ),
        HarborPolicySpec(
            "Diagonal-adaptive MPC", True, "diagonal", False, False, None, "energy", 2
        ),
        HarborPolicySpec(
            "Active diagonal MPC", True, "diagonal", False, True, None, "energy", 2
        ),
        HarborPolicySpec(
            "One-pass active MPC", True, "diagonal", False, True, None, "energy", 1
        ),
        HarborPolicySpec(
            "Information-aware MPC",
            True,
            "diagonal",
            False,
            True,
            None,
            "information",
            1,
        ),
        HarborPolicySpec(
            "Retained diagonal LMPC",
            False,
            "diagonal",
            True,
            False,
            "Diagonal-adaptive MPC",
            "energy",
            2,
        ),
        HarborPolicySpec(
            "Retained active-ID LMPC",
            False,
            "diagonal",
            True,
            False,
            "Active diagonal MPC",
            "energy",
            2,
        ),
        HarborPolicySpec(
            "Retained information-ID LMPC",
            False,
            "diagonal",
            True,
            False,
            "Information-aware MPC",
            "information",
            1,
        ),
    )
    return run_policy_trials(
        agents=agents,
        simulation=simulation,
        evaluation=evaluation,
        communication=communication,
        study_mpc=study_mpc,
        disturbance=disturbance,
        seed_result=seed.result,
        policies=policies,
    )


def run_actuator_fault_generalization(
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    communication: LinkConfig,
    mpc_config: HarborMPCConfig,
    base_disturbance: HarborDisturbanceConfig,
    study_config: HarborFaultStudyConfig,
    ensemble_config: HarborFaultEnsembleConfig,
    observation_noise: HarborObservationNoiseConfig | None = None,
) -> list[HarborFaultEnsembleCase]:
    """Compare equal-budget probe policies over stratified hidden faults."""
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
            learning_iterations=1,
            residual_adaptation=False,
            control_effectiveness_adaptation=False,
        ),
    )
    seed = min(
        (record for record in seed_iterations if record.admitted),
        key=lambda record: record.completion_step_sum,
    )
    if observation_noise is not None and observation_noise.enabled:
        policies = (
            HarborPolicySpec(
                "Instantaneous diagonal MPC",
                True,
                "diagonal",
                False,
                False,
                None,
                "energy",
                1,
            ),
            HarborPolicySpec(
                "Recursive diagonal MPC",
                True,
                "recursive_diagonal",
                False,
                False,
                None,
                "energy",
                1,
            ),
            HarborPolicySpec(
                "Recursive one-pass MPC",
                True,
                "recursive_diagonal",
                False,
                True,
                None,
                "energy",
                1,
            ),
            HarborPolicySpec(
                "Recursive information-aware MPC",
                True,
                "recursive_diagonal",
                False,
                True,
                None,
                "information",
                1,
            ),
        )
    else:
        policies = (
            HarborPolicySpec(
                "Passive diagonal MPC",
                True,
                "diagonal",
                False,
                False,
                None,
                "energy",
                1,
            ),
            HarborPolicySpec(
                "One-pass active MPC",
                True,
                "diagonal",
                False,
                True,
                None,
                "energy",
                1,
            ),
            HarborPolicySpec(
                "Information-aware MPC",
                True,
                "diagonal",
                False,
                True,
                None,
                "information",
                1,
            ),
        )
    cases = []
    for seed_value, disturbance in generate_fault_ensemble(
        agents, base_disturbance, ensemble_config
    ):
        case_noise = (
            replace(observation_noise, seed=observation_noise.seed + seed_value)
            if observation_noise is not None and observation_noise.enabled
            else None
        )
        evaluation = replace(
            simulation,
            guidance_update_interval_steps=study_mpc.replan_interval_steps,
            goal_hold_steps=disturbance.evaluation_hold_steps,
        )
        print(f"Fault ensemble seed {seed_value}", flush=True)
        trials = run_policy_trials(
            agents=agents,
            simulation=simulation,
            evaluation=evaluation,
            communication=communication,
            study_mpc=study_mpc,
            disturbance=disturbance,
            seed_result=seed.result,
            policies=policies,
            observation_noise=case_noise,
        )
        cases.append(
            HarborFaultEnsembleCase(
                seed=seed_value,
                disturbance=disturbance,
                observation_seed=(case_noise.seed if case_noise is not None else None),
                trials=tuple(trials),
            )
        )
    return cases
