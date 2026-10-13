"""Scheduled and ensemble temporary-actuator-fault study execution."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from scripts.contracts.harbor import HarborFaultEnsembleCase
from scripts.experiments.harbor.execution.ensembles import (
    generate_temporary_fault_ensemble,
)
from scripts.experiments.harbor.execution.trials import run_temporary_fault_trials
from scripts.experiments.harbor.learning import run_distributed_harbor_lmpc

if TYPE_CHECKING:
    from scripts.communication.harbor import LinkConfig
    from scripts.configuration.harbor_faults import (
        HarborFaultStudyConfig,
        HarborTemporaryFaultEnsembleConfig,
        HarborTimeVaryingFaultConfig,
    )
    from scripts.configuration.harbor_mpc import HarborMPCConfig
    from scripts.configuration.harbor_simulation import (
        HarborDisturbanceConfig,
        HarborObservationNoiseConfig,
        HarborSimulationConfig,
    )
    from scripts.contracts.harbor import HarborAgent


def run_time_varying_fault_study(
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    communication: LinkConfig,
    mpc_config: HarborMPCConfig,
    disturbance: HarborDisturbanceConfig,
    study_config: HarborFaultStudyConfig,
    experiment_config: HarborTimeVaryingFaultConfig,
    observation_noise: HarborObservationNoiseConfig,
) -> list[HarborFaultEnsembleCase]:
    """Compare fixed, threshold-adaptive, and CUSUM-adaptive scheduled-fault RLS."""
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
    cases = []
    for observation_seed in experiment_config.observation_seeds:
        noise = replace(observation_noise, enabled=True, seed=observation_seed)
        print(f"Scheduled-fault observation seed {observation_seed}", flush=True)
        trials = run_temporary_fault_trials(
            agents=agents,
            simulation=simulation,
            evaluation=evaluation,
            communication=communication,
            study_mpc=study_mpc,
            disturbance=disturbance,
            seed_result=seed.result,
            experiment_config=experiment_config,
            observation_noise=noise,
        )
        cases.append(
            HarborFaultEnsembleCase(
                seed=observation_seed,
                disturbance=disturbance,
                observation_seed=observation_seed,
                trials=tuple(trials),
            )
        )
    return cases


def run_temporary_fault_generalization(
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    communication: LinkConfig,
    mpc_config: HarborMPCConfig,
    base_disturbance: HarborDisturbanceConfig,
    study_config: HarborFaultStudyConfig,
    experiment_config: HarborTimeVaryingFaultConfig,
    ensemble_config: HarborTemporaryFaultEnsembleConfig,
    observation_noise: HarborObservationNoiseConfig,
    controller_labels: tuple[str, ...] | None = None,
    residual_adaptation: bool = False,
    residual_adaptation_kinds: tuple[str, ...] | None = None,
) -> list[HarborFaultEnsembleCase]:
    """Evaluate temporary-fault adaptation across hidden severities and timings."""
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
    cases = []
    generated = generate_temporary_fault_ensemble(
        agents, base_disturbance, ensemble_config
    )
    for case_seed, observation_seed, disturbance in generated:
        noise = replace(observation_noise, enabled=True, seed=observation_seed)
        evaluation = replace(
            simulation,
            guidance_update_interval_steps=study_mpc.replan_interval_steps,
            goal_hold_steps=disturbance.evaluation_hold_steps,
        )
        print(
            f"Temporary-fault case {case_seed}; observation seed {observation_seed}",
            flush=True,
        )
        trials = run_temporary_fault_trials(
            agents=agents,
            simulation=simulation,
            evaluation=evaluation,
            communication=communication,
            study_mpc=study_mpc,
            disturbance=disturbance,
            seed_result=seed.result,
            experiment_config=experiment_config,
            observation_noise=noise,
            controller_labels=controller_labels,
            residual_adaptation=residual_adaptation,
            residual_adaptation_kinds=residual_adaptation_kinds,
        )
        cases.append(
            HarborFaultEnsembleCase(
                seed=case_seed,
                disturbance=disturbance,
                observation_seed=observation_seed,
                trials=tuple(trials),
            )
        )
    return cases
