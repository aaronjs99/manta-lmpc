"""Execution of one Harbor policy and ordered matched trial batches."""

from __future__ import annotations

from dataclasses import replace

from scripts.contracts.harbor import HarborRobustnessTrial
from scripts.experiments.harbor.execution.policy import HarborPolicySpec
from scripts.experiments.harbor.execution.scenario import HarborTrialScenario
from scripts.experiments.harbor.execution.trial_result import assemble_trial_result
from scripts.mpc.harbor.controller import DistributedHarborMPC
from scripts.simulation.harbor.engine import run_harbor_simulation


def execute_policy_trial(
    scenario: HarborTrialScenario,
    policy: HarborPolicySpec,
    completed_trials: tuple[HarborRobustnessTrial, ...] = (),
) -> HarborRobustnessTrial:
    """Execute exactly one policy using the declared safe/source trajectory."""
    source = None
    if policy.source_controller is not None:
        source = next(
            trial
            for trial in completed_trials
            if trial.label == policy.source_controller
        )
    safe_result = source.result if source is not None else scenario.seed_result
    config = replace(
        scenario.mpc,
        **dict(policy.mpc_overrides),
        residual_adaptation=scenario.residual_adaptation,
        residual_adaptation_kinds=(
            scenario.mpc.residual_adaptation_kinds
            if scenario.residual_adaptation_kinds is None
            else scenario.residual_adaptation_kinds
        ),
        control_effectiveness_adaptation=policy.adaptive_effectiveness,
        effectiveness_estimator_mode=policy.estimator_mode,
        active_identification=policy.active_identification,
        identification_strategy=policy.identification_strategy,
        identification_min_probes_per_channel=policy.minimum_probes,
    )
    controller = DistributedHarborMPC(
        agents=scenario.agents,
        config=config,
        dt=scenario.simulation.dt,
        safe_states=safe_result.states,
        safe_controls=safe_result.controls,
        learning=policy.learning,
        initial_effectiveness_estimates=(
            source.final_effectiveness_estimates if source is not None else None
        ),
    )
    result = run_harbor_simulation(
        scenario.agents,
        scenario.evaluation,
        scenario.communication,
        control_provider=controller,
        disturbance=scenario.disturbance,
        observation_noise=scenario.observation_noise,
    )
    trial = assemble_trial_result(
        label=policy.label,
        source_controller=policy.source_controller,
        result=result,
        controller=controller,
        horizon=scenario.simulation.horizon,
    )
    telemetry = controller.telemetry
    print(
        f"{policy.label}: complete={result.all_goals_reached}, "
        f"safe={result.pairwise_violation_count == 0}, "
        f"cost={trial.completion_step_sum}, fallbacks={telemetry.retry.fallback_count}",
        flush=True,
    )
    return trial


def execute_policy_batch(
    scenario: HarborTrialScenario,
    policies: tuple[HarborPolicySpec, ...],
) -> list[HarborRobustnessTrial]:
    """Run policies in declared order; source policies must precede dependents."""
    trials: list[HarborRobustnessTrial] = []
    for policy in policies:
        trials.append(execute_policy_trial(scenario, policy, tuple(trials)))
    return trials
