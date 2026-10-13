"""Experiments / harbor / joint localization study."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from scripts.assembly.harbor import load_harbor_runtime as load_harbor_config
from scripts.cli.arguments.harbor.joint_localization_study import parse_args
from scripts.configuration.harbor_faults import (
    load_harbor_temporary_fault_ensemble_config,
    load_harbor_time_varying_fault_config,
)
from scripts.configuration.harbor_observation import (
    load_harbor_observation_noise_config,
)
from scripts.configuration.joint_localization import (
    _adaptive_config,
    _policy_controller_config,
)
from scripts.configuration.localization import load_range_aided_slam_config
from scripts.configuration.study_labels import POLICIES
from scripts.experiments.harbor.execution.ensembles import (
    generate_temporary_fault_ensemble,
)
from scripts.experiments.harbor.learning import run_distributed_harbor_lmpc
from scripts.metrics.harbor.joint_localization_study import (
    _current_rmse,
    _observability,
    _position_rmse,
    summarize,
)
from scripts.mpc.harbor.controller import DistributedHarborMPC
from scripts.plotting.harbor.joint_localization import _save_plot
from scripts.plotting.harbor.trajectories import save_harbor_animation
from scripts.reporting.json import write_json
from scripts.simulation.harbor.engine import run_harbor_simulation
from scripts.simulation.harbor.localization_modes import _localization


def main(argv: list[str] | None = None) -> None:
    """Run matched joint uncertainty cases without exposing plant truth."""
    args = parse_args(argv)
    agents, simulation, communication = load_harbor_config(args.config)
    controller_config = _adaptive_config(args.config)
    base_disturbance, _ = load_harbor_time_varying_fault_config(args.config)
    ensemble = load_harbor_temporary_fault_ensemble_config(
        args.config, "joint_localization_development"
    )
    generated = generate_temporary_fault_ensemble(agents, base_disturbance, ensemble)
    if args.case_seed is not None:
        generated = [case for case in generated if case[0] == args.case_seed]
        if not generated:
            raise ValueError("case-seed is not present in the configured ensemble")
    if args.case_limit is not None:
        if args.case_limit <= 0:
            raise ValueError("case-limit must be positive")
        generated = generated[: args.case_limit]
    seed_iterations = run_distributed_harbor_lmpc(
        agents,
        simulation,
        communication,
        replace(
            controller_config,
            learning_iterations=1,
            residual_adaptation=False,
            control_effectiveness_adaptation=False,
        ),
    )
    seed = min(
        (record for record in seed_iterations if record.admitted),
        key=lambda record: record.completion_step_sum,
    )
    observation_base = load_harbor_observation_noise_config(args.config)
    localization_base = load_range_aided_slam_config(args.config)
    records = []
    selected_policies = tuple(args.policy or POLICIES)
    gif_result = None
    for case_seed, observation_seed, disturbance in generated:
        evaluation = replace(
            simulation,
            guidance_update_interval_steps=controller_config.replan_interval_steps,
            goal_hold_steps=disturbance.evaluation_hold_steps,
        )
        for policy in selected_policies:
            localization = _localization(policy, localization_base, observation_seed)
            controller = DistributedHarborMPC(
                agents=agents,
                config=_policy_controller_config(controller_config, policy),
                dt=simulation.dt,
                safe_states=seed.result.states,
                safe_controls=seed.result.controls,
                learning=False,
            )
            result = run_harbor_simulation(
                agents,
                evaluation,
                communication,
                control_provider=controller,
                disturbance=disturbance,
                observation_noise=replace(
                    observation_base, enabled=True, seed=observation_seed
                ),
                localization_provider=localization,
            )
            record = {
                "seed": case_seed,
                "observation_seed": observation_seed,
                "policy": policy,
                "all_goals_reached": result.all_goals_reached,
                "pairwise_violation_count": result.pairwise_violation_count,
                "minimum_pairwise_distance": result.min_pairwise_distance,
                "completion_step_sum": sum(
                    step if step is not None else simulation.horizon + 1
                    for step in result.first_goal_steps.values()
                ),
                "final_goal_errors": result.final_goal_errors,
                "final_orientation_errors": result.final_orientation_errors,
                "solver_fallbacks": controller.telemetry.retry.fallback_count,
                "max_collision_slack": controller.max_collision_slack,
                "max_dynamic_state_slack": controller.max_dynamic_state_slack,
                "dynamic_state_retries": sum(
                    controller.telemetry.retry.dynamic_state_retry_count_by_agent.values()
                ),
                "position_rmse": _position_rmse(result, agents),
                "current_rmse": _current_rmse(controller, agents, disturbance),
                "observability": _observability(localization, agents),
            }
            records.append(record)
            if policy == "Joint SLAM + belief retry" and gif_result is None:
                gif_result = result
            print(
                f"case {case_seed} {policy}: complete={result.all_goals_reached}, "
                f"safe={result.pairwise_violation_count == 0}, "
                f"fallbacks={controller.telemetry.retry.fallback_count}",
                flush=True,
            )
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.make_gif or args.gif_only:
        if gif_result is None:
            raise ValueError("GIF generation requires the belief-retry policy")
        gif_path = output / "joint_localization_candidate.gif"
        save_harbor_animation(
            gif_result,
            agents,
            simulation,
            gif_path,
            label="Joint SLAM + belief-feasibility retry",
        )
        print(f"Saved joint localization animation: {gif_path}")
    if args.gif_only:
        return
    data_path = output / "joint_localization_development.json"
    plot_path = output / "joint_localization_development.png"
    payload = {
        "ensemble": asdict(ensemble),
        "case_limit": args.case_limit,
        "controller_uses_plant_truth": False,
        "summary": summarize(records),
        "records": records,
    }
    write_json(data_path, payload)
    _save_plot(records, plot_path)
    print(f"Saved joint localization data: {data_path}")
    print(f"Saved joint localization plot: {plot_path}")
