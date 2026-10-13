"""Experiments / harbor / temporary fault generalization."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from scripts.assembly.harbor import load_harbor_runtime as load_harbor_config
from scripts.cli.arguments.harbor.temporary_fault_generalization import parse_args
from scripts.configuration.harbor_temporary_fault_criteria import (
    load_harbor_confirmation_criteria_config,
    load_harbor_recovery_confirmation_criteria_config,
)
from scripts.configuration.harbor_faults import (
    load_harbor_fault_study_config,
    load_harbor_temporary_fault_ensemble_config,
    load_harbor_time_varying_fault_config,
)
from scripts.configuration.harbor_mpc import load_harbor_mpc_config
from scripts.configuration.harbor_observation import (
    load_harbor_observation_noise_config,
)
from scripts.experiments.harbor.execution.temporary_faults import (
    run_temporary_fault_generalization,
)
from scripts.metrics.harbor.temporary_fault_generalization import (
    evaluate_confirmation,
    evaluate_recovery_confirmation,
    summarize_recovery_prior,
)
from scripts.metrics.harbor.time_varying_fault import summarize_time_varying_faults
from scripts.metrics.harbor.time_varying_fault_records import (
    build_time_varying_fault_records,
)
from scripts.plotting.harbor.recovery import save_temporary_fault_generalization_plot
from scripts.reporting.json import write_json


def main(argv: list[str] | None = None) -> None:
    """Run the stratified ensemble and replace its compact artifacts."""
    args = parse_args(argv)
    transient_mode = (
        args.transient_recovery_development or args.transient_recovery_confirmation
    )
    recovery_confirmation_mode = (
        args.recovery_confirmation or args.transient_recovery_confirmation
    )
    agents, simulation, communication = load_harbor_config(args.config)
    base_disturbance, experiment = load_harbor_time_varying_fault_config(args.config)
    if transient_mode:
        if args.transient_recovery_confirmation:
            ensemble_section = "temporary_fault_transient_recovery_confirmation"
            evaluation_role = "transient_recovery_confirmation"
            artifact_stem = "temporary_fault_transient_recovery_confirmation"
            recovery_criteria = load_harbor_recovery_confirmation_criteria_config(
                args.config,
                section=("temporary_fault_transient_recovery_confirmation_criteria"),
            )
            controller_labels = recovery_criteria.controller_labels
        else:
            ensemble_section = "temporary_fault_transient_recovery_development"
            evaluation_role = "transient_recovery_development"
            artifact_stem = "temporary_fault_transient_recovery_development"
            controller_labels = (
                "Fixed-covariance RLS",
                "Innovation-threshold RLS",
                "Transient-offset threshold RLS",
            )
            recovery_criteria = None
        criteria = None
    elif args.recovery_development or args.recovery_confirmation:
        if args.recovery_confirmation:
            ensemble_section = "temporary_fault_recovery_confirmation"
            evaluation_role = "recovery_confirmation"
            artifact_stem = "temporary_fault_recovery_confirmation"
            recovery_criteria = load_harbor_recovery_confirmation_criteria_config(
                args.config
            )
        else:
            ensemble_section = "temporary_fault_recovery_development"
            evaluation_role = "recovery_development"
            artifact_stem = "temporary_fault_recovery_development"
            recovery_criteria = None
        controller_labels = (
            "Fixed-covariance RLS",
            "Innovation-threshold RLS",
            "Recovery-prior threshold RLS",
        )
        criteria = None
    elif args.confirmation:
        ensemble_section = "temporary_fault_confirmation"
        evaluation_role = "confirmation"
        artifact_stem = "temporary_fault_confirmation"
        criteria = load_harbor_confirmation_criteria_config(args.config)
        controller_labels = criteria.controller_labels
        recovery_criteria = None
    elif args.holdout:
        ensemble_section = "temporary_fault_holdout"
        evaluation_role = "holdout"
        artifact_stem = "temporary_fault_holdout"
        controller_labels = None
        criteria = None
        recovery_criteria = None
    else:
        ensemble_section = "temporary_fault_ensemble"
        evaluation_role = "development"
        artifact_stem = "temporary_fault_generalization"
        controller_labels = None
        criteria = None
        recovery_criteria = None
    ensemble = load_harbor_temporary_fault_ensemble_config(
        args.config, section=ensemble_section
    )
    mpc_config = load_harbor_mpc_config(args.config)
    cases = run_temporary_fault_generalization(
        agents,
        simulation,
        communication,
        mpc_config,
        base_disturbance,
        load_harbor_fault_study_config(args.config),
        experiment,
        ensemble,
        load_harbor_observation_noise_config(args.config),
        controller_labels=controller_labels,
    )
    records = build_time_varying_fault_records(
        cases, agents, simulation, experiment.event_detection_window_steps
    )
    summary = summarize_time_varying_faults(records, ensemble.bootstrap_samples)
    if transient_mode:
        recovery_comparison = summarize_recovery_prior(
            records,
            ensemble.bootstrap_samples,
            candidate_label="Transient-offset threshold RLS",
        )
    elif args.recovery_development or args.recovery_confirmation:
        recovery_comparison = summarize_recovery_prior(
            records, ensemble.bootstrap_samples
        )
    else:
        recovery_comparison = None
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / f"{artifact_stem}.json"
    figure_path = output / f"{artifact_stem}.png"
    payload = {
        "ensemble": asdict(ensemble),
        "experiment": {
            "evaluation_role": evaluation_role,
            "change_threshold": experiment.change_threshold,
            "covariance_inflation": experiment.covariance_inflation,
            "cusum_drift": mpc_config.effectiveness_rls_cusum_drift,
            "cusum_threshold": mpc_config.effectiveness_rls_cusum_threshold,
            "event_detection_window_steps": experiment.event_detection_window_steps,
            "recovery_prior_gain": experiment.recovery_prior_gain,
            "recovery_offset_decay": mpc_config.effectiveness_recovery_offset_decay,
            "recovery_minimum_dwell_steps": (
                mpc_config.effectiveness_recovery_minimum_dwell_steps
            ),
            "recovery_require_full_rank": (
                mpc_config.effectiveness_recovery_require_full_rank
            ),
            "recovery_rank_tolerance": (
                mpc_config.effectiveness_recovery_rank_tolerance
            ),
            "recovery_episode_hysteresis": True,
            "recovery_max_episodes_per_agent": (
                mpc_config.effectiveness_recovery_max_episodes_per_agent
            ),
            "controller_blind_to_hidden_schedule": True,
            "deployment_candidate": (
                "Transient-offset threshold RLS"
                if transient_mode
                else (
                    "Recovery-prior threshold RLS"
                    if args.recovery_development or args.recovery_confirmation
                    else "Innovation-threshold RLS"
                )
            ),
            "candidate_status": (
                "confirmation under evaluation"
                if args.confirmation
                else "provisional pending a fresh confirmation ensemble"
            ),
            "triggered_probing_role": "ablation",
        },
        "summary": summary,
        "trials": records,
    }
    if recovery_comparison is not None:
        comparison_key = (
            "transient_recovery_vs_threshold"
            if transient_mode
            else "recovery_prior_vs_threshold"
        )
        payload[comparison_key] = recovery_comparison
        payload["experiment"]["candidate_status"] = (
            "transient recovery offset under development"
            if transient_mode
            else "recovery prior under development"
        )
    if recovery_confirmation_mode:
        assert recovery_comparison is not None and recovery_criteria is not None
        recovery_confirmation = evaluate_recovery_confirmation(
            recovery_comparison, recovery_criteria
        )
        result_prefix = (
            "transient_recovery_confirmation"
            if args.transient_recovery_confirmation
            else "recovery_confirmation"
        )
        payload[f"{result_prefix}_criteria"] = asdict(recovery_criteria)
        payload[f"{result_prefix}_result"] = recovery_confirmation
        method_name = (
            "transient recovery offset"
            if args.transient_recovery_confirmation
            else "recovery prior"
        )
        payload["experiment"]["candidate_status"] = (
            f"{method_name} confirmed in simulation"
            if recovery_confirmation["passed"]
            else f"{method_name} confirmation criteria not met"
        )
        payload["experiment"]["deployment_candidate"] = (
            (
                "Transient-offset threshold RLS"
                if args.transient_recovery_confirmation
                else "Recovery-prior threshold RLS"
            )
            if recovery_confirmation["passed"]
            else "Innovation-threshold RLS"
        )
    if args.confirmation:
        assert criteria is not None
        confirmation = evaluate_confirmation(summary, criteria)
        payload["confirmation_criteria"] = asdict(criteria)
        payload["confirmation_result"] = confirmation
        payload["experiment"]["candidate_status"] = (
            "confirmed in simulation"
            if confirmation["passed"]
            else "confirmation criteria not met"
        )
    write_json(data_path, payload)
    save_temporary_fault_generalization_plot(
        records,
        summary,
        figure_path,
        study_title=(
            (
                "Independent Transient Recovery-Offset Confirmation"
                if args.transient_recovery_confirmation
                else "Transient Recovery-Offset Development"
            )
            if transient_mode
            else (
                "Channel-Selective Recovery-Prior Development"
                if args.recovery_development
                else (
                    "Independent Recovery-Prior Confirmation"
                    if recovery_confirmation_mode
                    else (
                        "Independent Threshold-RLS Confirmation"
                        if args.confirmation
                        else "Generalization Across Hidden Temporary Actuator Faults"
                    )
                )
            )
        ),
        headline=(
            (
                f"{'transient-offset' if transient_mode else 'recovery-prior'} "
                f"wins {recovery_comparison['recovery_wins']}/"
                f"{recovery_comparison['trials']}; mean recovery RMSE reduction "
                f"{100.0 * recovery_comparison['mean_relative_recovery_rmse_reduction']:.1f}%; "
                f"fallback-free {100.0 * recovery_comparison['fallback_free_rate']:.0f}%"
                + (
                    f"; overall gate "
                    f"{'PASSED' if recovery_confirmation['passed'] else 'FAILED'}"
                    if recovery_confirmation_mode
                    else ""
                )
            )
            if recovery_comparison is not None
            else None
        ),
    )
    print(json.dumps(summary, indent=2))
    if args.confirmation:
        print(json.dumps(payload["confirmation_result"], indent=2))
    if recovery_confirmation_mode:
        result_key = (
            "transient_recovery_confirmation_result"
            if args.transient_recovery_confirmation
            else "recovery_confirmation_result"
        )
        print(json.dumps(payload[result_key], indent=2))
    print(f"Saved temporary-fault generalization data: {data_path}")
    print(f"Saved temporary-fault generalization plot: {figure_path}")
