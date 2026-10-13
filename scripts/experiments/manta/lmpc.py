"""Run, checkpoint, resume, and report the Manta APF/LMPC workflow."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import yaml

from scripts.cli.arguments.manta.lmpc import parse_args
from scripts.configuration.manta import (
    load_project_config,
    override_project_config,
    project_config_to_mapping,
)
from scripts.configuration.paths import DEFAULT_CONFIG_PATH, resolve_config_path
from scripts.experiments.manta.run_directory import claim_run_directory
from scripts.experiments.manta.runner import run_manta_lmpc
from scripts.reporting.manta import save_manta_progress_summary, save_manta_run_report
from scripts.storage.manta import load_manta_checkpoint, save_manta_checkpoint

if TYPE_CHECKING:
    import argparse

    from scripts.configuration.manta import ProjectConfig


def main(argv: list[str] | None = None) -> None:
    """Run simulations locally; user input errors have actionable CLI messages."""
    args = parse_args(argv)
    try:
        _execute(args)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(f"manta: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


def _execute(args: argparse.Namespace) -> None:
    resume_path = None
    checkpoint = None
    if args.resume:
        source = Path(args.resume).resolve()
        resume_path = source / "checkpoint.npz" if source.is_dir() else source
        checkpoint = load_manta_checkpoint(resume_path)
    config = _load_effective_config(args, checkpoint.config if checkpoint else None)
    if checkpoint and config.lmpc.iterations < checkpoint.result.completed_iterations:
        raise ValueError(
            "--iterations cannot be smaller than the completed iteration count"
        )
    root = (
        Path(args.output_dir)
        if args.output_dir
        else resume_path.parent
        if resume_path
        else config.output.root_dir
        / f"{config.output.run_prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    ).resolve()
    config_path = (
        resolve_config_path(args.config or DEFAULT_CONFIG_PATH)
        if args.config or checkpoint is None
        else resume_path
    )
    stop_path = Path(args.stop_file).resolve() if args.stop_file else root / "STOP_RUN"
    checkpoint_path = root / "checkpoint.npz"
    last_completed = checkpoint.result if checkpoint else None

    with claim_run_directory(root, resume_path=resume_path):
        # The initial read locates the destination. Re-read under its lock so
        # an in-place writer finishing between those operations cannot be lost.
        if checkpoint is not None and root == resume_path.parent:
            checkpoint = load_manta_checkpoint(resume_path)
            config = _load_effective_config(args, checkpoint.config)
            if config.lmpc.iterations < checkpoint.result.completed_iterations:
                raise ValueError(
                    "--iterations cannot be smaller than the completed iteration count"
                )
            last_completed = checkpoint.result

        def completed(result):
            nonlocal last_completed
            save_manta_checkpoint(checkpoint_path, result, config)
            last_completed = result
            save_manta_progress_summary(
                root, result, config, config_path=config_path, run_status="running"
            )
            if not config.quiet:
                print(
                    f"Saved checkpoint after iteration {result.completed_iterations}: {checkpoint_path}"
                )

        if not config.quiet:
            print(f"Run directory: {root}\nStop signal: {stop_path}")
        try:
            if checkpoint is not None:
                # Persist a changed target and an imported seed before the
                # first new solve, including resumes into another directory.
                completed(last_completed)
            result = run_manta_lmpc(
                config.scenario,
                config=config.lmpc,
                apf_config=config.apf,
                dynamics_config=config.dynamics,
                should_stop=stop_path.exists,
                resume_result=last_completed,
                on_iteration_completed=completed,
                verbose=not config.quiet,
            )
            # Also handles a resume with no new iterations or a new destination.
            save_manta_checkpoint(checkpoint_path, result, config)
            last_completed = result
            save_manta_progress_summary(
                root, result, config, config_path=config_path, run_status="reporting"
            )
            save_manta_run_report(root, result, config, config_path=config_path)
        except KeyboardInterrupt:
            _record_stopped_run(
                root, last_completed, config, config_path, "interrupted"
            )
            print("Stopped manta LMPC run.")
            if last_completed is not None and checkpoint_path.exists():
                print(
                    f"Retained iterations 0 through {last_completed.completed_iterations}. "
                    f'Resume with: python run.py --resume "{root}"'
                )
            else:
                print("No completed APF seed is available yet; rerun to start again.")
            raise SystemExit(130) from None
        except Exception:
            _record_stopped_run(root, last_completed, config, config_path, "failed")
            raise

    validation = result.validation_by_iteration[
        result.selected_iteration if result.selected_iteration is not None else -1
    ]
    print(f"Saved manta LMPC outputs to: {root}")
    print(
        f"Completed LMPC iterations: {result.completed_iterations}/{config.lmpc.iterations}; "
        f"selected iteration: {result.selected_iteration}; "
        f"valid trajectory: {validation.valid}; solver clean: {validation.solver_clean}"
    )


def _record_stopped_run(root, result, config, config_path, status):
    if result is not None and (root / "checkpoint.npz").exists():
        try:
            save_manta_progress_summary(
                root, result, config, config_path=config_path, run_status=status
            )
        except (OSError, ValueError) as exc:
            print(
                f"Could not update progress summary: {exc}. "
                "The previous checkpoint is retained.",
                file=sys.stderr,
            )


def _load_effective_config(
    args: argparse.Namespace,
    stored: ProjectConfig | None = None,
) -> ProjectConfig:
    """Resume from saved settings; numerical changes require a separate run."""
    config = (
        stored
        if stored is not None and args.config is None
        else load_project_config(
            args.config or DEFAULT_CONFIG_PATH,
            scenario_name=args.scenario or (stored.scenario.name if stored else None),
        )
    )
    if (
        stored is not None
        and args.scenario is not None
        and args.scenario != stored.scenario.name
    ):
        raise ValueError(
            "--scenario differs from the checkpoint; start a new run instead"
        )
    config = override_project_config(
        config,
        lmpc={
            "iterations": args.iterations,
            "max_steps": args.max_steps,
            "dt": args.dt,
            "prediction_horizon": args.mpc_horizon,
            "k_hull": args.k_hull,
            "goal_tolerance": args.goal_tolerance,
        },
        apf={"max_steps": args.apf_max_steps},
        make_video=args.make_video,
        quiet=args.quiet,
    )
    if stored is not None:
        previous = project_config_to_mapping(stored)
        current = project_config_to_mapping(config)
        changed = []
        for section in ("scenario", "dynamics", "apf", "lmpc"):
            old_values, new_values = previous[section].copy(), current[section].copy()
            if section == "lmpc":
                old_values.pop("iterations")
                new_values.pop("iterations")
            if json.dumps(old_values, sort_keys=True) != json.dumps(
                new_values, sort_keys=True
            ):
                changed.append(section)
        if changed:
            raise ValueError(
                f"resume settings differ in {', '.join(changed)}. "
                "Only the iteration target, output, verbosity, and video may change; "
                "start a new run for numerical changes."
            )
    return config
