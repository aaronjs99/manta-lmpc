"""Experiments / manta / lmpc."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.cli.arguments.manta.lmpc import parse_args
from scripts.configuration.manta import load_project_config, override_project_config
from scripts.experiments.manta.runner import run_manta_lmpc
from scripts.reporting.manta import save_manta_run_report

if TYPE_CHECKING:
    import argparse

    from scripts.configuration.manta import ProjectConfig


def main(argv: list[str] | None = None) -> None:
    """Run APF/LMPC from YAML config and save all diagnostics."""
    args = parse_args(argv)
    project_config = _load_effective_config(args)
    scenario = project_config.scenario
    lmpc_config = project_config.lmpc
    apf_config = project_config.apf

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (
        Path(args.output_dir)
        if args.output_dir
        else project_config.output.root_dir
        / f"{project_config.output.run_prefix}_{timestamp}"
    )
    root.mkdir(parents=True, exist_ok=True)
    stop_path = _resolve_stop_file(args.stop_file)
    if stop_path.exists():
        stop_path.unlink()

    try:
        result = run_manta_lmpc(
            scenario,
            config=lmpc_config,
            apf_config=apf_config,
            dynamics_config=project_config.dynamics,
            should_stop=stop_path.exists,
            verbose=not project_config.quiet,
        )
    except KeyboardInterrupt:
        if stop_path.exists():
            stop_path.unlink()
        print("Stopped manta LMPC run.")
        raise SystemExit(130) from None

    save_manta_run_report(
        root,
        result,
        project_config,
        config_path=args.config,
    )
    print(f"Saved manta LMPC outputs to: {root}")


def _load_effective_config(args: argparse.Namespace) -> ProjectConfig:
    """Load YAML config, then apply explicit CLI overrides."""
    config = load_project_config(args.config, scenario_name=args.scenario)
    return override_project_config(
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


def _resolve_stop_file(path: str | Path) -> Path:
    stop_path = Path(path)
    if not stop_path.is_absolute():
        stop_path = Path.cwd() / stop_path
    return stop_path
