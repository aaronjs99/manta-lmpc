"""Experiments / manta / sanity."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from scripts.cli.arguments.manta.sanity import parse_args
from scripts.configuration.manta import list_config_scenarios, load_project_config
from scripts.metrics.core import compute_rollout_metrics, pairwise_distances
from scripts.plotting.trajectories import plot_pairwise_distances, plot_trajectories
from scripts.reporting.json import write_json
from scripts.simulation.environment import (
    MantaEnvConfig,
    MultiMantaRayEnv,
    manta_rollout,
)


def main(argv: list[str] | None = None) -> None:
    """Execute zero-control manta rollouts and save metrics/plots."""
    args = parse_args(argv)
    names = (
        list_config_scenarios(args.config)
        if args.scenario == "all"
        else [args.scenario]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (
        Path(args.output_dir)
        if args.output_dir
        else Path("results") / f"sanity_{timestamp}"
    )
    root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, float | int | None]] = {}

    for name in names:
        project_config = load_project_config(args.config, scenario_name=name)
        scenario = project_config.scenario
        dt = project_config.lmpc.dt if args.dt is None else args.dt
        horizon = (
            project_config.lmpc.max_steps if args.horizon is None else args.horizon
        )
        run_dir = root / name
        run_dir.mkdir(parents=True, exist_ok=True)

        env = MultiMantaRayEnv(
            starts=scenario.starts,
            goals=scenario.goals,
            config=MantaEnvConfig(
                dt=dt,
                horizon=horizon,
                dynamics=project_config.dynamics,
            ),
        )
        states, controls = manta_rollout(env, controls=None)

        dists = pairwise_distances(states)
        metrics = compute_rollout_metrics(
            states=states,
            goals=scenario.goals,
            safety_distance=scenario.safety_distance,
            dt=dt,
            controls=controls,
        )
        summary[name] = metrics.to_dict()

        plot_trajectories(
            states=states,
            goals=scenario.goals,
            title=f"{name}: trajectories (zero control)",
            out_path=str(run_dir / "trajectories.png"),
        )
        plot_pairwise_distances(
            distances=dists,
            dt=dt,
            safety_distance=scenario.safety_distance,
            title=f"{name}: pairwise distances (zero control)",
            out_path=str(run_dir / "pairwise_distances.png"),
        )

        write_json(run_dir / "metrics.json", metrics.to_dict(), trailing_newline=False)

    write_json(root / "summary.json", summary, trailing_newline=False)

    print(f"Saved sanity-check outputs to: {root}")
