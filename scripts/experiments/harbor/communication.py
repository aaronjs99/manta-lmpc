"""Experiments / harbor / communication."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from scripts.assembly.harbor import load_harbor_runtime as load_harbor_config
from scripts.cli.arguments.harbor.communication import parse_args
from scripts.plotting.harbor.trajectories import (
    save_harbor_animation,
    save_harbor_comparison,
)
from scripts.reporting.json import write_json
from scripts.simulation.harbor.engine import run_harbor_simulation


def main(argv: list[str] | None = None) -> None:
    """Run one or both communication modes and print compact metrics."""
    args = parse_args(argv)
    agents, simulation, communication = load_harbor_config(args.config)
    if args.policy is not None:
        simulation = replace(simulation, coordination_policy=args.policy)
    modes = ("independent", "communication") if args.mode == "compare" else (args.mode,)
    records = []
    results = {}
    for mode in modes:
        link = replace(communication, enabled=mode == "communication")
        result = run_harbor_simulation(agents, simulation, link)
        label = mode
        if mode == "communication":
            label = simulation.coordination_policy
        results[label] = result
        records.append(
            {
                "mode": mode,
                "coordination_policy": (
                    simulation.coordination_policy if mode == "communication" else None
                ),
                "all_goals_reached": result.all_goals_reached,
                "first_goal_steps": result.first_goal_steps,
                "completion_step_sum": sum(
                    step if step is not None else simulation.horizon + 1
                    for step in result.first_goal_steps.values()
                ),
                "makespan": max(
                    step if step is not None else simulation.horizon + 1
                    for step in result.first_goal_steps.values()
                ),
                "final_goal_errors": result.final_goal_errors,
                "final_orientation_errors": result.final_orientation_errors,
                "min_pairwise_distance": result.min_pairwise_distance,
                "pairwise_violation_count": result.pairwise_violation_count,
                "messages_sent": result.messages_sent,
                "messages_delivered": result.messages_delivered,
                "messages_dropped": result.messages_dropped,
                "guidance_update_count": result.guidance_update_count,
            }
        )
    text = json.dumps(records, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, records)
    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        comparison = save_harbor_comparison(
            results, agents, simulation, plot_dir / "harbor_comparison.png"
        )
        coordinated = next(
            (result for name, result in results.items() if name != "independent"),
            None,
        )
        print(f"Saved comparison plot: {comparison}")
        if coordinated is not None:
            animation = save_harbor_animation(
                coordinated,
                agents,
                simulation,
                plot_dir / "harbor_coordination.gif",
            )
            print(f"Saved coordination animation: {animation}")
