"""Reporting / manta benchmark."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = [
        "scenario",
        "iterations",
        "max_steps",
        "terminal_slack_weight",
        "elapsed_seconds",
        "selected_iteration",
        "selected_valid",
        "selected_safe",
        "selected_solver_clean",
        "selected_min_pairwise",
        "selected_min_obstacle_clearance",
        "selected_fallback_count",
        "selected_safety_interventions",
        "selected_max_static_slack",
        "selected_max_hyperplane_slack",
        "selected_max_terminal_slack",
        "selected_nonzero_static_slack_steps",
        "selected_nonzero_hyperplane_slack_steps",
        "selected_nonzero_terminal_slack_steps",
        "latest_max_static_slack",
        "latest_max_hyperplane_slack",
        "latest_max_terminal_slack",
        "latest_nonzero_static_slack_steps",
        "latest_nonzero_hyperplane_slack_steps",
        "latest_nonzero_terminal_slack_steps",
        "selected_slack_by_agent",
        "latest_slack_by_agent",
        "latest_valid",
        "latest_safe",
        "latest_solver_clean",
        "latest_fallback_count",
        "latest_safety_interventions",
        "selected_costs",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = {field: record[field] for field in fields}
            row["selected_costs"] = json.dumps(row["selected_costs"])
            row["selected_slack_by_agent"] = json.dumps(row["selected_slack_by_agent"])
            row["latest_slack_by_agent"] = json.dumps(row["latest_slack_by_agent"])
            writer.writerow(row)


def _print_table(records: list[dict[str, object]]) -> None:
    print(
        "scenario, seconds, selected, valid, safe, selected_clean, latest_clean, "
        "safety_interventions, "
        "selected_static_slack, latest_static_slack, selected_terminal_slack, "
        "latest_terminal_slack, min_pairwise, "
        "min_obs_clearance, selected_costs"
    )
    for record in records:
        print(
            f"{record['scenario']}, {_fmt(record['elapsed_seconds'])}, "
            f"{record['selected_iteration']}, "
            f"{record['selected_valid']}, {record['selected_safe']}, "
            f"{record['selected_solver_clean']}, "
            f"{record['latest_solver_clean']}, "
            f"{record['selected_safety_interventions']}, "
            f"{_fmt(record['selected_max_static_slack'])}, "
            f"{_fmt(record['latest_max_static_slack'])}, "
            f"{_fmt(record['selected_max_terminal_slack'])}, "
            f"{_fmt(record['latest_max_terminal_slack'])}, "
            f"{_fmt(record['selected_min_pairwise'])}, "
            f"{_fmt(record['selected_min_obstacle_clearance'])}, "
            f"{record['selected_costs']}"
        )


def _fmt(value: object) -> str:
    return "None" if value is None else f"{float(value):.3f}"
