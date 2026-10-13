"""Save complete Manta learning histories in an atomic, non-pickle archive."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile
import zlib

import numpy as np

from scripts.configuration.manta import (
    ProjectConfig,
    project_config_from_mapping,
    project_config_to_mapping,
)
from scripts.contracts.manta import MantaLMPCRunResult
from scripts.configuration.manta_learning import manta_learning_settings
from scripts.metrics.manta.learning import revalidate_result

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MantaCheckpoint:
    """Effective configuration and revalidated data for completed iterations."""

    config: ProjectConfig
    result: MantaLMPCRunResult


def save_manta_checkpoint(
    path: str | Path,
    result: MantaLMPCRunResult,
    config: ProjectConfig,
) -> Path:
    """Atomically replace ``path`` after validating a complete run snapshot.

    Every history is retained, including completed attempts that were not
    admitted for learning. Derived verdicts are not stored as trusted data.
    """
    try:
        settings = manta_learning_settings(
            config.scenario, config.lmpc, config.apf, config.dynamics
        )
        if (
            result.learning_settings is not None
            and result.learning_settings != settings
        ):
            raise ValueError(
                "result numerical settings do not match the checkpoint configuration"
            )
        # Revalidation changes derived verdict fields, not retained arrays.
        # A shallow dataclass copy leaves the caller's result unchanged.
        snapshot = replace(result)
        revalidate_result(snapshot, config.scenario, config.lmpc)
        arrays: dict[str, np.ndarray] = {}
        for iteration, history in enumerate(snapshot.histories):
            for agent, states in history.items():
                name = f"history_{iteration}_{agent}"
                arrays[name] = _numeric_array(states, name)
        for iteration, controls in enumerate(snapshot.controls_by_iteration, 1):
            name = f"controls_{iteration}"
            arrays[name] = _numeric_array(controls, name)
        for iteration, slacks in enumerate(snapshot.slack_by_iteration, 1):
            name = f"slack_{iteration}"
            arrays[name] = _numeric_array(slacks, name)
        assert snapshot.initial_controls is not None  # checked by revalidation
        for agent, controls in snapshot.initial_controls.items():
            name = f"initial_controls_{agent}"
            arrays[name] = _numeric_array(controls, name)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "config": project_config_to_mapping(config),
            "completed_iterations": snapshot.completed_iterations,
            "statuses": snapshot.statuses_by_iteration,
        }
        arrays["metadata"] = np.asarray(
            json.dumps(
                metadata,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise ValueError(f"Invalid Manta checkpoint data: {exc}") from exc

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            # Passing a file avoids NumPy silently appending a .npz suffix.
            np.savez_compressed(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def load_manta_checkpoint(path: str | Path) -> MantaCheckpoint:
    """Read a versioned archive or a run directory containing checkpoint.npz.

    Loading never enables NumPy object deserialization. Missing files retain
    ``FileNotFoundError``; damaged or unsupported checkpoints raise ``ValueError``.
    """
    source = Path(path)
    if source.is_dir():
        source = source / "checkpoint.npz"
    try:
        # Own the handle even when NumPy fails while initializing a damaged ZIP.
        with (
            source.open("rb") as checkpoint_file,
            np.load(
                checkpoint_file,
                allow_pickle=False,
            ) as archive,
        ):
            metadata = _read_metadata(archive)
            config = project_config_from_mapping(metadata["config"])
            completed = metadata["completed_iterations"]
            agents = range(len(config.scenario.starts))
            statuses = _read_statuses(metadata["statuses"], agents, completed)

            # Reject inconsistent counts before constructing a member set.
            expected_count = 1 + 2 * len(agents) + completed * (len(agents) + 2)
            if len(archive.files) != expected_count:
                raise ValueError(
                    "checkpoint archive members do not match iteration counts"
                )
            required = {"metadata"}
            required.update(f"initial_controls_{agent}" for agent in agents)
            required.update(
                f"history_{iteration}_{agent}"
                for iteration in range(completed + 1)
                for agent in agents
            )
            required.update(
                f"controls_{iteration}" for iteration in range(1, completed + 1)
            )
            required.update(
                f"slack_{iteration}" for iteration in range(1, completed + 1)
            )
            if set(archive.files) != required:
                raise ValueError("checkpoint archive has missing or unexpected members")

            result = MantaLMPCRunResult(
                scenario_name=config.scenario.name,
                histories=[
                    {
                        agent: _numeric_array(
                            archive[f"history_{iteration}_{agent}"],
                            f"history_{iteration}_{agent}",
                        )
                        for agent in agents
                    }
                    for iteration in range(completed + 1)
                ],
                controls_by_iteration=[
                    _numeric_array(
                        archive[f"controls_{iteration}"], f"controls_{iteration}"
                    )
                    for iteration in range(1, completed + 1)
                ],
                slack_by_iteration=[
                    _numeric_array(archive[f"slack_{iteration}"], f"slack_{iteration}")
                    for iteration in range(1, completed + 1)
                ],
                statuses_by_iteration=statuses,
                success_by_iteration=[],
                goal_reached_by_iteration=[],
                learned_by_iteration=[],
                validation_by_iteration=[],
                selected_iteration=None,
                initial_controls={
                    agent: _numeric_array(
                        archive[f"initial_controls_{agent}"],
                        f"initial_controls_{agent}",
                    )
                    for agent in agents
                },
                learning_settings=manta_learning_settings(
                    config.scenario,
                    config.lmpc,
                    config.apf,
                    config.dynamics,
                ),
            )
        revalidate_result(result, config.scenario, config.lmpc)
        return MantaCheckpoint(config=config, result=result)
    except FileNotFoundError:
        raise
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        OverflowError,
        EOFError,
        OSError,
        zipfile.BadZipFile,
        zlib.error,
    ) as exc:
        raise ValueError(f"Invalid Manta checkpoint '{source}': {exc}") from exc


def _numeric_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"checkpoint {name} must contain real numeric data")
    return array


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"checkpoint metadata contains non-finite number {value}")


def _read_metadata(archive: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata" not in archive.files:
        raise ValueError("checkpoint archive is missing metadata")
    raw = archive["metadata"]
    if raw.shape != () or raw.dtype.kind != "U":
        raise ValueError("checkpoint metadata must be a JSON unicode scalar")
    metadata = json.loads(raw.item(), parse_constant=_reject_json_constant)
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a mapping")
    version = metadata.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema version {version!r}")
    completed = metadata.get("completed_iterations")
    if type(completed) is not int or completed < 0:
        raise ValueError(
            "checkpoint completed_iterations must be a nonnegative integer"
        )
    if not isinstance(metadata.get("config"), dict):
        raise ValueError("checkpoint config must be a mapping")
    if not isinstance(metadata.get("statuses"), list):
        raise ValueError("checkpoint statuses must be a list")
    return metadata


def _read_statuses(
    raw: list[Any],
    agents: range,
    completed: int,
) -> list[list[dict[int, str]]]:
    if len(raw) != completed:
        raise ValueError(
            "checkpoint status iteration count does not match completed_iterations"
        )
    keys = {str(agent) for agent in agents}
    statuses = []
    for iteration in raw:
        if not isinstance(iteration, list):
            raise ValueError("checkpoint iteration statuses must be a list")
        rows = []
        for row in iteration:
            if (
                not isinstance(row, dict)
                or set(row) != keys
                or any(
                    not isinstance(value, str) or not value for value in row.values()
                )
            ):
                raise ValueError(
                    "checkpoint statuses must identify every agent with a string"
                )
            rows.append({int(agent): status for agent, status in row.items()})
        statuses.append(rows)
    return statuses
