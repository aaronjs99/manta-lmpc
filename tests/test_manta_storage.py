"""Focused checks for durable, non-pickle Manta learning checkpoints."""

from dataclasses import replace
import gc
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from scripts.configuration.manta import load_project_config, override_project_config
from scripts.contracts.manta import MantaLMPCRunResult, Scenario, StaticObstacle


def checkpoint_fixture():
    """A valid earlier iteration followed by an incomplete latest attempt."""
    starts = np.array(
        [
            [0, 0, 0, 0.1, 0, 0.1, 0],
            [0, 3, 0, 0.1, 0, 0.1, 0],
        ],
        dtype=float,
    )
    goals = starts.copy()
    goals[:, 0] = 2
    scenario = Scenario(
        "storage_lanes", starts, goals, 0.6, StaticObstacle((20, 20), 0.5)
    )
    config = replace(
        load_project_config(), scenario=scenario, scenario_name=scenario.name
    )
    config = override_project_config(
        config,
        lmpc={
            "iterations": 3,
            "max_steps": 5,
            "prediction_horizon": 4,
            "k_hull": 2,
            "goal_tolerance": 0.1,
            "dt": 0.1,
        },
        apf={"max_steps": 42},
        make_video=True,
        quiet=False,
    )
    histories = []
    for x_positions in ([0, 1, 2], [0, 1.2, 2], [0, 0.4]):
        history = {}
        for agent in range(2):
            states = np.tile(starts[agent], (len(x_positions), 1))
            states[:, 0] = x_positions
            history[agent] = states
        histories.append(history)
    result = MantaLMPCRunResult(
        scenario_name=scenario.name,
        histories=histories,
        controls_by_iteration=[
            np.array([[[0.2, 0.1], [0.3, -0.1]], [[0.4, 0.0], [0.5, 0.0]]]),
            np.array([[[0.6, 0.2], [0.7, -0.2]]]),
        ],
        slack_by_iteration=[
            np.array(
                [
                    [[0, 0.1, 0.2], [np.nan, np.nan, np.nan]],
                    [[0.3, 0.4, 0.5], [0, 0, 0]],
                ]
            ),
            np.full((1, 2, 3), np.nan),
        ],
        statuses_by_iteration=[
            [{0: "ok", 1: "fallback_apf"}, {0: "ok", 1: "hold"}],
            [{0: "safety_filter_hold", 1: "fallback_apf"}],
        ],
        # Deliberately stale: persistence must derive these from the histories.
        success_by_iteration=[False, False, False],
        goal_reached_by_iteration=[False, False, False],
        learned_by_iteration=[False, False, False],
        validation_by_iteration=[],
        selected_iteration=None,
    )
    result.initial_controls = {
        0: np.array([[0.8, 0.0], [0.9, 0.1]]),
        1: np.array([[1.0, -0.1], [1.1, 0.0]]),
    }
    return config, result


def archive_members(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def set_metadata(members, **updates):
    metadata = json.loads(members["metadata"].item())
    metadata.update(updates)
    members["metadata"] = np.asarray(json.dumps(metadata))


class MantaStorageTests(unittest.TestCase):
    def test_real_result_cannot_be_saved_with_different_numerical_settings(self):
        from test_manta_resume import small_config, run

        storage = self.storage()
        config = small_config()
        result = run(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            storage.save_manta_checkpoint(path, result, config)
            previous = path.read_bytes()
            different = replace(config, lmpc=replace(config.lmpc, dt=0.1))
            with self.assertRaisesRegex(ValueError, "settings"):
                storage.save_manta_checkpoint(path, result, different)
            self.assertEqual(path.read_bytes(), previous)

    def storage(self):
        self.assertIsNotNone(
            importlib.util.find_spec("scripts.storage"),
            "Manta checkpoint persistence is missing",
        )
        module = importlib.import_module("scripts.storage.manta")
        self.assertTrue(callable(getattr(module, "save_manta_checkpoint", None)))
        self.assertTrue(callable(getattr(module, "load_manta_checkpoint", None)))
        return module

    def test_round_trip_retains_effective_settings_and_every_iteration(self):
        storage = self.storage()
        config, result = checkpoint_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            self.assertEqual(storage.save_manta_checkpoint(path, result, config), path)
            checkpoint = storage.load_manta_checkpoint(directory)
            restored = checkpoint.result
            self.assertEqual(checkpoint.config.lmpc.iterations, 3)
            self.assertEqual(checkpoint.config.lmpc.max_steps, 5)
            self.assertEqual(checkpoint.config.lmpc.prediction_horizon, 4)
            self.assertEqual(checkpoint.config.lmpc.k_hull, 2)
            self.assertEqual(checkpoint.config.lmpc.dt, 0.1)
            self.assertEqual(checkpoint.config.apf.max_steps, 42)
            self.assertTrue(checkpoint.config.make_video)
            self.assertFalse(checkpoint.config.quiet)
            self.assertEqual(checkpoint.config.scenario.name, "storage_lanes")
            np.testing.assert_array_equal(
                checkpoint.config.scenario.starts, config.scenario.starts
            )
            self.assertEqual(restored.completed_iterations, 2)
            self.assertEqual(restored.selected_iteration, 1)
            self.assertEqual(restored.success_by_iteration, [True, True, False])
            self.assertEqual(restored.goal_reached_by_iteration, [True, True, False])
            self.assertEqual(restored.learned_by_iteration, [True, True, False])
            self.assertEqual(restored.validation_by_iteration[1].fallback_count, 1)
            self.assertEqual(
                restored.validation_by_iteration[2].safety_intervention_count, 1
            )
            self.assertEqual(
                restored.statuses_by_iteration, result.statuses_by_iteration
            )
            for expected, actual in zip(result.histories, restored.histories):
                for agent in expected:
                    np.testing.assert_array_equal(actual[agent], expected[agent])
            for expected, actual in zip(
                result.controls_by_iteration, restored.controls_by_iteration
            ):
                np.testing.assert_array_equal(actual, expected)
            for expected, actual in zip(
                result.slack_by_iteration, restored.slack_by_iteration
            ):
                np.testing.assert_array_equal(actual, expected)
            for agent in result.initial_controls:
                np.testing.assert_array_equal(
                    restored.initial_controls[agent], result.initial_controls[agent]
                )
            members = archive_members(path)
            self.assertEqual(members["metadata"].shape, ())
            self.assertEqual(members["metadata"].dtype.kind, "U")
            self.assertTrue(
                all(
                    value.dtype.kind in "fiu"
                    for name, value in members.items()
                    if name != "metadata"
                )
            )

    def test_initial_seed_without_learning_iterations_round_trips(self):
        storage = self.storage()
        config, result = checkpoint_fixture()
        result.histories = result.histories[:1]
        result.controls_by_iteration = []
        result.slack_by_iteration = []
        result.statuses_by_iteration = []
        config = replace(config, lmpc=replace(config.lmpc, iterations=0))
        with tempfile.TemporaryDirectory() as directory:
            path = storage.save_manta_checkpoint(
                Path(directory) / "checkpoint.npz", result, config
            )
            restored = storage.load_manta_checkpoint(path).result
            self.assertEqual(restored.completed_iterations, 0)
            self.assertEqual(restored.selected_iteration, 0)
            self.assertEqual(restored.statuses_by_iteration, [])
            np.testing.assert_array_equal(
                restored.initial_controls[1], result.initial_controls[1]
            )

    def test_missing_checkpoint_preserves_file_not_found_path(self):
        storage = self.storage()
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / "checkpoint.npz"
            with self.assertRaises(FileNotFoundError) as caught:
                storage.load_manta_checkpoint(directory)
            self.assertEqual(Path(caught.exception.filename), expected)

    def test_damaged_archives_raise_clear_value_error(self):
        storage = self.storage()
        config, result = checkpoint_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = storage.save_manta_checkpoint(
                Path(directory) / "checkpoint.npz", result, config
            )
            complete = path.read_bytes()
            for content in (b"not a checkpoint", complete[: len(complete) // 2]):
                with self.subTest(size=len(content)):
                    path.write_bytes(content)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ResourceWarning)
                        with self.assertRaisesRegex(ValueError, "checkpoint"):
                            storage.load_manta_checkpoint(path)
                        gc.collect()
                    self.assertEqual(
                        [
                            item
                            for item in caught
                            if issubclass(item.category, ResourceWarning)
                        ],
                        [],
                    )

    def test_metadata_version_counts_and_types_are_checked(self):
        storage = self.storage()
        config, result = checkpoint_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = storage.save_manta_checkpoint(
                Path(directory) / "checkpoint.npz", result, config
            )
            original = archive_members(path)
            mutations = [
                {"schema_version": 99},
                {"schema_version": True},
                {"completed_iterations": -1},
                {"completed_iterations": 2.5},
                {"completed_iterations": 1},
                {"statuses": {}},
                {"statuses": [[{"0": "ok", "1": 2}], []]},
                {"config": []},
            ]
            for updates in mutations:
                with self.subTest(updates=updates):
                    members = dict(original)
                    set_metadata(members, **updates)
                    np.savez_compressed(path, **members)
                    with self.assertRaisesRegex(ValueError, "checkpoint"):
                        storage.load_manta_checkpoint(path)
            for metadata in (
                np.asarray("{bad json"),
                np.asarray(["{}"]),
                np.asarray(b"{}"),
                np.asarray("[]"),
            ):
                with self.subTest(metadata=metadata):
                    members = dict(original, metadata=metadata)
                    np.savez_compressed(path, **members)
                    with self.assertRaisesRegex(ValueError, "checkpoint"):
                        storage.load_manta_checkpoint(path)

    def test_missing_archive_members_and_object_arrays_are_rejected(self):
        storage = self.storage()
        config, result = checkpoint_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = storage.save_manta_checkpoint(
                Path(directory) / "checkpoint.npz", result, config
            )
            original = archive_members(path)
            members = dict(original)
            del members["controls_1"]
            np.savez_compressed(path, **members)
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                storage.load_manta_checkpoint(path)
            for member in ("metadata", "history_0_0"):
                with self.subTest(member=member):
                    members = dict(original)
                    members[member] = np.asarray({"untrusted": "pickle"}, dtype=object)
                    np.savez_compressed(path, **members)
                    with self.assertRaisesRegex(ValueError, "checkpoint"):
                        storage.load_manta_checkpoint(path)

    def test_numeric_and_step_alignment_damage_is_rejected(self):
        storage = self.storage()
        config, result = checkpoint_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = storage.save_manta_checkpoint(
                Path(directory) / "checkpoint.npz", result, config
            )
            original = archive_members(path)
            mutations = [
                ("history_1_0", np.full((3, 7), np.nan)),
                ("history_1_0", original["history_1_0"][:2]),
                ("history_0_0", np.ones((3, 7))),
                ("controls_1", np.zeros((1, 2, 2))),
                ("controls_1", np.full((2, 2, 2), np.inf)),
                ("slack_1", np.zeros((2, 2, 2))),
                ("slack_1", np.full((2, 2, 3), np.inf)),
                ("initial_controls_0", np.zeros((1, 2))),
            ]
            for member, replacement in mutations:
                with self.subTest(member=member, shape=replacement.shape):
                    members = dict(original)
                    members[member] = replacement
                    np.savez_compressed(path, **members)
                    with self.assertRaisesRegex(ValueError, "checkpoint"):
                        storage.load_manta_checkpoint(path)

    def test_interrupted_replace_preserves_old_archive_and_removes_temp_file(self):
        storage = self.storage()
        config, result = checkpoint_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = storage.save_manta_checkpoint(
                Path(directory) / "checkpoint.npz", result, config
            )
            previous = path.read_bytes()
            result.controls_by_iteration[0][0, 0, 0] = 1.9
            with patch(
                "scripts.storage.manta.os.replace", side_effect=KeyboardInterrupt
            ):
                with self.assertRaises(KeyboardInterrupt):
                    storage.save_manta_checkpoint(path, result, config)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            self.assertEqual(
                storage.load_manta_checkpoint(path).result.controls_by_iteration[0][
                    0, 0, 0
                ],
                0.2,
            )

    def test_failed_validation_preserves_old_archive(self):
        storage = self.storage()
        config, result = checkpoint_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = storage.save_manta_checkpoint(
                Path(directory) / "checkpoint.npz", result, config
            )
            previous = path.read_bytes()
            result.histories[-1][0][1, 0] = np.inf
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                storage.save_manta_checkpoint(path, result, config)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
