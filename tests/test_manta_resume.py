"""Regression checks for completed-iteration recovery with the real solver."""

from dataclasses import replace
import copy
import unittest
from unittest.mock import patch

import numpy as np

from scripts.configuration import manta
from scripts.contracts.manta import Scenario, StaticObstacle
from scripts.experiments.manta.runner import run_manta_lmpc


def small_config(iterations=0):
    config = manta.load_project_config()
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
        "parallel_lanes", starts, goals, 0.6, StaticObstacle((20, 20), 0.5)
    )
    return replace(
        config,
        scenario=scenario,
        scenario_name=scenario.name,
        apf=replace(
            config.apf,
            static_agent_radius_scales=(1.0,),
            compact_staging=False,
            max_steps=100,
        ),
        lmpc=replace(
            config.lmpc,
            iterations=iterations,
            max_steps=3,
            prediction_horizon=3,
            k_hull=3,
        ),
        make_video=False,
        quiet=True,
    )


def run(config, **kwargs):
    return run_manta_lmpc(
        config.scenario,
        config=config.lmpc,
        apf_config=config.apf,
        dynamics_config=config.dynamics,
        verbose=False,
        **kwargs,
    )


class MantaResumeTests(unittest.TestCase):
    def test_effective_settings_survive_mapping_round_trip(self):
        config = small_config(iterations=2)
        serialize = getattr(manta, "project_config_to_mapping", None)
        deserialize = getattr(manta, "project_config_from_mapping", None)
        self.assertTrue(callable(serialize), "effective config export is missing")
        self.assertTrue(callable(deserialize), "effective config import is missing")
        restored = deserialize(serialize(config))
        self.assertEqual(restored.lmpc.iterations, 2)
        self.assertEqual(restored.lmpc.max_steps, 3)
        self.assertEqual(restored.scenario.name, "parallel_lanes")
        np.testing.assert_array_equal(restored.scenario.starts, config.scenario.starts)
        self.assertEqual(restored.apf.static_agent_radius_scales, (1.0,))
        self.assertFalse(restored.make_video)

    def test_apf_completion_emits_reusable_seed_and_controls(self):
        snapshots = []
        result = run(small_config(), on_iteration_completed=snapshots.append)
        self.assertEqual([s.completed_iterations for s in snapshots], [0])
        self.assertTrue(result.validation_by_iteration[0].valid)
        for agent, states in result.histories[0].items():
            self.assertEqual(result.initial_controls[agent].shape, (len(states) - 1, 2))

    def test_resumed_learning_matches_uninterrupted_learning(self):
        config = small_config(iterations=2)
        uninterrupted = run(config)
        first = run(replace(config, lmpc=replace(config.lmpc, iterations=1)))
        original = copy.deepcopy(first)
        with patch(
            "scripts.experiments.manta.runner.build_staggered_safe_sets",
            side_effect=AssertionError("resume regenerated the APF seed"),
        ):
            resumed = run(config, resume_result=first)
        self.assertEqual(len(first.histories), 2)
        np.testing.assert_array_equal(first.histories[1][0], original.histories[1][0])
        self.assertEqual(
            resumed.success_by_iteration, uninterrupted.success_by_iteration
        )
        self.assertEqual(resumed.selected_iteration, uninterrupted.selected_iteration)
        for expected, actual in zip(uninterrupted.histories, resumed.histories):
            for agent in expected:
                np.testing.assert_allclose(
                    actual[agent], expected[agent], atol=1e-10, rtol=0
                )
        for expected, actual in zip(
            uninterrupted.controls_by_iteration, resumed.controls_by_iteration
        ):
            np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_cancellation_keeps_only_completed_iterations(self):
        snapshots = []
        stop = False

        def completed(result):
            nonlocal stop
            snapshots.append(result)
            stop = result.completed_iterations == 1

        with self.assertRaises(KeyboardInterrupt):
            run(
                small_config(iterations=2),
                on_iteration_completed=completed,
                should_stop=lambda: stop,
            )
        self.assertEqual([s.completed_iterations for s in snapshots], [0, 1])
        self.assertEqual(len(snapshots[-1].histories), 2)
        self.assertEqual(len(snapshots[-1].statuses_by_iteration), 1)

    def test_stop_request_before_seed_does_no_work(self):
        with patch(
            "scripts.experiments.manta.runner.build_staggered_safe_sets",
            side_effect=AssertionError("ignored cancellation"),
        ):
            with self.assertRaises(KeyboardInterrupt):
                run(small_config(), should_stop=lambda: True)

    def test_resume_rejects_smaller_iteration_target(self):
        first = run(small_config(iterations=1))
        with self.assertRaisesRegex(ValueError, "iterations"):
            run(small_config(), resume_result=first)

    def test_python_resume_rejects_changed_numerical_settings(self):
        config = small_config()
        seed = run(config)
        changed = (
            replace(config, lmpc=replace(config.lmpc, dt=0.1)),
            replace(
                config, dynamics=replace(config.dynamics, oscillator_frequency=3.0)
            ),
            replace(config, apf=replace(config.apf, repulsion_gain=5.0)),
        )
        for section, different in zip(("lmpc", "dynamics", "apf"), changed):
            with (
                self.subTest(section=section),
                self.assertRaisesRegex(ValueError, "settings"),
            ):
                run(different, resume_result=seed)

    def test_incomplete_apf_seed_is_saved_but_never_used_for_learning(self):
        config = small_config(iterations=1)
        config = replace(config, apf=replace(config.apf, max_steps=1))
        snapshots = []
        with patch(
            "scripts.experiments.manta.runner.MantaAgentOptimizer",
            side_effect=AssertionError("optimized from an invalid seed"),
        ):
            with self.assertRaisesRegex(ValueError, "seed"):
                run(config, on_iteration_completed=snapshots.append)
        self.assertEqual(len(snapshots), 1)
        self.assertFalse(snapshots[0].validation_by_iteration[0].usable_for_learning)


if __name__ == "__main__":
    unittest.main()
