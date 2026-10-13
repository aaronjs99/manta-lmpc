"""User workflow checks: saved settings, output safety, and cancellation."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.experiments.manta import lmpc
from scripts.cli.arguments.manta.lmpc import parse_args
from scripts.configuration.manta import load_project_config
from test_manta_resume import small_config, run


class MantaRunCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = small_config()
        cls.seed = run(cls.config)

    def test_resume_option_accepts_a_run_directory(self):
        args = parse_args(["--resume", "saved_run", "--iterations", "2"])
        self.assertEqual(args.resume, "saved_run")
        self.assertEqual(args.iterations, 2)

    def test_explicit_matching_yaml_keeps_archived_scenario(self):
        stored = load_project_config(scenario_name="manta_crossover")
        args = parse_args(["--resume", "saved_run", "--config", "config/manta.yaml"])
        restored = lmpc._load_effective_config(args, stored)
        self.assertEqual(restored.scenario.name, "manta_crossover")

    def test_locked_resume_does_not_overwrite_a_newer_checkpoint(self):
        from scripts.storage.manta import save_manta_checkpoint, load_manta_checkpoint
        from scripts.experiments.manta.run_directory import claim_run_directory

        newer_config = small_config(iterations=1)
        newer = run(newer_config)
        latest_bytes = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "checkpoint.npz"
            save_manta_checkpoint(path, self.seed, self.config)

            @contextlib.contextmanager
            def concurrent_writer_finishes(output, *, resume_path=None):
                save_manta_checkpoint(path, newer, newer_config)
                latest_bytes.append(path.read_bytes())
                with claim_run_directory(output, resume_path=resume_path):
                    yield output

            with patch.object(lmpc, "claim_run_directory", concurrent_writer_finishes):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    lmpc.main(
                        ["--resume", str(root), "--iterations", "0", "--no-video"]
                    )
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(path.read_bytes(), latest_bytes[0])
            self.assertEqual(load_manta_checkpoint(path).result.completed_iterations, 1)

    def test_interrupted_extension_keeps_new_target_and_imported_seed(self):
        from scripts.storage.manta import save_manta_checkpoint, load_manta_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            output = Path(directory) / "extended"
            save_manta_checkpoint(source / "checkpoint.npz", self.seed, self.config)
            with patch(
                "scripts.experiments.manta.runner.execute_manta_iteration",
                side_effect=KeyboardInterrupt,
            ):
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    lmpc.main(
                        [
                            "--resume",
                            str(source),
                            "--iterations",
                            "2",
                            "--output-dir",
                            str(output),
                            "--no-video",
                        ]
                    )
            self.assertEqual(error.exception.code, 130)
            checkpoint = load_manta_checkpoint(output)
            self.assertEqual(checkpoint.config.lmpc.iterations, 2)
            self.assertEqual(checkpoint.result.completed_iterations, 0)
            self.assertEqual(load_manta_checkpoint(source).config.lmpc.iterations, 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["run_status"], "interrupted")

    def test_new_run_refuses_to_overwrite_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root / "summary.json"
            sentinel.write_text("existing results", encoding="utf-8")
            with patch.object(lmpc, "_load_effective_config", return_value=self.config):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    lmpc.main(["--output-dir", str(root), "--no-video"])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(sentinel.read_text(), "existing results")

    def test_active_output_lock_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / ".manta-run.lock"
            lock.write_text("pid=123\n", encoding="utf-8")
            with patch.object(lmpc, "_load_effective_config", return_value=self.config):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    lmpc.main(["--output-dir", str(root), "--no-video"])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(lock.read_text(), "pid=123\n")
            self.assertFalse((root / "checkpoint.npz").exists())

    def test_cancel_does_not_delete_a_user_stop_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stop = root / "stop_signal"
            stop.write_text("please stop", encoding="utf-8")
            with patch.object(lmpc, "_load_effective_config", return_value=self.config):
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    lmpc.main(
                        [
                            "--output-dir",
                            str(root / "run"),
                            "--stop-file",
                            str(stop),
                            "--no-video",
                        ]
                    )
            self.assertEqual(error.exception.code, 130)
            self.assertEqual(stop.read_text(), "please stop")
            self.assertFalse((root / "run" / ".manta-run.lock").exists())

    def test_resume_uses_archived_settings_without_reading_yaml(self):
        from scripts.storage.manta import save_manta_checkpoint, load_manta_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_manta_checkpoint(root / "checkpoint.npz", self.seed, self.config)
            with patch.object(
                lmpc,
                "load_project_config",
                side_effect=AssertionError("resume read mutable YAML"),
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    lmpc.main(["--resume", str(root), "--no-video"])
            saved = load_manta_checkpoint(root)
            self.assertEqual(saved.config.scenario.name, "parallel_lanes")
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["run_status"], "completed")
            self.assertEqual(summary["completed_iterations"], 0)
            self.assertTrue((root / "final_trajectories.png").exists())
            self.assertFalse((root / ".manta-run.lock").exists())

    def test_no_video_report_excludes_an_older_gif_from_current_artifacts(self):
        from scripts.storage.manta import save_manta_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_manta_checkpoint(root / "checkpoint.npz", self.seed, self.config)
            old_gif = root / "final_iteration.gif"
            old_gif.write_bytes(b"previous generated animation")
            with contextlib.redirect_stdout(io.StringIO()):
                lmpc.main(["--resume", str(root), "--no-video"])
            summary = json.loads((root / "summary.json").read_text())
            self.assertIsNone(summary["artifacts"]["animation"])
            self.assertEqual(
                summary["artifacts"]["trajectory_plot"], "final_trajectories.png"
            )
            self.assertEqual(old_gif.read_bytes(), b"previous generated animation")

    def test_resume_rejects_changed_dynamics_and_preserves_checkpoint(self):
        from scripts.storage.manta import save_manta_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            save_manta_checkpoint(path, self.seed, self.config)
            original = path.read_bytes()
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                lmpc.main(["--resume", str(path), "--dt", "0.3"])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(path.read_bytes(), original)

    def test_interrupt_after_seed_leaves_checkpoint_and_summary(self):
        from scripts.storage.manta import load_manta_checkpoint

        config = small_config(iterations=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def interrupt(_request):
                raise KeyboardInterrupt

            with (
                patch.object(lmpc, "_load_effective_config", return_value=config),
                patch(
                    "scripts.experiments.manta.runner.execute_manta_iteration",
                    interrupt,
                ),
            ):
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    lmpc.main(["--output-dir", str(root), "--no-video"])
            self.assertEqual(error.exception.code, 130)
            checkpoint = load_manta_checkpoint(root)
            self.assertEqual(checkpoint.result.completed_iterations, 0)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["run_status"], "interrupted")
            self.assertEqual(summary["completed_iterations"], 0)
            self.assertEqual(summary["iterations"], 1)
            self.assertFalse((root / ".manta-run.lock").exists())


if __name__ == "__main__":
    unittest.main()
