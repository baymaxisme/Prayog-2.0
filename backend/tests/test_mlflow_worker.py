import os
import stat
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("POSTGRES_USER", "prayog")
os.environ.setdefault("POSTGRES_PASSWORD", "test-password")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "prayog_test")

from mlflow_worker import (
    build_docker_command,
    claim_next_first_time_run,
    recover_stale_claims,
    cancel_jobs_without_capable_node,
    extract_code_archive,
    prepare_input_directory,
    safe_output_file,
    validate_entrypoint,
    configured_node_inventory,
    configured_timeout,
    configured_image,
    _node_can_run,
)


class MLFlowWorkerTests(unittest.TestCase):
    def test_first_time_run_claim_is_committed_to_one_worker(self):
        from unittest.mock import MagicMock

        db = MagicMock()
        run = SimpleNamespace(id=41, status='QUEUED', started_at=None, worker_id=None, worker_claimed_at=None)
        query = db.query.return_value.filter.return_value
        query.order_by.return_value.with_for_update.return_value.first.return_value = run

        claimed = claim_next_first_time_run(db, "worker-1")

        self.assertEqual(claimed, 41)
        self.assertEqual(run.status, "RUNNING")
        self.assertIsNotNone(run.started_at)
        self.assertEqual(run.worker_id, "worker-1")
        self.assertIsNotNone(run.worker_claimed_at)
        db.commit.assert_called_once()
        db.rollback.assert_not_called()

    def test_extracts_archive_and_validates_relative_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "code.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("main.py", "print('ok')\n")
                output.writestr("lib/helper.py", "VALUE = 1\n")
            workspace = root / "workspace"

            extract_code_archive(archive, workspace)

            self.assertEqual(validate_entrypoint(workspace, "main.py"), Path("main.py"))
            self.assertEqual((workspace / "main.py").read_text(encoding="utf-8"), "print('ok')\n")
            self.assertFalse((workspace / "main.py").stat().st_mode & 0o222)

    def test_rejects_archive_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escape.py", "print('unsafe')\n")

            with self.assertRaisesRegex(ValueError, "unsafe path"):
                extract_code_archive(archive, root / "workspace")

            self.assertFalse((root / "escape.py").exists())

    def test_rejects_entrypoint_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with self.assertRaisesRegex(ValueError, "safe relative path"):
                validate_entrypoint(workspace, "../outside.py")

    def test_container_command_enforces_isolation_and_no_gpu_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = build_docker_command(
                "prayog-ml-runtime:python3.12",
                "prayog-test-container",
                root / "workspace",
                root / "input",
                Path("main.py"),
                12,
                "snapshot-id",
                {"learning_rate": 0.01},
                True,
                "test-completion-token",
            )

            self.assertIn("--network", command)
            self.assertEqual(command[command.index("--network") + 1], "none")
            self.assertIn("--read-only", command)
            self.assertIn("--cap-drop", command)
            self.assertIn("ALL", command)
            self.assertIn("--pids-limit", command)
            self.assertIn("--memory", command)
            self.assertIn("--cpus", command)
            self.assertIn("--pull=never", command)
            self.assertIn("PRAYOG_COMPLETION_TOKEN=test-completion-token", command)
            self.assertTrue(any("type=tmpfs,destination=/output" in value for value in command))
            self.assertIn("PRAYOG_INPUT_WEIGHTS=/input/weights.bin", command)
            self.assertNotIn("--gpus", command)


    def test_input_weights_are_copied_before_mount_is_made_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source-weights.bin"
            source.write_bytes(b"trained-weights")
            input_dir = root / "input"

            prepare_input_directory(input_dir, source)

            restored = input_dir / "weights.bin"
            self.assertEqual(restored.read_bytes(), b"trained-weights")
            self.assertFalse(input_dir.stat().st_mode & 0o222)
            self.assertFalse(restored.stat().st_mode & 0o222)


    def test_output_artifact_must_be_regular_file_not_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            weights = output / "weights.bin"
            weights.write_bytes(b"weights")
            self.assertEqual(safe_output_file(output, "weights.bin"), weights)

            link = output / "linked.bin"
            link.symlink_to(weights)
            self.assertIsNone(safe_output_file(output, "linked.bin"))


    def test_rejects_duplicate_archive_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "duplicate.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("main.py", "print('first')\\n")
                output.writestr("main.py", "print('second')\\n")

            with self.assertRaisesRegex(ValueError, "duplicate paths"):
                extract_code_archive(archive, root / "workspace")

    def test_rejects_symbolic_link_archive_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "symlink.zip"
            member = zipfile.ZipInfo("linked.py")
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr(member, "outside.py")

            with self.assertRaisesRegex(ValueError, "symbolic link"):
                extract_code_archive(archive, root / "workspace")

    def test_rejects_archive_entry_over_artifact_size_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "oversized.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("main.py", "12345")

            with patch("mlflow_worker.MAX_ARTIFACT_BYTES", 4):
                with self.assertRaisesRegex(ValueError, "size limit"):
                    extract_code_archive(archive, root / "workspace")

    def test_node_inventory_rejects_boolean_gpu_units(self):
        with patch.dict(os.environ, {
            "PRAYOG_NODE_CAPABILITY_INVENTORY":
                '[{"capabilities":["cuda"],"gpu_units":true}]'
        }):
            with self.assertRaisesRegex(RuntimeError, "invalid capabilities"):
                configured_node_inventory()

    def test_node_capability_match_requires_all_capabilities_and_gpu_units(self):
        inventory = [
            {"capabilities": {"cuda", "python"}, "gpu_units": 1},
            {"capabilities": {"python", "linux"}, "gpu_units": 4},
        ]

        self.assertTrue(_node_can_run(
            {"required_capabilities": ["cuda", "python"], "required_gpu_units": 1},
            inventory,
        ))
        self.assertFalse(_node_can_run(
            {"required_capabilities": ["cuda", "python"], "required_gpu_units": 2},
            inventory,
        ))
        self.assertFalse(_node_can_run(
            {"required_capabilities": ["cuda", "linux"], "required_gpu_units": 0},
            inventory,
        ))

    def test_invalid_node_requirements_do_not_trigger_false_ineligibility(self):
        inventory = [{"capabilities": set(), "gpu_units": 0}]

        self.assertTrue(_node_can_run(
            {"required_capabilities": "cuda", "required_gpu_units": 8},
            inventory,
        ))
        self.assertTrue(_node_can_run(
            {"required_capabilities": [], "required_gpu_units": True},
            inventory,
        ))


    def test_node_inventory_normalizes_capabilities_to_a_set(self):
        with patch.dict(os.environ, {
            "PRAYOG_NODE_CAPABILITY_INVENTORY":
                '[{"capabilities":["cuda","python"],"gpu_units":2}]'
        }):
            self.assertEqual(
                configured_node_inventory(),
                [{"capabilities": {"cuda", "python"}, "gpu_units": 2}],
            )

    def test_worker_timeout_is_clamped_to_supported_range(self):
        with patch.dict(os.environ, {"PRAYOG_WORKER_TIMEOUT_SECONDS": "99999"}):
            self.assertEqual(configured_timeout(), 1800)
        with patch.dict(os.environ, {"PRAYOG_WORKER_TIMEOUT_SECONDS": "-10"}):
            self.assertEqual(configured_timeout(), 1)
        with patch.dict(os.environ, {"PRAYOG_WORKER_TIMEOUT_SECONDS": "invalid"}):
            self.assertEqual(configured_timeout(), 1800)

    def test_worker_image_rejects_shell_metacharacters(self):
        with patch.dict(os.environ, {"PRAYOG_WORKER_IMAGE": "runtime;touch /tmp/pwned"}):
            with self.assertRaisesRegex(RuntimeError, "invalid characters"):
                configured_image()

    def test_empty_code_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "empty.zip"
            with zipfile.ZipFile(archive, "w"):
                pass

            with self.assertRaisesRegex(ValueError, "invalid number of entries"):
                extract_code_archive(archive, root / "workspace")


    def test_stale_rebuild_claim_fails_attempt_and_running_execution(self):
        now = datetime(2026, 11, 4, 10, 0, 0)
        attempt = SimpleNamespace(
            id=12, execution_run_id=41, user_id=7, state="REBUILD_INITIATED",
            worker_id="worker-dead", worker_claimed_at=now - timedelta(hours=1),
            failure_summary=None, finished_at=None,
        )
        run = SimpleNamespace(
            id=41, user_id=7, status="RUNNING", error_summary=None,
            finished_at=None, logging_state="COMPLETE", logging_warnings=[],
        )
        attempt_query = MagicMock()
        attempt_query.filter.return_value.with_for_update.return_value.all.return_value = [attempt]
        execution_query = MagicMock()
        execution_query.filter.return_value.with_for_update.return_value.first.return_value = run
        runs_query = MagicMock()
        runs_query.filter.return_value.with_for_update.return_value.all.return_value = []
        db = MagicMock()
        db.query.side_effect = [attempt_query, execution_query, runs_query]

        with patch("mlflow_worker.utcnow", return_value=now):
            recovered = recover_stale_claims(db, 1800)

        self.assertEqual(recovered, 1)
        self.assertEqual(attempt.state, "REBUILD_FAILED")
        self.assertEqual(attempt.worker_id, "worker-dead")
        self.assertEqual(attempt.finished_at, now)
        self.assertIn("stopped responding", attempt.failure_summary)
        self.assertEqual(run.status, "FAILED")
        self.assertEqual(run.error_summary, attempt.failure_summary)
        self.assertEqual(run.logging_state, "DEGRADED")
        self.assertEqual(run.logging_warnings, ["The worker ended without a final artifact report."])
        db.commit.assert_called_once()
        db.rollback.assert_not_called()

    def test_stale_first_time_run_is_requeued_and_retains_queue_position(self):
        now = datetime(2026, 11, 4, 10, 0, 0)
        queued_at = now - timedelta(days=2)
        run = SimpleNamespace(
            id=41, status="RUNNING", run_type="FIRST_TIME", worker_id="worker-dead",
            worker_claimed_at=now - timedelta(hours=1), interruption_count=2,
            last_interrupted_at=None, last_attempt_status=None, queued_at=queued_at,
            error_summary=None, started_at=now - timedelta(hours=1), finished_at=None,
            logging_state="COMPLETE", logging_warnings=[f"warning-{i}" for i in range(50)],
        )
        attempts_query = MagicMock()
        attempts_query.filter.return_value.with_for_update.return_value.all.return_value = []
        runs_query = MagicMock()
        runs_query.filter.return_value.with_for_update.return_value.all.return_value = [run]
        db = MagicMock()
        db.query.side_effect = [attempts_query, runs_query]

        with patch("mlflow_worker.utcnow", return_value=now):
            recovered = recover_stale_claims(db, 1800)

        self.assertEqual(recovered, 1)
        self.assertEqual(run.status, "QUEUED")
        self.assertEqual(run.worker_id, None)
        self.assertEqual(run.worker_claimed_at, None)
        self.assertEqual(run.started_at, None)
        self.assertEqual(run.finished_at, None)
        self.assertEqual(run.queued_at, queued_at)
        self.assertEqual(run.interruption_count, 3)
        self.assertEqual(run.last_interrupted_at, now)
        self.assertEqual(run.last_attempt_status, "INTERRUPTED")
        self.assertEqual(run.logging_state, "DEGRADED")
        self.assertEqual(len(run.logging_warnings), 50)
        self.assertIn("interruption #3", run.logging_warnings[-1])
        db.commit.assert_called_once()

    def test_stale_rebuild_without_execution_still_marks_attempt_failed(self):
        now = datetime(2026, 11, 4, 10, 0, 0)
        attempt = SimpleNamespace(
            id=13, execution_run_id=99, user_id=7, state="REBUILD_INITIATED",
            worker_id="worker-dead", worker_claimed_at=now - timedelta(hours=1),
            failure_summary=None, finished_at=None,
        )
        attempt_query = MagicMock()
        attempt_query.filter.return_value.with_for_update.return_value.all.return_value = [attempt]
        execution_query = MagicMock()
        execution_query.filter.return_value.with_for_update.return_value.first.return_value = None
        runs_query = MagicMock()
        runs_query.filter.return_value.with_for_update.return_value.all.return_value = []
        db = MagicMock()
        db.query.side_effect = [attempt_query, execution_query, runs_query]

        with patch("mlflow_worker.utcnow", return_value=now):
            recovered = recover_stale_claims(db, 1800)

        self.assertEqual(recovered, 1)
        self.assertEqual(attempt.state, "REBUILD_FAILED")
        self.assertEqual(attempt.finished_at, now)
        db.commit.assert_called_once()

    def test_recovery_with_no_stale_claims_rolls_back_without_commit(self):
        db = MagicMock()
        attempt_query = MagicMock()
        attempt_query.filter.return_value.with_for_update.return_value.all.return_value = []
        runs_query = MagicMock()
        runs_query.filter.return_value.with_for_update.return_value.all.return_value = []
        db.query.side_effect = [attempt_query, runs_query]

        self.assertEqual(recover_stale_claims(db, 1800), 0)
        db.rollback.assert_called_once()
        db.commit.assert_not_called()


    def test_impossible_run_and_rebuild_are_cancelled_with_owner_notifications(self):
        now = datetime(2026, 11, 4, 10, 0, 0)
        run = SimpleNamespace(
            id=41, user_id=7, status="QUEUED", run_type="FIRST_TIME",
            code_artifact_id="code-1", worker_id=None, worker_claimed_at=None,
            environment={"required_capabilities": ["cuda"], "required_gpu_units": 2},
            error_summary=None, finished_at=None,
        )
        attempt = SimpleNamespace(
            id=12, user_id=7, execution_run_id=42, state="REBUILD_INITIATED",
            worker_id=None, worker_claimed_at=None,
            environment={"required_capabilities": ["cuda"], "required_gpu_units": 2},
            failure_summary=None, finished_at=None,
        )
        execution = SimpleNamespace(
            id=42, user_id=7, status="RUNNING", error_summary=None,
            finished_at=None, worker_id="worker-a", worker_claimed_at=now,
        )
        runs_query = MagicMock()
        runs_query.filter.return_value.with_for_update.return_value.all.return_value = [run]
        attempts_query = MagicMock()
        attempts_query.filter.return_value.with_for_update.return_value.all.return_value = [attempt]
        execution_query = MagicMock()
        execution_query.filter.return_value.with_for_update.return_value.first.return_value = execution
        db = MagicMock()
        db.query.side_effect = [runs_query, attempts_query, execution_query]

        with patch("mlflow_worker.utcnow", return_value=now):
            cancelled = cancel_jobs_without_capable_node(
                db, [{"capabilities": {"python"}, "gpu_units": 0}]
            )

        self.assertEqual(cancelled, 2)
        self.assertEqual(run.status, "CANCELLED")
        self.assertEqual(run.finished_at, now)
        self.assertIsNone(run.worker_id)
        self.assertEqual(attempt.state, "REBUILD_CANCELLED")
        self.assertEqual(attempt.finished_at, now)
        self.assertEqual(execution.status, "CANCELLED")
        self.assertEqual(execution.finished_at, now)
        self.assertIsNone(execution.worker_id)
        notifications = [call.args[0] for call in db.add.call_args_list]
        self.assertEqual(
            {item.event_key for item in notifications},
            {"run:41:NO_CAPABLE_NODE", "rebuild:12:NO_CAPABLE_NODE"},
        )
        self.assertTrue(all(item.code == "NO_CAPABLE_NODE" for item in notifications))
        db.commit.assert_called_once()
        db.rollback.assert_not_called()

    def test_impossible_rebuild_does_not_cancel_already_terminal_execution(self):
        terminal_run = SimpleNamespace(
            id=42, user_id=7, status="SUCCEEDED", error_summary=None,
            finished_at=None, worker_id=None, worker_claimed_at=None,
        )
        attempt = SimpleNamespace(
            id=12, user_id=7, execution_run_id=42, state="REBUILD_INITIATED",
            worker_id=None, worker_claimed_at=None,
            environment={"required_capabilities": ["cuda"], "required_gpu_units": 1},
            failure_summary=None, finished_at=None,
        )
        runs_query = MagicMock()
        runs_query.filter.return_value.with_for_update.return_value.all.return_value = []
        attempts_query = MagicMock()
        attempts_query.filter.return_value.with_for_update.return_value.all.return_value = [attempt]
        execution_query = MagicMock()
        execution_query.filter.return_value.with_for_update.return_value.first.return_value = terminal_run
        db = MagicMock()
        db.query.side_effect = [runs_query, attempts_query, execution_query]

        cancelled = cancel_jobs_without_capable_node(
            db, [{"capabilities": {"python"}, "gpu_units": 1}]
        )

        self.assertEqual(cancelled, 1)
        self.assertEqual(attempt.state, "REBUILD_CANCELLED")
        self.assertEqual(terminal_run.status, "SUCCEEDED")
        self.assertEqual(terminal_run.error_summary, None)
        self.assertEqual(terminal_run.finished_at, None)
        db.add.assert_called_once()
        db.commit.assert_called_once()
        db.rollback.assert_not_called()

    def test_unconfigured_node_inventory_skips_cancellation_without_database_access(self):
        db = MagicMock()

        self.assertEqual(cancel_jobs_without_capable_node(db, None), 0)
        db.query.assert_not_called()
        db.commit.assert_not_called()
        db.rollback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
