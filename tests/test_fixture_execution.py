from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from sherpa.cli import main
from sherpa.runtime.calls import Calls, RunStopped
from sherpa.runtime.fixture_execution import FixtureExecutor, fixture_demo, prepare_fixture, small_file
from sherpa.runtime.store import Store, atomic_json, digest, process_identity, read_json

PROJECT = Path(__file__).resolve().parents[1]


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sherpa-e2-")
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / "data")

    def executor(self, *, seconds=20, max_calls=8, step_seconds=2, mutate=None):
        task, prepared = prepare_fixture(self.store, seconds=seconds, max_calls=max_calls, step_seconds=step_seconds)
        if mutate:
            value = task.snapshot()
            mutate(value)
            from sherpa.runtime.models import Task
            task = Task(**value)
        run = self.store.create_run(task, kind="environment", sources=[prepared])
        run.update("test_started", status="running", scope="fixture_execution", synthetic=True, owner=process_identity(os.getpid()))
        return FixtureExecutor(run, allow_fixtures=True)

    def build(self, executor, **fault):
        return executor.execute({"kind": "build"}, fault=fault)

    def infer(self, executor, build, **fault):
        return executor.execute({"kind": "run_once", "build_call": build.call_id}, fault=fault)

    def assert_clean(self, executor):
        for rec in executor.run.records("calls"):
            self.assertIsNotNone(rec.get("output_hash"))
            if rec.get("process"):
                self.assertTrue(rec["cleanup_complete"])
                self.assertTrue(Calls.group_gone(rec["process"]["pid"]))

    def test_complete_pipeline_links_calls_artifacts_and_contract(self):
        run = fixture_demo(self.store)
        state = run.state()
        self.assertEqual(state["status"], "completed", state["stop_reason"])
        self.assertEqual(state["result_status"], "fixture_passed")
        self.assertEqual(state["readiness"], "unverified")
        self.assertFalse(state["installed"])
        records = run.records("calls")
        self.assertEqual([r["name"] for r in records], ["fixture.probe", "fixture.build", "fixture.run_once"])
        self.assertEqual(records[-1]["caused_by"], records[1]["id"])
        for rec in records:
            value = read_json(run.path / rec["output"])
            self.assertTrue(rec["synthetic"])
            self.assertEqual(value["verification"]["status"], "pass")
            self.assertEqual(value["process"]["child"]["child_returncode"], 0)
            self.assertEqual(value["task_hash"], digest(state["task"]))
            self.assertEqual(rec["output_hash"], digest(value))
            self.assertTrue(Calls.group_gone(rec["process"]["pid"]))
        self.assertEqual(state["cost"]["calls"], 3)
        self.assertEqual(state["cost"]["model_cost_usd"], 0)

    def test_process_success_and_verification_failure_are_separate(self):
        e = self.executor()
        built = self.build(e)
        bad = self.infer(e, built, wrong_output=True)
        self.assertEqual(bad.status, "succeeded")
        self.assertEqual(bad.output["verification"]["status"], "fail")
        self.assert_clean(e)

    def test_missing_build_artifact_and_malformed_response(self):
        e = self.executor()
        missing = self.build(e, missing_artifact=True)
        self.assertEqual(missing.status, "succeeded")
        self.assertEqual(missing.output["verification"]["status"], "fail")
        rejected = self.infer(e, missing)
        self.assertFalse(rejected.output["executed"])
        built = self.build(e)
        malformed = self.infer(e, built, malformed=True)
        self.assertEqual(malformed.status, "succeeded")
        self.assertEqual(malformed.output["verification"]["status"], "inconclusive")
        self.assert_clean(e)

    def test_nonzero_exit_keeps_actual_exit_and_logs(self):
        e = self.executor()
        result = self.build(e, exit_code=7, stderr_bytes=64)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.output["process"]["child"]["child_returncode"], 7)
        self.assertEqual(result.output["process"]["reason"], "nonzero_exit")
        self.assertEqual((e.run.path / "calls" / result.call_id / "stderr.log").stat().st_size, 64)
        self.assert_clean(e)

    def test_changed_build_artifact_rejected_before_spawn(self):
        e = self.executor()
        built = self.build(e)
        p = e.run.path / built.output["artifact"]["path"]
        p.write_text('{"synthetic": true}')
        with patch("sherpa.runtime.fixture_execution.subprocess.Popen", side_effect=AssertionError("must not launch")):
            result = self.infer(e, built)
        self.assertEqual(result.status, "failed")
        self.assertIn("artifact changed", result.output["error"])
        self.assertFalse(result.output["executed"])

    def test_forged_or_cross_run_build_call_is_not_reused(self):
        first = self.executor()
        built = self.build(first)
        second = self.executor()
        result = self.infer(second, built)
        self.assertEqual(result.status, "failed")
        self.assertFalse(result.output["executed"])
        record = first.run.records("calls")[-1]
        output = read_json(first.run.path / record["output"])
        output["verification"]["status"] = "fail"
        atomic_json(first.run.path / record["output"], output)
        result = self.infer(first, built)
        self.assertIn("identity mismatch", result.output["error"])

    def test_task_change_and_frozen_input_change_are_rejected(self):
        e = self.executor()
        state = e.run.state()
        state["task"]["workload"]["request"]["input"] = "changed"
        atomic_json(e.run.path / "run.json", state)
        result = self.build(e)
        self.assertIn("task changed", result.output["error"])
        self.assertFalse(result.output["executed"])
        e = self.executor()
        source = Path(e.task["target"]["path"]) / "source.txt"
        original_inputs = e._inputs
        count = 0
        def check_inputs():
            nonlocal count
            count += 1
            if count == 2:
                source.write_text("mutated by another writer")
            original_inputs()
        with patch.object(e, "_inputs", side_effect=check_inputs):
            result = self.build(e)
        self.assertEqual(result.status, "failed")
        self.assertIn("frozen fixture input changed", result.output["error"])
        self.assert_clean(e)

    def test_actual_e1_approval_mismatch_cannot_launch(self):
        e = self.executor()
        e.approval["task_hash"] = "0" * 64
        with patch("sherpa.runtime.fixture_execution.subprocess.Popen", side_effect=AssertionError("must not launch")):
            result = self.build(e)
        self.assertEqual(result.status, "failed")
        self.assertIn("approval_mismatch", result.output["error"])
        self.assertFalse(result.output["executed"])

    def test_production_adapter_gpu_and_arbitrary_commands_are_rejected(self):
        for mutate in (lambda t: t["contract"]["execution"]["adapter"].update(name="llama.cpp"),
                       lambda t: t["resources"].update(device="gpu", gpus=["GPU-real"])):
            e = self.executor(mutate=mutate)
            result = self.build(e)
            self.assertEqual(result.status, "failed")
            self.assertFalse(result.output["executed"])
        e = self.executor()
        result = e.execute({"kind": "build", "argv": ["anything"]})
        self.assertFalse(result.output["executed"])
        result = self.build(e, command="anything")
        self.assertFalse(result.output["executed"])
        with self.assertRaises(ValueError):
            FixtureExecutor(e.run)

    def test_log_limits_cover_both_streams_exact_boundary_and_fast_exit(self):
        cap = 1024 * 1024
        for stdout, stderr, expected in ((cap, 0, "succeeded"), (0, cap, "succeeded"),
                                         (cap // 2, cap // 2, "succeeded"),
                                         (cap + 1, 0, "failed"), (0, cap + 1, "failed"),
                                         (600000, 600000, "failed")):
            with self.subTest(stdout=stdout, stderr=stderr):
                e = self.executor()
                result = self.build(e, stdout_bytes=stdout, stderr_bytes=stderr)
                self.assertEqual(result.status, expected, result.output)
                logs = e.run.path / "calls" / result.call_id
                self.assertLessEqual(sum((logs / name).stat().st_size for name in ("stdout.log", "stderr.log")), cap)
                if expected == "failed":
                    self.assertEqual(result.output["process"]["reason"], "output_limit")
                self.assert_clean(e)

    def test_live_output_limit_cleans_descendants(self):
        e = self.executor()
        result = self.build(e, stdout_bytes=2 * 1024 * 1024, delay=10, child=True)
        self.assertEqual(result.output["process"]["reason"], "output_limit")
        self.assert_clean(e)

    def test_local_timeout_returns_control_and_allows_new_attempt(self):
        e = self.executor(step_seconds=0.3)
        result = self.build(e, delay=10, child=True)
        self.assertEqual(result.status, "timed_out")
        self.assertEqual(result.output["process"]["reason"], "step_timeout")
        self.assertEqual(self.build(e).output["verification"]["status"], "pass")
        self.assert_clean(e)

    def test_budget_exhaustion_prevents_launch_but_does_not_cancel_last_allowed_call(self):
        e = self.executor(max_calls=1)
        result = self.build(e)
        self.assertEqual(result.output["verification"]["status"], "pass", result.output)
        with self.assertRaises(RunStopped) as error:
            self.build(e)
        self.assertEqual(error.exception.status, "budget_exhausted")
        self.assertEqual(len(e.run.records("calls")), 1)
        self.assert_clean(e)

    def test_total_timeout_is_recorded_and_propagated(self):
        e = self.executor(seconds=0.5, step_seconds=0.5)
        with self.assertRaises(RunStopped) as error:
            self.build(e, delay=10, child=True)
        self.assertEqual(error.exception.status, "timed_out")
        self.assertEqual(e.run.records("calls")[-1]["status"], "timed_out")
        self.assert_clean(e)

    def test_cancel_during_running_preserves_reason_and_cleans_group(self):
        e = self.executor()
        def cancel_when_ready():
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if list(e.run.path.glob("calls/*/ready.json")):
                    e.run.cancel("cancel from test")
                    return
                time.sleep(0.01)
        thread = threading.Thread(target=cancel_when_ready)
        thread.start()
        try:
            with self.assertRaises(RunStopped) as error:
                self.build(e, delay=10, child=True)
            self.assertEqual(error.exception.status, "cancelled")
            self.assertEqual(error.exception.reason, "cancel from test")
        finally:
            thread.join(timeout=6)
        self.assert_clean(e)

    def test_cancel_before_start_and_after_cleanup_never_accepts_success(self):
        e = self.executor()
        e.run.cancel("before start")
        with self.assertRaises(RunStopped):
            self.build(e)
        self.assertEqual(e.run.records("calls"), [])
        e = self.executor()
        cleanup = e.calls.cleanup
        def cancel_after(proc):
            result = cleanup(proc)
            e.run.cancel("during cleanup")
            return result
        with patch.object(e.calls, "cleanup", side_effect=cancel_after):
            with self.assertRaises(RunStopped) as error:
                self.build(e)
        self.assertEqual(error.exception.reason, "during cleanup")
        self.assertEqual(e.run.records("calls")[-1]["status"], "cancelled")
        self.assert_clean(e)

    def test_cleanup_failure_latches_and_blocks_new_executor(self):
        e = self.executor()
        cleanup = e.calls.cleanup
        def failed(proc):
            cleanup(proc)
            return False
        with patch.object(e.calls, "cleanup", side_effect=failed):
            with self.assertRaises(RunStopped) as error:
                self.build(e)
        self.assertEqual(error.exception.status, "cleanup_failed")
        self.assertEqual(e.run.records("calls")[-1]["status"], "cleanup_failed")
        with self.assertRaises(RunStopped):
            FixtureExecutor(e.run, allow_fixtures=True)
        with self.assertRaises(RunStopped), patch("subprocess.Popen", side_effect=AssertionError("must not launch")):
            self.build(e)
        self.assertEqual(len(e.run.records("calls")), 1)

    def test_reconstructing_executor_cannot_reset_calls_or_deadline(self):
        e = self.executor(max_calls=1)
        self.build(e)
        before = e.run.state()["cost"]
        with self.assertRaisesRegex(ValueError, "already claimed"):
            FixtureExecutor(e.run, allow_fixtures=True)
        self.assertEqual(e.run.state()["cost"], before)
        expired = self.executor(seconds=0.2, step_seconds=0.2)
        time.sleep(0.21)
        with self.assertRaisesRegex(ValueError, "already claimed"):
            FixtureExecutor(expired.run, allow_fixtures=True)
        with self.assertRaises(RunStopped) as error:
            self.build(expired)
        self.assertEqual(error.exception.status, "timed_out")

    def test_cleanup_exception_and_interrupt_are_latched_with_fallback_cleanup(self):
        for exception in (OSError("cleanup failure"), KeyboardInterrupt()):
            with self.subTest(exception=type(exception).__name__):
                e = self.executor(step_seconds=0.3)
                with patch.object(e.calls, "cleanup", side_effect=exception):
                    with self.assertRaises(RunStopped) as error:
                        self.build(e, delay=10, child=True)
                self.assertEqual(error.exception.status, "cleanup_failed")
                record = e.run.records("calls")[-1]
                self.assertFalse(record["cleanup_complete"])
                self.assertTrue(Calls.group_gone(record["process"]["pid"]))
                process = read_json(e.run.path / "calls" / record["id"] / "process.json")
                self.assertIn(type(exception).__name__, process["cleanup_error"])
                with self.assertRaises(RunStopped):
                    self.build(e)
                with self.assertRaises(RunStopped):
                    FixtureExecutor(e.run, allow_fixtures=True)

    def test_fast_worker_exit_does_not_leave_pipe_holding_descendant(self):
        for exit_code in (0, 7):
            with self.subTest(exit_code=exit_code):
                e = self.executor(step_seconds=1)
                result = self.build(e, child=True, exit_code=exit_code)
                self.assertEqual(result.status, "succeeded" if exit_code == 0 else "failed", result.output)
                self.assertNotEqual(result.output["process"]["reason"], "step_timeout")
                self.assert_clean(e)

    def test_worker_kill_also_terminates_its_own_descendant(self):
        e = self.executor(step_seconds=3)
        killed = []
        def kill_worker():
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                paths = list(e.run.path.glob("calls/*/ready.json"))
                if paths:
                    ready = read_json(paths[0])
                    os.kill(ready["pid"], 9)
                    killed.append(ready["pid"])
                    return
                time.sleep(0.01)
        thread = threading.Thread(target=kill_worker)
        thread.start()
        try:
            result = self.build(e, delay=10, child=True)
        finally:
            thread.join(timeout=6)
        self.assertTrue(killed)
        self.assertEqual(result.output["process"]["child"]["child_signal"], "SIGKILL")
        self.assertEqual(result.output["process"]["reason"], "nonzero_exit")
        self.assert_clean(e)

    def test_disk_budget_counts_previous_calls_and_fast_exit_output(self):
        def limit(task):
            task["contract"]["execution"]["limits"]["disk_mib"] = 1
        e = self.executor(mutate=limit)
        first = self.build(e, stdout_bytes=700000)
        self.assertEqual(first.output["verification"]["status"], "pass", first.output)
        second = self.build(e, stderr_bytes=400000)
        self.assertEqual(second.status, "failed", second.output)
        self.assertIn("disk_limit", json.dumps(second.output))
        with patch("subprocess.Popen", side_effect=AssertionError("must not launch")):
            third = self.build(e)
        self.assertFalse(third.output["executed"])
        self.assert_clean(e)

    def test_transient_status_rename_does_not_fail_disk_sampling(self):
        from sherpa.runtime.fixture_execution import disk_bytes
        root = Path(self.temp.name)
        (root / "kept").write_bytes(b"123")
        (root / "changing.tmp").write_bytes(b"temporary")
        original = Path.lstat
        def transient(path, *args, **kwargs):
            if path.name == "changing.tmp":
                raise FileNotFoundError(path)
            return original(path, *args, **kwargs)
        with patch.object(Path, "lstat", transient):
            self.assertEqual(disk_bytes(root), 3)

    def test_interrupt_during_verification_records_cancelled_call(self):
        e = self.executor()
        with patch.object(e, "_verify", side_effect=KeyboardInterrupt()):
            with self.assertRaises(RunStopped) as error:
                self.build(e)
        self.assertEqual(error.exception.status, "cancelled")
        self.assertEqual(e.run.records("calls")[-1]["status"], "cancelled")
        self.assert_clean(e)

    def test_frozen_input_change_during_verification_invalidates_result(self):
        e = self.executor()
        original = e._verify
        def mutate(kind, directory):
            result = original(kind, directory)
            (Path(e.task["target"]["path"]) / "source.txt").write_text("changed")
            return result
        with patch.object(e, "_verify", side_effect=mutate):
            result = self.build(e)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.output["verification"]["status"], "fail")
        self.assert_clean(e)

    def test_terminal_run_cannot_execute(self):
        e = self.executor()
        e.run.update("test_finished", status="completed")
        with self.assertRaises(ValueError):
            self.build(e)

    def test_no_target_git_probe_or_provider_paths_are_called(self):
        with ExitStack() as stack:
            for name in ("sherpa.agent.Router.call", "sherpa.platform.machine.probe", "sherpa.runtime.workspace.git", "sherpa.runtime.framework.prepare", "socket.socket"):
                stack.enter_context(patch(name, side_effect=AssertionError(name)))
            run = fixture_demo(self.store)
        self.assertEqual(run.state()["result_status"], "fixture_passed")

    def test_cli_normal_fixture_and_wrong_output_exit_codes(self):
        for scenario, expected in (("normal", 0), ("wrong-output", 2)):
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(["--data-root", str(self.store.root), "env", "test-executor", "--scenario", scenario])
            self.assertEqual(code, expected, out.getvalue())
            self.assertIn('"synthetic": true', out.getvalue())
            self.assertIn('"installed": false', out.getvalue())

    def test_fixture_path_reader_rejects_links_and_fifo(self):
        root = Path(self.temp.name)
        (root / "file").write_text("safe")
        (root / "link").symlink_to(root / "file")
        (root / "linked-dir").symlink_to(root, target_is_directory=True)
        os.mkfifo(root / "fifo")
        for p in (root / "link", root / "linked-dir" / "file", root / "fifo"):
            with self.subTest(path=p), self.assertRaises((ValueError, OSError)):
                small_file(p)

    def test_parent_death_kills_supervisor_worker_and_child(self):
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        proc = subprocess.Popen([sys.executable, "-B", "-m", "sherpa.cli", "--data-root", str(self.store.root),
                                 "env", "test-executor", "--scenario", "slow"], cwd=PROJECT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        run, child_record = None, None
        try:
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                for state in self.store.runs():
                    if state.get("scope") == "fixture_execution":
                        run = self.store.open_run(state["id"])
                        if list(run.path.glob("calls/*/child.json")):
                            child_record = run.records("calls")[-1]
                            break
                if child_record:
                    break
                if proc.poll() is not None:
                    self.fail(str(proc.communicate()))
                time.sleep(0.02)
            self.assertIsNotNone(child_record, "worker did not become ready")
            proc.kill()
            proc.communicate(timeout=4)
            deadline = time.monotonic() + 4
            while not Calls.group_gone(child_record["process"]["pid"]) and time.monotonic() < deadline:
                time.sleep(0.025)
            self.assertTrue(Calls.group_gone(child_record["process"]["pid"]))
            self.assertTrue(run.status()["interrupted"])
            self.assertEqual(run.records("calls")[-1]["status"], "running")
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.communicate(timeout=5)
            if child_record and not Calls.group_gone(child_record["process"]["pid"]):
                os.killpg(child_record["process"]["pid"], 9)


if __name__ == "__main__":
    unittest.main()
