from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from sherpa.runtime.calls import Calls
from sherpa.runtime.simulation import demo, SimulationEngine, ScriptedAgent
from sherpa.runtime.models import Task, ToolResult
from sherpa.runtime.store import BusyError, Store, atomic_json, read_json
from sherpa.runtime import workspace

PROJECT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sherpa-runtime-test-")
        self.root = Path(self.temporary.name)
        self.store = Store(self.root / "data")
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "main.py").write_text("VALUE = 1\n")
        self.task = Task("test", {"kind": "local", "path": str(self.source)}, {"id": "synthetic"})
        self.processes = []

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
        self.temporary.cleanup()

    def test_task_explicit_defaults_and_round_trip(self):
        data = self.task.snapshot()
        self.assertEqual(data["objective"]["metric"], "throughput")
        self.assertEqual(data["resources"]["gpus"], [])
        self.assertTrue(data["contract"]["preserve_behavior"])
        path = self.root / "task.json"
        atomic_json(path, data)
        self.assertEqual(Task.load(path), self.task)

    def test_invalid_budgets_rejected(self):
        for value in (float("nan"), float("inf"), -1, 0, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Task("test", self.task.target, self.task.model,
                     budget={"seconds": value, "max_calls": 2, "model_cost_usd": 0})

    def test_data_must_not_live_in_project(self):
        with self.assertRaises(ValueError):
            Store(PROJECT / "data")

    def test_identifiers_cannot_escape_store(self):
        for value in ("../source", "/tmp/x", "a/b", ".."):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.store.open_run(value)

    def test_run_types_and_reference_links(self):
        inner = self.store.create_run(self.task)
        outer = self.store.create_run(None, "reflection", sources=[inner.id], trigger="automatic")
        self.assertEqual(outer.state()["source_runs"], [inner.id])
        self.assertEqual(len(self.store.runs()), 2)
        with self.assertRaises(FileNotFoundError):
            self.store.create_run(None, "reflection", sources=["unknown"])

    def test_normal_demo_complete_chain(self):
        run, state = demo(self.store)
        self.assertEqual(state["status"], "completed", state.get("stop_reason"))
        self.assertTrue(state["version"]["snapshot"])
        self.assertTrue(state["version"]["content_id"])
        self.assertGreater(state["best"]["metric"], state["baseline"]["metric"])
        self.assertNotEqual(state["best"]["commit"], state["baseline"]["commit"])
        self.assertEqual(state["baseline"], read_json(run.path / "baseline/reference.json"))
        self.assertTrue(read_json(run.path / "deliverable/result.json")["synthetic"])
        calls = {call["id"]: call for call in run.records("calls")}
        exp = run.records("experiments")[0]
        self.assertEqual(exp["conclusion"], "accepted")
        decision = calls[exp["decision_call"]]
        context = read_json(run.path / decision["input"])
        self.assertEqual(context["baseline"]["commit"], state["baseline"]["commit"])
        self.assertIn("tools", context)
        self.assertIn("history", context)
        for call_id in exp["calls"]:
            self.assertEqual(calls[call_id]["experiment_id"], exp["id"])
        self.assertEqual(exp["measurement_contract"]["workload"]["concurrency"], 1)
        self.assertTrue(all(c.get("cleanup_complete", True) for c in calls.values()))

    def test_failure_is_evidence_not_automatic_escalation(self):
        run, state = demo(self.store, "failure")
        self.assertEqual(state["status"], "completed", state.get("stop_reason"))
        calls = run.records("calls")
        failed = next(c for c in calls if c["name"] == "fail")
        self.assertEqual(failed["status"], "failed")
        agents = [c for c in calls if c["kind"] == "agent"]
        contexts = [read_json(run.path / c["input"]) for c in agents]
        self.assertTrue(any(c["history"] and c["history"][-1]["status"] == "failed" for c in contexts))
        self.assertFalse(any(c["large_model_intervention"] for c in calls))

    def test_invalid_candidate_never_replaces_best(self):
        run, state = demo(self.store, "invalid")
        self.assertEqual(state["status"], "completed", state.get("stop_reason"))
        self.assertEqual(state["best"], state["baseline"])
        exp = run.records("experiments")[0]
        self.assertEqual(exp["conclusion"], "rejected")
        self.assertEqual(len(exp["calls"]), 1)

    def test_run_timeout_preserves_baseline_and_cleans_descendants(self):
        run, state = demo(self.store, "timeout", seconds=2)
        self.assertEqual(state["status"], "timed_out", state.get("stop_reason"))
        self.assertIsNotNone(state["baseline"])
        call = run.records("calls")[-1]
        self.assertEqual(call["status"], "timed_out")
        self.assertTrue(call["cleanup_complete"])
        self.assertTrue(Calls.group_gone(call["process"]["pid"]))

    def test_tool_timeout_is_not_itself_a_run_failure(self):
        run = self.store.create_run(self.task)
        calls = Calls(run)
        failed = calls.tool("inspect", {"delay": 2, "child": True}, timeout_s=0.25)
        self.assertEqual(failed.status, "timed_out")
        result = calls.tool("inspect", {})
        self.assertEqual(result.status, "succeeded")
        self.assertTrue(all(c["cleanup_complete"] for c in run.records("calls")))

    def test_call_budget_enforced(self):
        run, state = demo(self.store, seconds=20)
        calls = Calls(run, max_calls=1)
        from sherpa.runtime.calls import RunStopped
        calls.count = 1
        with self.assertRaises(RunStopped) as error:
            calls.check()
        self.assertEqual(error.exception.status, "budget_exhausted")

    def test_bad_tool_parameters_are_recorded(self):
        run = self.store.create_run(self.task)
        calls = Calls(run)
        result = calls.tool("inspect", {"shell": "do not execute"})
        self.assertEqual(result.status, "failed")
        self.assertIn("unknown tool parameters", result.output["error"])
        self.assertIsNone(run.records("calls")[0]["process"])
        self.assertEqual(calls.tool("inspect", {}).status, "succeeded")

    def start_slow_cli(self):
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        process = subprocess.Popen([sys.executable, "-B", "-m", "sherpa.cli", "--data-root", str(self.store.root),
                                    "demo", "--scenario", "slow", "--delay", "20"], cwd=PROJECT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.processes.append(process)
        run = None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            for record in self.store.runs():
                run = self.store.open_run(record["id"])
                children = list((run.path / "calls").glob("*/child.json"))
                if children:
                    return process, run
            if process.poll() is not None:
                self.fail(f"CLI exited early: {process.communicate()}")
            time.sleep(0.05)
        self.fail("mock tool did not start")

    def test_cli_cancel_is_observable_and_preserves_partial_trace(self):
        process, run = self.start_slow_cli()
        self.assertEqual(run.status()["status"], "running")
        self.assertTrue(run.cancel("test cancellation"))
        stdout, stderr = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 2, stderr)
        self.assertIn("run_id=", stdout)
        self.assertEqual(run.state()["status"], "cancelled")
        call = run.records("calls")[-1]
        self.assertEqual(call["status"], "cancelled")
        self.assertTrue(Calls.group_gone(call["process"]["pid"]))
        self.assertTrue((run.path / "calls" / call["id"] / "progress.json").exists())
        self.assertIn("cancel_requested", [e["event"] for e in run.events()])
        self.assertFalse(run.cancel())

    def test_launcher_death_stops_its_process_group(self):
        process, run = self.start_slow_cli()
        call = run.records("calls")[-1]
        process.kill()
        process.communicate(timeout=3)
        deadline = time.monotonic() + 3
        while not Calls.group_gone(call["process"]["pid"]) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(Calls.group_gone(call["process"]["pid"]))
        self.assertTrue(run.status()["interrupted"])

    def test_snapshot_recovery_from_event_journal(self):
        run = self.store.create_run(self.task)
        run.update("phase_changed", status="running", phase="baseline")
        (run.path / "run.json").unlink()
        run.recover_snapshots()
        self.assertEqual(run.state()["phase"], "baseline")

    def test_completed_run_cannot_restart(self):
        run = self.store.create_run(self.task)
        run.update("run_finished", status="completed")
        with self.assertRaises(ValueError):
            run.update("run_started", status="running")

    def test_plain_directory_gets_private_git_snapshot(self):
        (self.source / ".env").write_text("SECRET=not-for-snapshot")
        (self.source / "weights.safetensors").write_bytes(b"not-for-git")
        (self.source / "link").symlink_to(self.source / "main.py")
        result = workspace.prepare(self.task.target, self.store.root / "workspace/framework")
        path = Path(result["path"])
        self.assertFalse((self.source / ".git").exists())
        self.assertTrue((path / ".git").is_dir())
        self.assertEqual(workspace.git(path, "rev-parse", "HEAD"), result["commit"])
        self.assertFalse((path / ".env").exists())
        self.assertFalse((path / "weights.safetensors").exists())
        self.assertFalse((path / "link").exists())

    def test_dirty_git_source_preserved_without_modifying_source(self):
        workspace.git(self.source, "init")
        before = workspace.commit_all(self.source, "initial fixture")
        (self.source / "main.py").write_text("VALUE = 3\n")
        (self.source / "extra.py").write_text("NEW = True\n")
        source_status = workspace.git(self.source, "status", "--porcelain")
        result = workspace.prepare(self.task.target, self.store.root / "workspace/framework")
        path = Path(result["path"])
        self.assertTrue(result["dirty_source"])
        self.assertEqual(result["source_commit"], before)
        self.assertNotEqual(result["commit"], before)
        self.assertEqual(workspace.git(self.source, "status", "--porcelain"), source_status)
        self.assertEqual(workspace.git(self.source, "rev-parse", "HEAD"), before)
        self.assertEqual((path / "main.py").read_text(), "VALUE = 3\n")
        self.assertEqual(workspace.git(path, "status", "--porcelain"), "")

    def test_git_source_revision_pinned(self):
        workspace.git(self.source, "init")
        before = workspace.commit_all(self.source, "initial fixture")
        (self.source / "main.py").write_text("VALUE = 4\n")
        workspace.commit_all(self.source, "newer fixture")
        target = {"kind": "git", "url": str(self.source), "revision": before}
        result = workspace.prepare(target, self.store.root / "workspace/framework")
        self.assertEqual(result["commit"], before)
        self.assertEqual((Path(result["path"]) / "main.py").read_text(), "VALUE = 1\n")

    def test_framework_snapshot_contains_actual_uncommitted_code(self):
        project = self.root / "project"
        project.mkdir()
        (project / "module.py").write_text("VALUE = 1\n")
        workspace.git(project, "init")
        workspace.commit_all(project, "initial")
        (project / "module.py").write_text("VALUE = 2\n")
        version = workspace.capture_framework(project, self.root / "snapshot")
        self.assertTrue(version["dirty"])
        self.assertEqual((self.root / "snapshot/module.py").read_text(), "VALUE = 2\n")

    def test_proposal_requires_reflection_and_human_review(self):
        inner = self.store.create_run(self.task)
        outer = self.store.create_run(None, "reflection", sources=[inner.id])
        plan = {"problem": "mock parser slow", "change": "batch processing", "scope": "mock data only",
                "verification": "compare output and measured cost"}
        with self.assertRaises(ValueError):
            self.store.propose(inner.id, plan)
        proposal = self.store.propose(outer.id, plan)
        record = read_json(self.store.root / "proposals" / proposal / "proposal.json")
        self.assertEqual(record["state"], "proposed")
        with self.assertRaises(ValueError):
            self.store.review(proposal, "", True, "missing reviewer")
        reviewed = self.store.review(proposal, "human", True, "approve this bounded plan")
        self.assertEqual(reviewed["state"], "approved")
        with self.assertRaises(ValueError):
            self.store.review(proposal, "human", False, "already reviewed")

    def test_framework_update_boundary_is_exclusive(self):
        outer = self.store.create_run(None, "reflection")
        plan = {key: "mock" for key in ("problem", "change", "scope", "verification")}
        with self.store.exclusive():
            with self.assertRaises(BusyError):
                self.store.propose(outer.id, plan)

    def test_nested_data_root_rejected_before_any_write(self):
        store = Store(self.source / "nested-data")
        with self.assertRaises(ValueError):
            store.create_run(self.task)
        self.assertEqual([p.name for p in self.source.iterdir()], ["main.py"])
        with self.assertRaises(ValueError):
            workspace.prepare(self.task.target, self.source / "new/workspace")
        self.assertFalse((self.source / "new").exists())

    def test_gitignore_cannot_hide_captured_code_from_commit(self):
        (self.source / ".gitignore").write_text("main.py\n")
        result = workspace.prepare(self.task.target, self.store.root / "workspace/framework")
        committed = workspace.git(Path(result["path"]), "show", result["commit"] + ":main.py")
        self.assertEqual(committed, "VALUE = 1")

    def test_torn_event_tail_does_not_corrupt_next_append(self):
        run = self.store.create_run(self.task)
        with (run.path / "events.jsonl").open("ab") as handle:
            handle.write(b'{"event":"unfinished')
        run.cancel("after interrupted append")
        run.recover_snapshots()
        self.assertEqual([event["event"] for event in run.events()], ["run_created", "cancel_requested"])

    def test_baseline_cancellation_is_not_ordinary_failure(self):
        from unittest.mock import patch
        run = self.store.create_run(self.task)
        engine = SimulationEngine(run, self.task, ScriptedAgent("normal", 0))
        def cancel_baseline(*args, **kwargs):
            run.cancel("cancel during baseline")
            return ToolResult("cancelled", {}, "mock-cancelled-call")
        with patch.object(engine.calls, "tool", side_effect=cancel_baseline):
            result = engine.execute()
        self.assertEqual(result["status"], "cancelled", result["stop_reason"])

    def test_git_preparation_honors_cancellation_and_cleans_group(self):
        from unittest.mock import patch
        from sherpa.runtime.calls import RunStopped
        real_popen = subprocess.Popen
        captured = []
        def spawn(*args, **kwargs):
            captured.append(real_popen(*args, **kwargs))
            return captured[-1]
        checks = 0
        def stop():
            nonlocal checks
            checks += 1
            if checks > 1:
                raise RunStopped("cancelled", "cancel git preparation")
        with patch.object(workspace.subprocess, "Popen", side_effect=spawn):
            with workspace.interruptible(stop, scratch=self.root), self.assertRaises(RunStopped):
                workspace.git(self.source, "init")
        self.assertEqual(len(captured), 1)
        self.assertTrue(Calls.group_gone(captured[0].pid))

    def test_large_or_remote_repository_is_not_fetched_by_prototype(self):
        with self.assertRaises(ValueError):
            workspace.prepare({"kind": "git", "url": "https://example.invalid/repo.git"},
                              self.root / "not-created/framework")
        self.assertFalse((self.root / "not-created").exists())

    def test_cli_status_events_and_trace(self):
        run, state = demo(self.store)
        for command in ("status", "events", "trace"):
            result = subprocess.run([sys.executable, "-B", "-m", "sherpa.cli", "--data-root", str(self.store.root),
                                     command, run.id], cwd=PROJECT, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout))


if __name__ == "__main__":
    unittest.main()
