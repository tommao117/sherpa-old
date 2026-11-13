from __future__ import annotations

import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from sherpa.agent import AgentError, BackendResponse, Config, Request, Router
from sherpa.cli import main
from sherpa.platform.exploration import EvidenceTools, bounded_view
from sherpa.runtime.calls import Calls, RunStopped
from sherpa.runtime.explore import Investigation, POLICY, explore_environment
from sherpa.runtime.models import Task
from sherpa.runtime.store import Store, read_json
import tomllib


class ScenarioBackend:
    capabilities = frozenset({"json"})
    version = "fake-v1"

    def __init__(self, actions):
        self.actions, self.requests = list(actions), []

    def complete(self, model_id, request, context):
        self.requests.append((model_id, request, context))
        action = self.actions.pop(0)
        if isinstance(action, Exception):
            raise action
        result = action(request, context) if callable(action) else action
        return BackendResponse(json.dumps(result), model_id, {"input_tokens": 10, "output_tokens": 10}, 0.001)


def tool(name, **parameters):
    return {"action": "tool", "tool": name, "parameters": parameters, "reason": "collect evidence"}


def observation(request, name="read"):
    return next(e for e in request.context["observations"] if e["tool"] == name)


def route(evidence_id, quote="native CPU runtime", device="cpu"):
    return {"name": "candidate", "runtime": "native runner", "framework_source": "external source unresolved",
            "devices": [device], "python_requirement": "not required", "prerequisites": ["obtain runtime"],
            "steps": ["inspect upstream runtime requirements before installation"],
            "verification": ["start service and verify a real request later"],
            "evidence": [{"id": evidence_id, "quote": quote}], "limitations": ["runtime not tested"]}


def finish(request, context):
    return {"action": "finish", "summary": "weights plus documentation", "routes": [route(observation(request)["id"])],
            "selected_route": None, "unknowns": ["runtime code is external; no compatibility test performed"]}


class ExplorationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sherpa-explore-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input"
        self.source.mkdir()
        (self.source / "README.md").write_text("Weights use a native CPU runtime, no Python.\nUse GGUF with an appropriate framework.\n")
        (self.source / "model.gguf").write_bytes(b"do-not-read-weights")
        self.store = Store(self.root / "data")
        self.facts = {"host": {"system": "Linux"}, "gpus": [], "gpu_status": "unknown", "topology": {}, "tools": {}}

    def router(self, actions):
        backend = ScenarioBackend(actions)
        data = {"version": 1, "backends": {"fake": {"type": "fake"}},
                "pools": {"daily": ["small"], "expert": ["large", "large2"]},
                "models": {n: {"backend": "fake", "model_id": n} for n in ("small", "large", "large2")}}
        return Router(Config(data), {"fake": lambda cfg: backend}), backend

    def execute(self, actions, **kwargs):
        router, backend = self.router(actions)
        with patch("sherpa.platform.exploration.machine.probe", return_value=self.facts):
            run, proposal = explore_environment(self.store, self.source, router=router, allow_model_calls=True,
                allow_unpriced=True, model_budget=1, seconds=30, **kwargs)
        return run, proposal, backend

    def test_native_weight_directory_is_not_assumed_python(self):
        run, proposal, backend = self.execute([tool("list"), tool("read", path="README.md"), tool("machine"), finish], device="gpu")
        self.assertEqual(run.state()["status"], "completed", run.state()["stop_reason"])
        self.assertEqual(proposal["readiness"], "unverified")
        self.assertFalse(proposal["installed"])
        self.assertIsNone(proposal["plan"]["selected_route"])
        self.assertEqual(proposal["plan"]["routes"][0]["python_requirement"], "not required")
        self.assertTrue(all(model == "small" for model, _, _ in backend.requests))
        self.assertIsNone(run.state()["baseline"])
        calls = run.records("calls")
        for call in calls:
            self.assertTrue((run.path / call["input"]).is_file())
            self.assertTrue((run.path / call["output"]).is_file())
        self.assertEqual(read_json(run.path / "environment/proposal.json"), proposal)
        self.assertFalse((self.source / ".git").exists())
        self.assertFalse((self.store.root / "envs").exists())

    def test_expert_is_bounded_and_returns_to_daily(self):
        def delegate(request, context):
            return {"action": "delegate", "question": "Explain the runtime distinction", "reason": "need bounded help",
                    "completion_criteria": "explain only the quoted runtime requirements",
                    "evidence_ids": [observation(request)["id"]]}
        def expert(request, context):
            self.assertEqual(request.role, "expert")
            self.assertIsNotNone(request.caused_by)
            self.assertTrue(context.deadline <= time.monotonic() + 120)
            entry = request.context["evidence"][0]
            return {"advice": "native runtime does not require Python", "evidence": [{"id": entry["id"], "quote": "native CPU runtime"}],
                    "unknowns": ["runtime binaries not inspected"]}
        run, proposal, backend = self.execute([tool("read", path="README.md"), tool("machine"), delegate, expert, finish])
        self.assertIsNotNone(proposal, run.state()["stop_reason"])
        self.assertEqual([model for model, _, _ in backend.requests], ["small", "small", "small", "large", "small"])
        experts = [c for c in run.records("calls") if c.get("role") == "expert"]
        self.assertEqual(len(experts), 1)
        self.assertTrue(experts[0]["large_model_intervention"])
        self.assertTrue(backend.requests[-1][1].context["expert_advice"])
        self.assertIn("expert_returned_to_daily", [e["event"] for e in run.events()])

    def test_failed_tool_is_returned_to_daily_not_auto_expert(self):
        def after_failure(request, context):
            self.assertEqual(request.context["history"][-1]["result"]["status"], "failed")
            return tool("read", path="README.md")
        run, proposal, backend = self.execute([tool("read", path="missing.md"), after_failure, tool("machine"), finish])
        self.assertIsNotNone(proposal, run.state()["stop_reason"])
        self.assertEqual(sum(c["status"] == "failed" for c in run.records("calls")), 1)
        self.assertTrue(all(model == "small" for model, _, _ in backend.requests))

    def test_expert_failure_returns_control_without_creating_evidence(self):
        def delegate(request, context):
            return {"action": "delegate", "question": "Explain", "reason": "need help", "completion_criteria": "bounded explanation",
                    "evidence_ids": [observation(request)["id"]]}
        run, proposal, backend = self.execute([tool("read", path="README.md"), tool("machine"), delegate,
                                              AgentError("unavailable", "down"), AgentError("unavailable", "down"), finish])
        self.assertIsNotNone(proposal, run.state()["stop_reason"])
        self.assertEqual(backend.requests[-1][1].role, "daily")
        self.assertEqual(backend.requests[-1][1].context["expert_advice"][0]["status"], "failed")
        self.assertTrue(all(e["tool"] in {"read", "machine"} for e in proposal["evidence"]))

    def test_fabricated_citation_rejected_then_corrected_by_daily(self):
        def bad(request, context):
            value = finish(request, context)
            value["routes"][0]["evidence"][0]["quote"] = "This model supports all GPUs"
            return value
        run, proposal, backend = self.execute([tool("read", path="README.md"), tool("machine"), bad, finish])
        self.assertIsNotNone(proposal, run.state()["stop_reason"])
        failed = [c for c in run.records("calls") if c["status"] == "failed"]
        self.assertEqual(failed[0]["error_category"], "invalid_response")
        self.assertEqual(backend.requests[-1][1].context["history"][-1]["status"], "invalid_decision")

    def test_cpu_selection_cannot_satisfy_gpu_request(self):
        def bad(request, context):
            value = finish(request, context)
            value["selected_route"] = "candidate"
            return value
        run, proposal, _ = self.execute([tool("read", path="README.md"), tool("machine"), bad,
                                         {"action": "stop", "reason": "GPU runtime unresolved", "unknowns": ["GPU path missing"]}], device="gpu")
        self.assertIsNone(proposal)
        self.assertEqual(run.state()["result_status"], "blocked")
        self.assertTrue(any(c.get("error_category") == "invalid_response" for c in run.records("calls")))

    def test_native_and_python_fixtures_use_same_tools(self):
        (self.source / "README.md").write_text("Use Python 3.11 with the supplied project.\n")
        (self.source / "pyproject.toml").write_text('[project]\nrequires-python=">=3.11"\ndependencies=["torch==2.0.0"]\n')
        def python_finish(request, context):
            e = observation(request)
            r = route(e["id"], "Python 3.11")
            r.update(runtime="Python program", python_requirement=">=3.11")
            return {"action": "finish", "summary": "Python source project", "routes": [r], "selected_route": None, "unknowns": ["not installed"]}
        run, proposal, _ = self.execute([tool("declarations", path="pyproject.toml"), tool("read", path="README.md"), tool("machine"), python_finish])
        self.assertIsNotNone(proposal, run.state()["stop_reason"])
        self.assertEqual(proposal["plan"]["routes"][0]["runtime"], "Python program")

    def test_no_implicit_model_consent_or_zero_budget(self):
        for kwargs in ({}, {"allow_model_calls": True}, {"allow_model_calls": True, "allow_unpriced": True}):
            with self.subTest(kwargs=kwargs), patch("subprocess.Popen") as popen, self.assertRaises(ValueError):
                explore_environment(self.store, self.source, **kwargs)
            popen.assert_not_called()
        self.assertEqual(self.store.runs(), [])

    def test_round_budget_and_partial_evidence(self):
        run, proposal, _ = self.execute([tool("read", path="README.md")], max_rounds=1)
        self.assertIsNone(proposal)
        self.assertEqual(run.state()["status"], "budget_exhausted")
        self.assertEqual(len(run.records("calls")), 2)
        self.assertFalse((run.path / "environment/proposal.json").exists())

    def test_cancellation_stops_before_next_tool(self):
        def cancel(request, context):
            run = self.store.open_run(self.store.runs()[-1]["id"])
            run.cancel("test stop")
            return tool("read", path="README.md")
        run, proposal, backend = self.execute([cancel])
        self.assertIsNone(proposal)
        self.assertEqual(run.state()["status"], "cancelled")
        self.assertEqual(len(backend.requests), 1)
        self.assertTrue(all(c["kind"] == "agent" for c in run.records("calls")))

    def test_context_omits_old_bodies_and_recall_brings_them_back(self):
        task = Task("test", {"kind": "local", "path": str(self.source)}, {"id": "unknown"}, resources={"gpus": [], "device": "unspecified"},
                    budget={"seconds": 30, "max_calls": 20, "model_cost_usd": 1})
        run = self.store.create_run(task, "environment")
        router, _ = self.router([])
        investigation = Investigation(run, task, router, tomllib.loads(POLICY.read_text()))
        for i in range(5):
            investigation.evidence[f"id{i}"] = {"id": f"id{i}", "tool": "read", "parameters": {"path": "README.md"},
                                                "view": {"data": "x" * 10000, "truncated": False}}
        context = investigation.context(1)
        self.assertLessEqual(len(json.dumps(context, ensure_ascii=False).encode()), 28000)
        self.assertNotIn("id0", investigation.seen)
        with self.assertRaises(ValueError):
            investigation.citations([{"id": "id0", "quote": "xxxxx"}])
        investigation.focus = "id0"
        next_context = investigation.context(2)
        self.assertIn("id0", [e["id"] for e in next_context["observations"]])
        self.assertIn("id0", investigation.seen)

    def test_tool_paths_and_execution_boundaries(self):
        outside = self.root / "secret.md"
        outside.write_text("do not expose")
        (self.source / "link.md").symlink_to(outside)
        (self.source / "linked-dir").symlink_to(self.root, target_is_directory=True)
        os.mkfifo(self.source / "fifo.txt")
        tools = EvidenceTools(self.source, self.root, lambda: None)
        self.addCleanup(tools.close)
        with patch("subprocess.Popen", side_effect=AssertionError("no execution from read tools")):
            for path in ("../secret.md", str(outside), "link.md", "linked-dir/secret.md", "fifo.txt", ".env", "model.gguf"):
                with self.subTest(path=path), self.assertRaises((ValueError, OSError)):
                    tools.execute("read", {"path": path})
            with self.assertRaises(ValueError):
                tools.execute("shell", {"command": "touch /tmp/unsafe"})
            result = tools.execute("read", {"path": "README.md", "lines": 1})
        self.assertEqual(result["next_line"], 2)
        self.assertEqual(result["lines"][0]["line"], 1)
        self.assertTrue(result["sha256"])

    def test_binary_metadata_allowed_but_never_read(self):
        tools = EvidenceTools(self.source, self.root, lambda: None)
        self.addCleanup(tools.close)
        with patch.object(tools, "text", side_effect=AssertionError("metadata only")):
            listed = tools.execute("list", {})
        self.assertIn("model.gguf", [e["name"] for e in listed["entries"]])
        self.assertTrue(bounded_view({"long": "x" * 20000})["truncated"])

    def test_machine_tool_explains_path_lookup_limitations(self):
        tools = EvidenceTools(self.source, self.root, lambda: None)
        self.addCleanup(tools.close)
        facts = {**self.facts, "tools": {"nvcc": None, "git": "/usr/bin/git"}}
        with patch("sherpa.platform.exploration.machine.probe", return_value=facts):
            result = tools.execute("machine", {})
        self.assertFalse(result["tools"]["nvcc"])
        self.assertTrue(result["tools"]["git"])
        self.assertIn("not proof of not installed", result["tool_detection"]["false_means"])
        self.assertIn("not validated", result["tool_detection"]["true_means"])
        self.assertFalse(result["cuda_runtime_verified"])

    def test_expert_deadline_is_whole_request_not_per_candidate(self):
        router, backend = self.router([AgentError("unavailable", "down"), {"answer": "unused"}])
        task = Task("test", {"kind": "local", "path": str(self.source)}, {"id": "mock"}, budget={"seconds": 30, "max_calls": 10, "model_cost_usd": 1})
        calls = Calls(self.store.create_run(task), seconds=30)
        request = Request("expert", "bounded help", "help", {}, {}, intervention_reason="need help", completion_criteria="one answer")
        with self.assertRaises(AgentError) as error:
            router.call(request, calls, lambda d: None, allow_unpriced=True, request_timeout_s=5, max_attempts=1)
        self.assertEqual(error.exception.category, "attempt_limit")
        self.assertEqual(len(backend.requests), 1)
        self.assertLessEqual(backend.requests[0][2].deadline, time.monotonic() + 5)

    def test_late_expert_result_rejected_but_cost_kept(self):
        def late(request, context):
            time.sleep(0.04)
            return {"answer": "late"}
        router, backend = self.router([late])
        task = Task("test", {"kind": "local", "path": str(self.source)}, {"id": "mock"}, budget={"seconds": 30, "max_calls": 10, "model_cost_usd": 1})
        run = self.store.create_run(task)
        calls = Calls(run, seconds=30)
        with self.assertRaises(AgentError) as error:
            router.call(Request("daily", "test", "test", {}, {}), calls, lambda d: None,
                        allow_unpriced=True, request_timeout_s=0.02)
        self.assertEqual(error.exception.category, "request_timeout")
        self.assertEqual(calls.cost_summary()["known_model_cost_usd"], 0.001)

    def test_metadata_is_not_a_valid_content_quote(self):
        def bad(request, context):
            value = finish(request, context)
            value["routes"][0]["evidence"][0]["quote"] = "README.md"
            return value
        run, proposal, backend = self.execute([tool("read", path="README.md"), tool("machine"), bad, finish])
        self.assertIsNotNone(proposal, run.state()["stop_reason"])
        self.assertTrue(any(c.get("error_category") == "invalid_response" for c in run.records("calls")))

    def test_oversized_expert_advice_returns_to_daily_as_failure(self):
        def delegate(request, context):
            return {"action": "delegate", "question": "Explain", "reason": "need help", "completion_criteria": "bounded reply",
                    "evidence_ids": [observation(request)["id"]]}
        def huge(request, context):
            return {"advice": "a short explanation", "unknowns": ["x" * 1500] * 12,
                    "evidence": [{"id": request.context["evidence"][0]["id"], "quote": "native CPU runtime"}]}
        run, proposal, backend = self.execute([tool("read", path="README.md"), tool("machine"), delegate, huge, finish])
        self.assertIsNotNone(proposal, run.state()["stop_reason"])
        last = backend.requests[-1][1]
        self.assertEqual(last.role, "daily")
        self.assertEqual(last.context["expert_advice"][0]["category"], "invalid_response")
        self.assertLess(len(json.dumps(last.context).encode()), 28000)

    def test_expert_capability_failure_returns_to_daily(self):
        def delegate(request, context):
            return {"action": "delegate", "question": "Explain", "reason": "need help", "completion_criteria": "bounded reply",
                    "evidence_ids": [observation(request)["id"]]}
        run, proposal, backend = self.execute([tool("read", path="README.md"), tool("machine"), delegate,
                                              AgentError("capability", "expert adapter cannot answer"), finish])
        self.assertIsNotNone(proposal, run.state()["stop_reason"])
        self.assertEqual(backend.requests[-1][1].context["expert_advice"][0]["category"], "capability")
        self.assertIn("expert_returned_to_daily", [e["event"] for e in run.events()])

    def test_validation_time_is_part_of_request_deadline(self):
        router, backend = self.router([{"answer": "ok"}])
        task = Task("test", {"kind": "local", "path": str(self.source)}, {"id": "mock"}, budget={"seconds": 30, "max_calls": 10, "model_cost_usd": 1})
        run = self.store.create_run(task)
        calls = Calls(run, seconds=30)
        def slow_validation(result):
            time.sleep(0.04)
        with self.assertRaises(AgentError) as error:
            router.call(Request("daily", "test", "test", {}, {}), calls, slow_validation,
                        allow_unpriced=True, request_timeout_s=0.03)
        self.assertEqual(error.exception.category, "request_timeout")
        self.assertEqual(run.records("calls")[0]["status"], "failed")
        self.assertEqual(calls.cost_summary()["known_model_cost_usd"], 0.001)

    def test_unrecoverable_tool_error_closes_call_record(self):
        with patch.object(EvidenceTools, "execute", side_effect=RuntimeError("query cleanup failed")):
            run, proposal, backend = self.execute([tool("machine")])
        self.assertIsNone(proposal)
        self.assertEqual(run.state()["status"], "failed")
        self.assertEqual(run.records("calls")[-1]["status"], "failed")
        self.assertIsNone(run.state()["active_call"])

    def test_cli_opt_in_is_explicit(self):
        with patch("subprocess.Popen") as popen, redirect_stdout(io.StringIO()):
            code = main(["--data-root", str(self.store.root), "env", "explore", "--framework", str(self.source),
                         "--model", str(self.source), "--model-budget", "1"])
        self.assertEqual(code, 1)
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
