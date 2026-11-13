from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from sherpa.agent import AgentError, BackendResponse, Config, Router
from sherpa.cli import main
from sherpa.runtime.calls import RunStopped
from sherpa.runtime.continuation import lineage
from sherpa.runtime.explore import explore_environment
from sherpa.runtime.store import Store, atomic_json, read_json
from test_explore import ScenarioBackend, finish, observation, tool


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sherpa-resume-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "README.md").write_text("This uses a native CPU runtime.\n")
        self.store = Store(self.root / "data")
        self.facts = {"host": {"system": "Linux"}, "gpus": [], "gpu_status": "unknown", "topology": {}, "tools": {}}
        self.probe = patch("sherpa.platform.exploration.machine.probe", return_value=self.facts).start()
        self.addCleanup(patch.stopall)

    def router(self, actions):
        backend = ScenarioBackend(actions)
        data = {"version": 1, "backends": {"test": {"type": "test"}},
                "pools": {"daily": ["small"], "expert": ["large"]},
                "models": {n: {"backend": "test", "model_id": n} for n in ("small", "large")}}
        return Router(Config(data), {"test": lambda cfg: backend}), backend

    def run_actions(self, actions, resume_from=None, **kwargs):
        router, backend = self.router(actions)
        run, result = explore_environment(self.store, None if resume_from else self.source,
            router=router, resume_from=resume_from, allow_model_calls=True, allow_unpriced=True,
            model_budget=1, seconds=30, **kwargs)
        return run, result, backend

    def original(self):
        return self.run_actions([tool("read", path="README.md"), tool("machine")], max_rounds=2, device="gpu")[0]

    def test_resumes_old_format_trace_without_modifying_or_replaying_it(self):
        old = self.original()
        before = {str(p.relative_to(old.path)): p.read_bytes() for p in old.path.rglob("*") if p.is_file()}
        old_read = next(c for c in old.records("calls") if c["name"] == "environment.read")
        def resumed(request, context):
            self.assertEqual(request.context["requested_device"], "gpu")
            self.assertEqual(observation(request)["id"], old_read["id"])
            self.assertEqual(observation(request)["origin_run"], old.id)
            self.assertFalse(any(e["tool"] == "machine" for e in request.context["observations"]))
            self.assertEqual(request.context["remaining"]["decision_rounds_including_this"], 2)
            return tool("machine")
        new, result, backend = self.run_actions([resumed, finish], old.id, max_rounds=2)
        self.assertIsNotNone(result, new.state()["stop_reason"])
        self.assertNotEqual(new.id, old.id)
        self.assertEqual(new.state()["source_runs"], [old.id])
        self.assertEqual(new.state()["resume_from"], old.id)
        self.assertEqual(len(backend.requests), 2)
        self.assertEqual([c["name"] for c in new.records("calls") if c["kind"] == "tool"], ["environment.machine"])
        self.assertEqual(before, {str(p.relative_to(old.path)): p.read_bytes() for p in old.path.rglob("*") if p.is_file()})
        self.assertEqual(new.state()["cost"]["calls"], 3)
        self.assertEqual(new.state()["cumulative_cost"]["calls"], 7)
        self.assertTrue((new.path / "environment/continuation.json").is_file())

    def test_refresh_instruction_tracks_success_not_just_historical_invalidation(self):
        old = self.original()
        old_machine_id = next(c["id"] for c in old.records("calls") if c["name"] == "environment.machine")
        def before_refresh(request, context):
            continuation = request.context["continuation"]
            self.assertTrue(continuation["machine_refresh_required"])
            self.assertIsNone(continuation["current_machine_evidence_id"])
            return tool("machine")
        def after_refresh(request, context):
            continuation = request.context["continuation"]
            self.assertFalse(continuation["machine_refresh_required"])
            self.assertEqual(continuation["current_machine_evidence_id"], observation(request, "machine")["id"])
            self.assertIn(old_machine_id, continuation["invalidated_evidence_ids"])
            self.assertIn("do not require another refresh", continuation["instructions"])
            return finish(request, context)
        new, result, _ = self.run_actions([before_refresh, after_refresh], old.id, max_rounds=2)
        self.assertIsNotNone(result, new.state()["stop_reason"])
        initial = read_json(new.path / "environment/continuation.json")
        self.assertEqual(initial["continuation"]["discarded_evidence"][0]["id"], old_machine_id)

    def test_failed_refresh_does_not_mark_machine_requirement_satisfied(self):
        old = self.original()
        def check_failure(request, context):
            self.assertTrue(request.context["continuation"]["machine_refresh_required"])
            self.assertIsNone(request.context["continuation"]["current_machine_evidence_id"])
            self.assertEqual(request.context["history"][-1]["result"]["status"], "failed")
            return {"action": "stop", "reason": "machine query failed", "unknowns": ["machine state unavailable"]}
        with patch("sherpa.platform.exploration.machine.probe", side_effect=OSError("test probe failure")):
            new, result, _ = self.run_actions([tool("machine"), check_failure], old.id, max_rounds=2)
        self.assertIsNone(result)
        self.assertEqual(new.state()["result_status"], "blocked")

    def test_machine_must_be_refreshed_before_finishing(self):
        old = self.original()
        new, result, backend = self.run_actions([finish, tool("machine"), finish], old.id, max_rounds=3)
        self.assertIsNotNone(result, new.state()["stop_reason"])
        first = new.records("calls")[0]
        self.assertEqual(first["error_category"], "invalid_response")
        self.assertIn("machine", read_json(new.path / first["output"])["error"])

    def test_changed_file_is_invalidated_not_silently_reused(self):
        old = self.original()
        (self.source / "README.md").write_text("Different implementation now.\n")
        def inspect_state(request, context):
            self.assertFalse(any(e["tool"] == "read" for e in request.context["observations"]))
            self.assertTrue(request.context["continuation"]["invalidated_evidence_ids"])
            return {"action": "stop", "reason": "need new source evidence", "unknowns": ["source changed"]}
        new, result, backend = self.run_actions([inspect_state], old.id)
        self.assertIsNone(result)
        self.assertEqual(new.state()["result_status"], "blocked")
        snapshot = read_json(new.path / "environment/continuation.json")
        self.assertTrue(any("changed" in e["reason"] for e in snapshot["continuation"]["discarded_evidence"]))

    def test_corrupt_call_artifact_fails_before_provider_call(self):
        old = self.original()
        record = old.records("calls")[1]
        atomic_json(old.path / record["output"], {"tampered": True})
        new, result, backend = self.run_actions([], old.id)
        self.assertEqual(new.state()["status"], "failed")
        self.assertIn("checksum mismatch", new.state()["stop_reason"])
        self.assertEqual(backend.requests, [])
        self.assertIsNone(result)
        self.assertEqual(new.state()["cumulative_cost"]["calls"], old.state()["cost"]["calls"])
        self.assertEqual(new.state()["cumulative_cost"]["known_model_cost_usd"], old.state()["cost"]["known_model_cost_usd"])

    def test_rejected_final_draft_is_restored_only_as_draft(self):
        def bad(request, context):
            value = finish(request, context)
            value["routes"][0]["evidence"][0]["quote"] = "not really observed"
            return value
        old, _, _ = self.run_actions([tool("read", path="README.md"), tool("machine"), bad], max_rounds=3)
        def inspect_draft(request, context):
            draft = request.context["continuation"]["draft"]
            self.assertEqual(draft["status"], "rejected_draft")
            self.assertEqual(draft["view"]["data"]["action"], "finish")
            self.assertIn("quote absent", draft["error"])
            return tool("machine")
        new, result, backend = self.run_actions([inspect_draft, finish], old.id, max_rounds=2)
        self.assertIsNotNone(result, new.state()["stop_reason"])
        self.assertTrue(all(e["tool"] in {"read", "machine"} for e in result["evidence"]))

    def test_uncommitted_successful_plan_is_restored_as_draft(self):
        def fail_proposal(path, value):
            if path.name == "proposal.json":
                raise OSError("simulated proposal write failure")
            return atomic_json(path, value)
        with patch("sherpa.runtime.explore.atomic_json", side_effect=fail_proposal):
            old, result, _ = self.run_actions([tool("read", path="README.md"), tool("machine"), finish], max_rounds=3)
        self.assertIsNone(result)
        self.assertEqual(old.state()["status"], "failed")
        self.assertEqual(old.records("calls")[-1]["status"], "succeeded")
        def inspect_draft(request, context):
            draft = request.context["continuation"]["draft"]
            self.assertEqual(draft["status"], "uncommitted_draft")
            self.assertEqual(draft["view"]["data"]["action"], "finish")
            return tool("machine")
        new, result, _ = self.run_actions([inspect_draft, finish], old.id, max_rounds=2)
        self.assertIsNotNone(result, new.state()["stop_reason"])

    def test_large_catalog_is_paged_with_room_for_focused_observation(self):
        import tomllib
        from sherpa.runtime.explore import Investigation, POLICY
        from sherpa.runtime.models import Task
        old = self.original()
        task = Task(**old.state()["task"])
        router, _ = self.router([])
        investigation = Investigation(old, task, router, tomllib.loads(POLICY.read_text()))
        for i in range(400):
            investigation.evidence[f"call-{i:04d}"] = {"id": f"call-{i:04d}", "tool": "read",
                "parameters": {"path": "d/" * 200 + "README.md"},
                "view": {"data": {"lines": [{"line": 1, "text": "x" * 8000}]}, "truncated": False}}
        investigation.focus = "call-0399"
        context = investigation.context(1)
        self.assertLessEqual(len(json.dumps(context, ensure_ascii=False).encode()), 28000)
        self.assertIn("call-0399", [e["id"] for e in context["observations"]])
        self.assertIsNotNone(context["catalog_next_offset"])
        self.assertLessEqual(len(context["omitted_evidence_ids"]), 20)
        offset = context["catalog_next_offset"]
        investigation.validate({"action": "catalog", "offset": offset, "reason": "next page"})
        investigation.catalog_offset = offset
        page = investigation.context(2)
        self.assertEqual(page["catalog"][0]["id"], f"call-{offset:04d}")

    def test_multi_segment_cost_counts_each_run_once_and_keeps_unknown(self):
        router, backend = self.router([tool("read", path="README.md")])
        original_complete = backend.complete
        def unpriced(*args):
            reply = original_complete(*args)
            return BackendResponse(reply.text, reply.reported_model)
        with patch.object(backend, "complete", side_effect=unpriced):
            old, _ = explore_environment(self.store, self.source, router=router, allow_model_calls=True,
                allow_unpriced=True, model_budget=1, seconds=30, max_rounds=1)
        middle, _, _ = self.run_actions([tool("machine")], old.id, max_rounds=1)
        new, result, backend = self.run_actions([tool("machine"), finish], middle.id, max_rounds=2)
        self.assertIsNotNone(result, new.state()["stop_reason"])
        self.assertEqual(new.state()["source_chain"], [old.id, middle.id])
        self.assertEqual(new.state()["cumulative_cost"]["calls"], 7)
        self.assertEqual(new.state()["cumulative_cost"]["unpriced_model_calls"], 1)
        self.assertIsNone(new.state()["cumulative_cost"]["model_cost_usd"])
        self.assertEqual(new.state()["cost"]["calls"], 3)

    def test_unsafe_run_states_are_rejected_before_new_run(self):
        for status in ("running", "created", "cleanup_failed", "completed"):
            with self.subTest(status=status):
                old = self.original()
                state = old.state()
                state["status"] = status
                atomic_json(old.path / "run.json", state)
                before = len(self.store.runs())
                with self.assertRaises(ValueError):
                    self.run_actions([], old.id)
                self.assertEqual(len(self.store.runs()), before)

    def test_unfinished_call_and_bad_scope_are_rejected(self):
        old = self.original()
        record = old.records("calls")[0]
        record["status"] = "running"
        atomic_json(old.path / "calls" / record["id"] / "record.json", record)
        with self.assertRaises(ValueError):
            lineage(self.store, old.id)
        state = old.state()
        state["scope"] = "transport_diagnostic_only"
        atomic_json(old.path / "run.json", state)
        with self.assertRaisesRegex(ValueError, "only environment exploration"):
            lineage(self.store, old.id)

    def test_new_budget_is_explicit_and_no_consent_means_no_calls(self):
        old = self.original()
        before = len(self.store.runs())
        with patch("subprocess.Popen") as popen, self.assertRaises(ValueError):
            explore_environment(self.store, None, resume_from=old.id, model_budget=1)
        popen.assert_not_called()
        self.assertEqual(len(self.store.runs()), before)
        with self.assertRaisesRegex(ValueError, "inherits"):
            explore_environment(self.store, None, resume_from=old.id, device="cpu",
                                allow_model_calls=True, allow_unpriced=True, model_budget=1)

    def test_cli_resume_inherits_target_and_reports_cumulative_cost(self):
        old = self.original()
        router, backend = self.router([tool("machine"), finish])
        stream = io.StringIO()
        with patch("sherpa.runtime.explore.Router", return_value=router), redirect_stdout(stream):
            code = main(["--data-root", str(self.store.root), "env", "resume", old.id, "--allow-model-calls",
                         "--allow-unpriced", "--model-budget", "1", "--max-rounds", "2", "--json"])
        self.assertEqual(code, 0, stream.getvalue())
        response = json.loads(stream.getvalue())
        self.assertEqual(response["resume_from"], old.id)
        self.assertEqual(response["cost"]["calls"], 3)
        self.assertEqual(response["cumulative_cost"]["calls"], 7)

    def test_supplemental_machine_quote_is_allowed_with_real_document_quote(self):
        def with_machine(request, context):
            value = finish(request, context)
            value["routes"][0]["evidence"].append({"id": observation(request, "machine")["id"], "quote": '"system": "Linux"'})
            return value
        run, result, backend = self.run_actions([tool("read", path="README.md"), tool("machine"), with_machine], max_rounds=3)
        self.assertIsNotNone(result, run.state()["stop_reason"])

    def test_machine_or_listing_alone_cannot_prove_runtime_support(self):
        def no_contents(request, context):
            value = finish(request, context)
            value["routes"][0]["evidence"] = [{"id": observation(request, "machine")["id"], "quote": '"system": "Linux"'}]
            return value
        run, result, backend = self.run_actions([tool("read", path="README.md"), tool("machine"), no_contents], max_rounds=3)
        self.assertIsNone(result)
        self.assertIn("supplemental only", read_json(run.path / run.records("calls")[-1]["output"])["error"])

    def test_wrong_supplemental_quote_reports_the_exact_id(self):
        def wrong_id(request, context):
            value = finish(request, context)
            value["routes"][0]["evidence"].append({"id": observation(request, "machine")["id"], "quote": "file-not-here.gguf"})
            return value
        run, result, backend = self.run_actions([tool("read", path="README.md"), tool("machine"), wrong_id], max_rounds=3)
        self.assertIsNone(result)
        error = read_json(run.path / run.records("calls")[-1]["output"])["error"]
        machine_id = next(c["id"] for c in run.records("calls") if c["name"] == "environment.machine")
        self.assertIn(machine_id, error)
        self.assertIn("quote absent", error)


if __name__ == "__main__":
    unittest.main()
