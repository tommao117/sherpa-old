from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from sherpa.agent import Config, Router
from sherpa.cli import main
from sherpa.platform.scoped_evidence import ScopedEvidence
from sherpa.runtime import framework, workspace
from sherpa.runtime.calls import RunStopped
from sherpa.runtime.continuation import lineage
from sherpa.runtime.explore import explore_environment
from sherpa.runtime.models import Task
from sherpa.runtime.store import Store, atomic_json, read_json
from test_explore import ScenarioBackend, tool, route


def scoped_observation(request, scope):
    return next(e for e in request.context["observations"] if e["tool"] == "read" and e["parameters"].get("scope") == scope)


def dual_finish(request, context):
    code = scoped_observation(request, "framework")
    model = scoped_observation(request, "model")
    candidate = route(code["id"], "Framework source")
    candidate.pop("runtime")
    candidate.pop("framework_source")
    candidate["model_files"] = ["weights.gguf"]
    candidate["evidence"] = [{"id": e["id"], "start": 1, "end": 1} for e in (code, model)]
    return {"action": "finish", "summary": "Check specified framework with specified model", "routes": [candidate],
            "selected_route": None, "unknowns": ["No runtime behavior has been tested"]}


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sherpa-inputs-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.code = self.root / "code"
        self.model = self.root / "model"
        self.code.mkdir()
        self.model.mkdir()
        (self.code / "README.md").write_text("Framework source implements a native runtime.\n")
        (self.code / "main.cpp").write_text("int main() { return 0; }\n")
        (self.model / "README.md").write_text("Model metadata describes supplied weights.\n")
        (self.model / "weights.gguf").write_bytes(b"not to be loaded")
        self.store = Store(self.root / "data")
        self.facts = {"host": {"system": "Linux"}, "gpus": [], "gpu_status": "unknown", "topology": {}, "tools": {}}
        self.probe = patch("sherpa.platform.exploration.machine.probe", return_value=self.facts).start()
        self.addCleanup(patch.stopall)

    def router(self, actions):
        backend = ScenarioBackend(actions)
        data = {"version": 1, "backends": {"test": {"type": "test"}}, "pools": {"daily": ["small"], "expert": ["large"]},
                "models": {n: {"backend": "test", "model_id": n} for n in ("small", "large")}}
        return Router(Config(data), {"test": lambda cfg: backend}), backend

    def run_inputs(self, actions, **options):
        router, backend = self.router(actions)
        args = {"framework": str(self.code), "model": self.model} if not options.get("resume_from") else {}
        args.update(options)
        run, proposal = explore_environment(self.store, router=router, allow_model_calls=True, allow_unpriced=True,
                                           model_budget=1, seconds=60, **args)
        return run, proposal, backend

    def actions(self):
        return [tool("read", scope="framework", path="README.md"), tool("read", scope="model", path="README.md"), tool("machine"), dual_finish]

    def test_two_roots_same_names_not_interchangeable(self):
        run, proposal, backend = self.run_inputs(self.actions(), max_rounds=4)
        self.assertIsNotNone(proposal, run.state()["stop_reason"])
        self.assertEqual(proposal["inputs"]["model"]["path"], str(self.model))
        self.assertEqual(proposal["inputs"]["framework"]["commit"], run.state()["framework"]["commit"])
        self.assertEqual(run.state()["task"]["target"]["path"], str(self.code))
        reads = [c for c in run.records("calls") if c["name"] == "environment.read"]
        first, second = [read_json(run.path / r["output"]) for r in reads]
        self.assertEqual(first["subject"]["scope"], "framework")
        self.assertEqual(second["subject"]["scope"], "model")
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertFalse((self.code / ".git").exists())
        self.assertFalse((self.model / ".git").exists())
        self.assertEqual(run.records("calls")[0]["name"], "framework.prepare")
        self.assertEqual(len(backend.requests), 4)
        raw = read_json(run.path / run.records("calls")[-1]["output"])
        self.assertNotIn("framework_commit", raw["routes"][0])
        self.assertNotIn("quote", raw["routes"][0]["evidence"][0])
        self.assertEqual(proposal["plan"]["routes"][0]["framework_commit"], run.state()["framework"]["commit"])
        self.assertEqual(proposal["plan"]["routes"][0]["evidence"][0]["quote"], (self.code / "README.md").read_text().strip())
        self.assertTrue(all(model == "small" for model, _, _ in backend.requests))

    def test_prepare_only_pins_code_without_any_provider_or_probe(self):
        router, backend = self.router([])
        run, proposal = explore_environment(self.store, framework=str(self.code), model=self.model, prepare_only=True, router=router)
        self.assertEqual(run.state()["status"], "completed", run.state()["stop_reason"])
        self.assertEqual(run.state()["result_status"], "code_prepared")
        self.assertEqual(backend.requests, [])
        self.probe.assert_not_called()
        self.assertIsNone(proposal)
        self.assertEqual(run.state()["cost"]["model_cost_usd"], 0)
        path = Path(run.state()["framework"]["path"])
        self.assertEqual(workspace.git(path, "rev-parse", "HEAD"), run.state()["framework"]["commit"])
        self.assertFalse((path / "weights.gguf").exists())

    def test_search_windows_survive_resume_and_unpin_is_final(self):
        def pin(request, context):
            refs = [{"id": e["id"], "start": 1, "end": 1} for e in request.context["observations"] if e["tool"] == "search"]
            self.assertEqual(len(refs), 2)
            return {"action": "pin", "evidence": refs, "reason": "keep the two input facts"}
        original, _, _ = self.run_inputs([tool("search", scope="framework", path="README.md", query="Framework"),
            tool("search", scope="model", path="README.md", query="Model"), tool("machine"), pin], max_rounds=4)
        self.assertEqual(len(original.state()["evidence_windows"]), 2)
        before = {p: p.read_bytes() for p in original.path.rglob("*") if p.is_file()}
        def release(request, context):
            self.assertEqual(len(request.context["retained_windows"]), 2)
            self.assertEqual(sum(e.get("pinned", False) for e in request.context["observations"]), 2)
            return {"action": "pin", "evidence": [], "reason": "no longer needed"}
        next_run, _, _ = self.run_inputs([release], resume_from=original.id, max_rounds=1)
        self.assertEqual(next_run.state()["evidence_windows"], [])
        self.assertEqual(before, {p: p.read_bytes() for p in original.path.rglob("*") if p.is_file()})
        last, _, _ = self.run_inputs([], resume_from=next_run.id, prepare_only=True)
        self.assertEqual(last.state()["evidence_windows"], [])

    def test_changed_model_invalidates_only_its_retained_window(self):
        def pin(request, context):
            refs = [{"id": e["id"], "start": 1, "end": 1} for e in request.context["observations"] if e["tool"] == "read"]
            return {"action": "pin", "evidence": refs, "reason": "retain observed inputs"}
        old, _, _ = self.run_inputs(self.actions()[:-1] + [pin], max_rounds=4)
        (self.model / "README.md").write_text("Different model metadata now.\n")
        new, _, _ = self.run_inputs([], resume_from=old.id, prepare_only=True)
        self.assertEqual(new.state()["status"], "completed", new.state()["stop_reason"])
        self.assertEqual(len(new.state()["evidence_windows"]), 1)
        kept = new.state()["evidence_windows"][0]["id"]
        self.assertEqual(read_json(old.path / "calls" / kept / "input.json")["scope"], "framework")

    def test_line_report_accepts_search_but_never_crosses_match_gaps(self):
        def final(request, context):
            evidence = [e for e in request.context["observations"] if e["tool"] == "search"]
            candidate = route(evidence[0]["id"])
            candidate.pop("runtime")
            candidate.pop("framework_source")
            candidate.update(model_files=["weights.gguf"], evidence=[{"id": e["id"], "start": 1, "end": 1} for e in evidence])
            return {"action": "finish", "summary": "specific next check", "routes": [candidate], "selected_route": None,
                    "unknowns": ["runtime not tested"]}
        run, proposal, _ = self.run_inputs([tool("search", scope="framework", path="README.md", query="Framework"),
            tool("search", scope="model", path="README.md", query="Model"), tool("machine"), final], max_rounds=4)
        self.assertIsNotNone(proposal, run.state()["stop_reason"])
        self.assertEqual(len(proposal["plan"]["routes"][0]["evidence"]), 2)
        self.assertTrue(all(e["quote"].endswith(".") for e in proposal["plan"]["routes"][0]["evidence"]))

    def test_pin_validation_failure_does_not_apply_windows(self):
        def invalid_pin(request, context):
            e = next(e for e in request.context["observations"] if e["tool"] == "read")
            return {"action": "pin", "evidence": [{"id": e["id"], "start": 1, "end": 3}], "reason": "unseen lines"}
        run, _, _ = self.run_inputs([tool("read", scope="model", path="README.md"), invalid_pin], max_rounds=2)
        self.assertNotIn("evidence_windows", run.state())
        self.assertEqual(run.records("calls")[-1]["error_category"], "invalid_response")

    def test_bounded_context_retains_both_input_anchors_and_machine(self):
        import tomllib
        from sherpa.runtime.explore import Investigation, POLICY
        run, _, _ = self.run_inputs([], prepare_only=True)
        router, _ = self.router([])
        investigation = Investigation(run, Task(**run.state()["task"]), router, tomllib.loads(POLICY.read_text()))
        investigation.framework_ref = run.state()["framework"]
        investigation.tools = ScopedEvidence(investigation.framework_ref, {"path": str(self.model)}, self.root, lambda: None)
        self.addCleanup(investigation.tools.close)
        for i, scope in enumerate(("model", "framework", "framework", "framework", "framework")):
            investigation.evidence[f"read-{i}"] = {"id": f"read-{i}", "tool": "read", "parameters": {"scope": scope, "path": f"doc{i}.md"},
                "view": {"data": {"lines": [{"line": n, "text": "x" * 190} for n in range(1, 31)]}, "truncated": False}}
        investigation.evidence["machine"] = {"id": "machine", "tool": "machine", "parameters": {},
                                              "view": {"data": self.facts, "truncated": False}}
        investigation.evidence["inventory"] = {"id": "inventory", "tool": "list", "parameters": {"scope": "model", "path": "."},
                                                "view": {"data": {"path": ".", "entries": [{"name": "weights.gguf", "kind": "file"}]}, "truncated": False}}
        investigation.continuation = {"source_run": "old", "budget_semantics": "additional", "instructions": "refresh",
            "discarded_evidence": [], "draft": {"call_id": "old-call", "status": "rejected_draft", "error": "bad quote",
                                                  "view": {"data": {"summary": "d" * 6500}, "truncated": False}}}
        investigation.focus = "read-4"
        investigation.history = [{"action": "recall"}]
        context = investigation.context(1)
        visible = [e["id"] for e in context["observations"]]
        self.assertEqual(visible[:3], ["read-4", "read-0", "read-1"])
        self.assertEqual(context["facts"]["machine"]["evidence_id"], "machine")
        self.assertEqual(context["facts"]["model_inventory"][0]["path"], "weights.gguf")
        self.assertLess(len(investigation.visible_lines["read-0"]), 30)
        self.assertLessEqual(len(json.dumps(context, ensure_ascii=False).encode()), 28000)
        self.assertEqual(context["continuation"]["draft"]["error"], "bad quote")

    def test_python_declaration_fixture_uses_same_line_assembly(self):
        (self.code / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.11"\ndependencies = ["example==1.0"]\n')
        (self.model / "config.json").write_text('{"architecture": "a_test_architecture"}\n')
        def python_plan(request, context):
            value = dual_finish(request, context)
            value["routes"][0]["python_requirement"] = ">=3.11"
            value["routes"][0]["evidence"][0].update(start=2, end=3)
            return value
        run, proposal, backend = self.run_inputs([tool("read", scope="framework", path="pyproject.toml"),
            tool("read", scope="model", path="config.json"), tool("machine"), python_plan], max_rounds=4)
        self.assertIsNotNone(proposal, run.state()["stop_reason"])
        self.assertIn('requires-python = ">=3.11"', proposal["plan"]["routes"][0]["evidence"][0]["quote"])
        self.assertIn("a_test_architecture", proposal["plan"]["routes"][0]["evidence"][1]["quote"])
        self.assertEqual(len(backend.requests), 4)
        self.assertTrue(all(model == "small" for model, _, _ in backend.requests))

    def test_old_quote_draft_is_preserved_but_not_executed_on_resume(self):
        def old_format(request, context):
            code = scoped_observation(request, "framework")
            value = dual_finish(request, context)
            value["routes"][0].update(runtime="unverified runner", framework_commit=request.context["inputs"]["framework"]["commit"],
                                      framework_source="old source text", evidence=[{"id": code["id"], "quote": "Framework source"}])
            return value
        old, _, _ = self.run_inputs(self.actions()[:-1] + [old_format], max_rounds=4)
        self.assertEqual(old.state()["status"], "budget_exhausted")
        before = {p: p.read_bytes() for p in old.path.rglob("*") if p.is_file()}
        def resume(request, context):
            self.assertEqual(request.context["continuation"]["draft"]["status"], "rejected_draft")
            self.assertNotIn("framework_commit", request.output_contract["finish"]["routes"][0])
            return tool("machine")
        new, proposal, backend = self.run_inputs([resume, dual_finish], resume_from=old.id, max_rounds=2)
        self.assertIsNotNone(proposal, new.state()["stop_reason"])
        self.assertEqual(before, {p: p.read_bytes() for p in old.path.rglob("*") if p.is_file()})
        self.assertEqual(proposal["assembly"]["line_selections"][0][0]["start"], 1)
        self.assertEqual(len(backend.requests), 2)

    def test_missing_model_files_has_actionable_feedback_without_guessing_selection(self):
        def missing(request, context):
            value = dual_finish(request, context)
            value["routes"][0].pop("model_files")
            return value
        run, proposal, _ = self.run_inputs(self.actions()[:-1] + [missing, dual_finish], max_rounds=5)
        self.assertIsNotNone(proposal, run.state()["stop_reason"])
        failed = next(c for c in run.records("calls") if c.get("error_category") == "invalid_response")
        self.assertIn("missing fields: ['model_files']", read_json(run.path / failed["output"])["error"])
        self.assertNotIn("model_files", json.loads(read_json(run.path / "calls" / failed["id"] / "raw.json")["text"])["routes"][0])
        self.assertEqual(proposal["plan"]["routes"][0]["model_files"], ["weights.gguf"])

    def test_invalid_line_range_returns_control_for_correction(self):
        def bad_lines(request, context):
            value = dual_finish(request, context)
            value["routes"][0]["evidence"][0]["end"] = 8
            return value
        run, proposal, backend = self.run_inputs(self.actions()[:-1] + [bad_lines, dual_finish], max_rounds=5)
        self.assertIsNotNone(proposal, run.state()["stop_reason"])
        invalid = next(c for c in run.records("calls") if c.get("error_category") == "invalid_response")
        self.assertIn("not currently displayed", read_json(run.path / invalid["output"])["error"])
        self.assertTrue(all(model == "small" for model, _, _ in backend.requests))

    def test_missing_scope_rejected_not_guessed(self):
        run, proposal, _ = self.run_inputs([tool("read", path="README.md"), {"action": "stop", "reason": "fix scope", "unknowns": []}], max_rounds=2)
        invalid = next(c for c in run.records("calls") if c.get("error_category") == "invalid_response")
        self.assertIn("scope", read_json(run.path / invalid["output"])["error"])
        self.assertIsNone(proposal)

    def test_final_needs_both_scopes_and_fixed_commit(self):
        def wrong_commit(request, context):
            value = dual_finish(request, context)
            value["routes"][0]["framework_commit"] = "different"
            return value
        def missing_code(request, context):
            value = dual_finish(request, context)
            value["routes"][0]["evidence"] = [value["routes"][0]["evidence"][1]]
            return value
        actions = self.actions()[:-1] + [wrong_commit, missing_code, dual_finish]
        run, proposal, _ = self.run_inputs(actions, max_rounds=6)
        self.assertIsNotNone(proposal, run.state()["stop_reason"])
        failures = [c for c in run.records("calls") if c["status"] == "failed"]
        self.assertEqual(len(failures), 2)

    def test_invalid_model_selection_cannot_escape_or_reference_missing_file(self):
        def absent(request, context):
            value = dual_finish(request, context)
            value["routes"][0]["model_files"] = ["missing.gguf"]
            return value
        run, proposal, _ = self.run_inputs(self.actions()[:-1] + [absent, dual_finish], max_rounds=5)
        self.assertIsNotNone(proposal, run.state()["stop_reason"])
        self.assertEqual(run.state()["status"], "completed")
        invalid = next(c for c in run.records("calls") if c.get("error_category") == "invalid_response")
        self.assertIn("missing.gguf", read_json(run.path / invalid["output"])["error"])
        self.assertFalse(any(c["kind"] == "tool" and c["name"] == "environment.read" and
                             "weights" in str(read_json(run.path / c["input"])) for c in run.records("calls")))

    def test_resume_uses_commit_not_advanced_source_branch(self):
        workspace.git(self.code, "init")
        original = workspace.commit_all(self.code, "initial fixture")
        first, _, _ = self.run_inputs(self.actions()[:-1], revision="HEAD", max_rounds=3)
        self.assertEqual(first.state()["framework"]["commit"], original)
        (self.code / "main.cpp").write_text("int main() { return 1; }\n")
        workspace.commit_all(self.code, "advance source branch")
        second, proposal, backend = self.run_inputs([tool("machine"), dual_finish], resume_from=first.id, max_rounds=2)
        self.assertIsNotNone(proposal, second.state()["stop_reason"])
        self.assertEqual(second.state()["framework"]["commit"], original)
        self.assertNotEqual(first.state()["framework"]["path"], second.state()["framework"]["path"])
        self.assertIn("return 0", (Path(second.state()["framework"]["path"]) / "main.cpp").read_text())
        self.assertEqual(len(backend.requests), 2)
        self.assertEqual(len(second.state()["source_chain"]), 1)

    def test_shallow_repository_resume_copies_shallow_boundary_without_transport(self):
        workspace.git(self.code, "init")
        workspace.commit_all(self.code, "first")
        (self.code / "main.cpp").write_text("new source")
        workspace.commit_all(self.code, "second")
        old, _, _ = self.run_inputs([], revision="HEAD", prepare_only=True)
        source = Path(old.state()["framework"]["path"])
        self.assertTrue((source / ".git/shallow").is_file())
        new, _, _ = self.run_inputs([], resume_from=old.id, prepare_only=True)
        self.assertEqual(new.state()["status"], "completed", new.state()["stop_reason"])
        destination = Path(new.state()["framework"]["path"])
        self.assertEqual((source / ".git/shallow").read_bytes(), (destination / ".git/shallow").read_bytes())
        self.assertEqual(workspace.git(destination, "rev-parse", "HEAD"), old.state()["framework"]["commit"])
        self.assertEqual(workspace.git(destination, "fsck", "--no-dangling"), "")
        self.assertFalse((destination / ".git/objects/info/alternates").exists())
        for log in new.path.glob("calls/*/git-*/request.json"):
            self.assertNotIn(read_json(log)["args"][0], {"clone", "fetch", "repack"})

    def test_object_copy_rejects_links_and_honors_cancellation(self):
        old, _, _ = self.run_inputs([], prepare_only=True)
        source = Path(old.state()["framework"]["path"])
        destination = self.root / "copy-target"
        destination.mkdir()
        workspace.git(destination, "init")
        def cancel():
            raise RunStopped("cancelled", "stop copying")
        with self.assertRaises(RunStopped):
            framework.copy_pinned_objects(source, destination, 1024 * 1024, cancel)
        (source / ".git/objects/linked").symlink_to(self.code, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinked"):
            framework.copy_pinned_objects(source, destination, 1024 * 1024, lambda: None)

    def test_prepared_checkpoint_can_enter_exploration(self):
        first, _, _ = self.run_inputs([], prepare_only=True)
        second, proposal, _ = self.run_inputs(self.actions(), resume_from=first.id, max_rounds=4)
        self.assertIsNotNone(proposal, second.state()["stop_reason"])
        self.assertEqual(second.state()["framework"]["commit"], first.state()["framework"]["commit"])

    def test_changed_framework_workspace_rejected_before_model(self):
        first, _, _ = self.run_inputs([tool("read", scope="framework", path="README.md")], max_rounds=1)
        (Path(first.state()["framework"]["path"]) / "main.cpp").write_text("changed")
        second, proposal, backend = self.run_inputs([], resume_from=first.id)
        self.assertEqual(second.state()["status"], "failed")
        self.assertEqual(backend.requests, [])
        self.assertIn("changed", second.state()["stop_reason"])

    def test_changed_pinned_commit_must_match_original_prepare_artifact(self):
        first, _, _ = self.run_inputs([], prepare_only=True)
        path = Path(first.state()["framework"]["path"])
        (path / "main.cpp").write_text("a different commit")
        altered = workspace.commit_all(path, "altered fixture")
        state = first.state()
        state["framework"]["commit"] = altered
        atomic_json(first.path / "run.json", state)
        with self.assertRaisesRegex(ValueError, "verified prepare output"):
            lineage(self.store, first.id)

    def test_inherited_git_settings_and_templates_do_not_leak_into_import(self):
        import os
        template = self.root / "external-template"
        template.mkdir()
        (template / "config").write_text("[injected]\n marker = from-template\n")
        environment = {"GIT_TEMPLATE_DIR": str(template), "GIT_CONFIG_PARAMETERS": "'injected.marker=from-environment'",
                       "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "injected.marker", "GIT_CONFIG_VALUE_0": "from-count"}
        with patch.dict(os.environ, environment):
            first, _, _ = self.run_inputs([], prepare_only=True)
        self.assertEqual(first.state()["status"], "completed", first.state()["stop_reason"])
        path = Path(first.state()["framework"]["path"])
        self.assertNotIn("injected", (path / ".git/config").read_text())
        self.assertEqual(workspace.git(path, "config", "--get", "injected.marker", check=False), "")

    def test_git_resource_settings_are_process_local_and_logged(self):
        import os
        workspace.git(self.code, "init")
        before = (self.code / ".git/config").read_bytes()
        environment = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "pack.threads", "GIT_CONFIG_VALUE_0": "128"}
        with patch.dict(os.environ, environment):
            for key, expected in workspace.GIT_RESOURCE_SETTINGS.items():
                self.assertEqual(workspace.git(self.code, "config", "--get", key), expected)
        self.assertEqual((self.code / ".git/config").read_bytes(), before)
        old, _, _ = self.run_inputs([], prepare_only=True)
        source = Path(old.state()["framework"]["path"])
        original_config = (source / ".git/config").read_bytes()
        new, _, _ = self.run_inputs([], resume_from=old.id, prepare_only=True)
        self.assertEqual(new.state()["status"], "completed", new.state()["stop_reason"])
        operations = [read_json(p) for p in new.path.glob("calls/*/git-*/request.json")]
        self.assertNotIn("fetch", [op["args"][0] for op in operations])
        self.assertNotIn("clone", [op["args"][0] for op in operations])
        self.assertEqual(operations[0]["resource_settings"]["pack.threads"], "1")
        self.assertEqual(operations[0]["memory_limit_mib"], 512)
        self.assertEqual(new.state()["framework"]["import_mode"], "pinned_local_copy")
        destination = Path(new.state()["framework"]["path"])
        for original_object in (source / ".git/objects").rglob("*"):
            if original_object.is_file():
                copied = destination / original_object.relative_to(source)
                self.assertEqual(original_object.read_bytes(), copied.read_bytes())
                self.assertNotEqual(original_object.stat().st_ino, copied.stat().st_ino)
        self.assertEqual((source / ".git/config").read_bytes(), original_config)

    def test_prepare_failure_preserves_prior_cost_and_can_be_resumed(self):
        old, _, _ = self.run_inputs([tool("read", scope="model", path="README.md")], max_rounds=1)
        with patch("sherpa.runtime.framework.import_code", side_effect=RuntimeError("thread creation failed")):
            failed, _, backend = self.run_inputs([], resume_from=old.id)
        self.assertEqual(backend.requests, [])
        self.assertEqual(failed.state()["status"], "failed")
        self.assertEqual(failed.state()["cumulative_cost"]["calls"], old.state()["cost"]["calls"] + 1)
        self.assertEqual(failed.state()["cumulative_cost"]["model_cost_usd"], old.state()["cost"]["model_cost_usd"])
        continued, _, backend = self.run_inputs([], resume_from=failed.id, prepare_only=True)
        self.assertEqual(continued.state()["status"], "completed", continued.state()["stop_reason"])
        self.assertEqual(continued.state()["framework"]["commit"], old.state()["framework"]["commit"])
        self.assertEqual(continued.state()["source_chain"], [old.id, failed.id])
        self.assertEqual(continued.state()["cumulative_cost"]["calls"], old.state()["cost"]["calls"] + 2)

    def test_direct_import_does_not_create_directories_under_input(self):
        with self.assertRaisesRegex(ValueError, "inside input"):
            framework.import_code({"kind": "local", "path": str(self.code)}, self.code / "generated/repo",
                                  self.root / "logs", lambda: None)
        self.assertFalse((self.code / "generated").exists())

    def test_git_disk_probe_tolerates_transient_files(self):
        original = Path.lstat
        temporary = self.code / "pack.tmp"
        temporary.write_text("temporary")
        def transient(path, *args, **kwargs):
            if path.name == "pack.tmp":
                raise FileNotFoundError(path)
            return original(path, *args, **kwargs)
        with patch.object(Path, "lstat", transient):
            framework.check_tree(self.code, 1024 * 1024, lambda: None)

    def test_oversized_framework_is_stopped_before_provider_calls(self):
        (self.code / "large.cpp").write_text("x" * (1024 * 1024))
        run, result, backend = self.run_inputs([], framework_max_mib=1)
        self.assertEqual(run.state()["status"], "failed")
        self.assertIn("budget", run.state()["stop_reason"])
        self.assertEqual(backend.requests, [])

    def test_model_identity_cannot_change_in_continuation_chain(self):
        first, _, _ = self.run_inputs([], prepare_only=True)
        second, _, _ = self.run_inputs([tool("machine")], resume_from=first.id, max_rounds=1)
        state = second.state()
        state["task"]["model"]["path"] = str(self.root)
        atomic_json(second.path / "run.json", state)
        with self.assertRaisesRegex(ValueError, "model mismatch"):
            lineage(self.store, second.id)

    def test_model_directory_is_protected_from_run_data(self):
        nested = Store(self.model / "data")
        with self.assertRaisesRegex(ValueError, "model input"):
            explore_environment(nested, framework=str(self.code), model=self.model, prepare_only=True)
        self.assertFalse(nested.root.exists())

    def test_scoped_cpp_read_and_cross_root_symlink_rejection(self):
        ref = {"path": str(self.code), "source": {"kind": "local", "path": str(self.code)}, "commit": "fixed"}
        tools = ScopedEvidence(ref, {"path": str(self.model)}, self.root, lambda: None)
        self.addCleanup(tools.close)
        result = tools.execute("read", {"scope": "framework", "path": "main.cpp"})
        self.assertIn("int main", result["lines"][0]["text"])
        (self.code / "linked").symlink_to(self.model, target_is_directory=True)
        for scope, path in (("framework", "linked/README.md"), ("model", "../code/main.cpp"), ("model", "weights.gguf")):
            with self.subTest(scope=scope, path=path), self.assertRaises((ValueError, OSError)):
                tools.execute("read", {"scope": scope, "path": path})

    def test_local_dirty_git_snapshot_and_ignored_code_are_committed(self):
        workspace.git(self.code, "init")
        workspace.commit_all(self.code, "initial")
        (self.code / "main.cpp").write_text("changed uncommitted code")
        (self.code / ".gitignore").write_text("extra.cpp\n")
        (self.code / "extra.cpp").write_text("ignored source")
        status = workspace.git(self.code, "status", "--porcelain")
        run, _, _ = self.run_inputs([], prepare_only=True)
        self.assertEqual(run.state()["status"], "completed", run.state()["stop_reason"])
        path = Path(run.state()["framework"]["path"])
        self.assertEqual(workspace.git(path, "show", "HEAD:extra.cpp"), "ignored source")
        self.assertEqual(workspace.git(path, "show", "HEAD:main.cpp"), "changed uncommitted code")
        self.assertEqual(workspace.git(self.code, "status", "--porcelain"), status)

    def test_git_revision_is_pinned_and_no_submodule_fetch(self):
        workspace.git(self.code, "init")
        first = workspace.commit_all(self.code, "first")
        (self.code / "main.cpp").write_text("newer")
        workspace.commit_all(self.code, "second")
        run, _, _ = self.run_inputs([], revision=first, prepare_only=True)
        self.assertEqual(run.state()["status"], "completed", run.state()["stop_reason"])
        self.assertEqual(run.state()["framework"]["commit"], first)
        logs = list(run.path.glob("calls/*/git-*/request.json"))
        fetches = [read_json(p)["args"] for p in logs if read_json(p)["args"][0] == "fetch"]
        self.assertEqual(len(fetches), 1)
        self.assertIn("--depth=1", fetches[0])
        self.assertIn("--no-recurse-submodules", fetches[0])

    def test_https_url_preserved_and_bad_urls_or_refs_rejected(self):
        url = "https://github.com/ggml-org/llama.cpp"
        self.assertEqual(framework.target(url), {"kind": "git", "url": url, "revision": "HEAD"})
        for value in ("https://token@example.com/repo", "http://example.com/repo", "ext::anything", "https://example.com/repo?token=secret"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                framework.target(value)
        for ref in ("--upload-pack=bad", "../HEAD", "main:other", "HEAD^{tree}"):
            with self.assertRaises(ValueError):
                framework.target(url, ref)

    def test_download_failure_closes_call_and_never_calls_model(self):
        with patch("sherpa.runtime.framework.import_code", side_effect=RuntimeError("fetch failed")):
            run, result, backend = self.run_inputs([])
        self.assertEqual(run.state()["status"], "failed")
        self.assertEqual(run.records("calls")[-1]["status"], "failed")
        self.assertEqual(backend.requests, [])

    def test_cancel_prepare_preserves_status_and_no_provider(self):
        with patch("sherpa.runtime.framework.import_code", side_effect=RunStopped("cancelled", "cancel fetch")):
            run, result, backend = self.run_inputs([])
        self.assertEqual(run.state()["status"], "cancelled")
        self.assertEqual(backend.requests, [])
        self.assertEqual(run.records("calls")[-1]["status"], "cancelled")

    def test_missing_cli_framework_is_rejected_before_any_execution(self):
        with patch("subprocess.Popen") as popen, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["env", "explore", "--model", str(self.model), "--model-budget", "1"])
        popen.assert_not_called()

    def test_cli_prepare_code_needs_no_model_consent(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = main(["--data-root", str(self.store.root), "env", "prepare-code", "--framework", str(self.code),
                         "--model", str(self.model), "--device", "gpu", "--json"])
        self.assertEqual(code, 0, stream.getvalue())
        result = json.loads(stream.getvalue())
        self.assertEqual(result["result_status"], "code_prepared")
        self.assertEqual(result["model"]["path"], str(self.model))
        self.assertIsNotNone(result["framework"]["commit"])
        self.assertEqual(result["cost"]["model_cost_usd"], 0)


if __name__ == "__main__":
    unittest.main()
