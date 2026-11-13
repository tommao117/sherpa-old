from __future__ import annotations

import copy
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sherpa.cli import main
from sherpa.runtime.execution_contract import load_document, metadata, validate_execution
from sherpa.runtime.models import Task
from sherpa.runtime.store import digest


def fixture(root):
    return Task("contract-test", {"kind": "git", "url": "https://example.com/framework.git"},
        {"kind": "local", "path": str(root), "selected_set": "q4", "sets": [
            {"id": "q4", "layout": "single", "weights": [{"path": "weights.gguf", "bytes": 4,
              "sha256": hashlib.sha256(b"test").hexdigest()}], "support": []},
            {"id": "q8", "layout": "single", "weights": [{"path": "alternative.gguf", "bytes": 8,
              "sha256": "b" * 64}], "support": []}]},
        resources={"device": "gpu", "gpus": ["GPU-test-uuid"]},
        workload={"request": {"interface": "cli", "input": "test request", "parameters": {"seed": 0, "max_tokens": 8}}},
        budget={"seconds": 60, "max_calls": 8, "model_cost_usd": 0},
        contract={"preserve_behavior": True, "execution": {
            "version": 1,
            "framework": {"commit": "a" * 40, "prepare_run": "environment-source", "prepare_call": "call-0001-fixed", "prepare_output_hash": "d" * 64},
            "adapter": {"name": "native-cli", "version": "1", "build_parameters": {"cuda": True}},
            "permissions": {"network": "none", "write_scopes": ["build", "scratch"], "tools": ["cmake", "cc"], "trusted_code": True},
            "limits": {"step_seconds": 20, "cpu_threads": 1, "memory_mib": 512, "gpu_memory_mib": 2048, "disk_mib": 256, "output_mib": 16},
            "acceptance": {"scope": "smoke", "parser": {"name": "plain-text", "version": "1"},
                "verifier": {"name": "smoke", "version": "1"}, "reference": None, "tolerances": None}}})


def grant(task, actions=None):
    value = task.snapshot() if isinstance(task, Task) else task
    return {"task_hash": digest(value), "approved_by": "test-user", "actions": actions or ["probe", "build", "run_once"]}


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sherpa-contract-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "weights.gguf").write_bytes(b"test")
        self.task = fixture(self.root).snapshot()
        self.action = {"kind": "run_once", "build_call": "call-build-001"}

    def check(self, task=None, action=None):
        task = self.task if task is None else task
        return validate_execution(task, self.action if action is None else action, grant(task))

    def codes(self, result):
        return {d["code"] for d in result["diagnostics"]}

    def test_valid_single_model_produces_data_only_pending_action(self):
        before = copy.deepcopy(self.task)
        result = self.check()
        self.assertEqual(result["status"], "valid", result["diagnostics"])
        self.assertEqual(result["pending_action"]["task"]["model"]["selected_set"], "q4")
        self.assertEqual(result["task_hash"], digest(before))
        self.assertEqual(result["readiness"], "unverified")
        self.assertFalse(result["executed"])
        self.assertFalse(result["execution_authorized"])
        self.assertEqual(result["pending_action"]["status"], "requires_runtime_checks")
        self.assertIn("successful_build_call_and_artifact_binding", result["pending_action"]["required_runtime_checks"])
        self.assertEqual(self.task, before)
        result["pending_action"]["task"]["model"]["selected_set"] = "changed"
        self.assertEqual(self.task, before)

    def test_accepts_existing_task_type_without_changing_task_defaults(self):
        task = fixture(self.root)
        self.assertEqual(self.check(task)["status"], "valid")
        legacy = Task("old", {"kind": "local", "path": str(self.root)}, {"path": str(self.root)})
        self.assertEqual(legacy.contract, {"preserve_behavior": True})
        self.assertEqual(self.check(legacy)["status"], "invalid")

    def test_shards_and_support_files_are_a_single_selected_set(self):
        for name in ("part-1.gguf", "part-2.gguf", "tokenizer.json"):
            (self.root / name).write_bytes(b"test")
        item = self.task["model"]["sets"][0]
        item["layout"] = "sharded"
        base = item["weights"][0]
        item["weights"] = [{**base, "path": "part-1.gguf"}, {**base, "path": "part-2.gguf"}]
        item["support"] = [{**base, "path": "tokenizer.json"}]
        result = self.check()
        self.assertEqual(result["status"], "valid", result["diagnostics"])
        self.assertIn("selected_file_content_and_shard_identity", result["pending_action"]["required_runtime_checks"])

    def test_two_alternative_weights_cannot_masquerade_as_single_file_layout(self):
        self.task["model"]["sets"][0]["weights"].append(copy.deepcopy(self.task["model"]["sets"][1]["weights"][0]))
        result = self.check()
        self.assertIn("ambiguous_weights", self.codes(result))
        self.assertIsNone(result["pending_action"])

    def test_no_implicit_choice_even_with_one_set(self):
        self.task["model"]["sets"] = self.task["model"]["sets"][:1]
        for value in (None, [], ["q4", "q8"], "missing"):
            with self.subTest(value=value):
                self.task["model"]["selected_set"] = value
                self.assertIn("selection_required", self.codes(self.check()))

    def test_invalid_selected_file_does_not_fall_back_to_other_set(self):
        self.task["model"]["selected_set"] = "q8"
        result = self.check()
        self.assertIn("file_unavailable", self.codes(result))
        self.assertEqual(self.task["model"]["selected_set"], "q8")
        self.assertIsNone(result["pending_action"])

    def test_duplicate_sets_and_files_are_reported(self):
        self.task["model"]["sets"].append(copy.deepcopy(self.task["model"]["sets"][0]))
        first = self.task["model"]["sets"][0]
        first["support"].append(copy.deepcopy(first["weights"][0]))
        codes = self.codes(self.check())
        self.assertTrue({"duplicate_set", "duplicate_file"} <= codes)

    def test_probe_does_not_require_model_request_or_prepared_framework(self):
        self.task.pop("model")
        self.task.pop("workload")
        execution = self.task["contract"]["execution"]
        for field in ("framework", "adapter", "acceptance"):
            execution.pop(field)
        self.task["resources"]["gpus"] = []
        execution["permissions"]["write_scopes"] = []
        execution["permissions"]["trusted_code"] = False
        with patch("sherpa.runtime.execution_contract.metadata", side_effect=AssertionError("probe must not open model")):
            result = self.check(action={"kind": "probe", "targets": ["host", "toolchain", "gpu"]})
        self.assertEqual(result["status"], "valid", result["diagnostics"])
        self.assertNotIn("model_files", [c["name"] for c in result["checks"]])
        self.assertFalse(result["executed"])

    def test_build_does_not_require_model_request_or_acceptance(self):
        self.task["model"] = None
        self.task["workload"] = None
        self.task["contract"]["execution"].pop("acceptance")
        self.task["resources"]["gpus"] = []
        with patch("sherpa.runtime.execution_contract.metadata", side_effect=AssertionError("build does not read models")):
            result = self.check(action={"kind": "build"})
        self.assertEqual(result["status"], "valid", result["diagnostics"])

    def test_collects_independent_errors_and_skips_only_dependents(self):
        self.task["model"]["selected_set"] = None
        self.task["workload"] = {}
        self.task["contract"]["execution"]["framework"]["commit"] = "HEAD"
        with patch("sherpa.runtime.execution_contract.metadata", side_effect=AssertionError("not selected")):
            result = self.check()
        self.assertTrue({"selection_required", "missing_field", "unpinned_framework"} <= self.codes(result))
        checks = {c["name"]: c for c in result["checks"]}
        self.assertEqual(checks["model_files"]["blocked_by"], ["model_selection"])
        self.assertEqual(checks["request"]["status"], "failed")
        self.assertEqual(checks["resources"]["status"], "passed")
        self.assertTrue(all(d["field"] and d["remedy"] for d in result["diagnostics"]))

    def test_missing_action_parameter_does_not_hide_independent_errors(self):
        self.task["resources"]["gpus"] = []
        self.task["contract"]["execution"]["permissions"]["trusted_code"] = False
        result = validate_execution(self.task, {"kind": "run_once"})
        self.assertTrue({"build_reference_required", "untrusted_code", "gpu_identity_required", "object_required"} <= self.codes(result))
        checks = {c["name"]: c["status"] for c in result["checks"]}
        for key in ("action", "permissions", "resources", "approval"):
            self.assertEqual(checks[key], "failed")
        self.assertIsNone(result["pending_action"])
        unknown = validate_execution(self.task, {"kind": []})
        checks = {c["name"]: c for c in unknown["checks"]}
        self.assertEqual(checks["permissions"]["blocked_by"], ["action_kind"])
        self.assertEqual(checks["approval"]["status"], "skipped")
        self.assertIsNone(unknown["pending_action"])

    def test_authorization_is_required_and_not_inferred_from_report(self):
        result = validate_execution(self.task, self.action)
        self.assertEqual(result["status"], "invalid")
        result = validate_execution(self.task, self.action, {"status": "proposed", "installed": False})
        self.assertIsNone(result["pending_action"])
        result = validate_execution({"plan": self.task, "readiness": "unverified"}, self.action, grant(self.task))
        self.assertIsNone(result["pending_action"])

    def test_approval_binds_source_selection_request_budget_and_action_scope(self):
        approval = grant(self.task)
        for path, value in ((["target", "url"], "https://example.com/other.git"),
                            (["model", "selected_set"], "q8"),
                            (["workload", "request", "input"], "other request"),
                            (["budget", "seconds"], 120),
                            (["contract", "execution", "adapter", "version"], "2")):
            task = copy.deepcopy(self.task)
            node = task
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
            with self.subTest(path=path):
                self.assertIn("approval_mismatch", self.codes(validate_execution(task, self.action, approval)))
        self.assertIn("action_not_approved", self.codes(validate_execution(self.task, self.action, grant(self.task, ["probe"]))))

    def test_path_traversal_absolute_and_noncanonical_paths_are_rejected(self):
        file = self.task["model"]["sets"][0]["weights"][0]
        for value in ("../weights.gguf", "/weights.gguf", "a/../weights.gguf", "./weights.gguf", "a//weights.gguf", ".", "", "a\\b", "secrets.json"):
            with self.subTest(value=value):
                file["path"] = value
                self.assertIn("invalid_path", self.codes(self.check()))

    def test_links_and_special_files_are_rejected_without_blocking(self):
        (self.root / "linked.gguf").symlink_to(self.root / "weights.gguf")
        (self.root / "linked-dir").symlink_to(self.root, target_is_directory=True)
        os.mkfifo(self.root / "fifo.gguf")
        for value in ("linked.gguf", "linked-dir/weights.gguf", "fifo.gguf"):
            with self.subTest(value=value):
                self.task["model"]["sets"][0]["weights"][0]["path"] = value
                self.assertIn("file_unavailable", self.codes(self.check()))
        self.task["model"]["sets"][0]["weights"][0]["path"] = "weights.gguf"
        self.task["model"]["path"] = str(self.root / "linked-dir")
        self.assertIn("file_unavailable", self.codes(self.check()))

    def test_file_size_change_is_reported_without_hashing_weights(self):
        (self.root / "weights.gguf").write_bytes(b"changed")
        self.assertIn("file_changed", self.codes(self.check()))

    def test_metadata_opens_only_opath_and_keeps_hash_verification_pending(self):
        original = os.open
        opened = []
        def track(path, flags, *args, **kwargs):
            opened.append((path, flags))
            self.assertTrue(flags & os.O_PATH)
            return original(path, flags, *args, **kwargs)
        with patch("sherpa.runtime.execution_contract.os.open", side_effect=track):
            result = self.check()
        self.assertEqual(result["status"], "valid")
        self.assertTrue(opened)
        self.assertEqual(metadata(str(self.root), "weights.gguf")["bytes"], 4)

    def test_finite_budget_and_resource_rules(self):
        for value in (True, 0, -1, float("inf"), float("nan"), "20"):
            with self.subTest(value=value):
                task = copy.deepcopy(self.task)
                task["contract"]["execution"]["limits"]["step_seconds"] = value
                approval = grant(self.task) if isinstance(value, float) else grant(task)
                result = validate_execution(task, self.action, approval)
                self.assertEqual(result["status"], "invalid")
        self.task["contract"]["execution"]["limits"]["step_seconds"] = 61
        self.assertIn("exceeds_budget", self.codes(self.check()))
        self.task["resources"]["gpus"] = []
        self.assertIn("gpu_identity_required", self.codes(self.check()))

    def test_permissions_cannot_expand_to_arbitrary_writes_network_or_untrusted_code(self):
        p = self.task["contract"]["execution"]["permissions"]
        p.update(network="any", write_scopes=["/home"], trusted_code=False)
        self.assertTrue({"unsupported_permission", "invalid_list", "write_scope_required", "untrusted_code"} <= self.codes(self.check()))

    def test_behavior_reference_is_required_and_smoke_does_not_claim_behavior(self):
        a = self.task["contract"]["execution"]["acceptance"]
        a["scope"] = "behavior"
        self.assertEqual(self.check()["status"], "invalid")
        a.update(reference={"path": str(self.root / "reference.json"), "sha256": "c" * 64}, tolerances={"absolute": 0, "relative": 0.01})
        self.assertIn("reference_unavailable", self.codes(self.check()))
        (self.root / "reference.json").write_text('{"expected": 0.5}')
        result = self.check()
        self.assertIn("reference_content_identity_and_comparison_rules", result["pending_action"]["required_runtime_checks"])
        self.assertEqual(result["status"], "valid", result["diagnostics"])
        self.assertEqual(result["readiness"], "unverified")
        a["scope"] = "smoke"
        self.assertIn("scope_conflict", self.codes(self.check()))

    def test_bad_shapes_never_escape_as_exceptions(self):
        for field in ("target", "model", "resources", "objective", "workload", "contract", "budget"):
            for value in (None, True, [], 42, "wrong"):
                task = copy.deepcopy(self.task)
                task[field] = value
                with self.subTest(field=field, value=value):
                    self.assertEqual(self.check(task)["status"], "invalid")
        for value in (None, [], 3, "finish", {"kind": []}, {"kind": {}}, {"kind": "tool"}, {"kind": "build", "command": "arbitrary"}):
            with self.subTest(action=value):
                self.assertEqual(validate_execution(self.task, value, grant(self.task))["status"], "invalid")
        for field in ("framework", "adapter", "permissions", "limits", "acceptance"):
            task = copy.deepcopy(self.task)
            task["contract"]["execution"][field] = []
            self.assertEqual(self.check(task)["status"], "invalid")

    def test_nested_field_type_mutations_always_return_diagnostics(self):
        paths = []
        def collect(value, path=()):
            if isinstance(value, dict):
                for key, child in value.items():
                    paths.append(path + (key,))
                    collect(child, path + (key,))
            elif isinstance(value, list):
                for i, child in enumerate(value):
                    paths.append(path + (i,))
                    collect(child, path + (i,))
        collect(self.task)
        for path in paths:
            for value in (None, False, 1, [], {}, ""):
                task = copy.deepcopy(self.task)
                target = task
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.subTest(path=path, value=value):
                    report = validate_execution(task, self.action)
                    self.assertEqual(report["status"], "invalid")
                    self.assertIsNone(report["pending_action"])
                    self.assertFalse(report["executed"])

    def test_content_identity_is_not_falsely_verified_by_equal_size(self):
        (self.root / "weights.gguf").write_bytes(b"fake")
        result = self.check()
        self.assertEqual(result["status"], "valid")
        self.assertFalse(result["execution_authorized"])
        self.assertIn("selected_file_content_and_shard_identity", result["pending_action"]["required_runtime_checks"])
        self.assertEqual(result["pending_action"]["approval_hash"], digest(grant(self.task)))

    def test_invalid_hash_and_unknown_fields_reported(self):
        file = self.task["model"]["sets"][0]["weights"][0]
        file.update(sha256="unknown", unexpected=True)
        self.task["contract"]["execution"]["framework"]["commit"] = "abc123"
        self.assertTrue({"invalid_sha256", "unexpected_field", "unpinned_framework"} <= self.codes(self.check()))

    def test_validator_calls_no_provider_process_probe_or_git(self):
        with ExitStack() as stack:
            for name in ("subprocess.Popen", "socket.socket", "os.system", "sherpa.platform.machine.probe",
                         "sherpa.runtime.framework.prepare", "sherpa.runtime.workspace.git", "sherpa.agent.Router.call"):
                stack.enter_context(patch(name, side_effect=AssertionError(name)))
            self.assertEqual(self.check()["status"], "valid")
            self.assertEqual(self.check(action={"kind": "build"})["status"], "valid")
            self.assertEqual(self.check(action={"kind": "probe", "targets": ["host"]})["status"], "valid")

    def test_diagnostics_are_deterministic(self):
        self.task["model"]["selected_set"] = None
        self.task["workload"] = {}
        self.assertEqual(self.check(), self.check())

    def test_cli_checks_without_store_creation_or_writes(self):
        paths = [self.root / name for name in ("task.json", "action.json", "approval.json")]
        for p, value in zip(paths, (self.task, self.action, grant(self.task))):
            p.write_text(json.dumps(value))
        before = {p: p.read_bytes() for p in self.root.iterdir()}
        out = io.StringIO()
        with patch("sherpa.cli.Store", side_effect=AssertionError("no run should be created")), redirect_stdout(out):
            code = main(["--data-root", str(self.root / "absent"), "env", "check-contract", str(paths[0]), "--action", str(paths[1]), "--approval", str(paths[2])])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["status"], "valid")
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.iterdir()})
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["env", "check-contract", str(paths[0]), "--action", str(paths[1])])
        self.assertEqual(code, 2)
        self.assertIsNone(json.loads(out.getvalue())["pending_action"])

    def test_loader_rejects_duplicate_nonfinite_huge_nested_and_special_inputs(self):
        p = self.root / "input.json"
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', '"' + 'x' * 131072 + '"', '[' * 30 + '0' + ']' * 30):
            with self.subTest(raw=raw[:30]):
                p.write_text(raw)
                with self.assertRaises(ValueError):
                    load_document(p)
        p.unlink()
        os.mkfifo(p)
        with self.assertRaises(ValueError):
            load_document(p)
        p.unlink()
        p.symlink_to(self.root / "weights.gguf")
        with self.assertRaises(OSError):
            load_document(p)

    def test_cli_parse_error_is_not_a_valid_contract(self):
        p = self.root / "bad.json"
        p.write_text('{"x":1,"x":2}')
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(["env", "check-contract", str(p), "--action", str(p)]), 1)


if __name__ == "__main__":
    unittest.main()
