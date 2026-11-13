from __future__ import annotations

from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sherpa.platform.llama_completion import plan_completion
from sherpa.runtime.calls import RunStopped
from sherpa.runtime.execution_records import ARTIFACT_LIMIT, JSON_LIMIT, RecordReader, check_completion_records
from sherpa.runtime.store import atomic_json, digest
from test_llama_completion import inputs


class RecordTests(unittest.TestCase):
    """Handwritten trace fixtures test real-record shapes, not actual build evidence."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sherpa-records-")
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / "data"
        self.root = self.data / "runs" / "run-1"
        self.task, self.loc = inputs()
        self.loc.update(run_root=str(self.root), source=str(self.root / "workspace/framework"),
                        build=str(self.root / "build"), scratch=str(self.root / "scratch"))
        self.prepared = {"commit": self.task["contract"]["execution"]["framework"]["commit"],
                         "source_commit": self.task["contract"]["execution"]["framework"]["commit"],
                         "source": self.task["target"], "path": self.loc["source"], "import_version": "framework-import-v2",
                         "import_mode": "pinned_local_copy", "model_weights_copied": False,
                         "framework_max_mib": 512, "submodules_not_fetched": [], "symlinks_not_read_by_tools": []}
        self.task["contract"]["execution"]["framework"].update(
            prepare_run="run-1", prepare_call="call-prepare", prepare_output_hash=digest(self.prepared))
        self.state = {"id": "run-1", "kind": "environment", "scope": "environment_execution", "status": "running",
                      "task": self.task, "framework": self.prepared, "active_call": None}
        atomic_json(self.root / "run.json", self.state)
        self.make_call("call-prepare", "framework.prepare", "framework-import-v2",
                       {"target": self.task["target"], "max_mib": 512, "pinned_commit": self.prepared["commit"]}, self.prepared)
        self.plan = plan_completion(self.task, self.loc)
        self.assertEqual(self.plan["status"], "draft", self.plan["diagnostics"])
        for i, phase in enumerate(("configure", "build")):
            inp = {"task_hash": digest(self.task), "plan_hash": self.plan["plan_hash"], "command": self.plan["commands"][i]}
            out = {"process": {"command": inp["command"], "status": "succeeded", "exit_code": 0, "cleanup_complete": True},
                   "verification": {"status": "pass", "verifier": "completion-" + phase + "-v1"}}
            if phase == "build":
                config = self.json("call-configure", "record")
                inp.update(configure_call="call-configure", configure_output_hash=config["output_hash"])
                binary = self.root / "build/bin/llama-completion"
                binary.parent.mkdir(parents=True)
                binary.write_bytes(b"not a real binary: never executed")
                binary.chmod(0o700)
                out["artifact"] = {"path": "build/bin/llama-completion", "bytes": binary.stat().st_size,
                                   "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
            self.make_call("call-" + phase, "completion." + phase, "completion-executor-v1", inp, out, process=True,
                           caused_by="call-configure" if phase == "build" else None)

    def make_call(self, call, name, version, inp, out, process=False, caused_by=None):
        base = self.root / "calls" / call
        record = {"id": call, "run_id": "run-1", "kind": "tool", "name": name, "tool_version": version,
                  "status": "succeeded", "synthetic": False, "input": f"calls/{call}/input.json", "input_hash": digest(inp),
                  "output": f"calls/{call}/output.json", "output_hash": digest(out), "caused_by": caused_by}
        if process:
            record.update(cleanup_complete=True, process={"pid": 12345, "start_ticks": "123456"})
        for file, value in (("record", record), ("input", inp), ("output", out)):
            atomic_json(base / (file + ".json"), value)

    def json(self, call, name):
        return json.loads((self.root / "calls" / call / (name + ".json")).read_text())

    def edit(self, call, name, fn, rehash=True):
        value = self.json(call, name)
        fn(value)
        atomic_json(self.root / "calls" / call / (name + ".json"), value)
        if rehash and name != "record":
            record = self.json(call, "record")
            record[name + "_hash"] = digest(value)
            atomic_json(self.root / "calls" / call / "record.json", record)

    def check(self, build="call-build", **kwargs):
        return check_completion_records(self.data, self.task, self.loc, build_call=build, **kwargs)

    def invalid(self, result):
        self.assertEqual(result["status"], "invalid", result)
        self.assertTrue(result["diagnostics"])
        self.assertIsNone(result["artifact"])
        self.assertFalse(result["execution_authorized"])
        return str(result["diagnostics"])

    def test_successful_binding_has_no_execution_or_hardware_verdict(self):
        before = {str(p): p.read_bytes() for p in self.data.rglob("*") if p.is_file()}
        result = self.check()
        self.assertEqual(result["status"], "records_verified", result)
        self.assertEqual(result["artifact"]["build_call"], "call-build")
        self.assertEqual(result["prepare"]["output_hash"], digest(self.prepared))
        self.assertFalse(result["executed"])
        self.assertFalse(result["execution_authorized"])
        self.assertEqual(result["readiness"], "unverified")
        self.assertIn("authenticated_execution_grant", result["pending_checks"])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.data.rglob("*") if p.is_file()})

    def test_prepare_only_does_not_require_build_cleanup_or_read_binary(self):
        for p in (self.root / "build/bin").iterdir():
            p.unlink()
        result = self.check(build=None)
        self.assertEqual(result["status"], "records_verified", result)
        self.assertIsNone(result["artifact"])
        self.assertEqual([c["name"] for c in result["checks"]], ["plan", "run", "prepare"])

    def test_synthetic_or_wrong_calls_are_rejected(self):
        original = self.json("call-prepare", "record")
        for key, value in (("synthetic", True), ("synthetic", None), ("name", "fixture.prepare"),
                           ("id", "other"), ("run_id", "other"), ("kind", "agent"), ("tool_version", "v1"),
                           ("status", "failed"), ("cleanup_complete", False), ("input", "../input.json")):
            with self.subTest(key=key):
                atomic_json(self.root / "calls/call-prepare/record.json", {**original, key: value})
                result = self.check()
                self.invalid(result)
                self.assertEqual(result["checks"][-1]["blocked_by"], ["prepare"])
        atomic_json(self.root / "calls/call-prepare/record.json", original)
        self.edit("call-build", "record", lambda r: r.update(synthetic=True))
        self.invalid(self.check())

    def test_prepare_output_and_state_must_match_contract(self):
        original = copy.deepcopy(self.prepared)
        for field, value in (("source", {"kind": "git", "url": "https://elsewhere"}), ("commit", "0" * 40),
                             ("path", str(self.root / "wrong")), ("model_weights_copied", True),
                             ("import_mode", "working_tree_snapshot"), ("submodules_not_fetched", [{"path": "external"}])):
            changed = {**original, field: value}
            self.task["contract"]["execution"]["framework"]["prepare_output_hash"] = digest(changed)
            self.state.update(task=self.task, framework=changed)
            atomic_json(self.root / "run.json", self.state)
            self.edit("call-prepare", "output", lambda r: (r.clear(), r.update(changed)))
            with self.subTest(field=field):
                self.invalid(self.check(build=None))

    def test_snapshot_task_and_scope_cannot_be_substituted(self):
        state = copy.deepcopy(self.state)
        for key, value in (("id", "other"), ("scope", "fixture_execution"), ("synthetic", True),
                           ("kind", "reflection"), ("status", "cleanup_failed"), ("active_call", "unfinished"), ("task", {})):
            atomic_json(self.root / "run.json", {**state, key: value})
            with self.subTest(key=key):
                result = self.check()
                self.invalid(result)
                self.assertEqual(result["checks"][-2]["blocked_by"], ["run"])

    def test_json_types_are_part_of_task_framework_and_command_identity(self):
        state = copy.deepcopy(self.state)
        state["task"]["workload"]["request"]["parameters"]["seed"] = False
        atomic_json(self.root / "run.json", state)
        self.assertIn("Run task differs", self.invalid(self.check()))
        state = copy.deepcopy(self.state)
        state["framework"]["model_weights_copied"] = 0
        atomic_json(self.root / "run.json", state)
        self.assertIn("framework snapshot differs", self.invalid(self.check()))
        atomic_json(self.root / "run.json", self.state)
        self.edit("call-build", "output", lambda r: r["process"]["command"].update(inherit_environment=0))
        self.assertIn("actual command differs", self.invalid(self.check()))

    def test_changed_plan_locations_and_source_cannot_reuse_records(self):
        self.loc["cmake"] = "/new/cmake"
        self.assertIn("task or plan mismatch", self.invalid(self.check()))
        self.loc["run_root"] = str(self.data / "other/run-1")
        self.invalid(self.check())

    def test_wrong_prepare_run_cannot_bind_an_external_checkpoint_in_place(self):
        self.task["contract"]["execution"]["framework"]["prepare_run"] = "older-run"
        self.assertIn("prepare must belong", self.invalid(self.check()))

    def test_input_or_output_byte_hash_mismatch_is_detected(self):
        self.edit("call-build", "output", lambda r: r.update(extra=True), rehash=False)
        self.assertIn("hash mismatch", self.invalid(self.check()))
        self.edit("call-build", "output", lambda r: r.pop("extra"))
        self.edit("call-configure", "input", lambda r: r.update(extra=True), rehash=False)
        self.assertIn("hash mismatch", self.invalid(self.check()))

    def test_configure_link_and_actual_command_are_checked(self):
        self.edit("call-build", "input", lambda r: r.update(configure_output_hash="0" * 64))
        self.assertIn("configure output binding", self.invalid(self.check()))
        self.edit("call-build", "input", lambda r: r.update(configure_output_hash=self.json("call-configure", "record")["output_hash"]))
        self.edit("call-build", "output", lambda r: r["process"]["command"].update(argv=["wrong-command"]))
        self.assertIn("actual command", self.invalid(self.check()))

    def test_failed_process_missing_cleanup_and_verifier_rejected(self):
        original = self.json("call-build", "output")
        for field, value in (("exit_code", 7), ("exit_code", False), ("status", "failed"), ("cleanup_complete", False)):
            candidate = copy.deepcopy(original)
            candidate["process"][field] = value
            self.edit("call-build", "output", lambda r: (r.clear(), r.update(candidate)))
            with self.subTest(field=field):
                self.invalid(self.check())
        self.edit("call-build", "output", lambda r: (r.clear(), r.update(original)))
        self.edit("call-build", "output", lambda r: r["verification"].update(status="inconclusive"))
        self.assertIn("verification required", self.invalid(self.check()))
        self.edit("call-build", "record", lambda r: r.pop("cleanup_complete"))
        self.invalid(self.check())

    def test_binary_replacement_even_same_size_is_rejected(self):
        path = self.root / "build/bin/llama-completion"
        path.write_bytes(b"x" * path.stat().st_size)
        self.assertIn("content identity", self.invalid(self.check()))

    def test_wrong_artifact_path_and_mode_are_rejected(self):
        binary = self.root / "build/bin/llama-completion"
        binary.chmod(0o600)
        self.assertIn("not executable", self.invalid(self.check()))
        binary.chmod(0o700)
        self.edit("call-build", "output", lambda r: r["artifact"].update(path="../../outside"))
        self.assertIn("unexpected binary path", self.invalid(self.check()))

    def test_links_fifo_and_directory_cannot_be_read(self):
        path = self.root / "build/bin/llama-completion"
        outside = Path(self.temp.name) / "outside"
        outside.write_text("must not follow")
        path.unlink()
        path.symlink_to(outside)
        self.invalid(self.check())
        path.unlink()
        os.mkfifo(path)
        self.invalid(self.check())
        path.unlink()
        path.mkdir()
        self.invalid(self.check())
        call_dir = self.root / "calls/call-build"
        moved = self.root / "moved-call"
        call_dir.rename(moved)
        call_dir.symlink_to(moved, target_is_directory=True)
        self.invalid(self.check())

    def test_reader_blocks_symlink_roots_and_oversized_files(self):
        linked = Path(self.temp.name) / "linked"
        linked.symlink_to(self.data, target_is_directory=True)
        with self.assertRaises(OSError):
            RecordReader(linked, lambda: None).json("runs/run-1/run.json")
        path = self.root / "calls/call-build/output.json"
        with path.open("wb") as f:
            f.truncate(JSON_LIMIT + 1)
        self.assertIn("exceeds read limit", self.invalid(self.check()))
        with self.assertRaises(ValueError):
            RecordReader("//tmp", lambda: None)
        for relative in ("../secret", "/etc/passwd", "a/../b", "a//b"):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                RecordReader(self.data, lambda: None).read(relative)

    def test_duplicate_keys_nonfinite_or_nonobject_records_rejected(self):
        path = self.root / "calls/call-build/output.json"
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '[]', '{bad'):
            path.write_text(text)
            with self.subTest(text=text):
                self.invalid(self.check())

    def test_large_binary_is_bounded_before_content_read(self):
        path = self.root / "build/bin/llama-completion"
        with path.open("wb") as f:
            f.truncate(ARTIFACT_LIMIT + 1)
        self.assertIn("exceeds read limit", self.invalid(self.check()))

    def test_cancellation_during_binary_read_propagates(self):
        calls = []
        original = RecordReader.read
        def counted(reader, relative, *args, **kwargs):
            if kwargs.get("hash_only"):
                calls.append(relative)
                reader.check = lambda: (_ for _ in ()).throw(RunStopped("cancelled", "test interruption"))
            return original(reader, relative, *args, **kwargs)
        with patch.object(RecordReader, "read", counted), self.assertRaises(RunStopped):
            self.check()
        self.assertEqual(calls, ["runs/run-1/build/bin/llama-completion"])

    def test_no_process_network_model_or_source_content_access(self):
        original = os.open
        opened = []
        def track(path, flags, *args, **kwargs):
            opened.append(str(path))
            if not flags & os.O_PATH:
                self.assertIn(str(path), ("run.json", "record.json", "input.json", "output.json", "llama-completion"))
            return original(path, flags, *args, **kwargs)
        with ExitStack() as stack:
            for name in ("subprocess.Popen", "socket.socket", "sherpa.platform.machine.probe", "sherpa.runtime.workspace.git", "sherpa.runtime.framework.prepare", "sherpa.agent.Router.call"):
                stack.enter_context(patch(name, side_effect=AssertionError(name)))
            stack.enter_context(patch("os.open", side_effect=track))
            result = self.check()
        self.assertEqual(result["status"], "records_verified", result)
        self.assertNotIn("alternative.gguf", opened)
        self.assertNotIn("weights.gguf", opened)

    def test_invalid_shapes_never_become_verified(self):
        for value in (None, [], True, 3, {}, "task"):
            result = check_completion_records(self.data, value, self.loc, build_call="call-build")
            self.invalid(result)
        for value in ("../build", [], {}, "missing-call"):
            self.invalid(self.check(build=value))


if __name__ == "__main__":
    unittest.main()
