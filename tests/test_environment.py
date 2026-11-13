from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from sherpa.cli import main
from sherpa.platform import environment, machine
from sherpa.runtime.environment import build_plan, plan_environment
from sherpa.runtime.models import Task
from sherpa.runtime.store import Store, read_json


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sherpa-env-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "package"
        self.source.mkdir()
        self.store = Store(self.root / "data")
        self.write("setup_jev.sh", 'PYTHON="${PYTHON:-python3.11}"\n'
                   'python -m pip install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu130\n'
                   'python -m pip install -r "$ROOT_DIR/requirements.txt"\n')
        self.write("setup_prism.sh", 'PYTHON="${PYTHON:-python3.12}"\n'
                   'python -m pip install "torch==9.0.0"\n'
                   'python -m pip install -r "$ROOT_DIR/External/requirements.txt"\n')
        self.write("requirements.txt", "transformers==5.8.0\npeft==0.18.1\n")
        self.write("jev/environment.json", json.dumps({"python": "3.11.6 [GCC info]",
                                                       "packages": {"torch": "2.14.0", "triton": "3.8.0"}}))
        self.write("README.md", "Comparison instructions for JEV and Prism. Further requirements need review.")
        self.facts = {"host": {"system": "Linux", "controller_python": "3.12.0"},
                      "tools": {"uv": "/tools/uv", "git": "/usr/bin/git", "nsys": None},
                      "gpu_status": "observed", "queries": {}, "topology": {},
                      "gpus": [{"index": 0, "uuid": "GPU-a", "name": "Example GPU",
                                "occupancy": "idle_observed", "memory_total_mib": 16000,
                                "memory_used_mib": 1, "compute_capability": "12.0", "driver": "test"}],
                      "cuda_runtime_verified": False}

    def write(self, name, text):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_profiles_are_separate_and_not_silently_guessed(self):
        result = environment.inspect(self.source)
        self.assertEqual(set(result["profiles"]), {"jev", "prism"})
        self.assertIsNone(result["selected_profile"])
        self.assertEqual(result["packages"], [])
        self.assertTrue(any("multiple" in issue for issue in result["issues"]))
        selected = environment.inspect(self.source, "jev")
        packages = {p["name"]: p for p in selected["packages"]}
        self.assertEqual(packages["torch"]["versions"], ["2.14.0"])
        self.assertEqual(packages["torch"]["indexes"], ["https://download.pytorch.org/whl/cu130"])
        self.assertTrue(packages["triton"]["only_snapshot"])
        self.assertEqual([p["value"] for p in selected["python"]], ["3.11.6", "3.11"])
        self.assertNotIn("setup_prism.sh", selected["included_sources"])
        self.assertFalse(any("External" in issue for issue in selected["issues"]))

    def test_external_requirement_is_explicitly_unresolved(self):
        result = environment.inspect(self.source, "prism")
        self.assertTrue(any("External/requirements.txt" in issue for issue in result["issues"]))
        self.assertNotIn("requirements.txt", result["included_sources"])
        self.assertFalse(any(p["name"] == "triton" for p in result["packages"]))

    def test_snapshot_does_not_silently_override_conflicting_install_pin(self):
        self.write("jev/environment.json", '{"python":"3.11.6","packages":{"torch":"2.13.0"}}')
        result = environment.inspect(self.source, "jev")
        package = next(p for p in result["packages"] if p["name"] == "torch")
        self.assertTrue(package["conflict"])
        self.assertEqual(package["versions"], ["2.13.0", "2.14.0"])
        self.assertEqual(len(package["evidence"]), 2)

    def test_distinct_local_builds_are_a_conflict(self):
        self.write("requirements.txt", "torch==2.14.0+cu128\ntorch==2.14.0+cu130\n")
        result = environment.inspect(self.source, "jev")
        self.assertTrue(next(p for p in result["packages"] if p["name"] == "torch")["conflict"])

    def test_scripts_are_never_executed(self):
        sentinel = self.root / "MUST_NOT_EXIST"
        script = self.source / "setup_jev.sh"
        script.write_text(script.read_text() + f"\ntouch {sentinel}\n")
        before = {p.relative_to(self.source).as_posix(): p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        with patch("subprocess.Popen", side_effect=AssertionError("target scripts must not execute")):
            result = environment.inspect(self.source, "jev")
        after = {p.relative_to(self.source).as_posix(): p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse(sentinel.exists())
        self.assertEqual(result["selected_profile"], "jev")

    def test_unsupported_ranges_and_python_constraints_are_preserved(self):
        result = environment.parse({"path": "pyproject.toml", "kind": "pyproject", "text":
                                    '[project]\nrequires-python = ">=3.11,<3.12"\ndependencies = ["torch>=2.0"]'})
        self.assertEqual(result["python"][0]["value"], ">=3.11,<3.12")
        self.assertEqual(result["packages"], [])
        self.assertEqual(result["unresolved"][0]["declaration"], "torch>=2.0")

    def test_conditional_script_requires_review(self):
        script = self.source / "setup_jev.sh"
        script.write_text("if [ true ]; then\n" + script.read_text() + "\nfi\n")
        result = environment.inspect(self.source, "jev")
        self.assertTrue(any("conditional" in item["reason"] for item in result["unresolved"]))

    def test_unread_documents_are_not_treated_as_verified_requirements(self):
        result = environment.inspect(self.source, "jev")
        self.assertTrue(any("no LLM" in item["reason"] for item in result["unresolved"]))
        plan = build_plan(result, self.facts)
        self.assertEqual(plan["readiness"], "unverified")
        self.assertEqual(plan["status"], "needs_review")
        self.assertFalse(plan["installed"])
        self.assertNotIn("environment_id", plan)

    def test_symlinks_and_fifo_are_not_read(self):
        outside = self.root / "outside"
        outside.write_text("not to read")
        (self.source / "environment.json").symlink_to(outside)
        os.mkfifo(self.source / "requirements-extra.txt")
        result = environment.inspect(self.source, "jev")
        self.assertTrue(any("symlink" in warning for warning in result["warnings"]))
        self.assertTrue(any("not a regular" in warning for warning in result["warnings"]))

    def test_scan_bounds_and_large_weights_are_not_read(self):
        self.write("model.safetensors", "not inspected")
        self.write(".venv/site-packages/requirements.txt", "not inspected")
        self.write("README-large", "x" * (environment.MAX_BYTES + 1))
        result = environment.inspect(self.source, "jev")
        self.assertFalse(any(".venv" in item["path"] or "safetensors" in item["path"] for item in result["sources"]))
        self.assertTrue(any("128 KiB" in warning for warning in result["warnings"]))
        with patch.object(environment, "MAX_ENTRIES", 1), self.assertRaises(ValueError):
            environment.inspect(self.source)

    def test_include_cannot_escape_target(self):
        self.write("requirements.txt", "-r ../outside.txt\n")
        result = environment.inspect(self.source, "jev")
        self.assertTrue(any("unsafe" in issue for issue in result["issues"]))

    def test_unknown_profile_is_not_accepted(self):
        with self.assertRaises(ValueError):
            environment.inspect(self.source, "invented")

    def test_machine_parser_keeps_unknown_memory_and_sparse_indices(self):
        parsed = machine.parse_gpus("0, GPU-a, A, 16000, 1, 1.0, 12.0\n3, GPU-b, B, [N/A], [N/A], 1.0, [N/A]\n")
        self.assertEqual(parsed[1]["index"], 3)
        self.assertIsNone(parsed[1]["memory_used_mib"])
        self.assertIsNone(parsed[1]["compute_capability"])
        matrix = machine.topology("\tGPU0\tGPU3\tCPU Affinity\nGPU0\tX\tNV4\t0-7\nGPU3\tNV4\tX\t0-7\n")
        self.assertEqual(matrix["GPU0"]["GPU3"], "NV4")

    def test_multiple_gpus_explicit_selection_and_duplicates(self):
        self.facts["gpus"].append({**self.facts["gpus"][0], "uuid": "GPU-b", "index": 3, "occupancy": "busy"})
        selected = machine.select_gpus(self.facts, ["GPU-a", "3"])
        self.assertEqual(selected["uuids"], ["GPU-a", "GPU-b"])
        self.assertTrue(selected["issues"])
        self.assertFalse(selected["reserved"])
        self.assertEqual(machine.select_gpus(self.facts)["uuids"], ["GPU-a"])
        with self.assertRaises(ValueError):
            machine.select_gpus(self.facts, ["0", "GPU-a"])
        self.assertTrue(machine.select_gpus(self.facts, count=2)["issues"])

    def test_missing_nvidia_smi_is_unknown_not_no_gpu(self):
        with patch.object(machine.shutil, "which", return_value=None), patch("subprocess.Popen") as popen:
            result = machine.probe(self.root)
        popen.assert_not_called()
        self.assertEqual(result["gpu_status"], "unknown")
        self.assertIn("not proof", result["gpu_reason"])

    def test_probe_processes_unknown_means_cards_not_claimed_free(self):
        replies = [{"status": "observed", "raw": "0, GPU-a, A, 16000, 0, 1.0, 12.0\n"},
                   {"status": "unknown", "reason": "unsupported"}, {"status": "unknown"}]
        with patch.object(machine.shutil, "which", return_value="/tools/test"), patch.object(machine, "query", side_effect=replies):
            result = machine.probe(self.root)
        self.assertEqual(result["gpus"][0]["occupancy"], "unknown")
        self.assertEqual(machine.select_gpus(result)["uuids"], [])

    def test_readonly_plan_has_real_trace_not_mock_calls(self):
        with patch.object(machine, "probe", return_value=self.facts):
            run, plan = plan_environment(self.store, self.source, "jev", ["0"])
        self.assertEqual(run.state()["kind"], "environment")
        self.assertEqual(run.state()["status"], "completed", run.state().get("stop_reason"))
        self.assertEqual(plan["readiness"], "unverified")
        self.assertTrue(run.state()["version"]["source_id"])
        self.assertEqual(len(run.records("calls")), 2)
        for call in run.records("calls"):
            self.assertFalse(call["synthetic"])
            self.assertEqual(call["kind"], "tool")
            self.assertTrue((run.path / call["input"]).is_file())
            self.assertTrue((run.path / call["output"]).is_file())
        self.assertEqual(plan, read_json(run.path / "environment/plan.json"))
        self.assertFalse((self.source / ".git").exists())
        self.assertFalse((self.source / ".venv").exists())
        self.assertFalse((self.store.root / "envs").exists())
        self.assertIsNone(run.state()["best"])
        self.assertIsNone(run.state()["baseline"])

    def test_failed_scan_finishes_call_and_run(self):
        with patch.object(environment, "inspect", side_effect=ValueError("bad declaration")), patch.object(machine, "probe") as probe:
            run, plan = plan_environment(self.store, self.source, "jev")
        probe.assert_not_called()
        self.assertIsNone(plan)
        self.assertEqual(run.state()["status"], "failed")
        self.assertEqual(run.records("calls")[0]["status"], "failed")
        self.assertIsNone(run.state()["active_call"])

    def test_single_profile_does_not_absorb_an_unrelated_snapshot(self):
        (self.source / "setup_prism.sh").unlink()
        self.write("prism/environment.json", '{"python":"3.12","packages":{"other-model":"9.0"}}')
        result = environment.inspect(self.source, "jev")
        self.assertNotIn("other-model", [p["name"] for p in result["packages"]])
        self.assertNotIn("prism/environment.json", result["included_sources"])
        self.assertTrue(any(item.get("source") == "prism/environment.json" for item in result["unresolved"]))

    def test_fifo_is_not_opened_for_reading(self):
        fifo = self.source / "requirements-fifo.txt"
        os.mkfifo(fifo)
        original_open = os.open
        flags_seen = []
        def tracked_open(path, flags, *args, **kwargs):
            flags_seen.append(flags)
            return original_open(path, flags, *args, **kwargs)
        with patch.object(environment.os, "open", side_effect=tracked_open):
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                environment.read_bounded(fifo)
        self.assertEqual(len(flags_seen), 1)
        self.assertTrue(flags_seen[0] & os.O_PATH)

    def test_cancellation_after_build_does_not_return_a_plan(self):
        def build_and_cancel(*args, **kwargs):
            value = build_plan(*args, **kwargs)
            run_id = self.store.runs()[-1]["id"]
            self.store.open_run(run_id).cancel("cancel after computing plan")
            return value
        with patch.object(machine, "probe", return_value=self.facts), patch(
                "sherpa.runtime.environment.build_plan", side_effect=build_and_cancel):
            run, plan = plan_environment(self.store, self.source, "jev")
        self.assertIsNone(plan)
        self.assertEqual(run.state()["status"], "cancelled")
        self.assertFalse((run.path / "environment/plan.json").exists())

    def test_saving_plan_failure_is_a_failed_run(self):
        from sherpa.runtime.store import atomic_json
        def fail_plan_write(path, value):
            if path.name == "plan.json":
                raise OSError("simulated disk failure")
            return atomic_json(path, value)
        with patch.object(machine, "probe", return_value=self.facts), patch(
                "sherpa.runtime.environment.atomic_json", side_effect=fail_plan_write):
            run, plan = plan_environment(self.store, self.source, "jev")
        self.assertIsNone(plan)
        self.assertEqual(run.state()["status"], "failed")
        self.assertIn("disk failure", run.state()["stop_reason"])

    def test_environment_requires_task_and_input_is_protected(self):
        with self.assertRaises(ValueError):
            self.store.create_run(None, "environment")
        nested = Store(self.source / "runtime")
        with self.assertRaises(ValueError):
            plan_environment(nested, self.source, "jev")
        self.assertFalse(nested.root.exists())

    def test_cli_json_is_valid_and_does_not_hide_unresolved_state(self):
        stream = io.StringIO()
        with redirect_stdout(stream), patch.object(machine, "probe", return_value=self.facts):
            code = main(["--data-root", str(self.store.root), "env", "plan", str(self.source),
                         "--profile", "jev", "--gpus", "0", "--json"])
        self.assertEqual(code, 0)
        result = json.loads(stream.getvalue())
        self.assertEqual(result["plan"]["status"], "needs_review")
        self.assertFalse(result["plan"]["installed"])
        self.assertTrue(result["run_id"].startswith("environment-"))


if __name__ == "__main__":
    unittest.main()
