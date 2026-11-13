"""E2 fixture-only execution harness; deliberately not a user-program executor."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import sys
import time
import uuid

from .calls import Calls, RunStopped
from .execution_contract import load_document, validate_execution
from .models import Task, identifier
from .store import atomic_json, digest, now, process_identity, read_json

VERSION = "fixture-executor-v1"
SOURCE = b"sherpa e2 synthetic source v1\n"
WEIGHTS = b"sherpa e2 synthetic weights v1\n"
SUPERVISOR = Path(__file__).resolve().parents[1] / "agent" / "supervisor.py"
WORKER = Path(__file__).with_name("fixture_worker.py")
FAULTS = {"delay": 0, "child": False, "stdout_bytes": 0, "stderr_bytes": 0, "exit_code": 0,
          "missing_artifact": False, "malformed": False, "wrong_output": False}
SCENARIOS = ("normal", "failure", "missing-artifact", "wrong-output", "malformed", "timeout", "slow", "output-limit")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def small_file(path, limit=131072):
    """Read only bounded harness-owned files, rejecting symlink components."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("absolute harness artifact path required")
    fd = os.open("/", os.O_PATH | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            child = os.open(component, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        child = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    finally:
        os.close(fd)
    with os.fdopen(child, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("regular harness artifact required")
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError("harness artifact exceeds byte limit")
    return data


def disk_bytes(root):
    total = 0
    for directory, _, names in os.walk(root, followlinks=False):
        for name in names:
            try:
                info = (Path(directory) / name).lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def fault_options(value):
    if not isinstance(value, dict) or set(value) - set(FAULTS):
        raise ValueError("unsupported fixture fault parameter")
    result = {**FAULTS, **value}
    for name in ("child", "missing_artifact", "malformed", "wrong_output"):
        if type(result[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    if type(result["delay"]) not in (int, float) or not math.isfinite(result["delay"]) or not 0 <= result["delay"] <= 120:
        raise ValueError("fixture delay must be in [0, 120]")
    for name, maximum in (("stdout_bytes", 4 * 1024 * 1024), ("stderr_bytes", 4 * 1024 * 1024), ("exit_code", 125)):
        if type(result[name]) is not int or not 0 <= result[name] <= maximum:
            raise ValueError(f"{name} outside fixture limits")
    return result


def _fixture_task(root, framework):
    model = {"kind": "local", "path": str(root / "model"), "selected_set": "fixture", "sets": [
        {"id": "fixture", "layout": "single", "weights": [{"path": "weights.bin", "bytes": len(WEIGHTS), "sha256": sha(WEIGHTS)}], "support": []}]}
    return Task("fixture-execution", {"kind": "local", "path": str(root / "source")}, model,
        resources={"device": "cpu", "gpus": []},
        workload={"request": {"interface": "cli", "input": "fixture request", "parameters": {}}},
        budget={"seconds": 30, "max_calls": 3, "model_cost_usd": 0},
        contract={"preserve_behavior": True, "execution": {
            "version": 1, "framework": framework,
            "adapter": {"name": "fixture", "version": "1", "build_parameters": {}},
            "permissions": {"network": "none", "write_scopes": ["build", "scratch"], "tools": ["fixture-python"], "trusted_code": True},
            "limits": {"step_seconds": 5, "cpu_threads": 1, "memory_mib": 256, "gpu_memory_mib": 0, "disk_mib": 16, "output_mib": 1},
            "acceptance": {"scope": "smoke", "parser": {"name": "fixture-json", "version": "1"},
                           "verifier": {"name": "fixture-smoke", "version": "1"}, "reference": None, "tolerances": None}}})


def prepare_fixture(store, *, seconds=30, max_calls=3, step_seconds=5):
    """Create fixed synthetic inputs and a traceable synthetic preparation record."""
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 180:
        raise ValueError("fixture seconds must be in (0, 180]")
    if type(max_calls) is not int or not 1 <= max_calls <= 12:
        raise ValueError("fixture max_calls must be in [1, 12]")
    if type(step_seconds) not in (int, float) or not math.isfinite(step_seconds) or not 0 < step_seconds <= seconds:
        raise ValueError("fixture step timeout must fit total time")
    root = store.root / "fixtures" / ("fixture-" + uuid.uuid4().hex[:12])
    (root / "source").mkdir(parents=True)
    (root / "model").mkdir()
    (root / "source" / "source.txt").write_bytes(SOURCE)
    (root / "model" / "weights.bin").write_bytes(WEIGHTS)
    task = _fixture_task(root, {})
    prepared = store.create_run(task, kind="environment")
    prepared.update("fixture_prepare_started", status="running", scope="fixture_prepare", synthetic=True)
    calls = Calls(prepared)
    record, directory = calls.new("tool", "fixture.prepare", {"root": str(root)})
    record["synthetic"] = True
    result = {"synthetic": True, "source": task.target, "root": str(root), "commit": sha(SOURCE)[:40],
              "files": {"source/source.txt": sha(SOURCE), "model/weights.bin": sha(WEIGHTS)}}
    calls.finish(record, directory, "succeeded", result, time.monotonic())
    prepared.update("fixture_prepared", status="completed", ended_at=now(), cost=calls.cost_summary())
    binding = {"commit": result["commit"], "prepare_run": prepared.id, "prepare_call": record["id"], "prepare_output_hash": digest(result)}
    snapshot = _fixture_task(root, binding).snapshot()
    snapshot["budget"].update(seconds=seconds, max_calls=max_calls)
    snapshot["contract"]["execution"]["limits"]["step_seconds"] = step_seconds
    return Task(**snapshot), prepared.id


class FixtureExecutor:
    """Only the repository fixture adapter is executable, even with a valid E1 plan."""
    def __init__(self, run, *, allow_fixtures=False):
        if allow_fixtures is not True or run.state().get("scope") != "fixture_execution":
            raise ValueError("explicit fixture-only scope required")
        self.run = run
        self.task = copy.deepcopy(run.state()["task"])
        self.task_hash = digest(self.task)
        self.code_hash = {str(path): sha(small_file(path)) for path in (WORKER, SUPERVISOR)}
        budget = self.task["budget"]
        self.calls = Calls(run, seconds=budget["seconds"], max_calls=budget["max_calls"])
        self.halted = False
        self.approval = {"task_hash": self.task_hash, "approved_by": "fixture-test-driver", "actions": ["probe", "build", "run_once"]}
        from .store import locked
        with locked(run.path / ".fixture-claim.lock", blocking=False):
            if any(r.get("cleanup_complete") is False for r in run.records("calls")):
                raise RunStopped("cleanup_failed", "previous cleanup was not confirmed")
            if run.state().get("fixture_executor_claim") or run.records("calls"):
                raise ValueError("fixture executor already claimed; in-place budget reset/resume is not supported")
            if run.state()["status"] != "running":
                raise ValueError("fixture run must be running")
            run.update("fixture_executor_claimed", fixture_executor_claim={"owner": process_identity(os.getpid()),
                       "task_hash": self.task_hash, "started_at": now()})

    def _inputs(self):
        if digest(self.run.state()["task"]) != self.task_hash:
            raise ValueError("task changed after fixture execution started")
        t = self.task
        e = t["contract"]["execution"]
        if e["adapter"] != {"name": "fixture", "version": "1", "build_parameters": {}}:
            raise ValueError("only the fixed fixture adapter is executable")
        if t["resources"] != {"device": "cpu", "gpus": []} or e["limits"]["gpu_memory_mib"] != 0:
            raise ValueError("fixture executor never accepts GPU tasks")
        if e["permissions"] != {"network": "none", "write_scopes": ["build", "scratch"], "tools": ["fixture-python"], "trusted_code": True}:
            raise ValueError("unsupported fixture permissions")
        if e["acceptance"] != {"scope": "smoke", "parser": {"name": "fixture-json", "version": "1"},
                                "verifier": {"name": "fixture-smoke", "version": "1"}, "reference": None, "tolerances": None}:
            raise ValueError("only fixture smoke verification is available")
        if t["workload"]["request"]["parameters"] != {}:
            raise ValueError("fixture inference has no configurable execution parameters")
        if e["limits"]["memory_mib"] < 128 or e["limits"]["memory_mib"] > 512 or e["limits"]["cpu_threads"] != 1:
            raise ValueError("fixture supports one thread and 128..512 MiB only")
        if e["limits"]["output_mib"] > 4 or e["limits"]["disk_mib"] < 1:
            raise ValueError("fixture output budget must be at most 4 MiB with positive disk allowance")
        binding = e["framework"]
        prepared = self.run.store.open_run(binding["prepare_run"])
        if prepared.state().get("scope") != "fixture_prepare" or prepared.state()["status"] != "completed":
            raise ValueError("synthetic prepare run required")
        call_id = identifier(binding["prepare_call"])
        directory = prepared.path / "calls" / call_id
        rec = load_document(directory / "record.json")
        output = load_document(directory / "output.json")
        if rec["name"] != "fixture.prepare" or rec["status"] != "succeeded" or rec.get("synthetic") is not True:
            raise ValueError("synthetic prepare call required")
        if rec["output"] != f"calls/{call_id}/output.json" or digest(output) != binding["prepare_output_hash"] or digest(output) != rec["output_hash"]:
            raise ValueError("prepare output binding mismatch")
        root = Path(output["root"])
        if root.parent != self.run.store.root / "fixtures" or not root.name.startswith("fixture-"):
            raise ValueError("only harness-owned fixture roots are allowed")
        if output["source"] != t["target"] or t["target"] != {"kind": "local", "path": str(root / "source")}:
            raise ValueError("fixture source binding mismatch")
        expected_model = _fixture_task(root, binding).model
        if t["model"] != expected_model or binding["commit"] != sha(SOURCE)[:40]:
            raise ValueError("only fixed fixture model inputs are allowed")
        for relative, expected in (("source/source.txt", SOURCE), ("model/weights.bin", WEIGHTS)):
            self.calls.check_interrupt()
            if small_file(root / relative) != expected or output["files"].get(relative) != sha(expected):
                raise ValueError("frozen fixture input changed")
        for path, expected in self.code_hash.items():
            if sha(small_file(Path(path))) != expected:
                raise ValueError("fixture worker or supervisor changed during execution")

    def _build(self, call_id):
        call_id = identifier(call_id)
        directory = self.run.path / "calls" / call_id
        record = load_document(directory / "record.json")
        output = load_document(directory / "output.json")
        inputs = load_document(directory / "input.json")
        if record["name"] != "fixture.build" or record["status"] != "succeeded" or record.get("cleanup_complete") is not True:
            raise ValueError("build call was not successfully completed and cleaned")
        if record.get("synthetic") is not True or record["run_id"] != self.run.id:
            raise ValueError("build must belong to the same synthetic run")
        if record["output"] != f"calls/{call_id}/output.json" or record["input"] != f"calls/{call_id}/input.json":
            raise ValueError("build call artifact location mismatch")
        if record["output_hash"] != digest(output) or record["input_hash"] != digest(inputs) or inputs["task_hash"] != self.task_hash:
            raise ValueError("build call identity mismatch")
        if output["verification"]["status"] != "pass":
            raise ValueError("build verification did not pass")
        artifact = output.get("artifact", {})
        if artifact.get("path") != f"calls/{call_id}/artifact.json":
            raise ValueError("unexpected fixture build artifact")
        raw = small_file(directory / "artifact.json")
        expected = {"synthetic": True, "task_hash": self.task_hash, "format": "fixture-artifact-v1"}
        if sha(raw) != artifact.get("sha256") or json.loads(raw) != expected:
            raise ValueError("build artifact changed")
        return artifact

    def _process(self, record, directory, payload):
        limits = self.task["contract"]["execution"]["limits"]
        cap = limits["output_mib"] * 1024 * 1024
        disk_cap = limits["disk_mib"] * 1024 * 1024
        started = time.monotonic()
        deadline = started + limits["step_seconds"]
        proc = None
        reason, status, stopped = None, "succeeded", None
        saved = {"stdout": 0, "stderr": 0}
        observed = {"stdout": 0, "stderr": 0}
        atomic_json(directory / "worker-input.json", payload)
        argv = [sys.executable, "-I", "-B", str(SUPERVISOR), str(os.getpid()),
                sys.executable, "-I", "-B", str(WORKER), str(directory / "worker-input.json")]
        env = {"PATH": os.defpath, "HOME": str(directory), "TMPDIR": str(directory), "CUDA_VISIBLE_DEVICES": "",
               "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
               "SHERPA_AGENT_PROCESS_STATUS": str(directory / "child-status.json")}
        metadata = {"argv": argv, "cwd": str(directory), "env": env, "worker_hashes": self.code_hash,
                    "output_limit_bytes": cap, "disk_limit_bytes": disk_cap, "cleanup_complete": True}
        try:
            with selectors.DefaultSelector() as selector, (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err:
                try:
                    self.calls.check_interrupt()
                    proc = subprocess.Popen(argv, cwd=directory, env=env, stdin=subprocess.DEVNULL,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                    metadata["cleanup_complete"] = False
                    record["cleanup_complete"] = False
                    record["process"] = process_identity(proc.pid)
                    record["logs"] = [f"calls/{record['id']}/{stream}.log" for stream in saved]
                    self.run.save("calls", record["id"], record, "fixture_process_started")
                    for name, pipe, dest in (("stdout", proc.stdout, out), ("stderr", proc.stderr, err)):
                        os.set_blocking(pipe.fileno(), False)
                        selector.register(pipe, selectors.EVENT_READ, (name, dest))
                    while selector.get_map() or proc.poll() is None:
                        self.calls.check_interrupt()
                        if time.monotonic() >= deadline:
                            status, reason = "timed_out", "step_timeout"
                            break
                        if disk_bytes(self.run.path) > disk_cap:
                            status, reason = "failed", "disk_limit"
                            break
                        for key, _ in selector.select(0.025):
                            try:
                                block = os.read(key.fileobj.fileno(), 16384)
                            except BlockingIOError:
                                continue
                            if not block:
                                selector.unregister(key.fileobj)
                                continue
                            name, dest = key.data
                            observed[name] += len(block)
                            amount = min(len(block), max(0, cap - sum(saved.values())))
                            dest.write(block[:amount])
                            dest.flush()
                            saved[name] += amount
                            if sum(observed.values()) > cap:
                                status, reason = "failed", "output_limit"
                                break
                        if reason:
                            break
                finally:
                    if proc is not None:
                        try:
                            metadata["cleanup_complete"] = self.calls.cleanup(proc) is True
                        except (Exception, KeyboardInterrupt) as error:
                            metadata["cleanup_error"] = f"{type(error).__name__}: {error}"
                            try:
                                os.killpg(proc.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            try:
                                proc.wait(timeout=1)
                            except subprocess.TimeoutExpired:
                                pass
                        finally:
                            record["cleanup_complete"] = metadata["cleanup_complete"]
                            for pipe in (proc.stdout, proc.stderr):
                                pipe.close()
        except RunStopped as error:
            status, reason, stopped = error.status, error.reason, error
        except KeyboardInterrupt:
            status, reason = "cancelled", "keyboard interrupt"
            stopped = RunStopped(status, reason)
        except Exception as error:
            status, reason = "failed", f"{type(error).__name__}: {error}"
        if not metadata["cleanup_complete"]:
            self.halted = True
            status, reason = "cleanup_failed", "owned fixture process group did not stop"
            stopped = RunStopped(status, reason)
        else:
            try:
                self.calls.check_interrupt()
            except RunStopped as error:
                status, reason, stopped = error.status, error.reason, error
        child = {}
        try:
            if (directory / "child-status.json").exists():
                child = load_document(directory / "child-status.json")
        except (OSError, ValueError) as error:
            metadata["child_status_error"] = str(error)
            if status == "succeeded":
                status, reason = "failed", "child_status_unavailable"
        metadata.update(status=status, reason=reason, supervisor_returncode=proc.returncode if proc else None,
                        child=child, captured_bytes=saved, observed_bytes=observed,
                        capture_complete=reason is None, seconds=time.monotonic() - started)
        if status == "succeeded" and (proc is None or proc.returncode != 0 or child.get("child_returncode") != 0):
            status, reason = "failed", "nonzero_exit"
            metadata.update(status=status, reason=reason)
        atomic_json(directory / "process.json", metadata)
        record["cleanup_complete"] = metadata["cleanup_complete"]
        return metadata, stopped

    def _verify(self, kind, directory):
        try:
            result = load_document(directory / "result.json")
        except (OSError, ValueError) as error:
            return {"status": "inconclusive", "reason": f"response unavailable: {error}", "synthetic": True}, None
        artifact = None
        if kind == "probe":
            expected = {"synthetic": True, "device": "cpu-fixture", "tools": ["fixture-python"]}
        elif kind == "build":
            expected = {"synthetic": True, "task_hash": self.task_hash, "format": "fixture-artifact-v1"}
            try:
                raw = small_file(directory / "artifact.json")
                if json.loads(raw) != expected:
                    raise ValueError("invalid fixture artifact")
                artifact = {"path": str((directory / "artifact.json").relative_to(self.run.path)), "sha256": sha(raw), "bytes": len(raw)}
            except (OSError, ValueError) as error:
                return {"status": "fail", "reason": f"build artifact missing or invalid: {error}", "synthetic": True}, None
        else:
            q = self.task["workload"]["request"]
            expected = {"synthetic": True, "task_hash": self.task_hash, "request_hash": digest(q),
                        "device": "cpu-fixture", "output": q["input"].upper()}
        passed = result == expected
        return {"status": "pass" if passed else "fail", "reason": "fixture contract matched" if passed else "fixture response mismatch",
                "synthetic": True, "verifier": "fixture-smoke-v1", "result_hash": digest(result)}, artifact

    def execute(self, action, *, fault=None):
        if self.halted or any(r.get("cleanup_complete") is False for r in self.run.records("calls")):
            raise RunStopped("cleanup_failed", "cleanup failure latched; no new process allowed")
        if self.run.state()["status"] != "running":
            raise ValueError("fixture run must be running; terminal runs are not resumed")
        self.calls.check()
        action = copy.deepcopy(action)
        started = time.monotonic()
        record, directory = self.calls.new("tool", "fixture." + str(action.get("kind")),
            {"task_hash": self.task_hash, "action": action, "fault": fault or {}}, caused_by=action.get("build_call"))
        record.update(synthetic=True, tool_version=VERSION, cleanup_complete=True)
        status, output, stopped = "failed", {"synthetic": True, "executed": False}, None
        try:
            self._inputs()
            checked = validate_execution(self.task, action, self.approval)
            atomic_json(directory / "admission.json", checked)
            if checked["status"] != "valid":
                raise ValueError("invalid fixture contract: " + json.dumps(checked["diagnostics"]))
            fault = fault_options({} if fault is None else fault)
            if disk_bytes(self.run.path) > self.task["contract"]["execution"]["limits"]["disk_mib"] * 1024 * 1024:
                raise ValueError("disk_limit: cumulative run data exceeded budget")
            build = self._build(action["build_call"]) if action["kind"] == "run_once" else None
            q = self.task["workload"]["request"]
            payload = {"action": action, "fault": fault, "task_hash": self.task_hash, "request_hash": digest(q),
                       "text": q["input"], "build": build, "memory_mib": self.task["contract"]["execution"]["limits"]["memory_mib"]}
            process, stopped = self._process(record, directory, payload)
            status = process["status"]
            output.update(executed=record["process"] is not None, process=process, task_hash=self.task_hash,
                          verification={"status": "inconclusive", "reason": "process did not complete successfully", "synthetic": True})
            if status == "succeeded":
                self._inputs()
                if build is not None:
                    self._build(action["build_call"])
                verification, artifact = self._verify(action["kind"], directory)
                self._inputs()
                if build is not None:
                    self._build(action["build_call"])
                if disk_bytes(self.run.path) > self.task["contract"]["execution"]["limits"]["disk_mib"] * 1024 * 1024:
                    raise ValueError("disk_limit: cumulative run data exceeded budget")
                self.calls.check_interrupt()
                output.update(verification=verification, artifact=artifact)
        except RunStopped as error:
            status, stopped = error.status, error
            output.update(error=error.reason, verification={"status": "inconclusive", "reason": error.reason, "synthetic": True})
        except KeyboardInterrupt:
            status, stopped = "cancelled", RunStopped("cancelled", "keyboard interrupt")
            output.update(error=stopped.reason, verification={"status": "inconclusive", "reason": stopped.reason, "synthetic": True})
        except Exception as error:
            output.update(error=f"{type(error).__name__}: {error}", verification={"status": "fail", "reason": str(error), "synthetic": True})
            status = "failed"
        result = self.calls.finish(record, directory, status, output, started)
        if stopped is not None:
            raise stopped
        return result


def fixture_demo(store, scenario="normal", *, seconds=30, max_calls=3, announce=None):
    if scenario not in SCENARIOS:
        raise ValueError("unsupported fixture scenario")
    with store.exclusive():
        task, prepared = prepare_fixture(store, seconds=seconds, max_calls=max_calls,
                                         step_seconds=min(seconds, 0.3 if scenario == "timeout" else 5))
        run = store.create_run(task, kind="environment", sources=[prepared])
        run.update("fixture_run_started", status="running", phase="fixture_execution", scope="fixture_execution",
                   synthetic=True, owner=process_identity(os.getpid()), executor_version=VERSION)
        executor = FixtureExecutor(run, allow_fixtures=True)
        if announce:
            announce(run.id)
        outcome, reason, accepted = "completed", "fixture pipeline completed; no target inference", []
        try:
            previous = None
            for kind in ("probe", "build", "run_once"):
                action = {"kind": kind}
                if kind == "probe":
                    action["targets"] = ["host", "toolchain"]
                if kind == "run_once":
                    action["build_call"] = previous
                fault = {}
                if kind == "build":
                    fault = {"failure": {"exit_code": 7}, "missing-artifact": {"missing_artifact": True},
                             "timeout": {"delay": 10, "child": True}, "slow": {"delay": 20, "child": True},
                             "output-limit": {"stdout_bytes": 2 * 1024 * 1024, "child": True}}.get(scenario, {})
                if kind == "run_once":
                    fault = {"wrong-output": {"wrong_output": True}, "malformed": {"malformed": True}}.get(scenario, {})
                result = executor.execute(action, fault=fault)
                if result.status != "succeeded" or result.output["verification"]["status"] != "pass":
                    outcome, reason = "failed", f"{kind}: {result.output.get('error') or result.output['verification']['reason']}"
                    break
                accepted.append(result.call_id)
                previous = result.call_id
        except RunStopped as error:
            outcome, reason = error.status, error.reason
        except KeyboardInterrupt:
            outcome, reason = "cancelled", "keyboard interrupt"
        except Exception as error:
            outcome, reason = "failed", f"{type(error).__name__}: {error}"
        finally:
            run.update("fixture_run_finished", status=outcome, phase="finished", ended_at=now(), active_call=None,
                       stop_reason=reason, cost=executor.calls.cost_summary(), accepted_calls=accepted,
                       readiness="unverified", installed=False, result_status="fixture_passed" if outcome == "completed" else "fixture_incomplete")
    return run
