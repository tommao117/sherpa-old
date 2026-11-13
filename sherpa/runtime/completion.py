"""Two-step, locally approved llama.cpp smoke runner. No automatic retries."""
from __future__ import annotations

import copy
import getpass
import hashlib
import os
from pathlib import Path
import re
import time

from ..platform.llama_completion import SUPPORTED_COMMIT, parse_completion_log, plan_completion
from . import framework
from .calls import Calls, RunStopped
from .completion_host import CompletionHost, bus_id, file_identity, model_identity, selected_weight
from .execution_contract import _bounded, load_document, validate_execution
from .execution_records import BUILD_VERSION, RecordReader, absolute, check_completion_records, require
from .managed_process import SUPERVISOR, disk_bytes, run_process
from .models import Task
from .store import atomic_json, digest, locked, now, process_identity

VERSION = "completion-runner-v1"
TOOL_NAMES = ("cmake", "make", "cc", "cxx", "nvcc", "nvidia_smi")


def completion_template():
    """Suggested values only; paths/device must be filled and execution approved."""
    return {"pinned_source": "/replace/with/clean/local/llama.cpp",
            "tools": {key: "/replace/with/" + key for key in TOOL_NAMES},
            "task": Task("completion-smoke", {"kind": "git", "url": "https://github.com/ggml-org/llama.cpp", "revision": SUPPORTED_COMMIT},
                {"kind": "local", "path": "/replace/with/model-directory", "selected_set": "q8", "sets": [
                    {"id": "q8", "layout": "single", "weights": [{"path": "jevos-v4-q8_0.gguf", "sha256": None, "bytes": None}], "support": []}]},
                resources={"device": "gpu", "gpus": ["REPLACE_WITH_GPU_UUID"]},
                workload={"request": {"interface": "cli", "input": "Hello", "parameters": {"seed": 0, "max_tokens": 8, "ctx_size": 512, "gpu_layers": 99, "temperature": 0}}},
                budget={"seconds": 900, "max_calls": 8, "model_cost_usd": 0},
                contract={"preserve_behavior": True, "execution": {"version": 1,
                    "framework": {"commit": SUPPORTED_COMMIT, "prepare_run": None, "prepare_call": None, "prepare_output_hash": None},
                    "adapter": {"name": "llama-completion", "version": "1", "build_parameters": {"cuda_arch": "REPLACE"}},
                    "permissions": {"network": "none", "write_scopes": ["build", "scratch"], "tools": ["cmake", "make", "cc", "cxx", "nvcc", "llama-completion"], "trusted_code": True},
                    "limits": {"step_seconds": 600, "cpu_threads": 2, "memory_mib": 8192, "gpu_memory_mib": 6144, "disk_mib": 4096, "output_mib": 16},
                    "acceptance": {"scope": "smoke", "parser": {"name": "llama-completion", "version": "1"}, "verifier": {"name": "gpu-smoke", "version": "1"}, "reference": None, "tolerances": None}}}).snapshot()}


def controller_identity(check):
    paths = [Path(__file__), SUPERVISOR, Path(__file__).with_name("managed_process.py"),
             Path(__file__).with_name("completion_host.py"), Path(__file__).with_name("execution_records.py"),
             Path(__file__).with_name("execution_contract.py"), Path(framework.__file__),
             Path(__file__).with_name("workspace.py"), Path(__file__).with_name("git_worker.py"),
             Path(__file__).with_name("calls.py"), Path(__file__).with_name("store.py"), Path(__file__).with_name("models.py"),
             Path(__file__).parents[1] / "platform/llama_completion.py"]
    return {p.name: file_identity(p, check)["sha256"] for p in paths}


def _locations(run, tools):
    root = run.path
    return {"run_root": str(root), "source": str(root / "workspace/framework"),
            "build": str(root / "build"), "scratch": str(root / "scratch"),
            **{k: tools[k]["path"] for k in TOOL_NAMES if k != "nvidia_smi"}}


def _check_limits(task):
    b, limits = task["budget"], task["contract"]["execution"]["limits"]
    require(type(b["seconds"]) in (int, float) and 1 <= b["seconds"] <= 3600, "total seconds must be in [1, 3600]")
    require(type(b["max_calls"]) is int and 8 <= b["max_calls"] <= 32, "max_calls must be in [8, 32]")
    require(b["model_cost_usd"] == 0, "smoke runner never invokes decision models")
    require(type(limits["cpu_threads"]) is int and 1 <= limits["cpu_threads"] <= 4, "cpu_threads must be in [1, 4]")
    require(type(limits["memory_mib"]) is int and 256 <= limits["memory_mib"] <= 16384, "memory_mib must be in [256, 16384]")
    require(type(limits["disk_mib"]) is int and 512 <= limits["disk_mib"] <= 16384, "disk_mib must be in [512, 16384]")
    require(type(limits["output_mib"]) is int and 1 <= limits["output_mib"] <= 16, "output_mib must be in [1, 16]")
    require(0 < limits["step_seconds"] <= b["seconds"], "step timeout must fit total budget")


def stage_completion(store, task, pinned_source, tools, *, allow_input_preparation=False):
    """Permission covers local source copy, tool inspection and selected-weight hashing, not GPU use."""
    require(allow_input_preparation is True, "explicit --allow-input-preparation required")
    t, tool_paths = _bounded(task), _bounded(tools)
    require(isinstance(tool_paths, dict) and set(tool_paths) == set(TOOL_NAMES), "explicit cmake/make/cc/cxx/nvcc/nvidia_smi paths required")
    try:
        Task(**t)
        _check_limits(t)
        binding = t["contract"]["execution"]["framework"]
        require(binding["commit"] == SUPPORTED_COMMIT, "unsupported fixed framework revision")
        weight = selected_weight(t)
        require(weight.get("sha256") is None and weight.get("bytes") is None,
                "staging input uses null weight sha256/bytes; actual identity will be recorded")
    except (TypeError, KeyError, AttributeError, IndexError) as error:
        raise ValueError("malformed completion task; start from completion-template") from error
    pinned_source = absolute(str(pinned_source))
    model_root = absolute(t["model"]["path"])
    for protected in (pinned_source, model_root):
        require(store.root != protected and store.root not in protected.parents and protected not in store.root.parents,
                "data root and protected inputs must not overlap")
    for path in tool_paths.values():
        absolute(path)
    validation = copy.deepcopy(t)
    selected_weight(validation).update(sha256="0" * 64, bytes=1)
    placeholder = store.root / "runs/staging-validation"
    locations = {"run_root": str(placeholder), "source": str(placeholder / "workspace/framework"),
                 "build": str(placeholder / "build"), "scratch": str(placeholder / "scratch"),
                 **{k: str(Path(v).resolve()) for k, v in tool_paths.items() if k != "nvidia_smi"}}
    require(plan_completion(validation, locations)["status"] == "draft", "staging task is not a supported completion profile")
    with store.exclusive():
        run = store.create_run(Task(**t), kind="environment")
        with locked(run.path / ".execution.lock", blocking=False):
            calls = Calls(run, seconds=t["budget"]["seconds"], max_calls=t["budget"]["max_calls"])
            started = time.monotonic()
            run.update("completion_staging", status="running", scope="environment_execution", phase="staging", owner=process_identity(os.getpid()), synthetic=False)
            try:
                record, directory = calls.new("tool", "completion.stage-inputs", {"pinned_source": str(pinned_source), "tools": tool_paths})
                record.update(synthetic=False, tool_version=VERSION)
                try:
                    git_config = RecordReader(pinned_source, calls.check_interrupt).read(".git/config", 65536).decode("utf-8-sig")
                    require(not re.search(r"(?im)^\s*\[\s*(?:filter|include|includeif|extensions)\b", git_config),
                            "pinned source uses custom Git filters/includes/extensions; use a plain isolated checkout")
                    identities = {key: file_identity(Path(value).resolve(), calls.check_interrupt, limit=128 * 1024 * 1024)
                                  for key, value in tool_paths.items()}
                    for item in identities.values():
                        require(item["executable"], "tool is not executable: " + item["path"])
                        path = Path(item["path"])
                        require(store.root not in path.parents and model_root not in path.parents and pinned_source not in path.parents,
                                "tool must be outside run and target inputs")
                    model = model_identity(t, calls.check_interrupt)
                    selected_weight(t).update(sha256=model["sha256"], bytes=model["bytes"])
                    code = controller_identity(calls.check_interrupt)
                    calls.finish(record, directory, "succeeded", {"tools": identities, "model": model, "controller": code}, started)
                except BaseException as error:
                    calls.finish(record, directory, error.status if isinstance(error, RunStopped) else "cancelled" if isinstance(error, KeyboardInterrupt) else "failed", {"error": str(error)}, started)
                    raise
                result = framework.prepare(run, calls, max_mib=min(512, t["contract"]["execution"]["limits"]["disk_mib"] // 2),
                                           pinned={"path": str(pinned_source), "commit": binding["commit"]})
                prepare_record = run.records("calls")[-1]
                binding.update(prepare_run=run.id, prepare_call=prepare_record["id"], prepare_output_hash=prepare_record["output_hash"])
                loc = _locations(run, identities)
                plan = plan_completion(t, loc)
                require(plan["status"] == "draft", "prepared task did not produce valid command plan")
                scratch = Path(loc["scratch"])
                for sub in ("home", "config", "cache", "tmp", "cuda-cache"):
                    (scratch / sub).mkdir(parents=True, exist_ok=False)
                CompletionHost.check_config(scratch)
                e1 = validate_execution(t, {"kind": "build"}, {"task_hash": digest(t), "approved_by": "staging-validation-only", "actions": ["build"]})
                require(e1["status"] == "valid", "prepared task failed E1 checks: " + str(e1["diagnostics"]))
                run.update("completion_inputs_bound", task=t, locations=loc)
                checked = check_completion_records(store.root, t, loc, check=calls.check_interrupt)
                require(checked["status"] == "records_verified", "prepare records failed: " + str(checked["diagnostics"]))
                ticket = {"version": VERSION, "run_id": run.id, "task": t, "locations": loc, "plan_hash": plan["plan_hash"],
                          "tools": identities, "model": model, "controller": code, "staging_calls": calls.count,
                          "staging_seconds": time.monotonic() - started,
                          "scope": "configure-build-gpu-smoke-once", "created_at": now()}
                require(ticket["staging_seconds"] < t["budget"]["seconds"], "budget exhausted by staging")
                atomic_json(run.path / "completion-ticket.json", ticket)
                atomic_json(run.path / "completion-plan.json", plan)
                require(disk_bytes(run.path) <= t["contract"]["execution"]["limits"]["disk_mib"] * 1024 * 1024, "staging exceeded disk budget")
                run.update("completion_awaiting_approval", status="created", phase="awaiting_approval", owner=None,
                           ticket_hash=digest(ticket), cost=calls.cost_summary(), readiness="unverified", installed=False)
            except BaseException as error:
                status = error.status if isinstance(error, RunStopped) else "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
                run.update("completion_staging_failed", status=status, phase="finished", ended_at=now(), active_call=None,
                           stop_reason=str(error), cost=calls.cost_summary(), readiness="unverified")
                raise
    return run


class CompletionRunner:
    def __init__(self, run, ticket):
        self.run, self.ticket = run, ticket
        self.task, self.locations = ticket["task"], ticket["locations"]
        self.plan = plan_completion(self.task, self.locations)
        require(self.plan["status"] == "draft" and self.plan["plan_hash"] == ticket["plan_hash"], "command plan changed")
        remaining = self.task["budget"]["seconds"] - ticket["staging_seconds"]
        self.calls = Calls(run, seconds=remaining, max_calls=self.task["budget"]["max_calls"])
        self.calls.count = ticket["staging_calls"]
        self.host = CompletionHost(run, self.calls, self.task, ticket["tools"])
        self.approval = {"task_hash": digest(self.task), "approved_by": getpass.getuser(), "actions": ["probe", "build", "run_once"]}
        self.configure, self.built = None, None
        self.next_disk_check = 0.0

    def guard(self, *, force=False):
        self.calls.check_interrupt()
        current = time.monotonic()
        if force or current >= self.next_disk_check:
            self.next_disk_check = current + 0.5
            if disk_bytes(self.run.path) > self.task["contract"]["execution"]["limits"]["disk_mib"] * 1024 * 1024:
                raise RunStopped("budget_exhausted", "cumulative run disk limit exceeded")

    def call(self, name, inputs, fn, caused_by=None):
        started = time.monotonic()
        record, directory = self.calls.new("tool", name, inputs, caused_by=caused_by)
        record.update(synthetic=False, tool_version=BUILD_VERSION)
        self.run.save("calls", record["id"], record, "completion_call_bound")
        status, output, stopped = "failed", {}, None
        try:
            output = fn(record, directory)
            status = output.get("process", {}).get("status", "succeeded")
            if status != "succeeded":
                raise RunStopped(status, output["process"]["reason"])
            if output.get("verification", {}).get("status", "pass") != "pass":
                raise RunStopped("failed", output["verification"].get("reason", "verification failed"))
            self.guard(force=True)
        except BaseException as error:
            status = error.status if isinstance(error, RunStopped) else "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
            process_path = directory / "process/process.json"
            if not output and process_path.is_file():
                output["process"] = load_document(process_path)
                output["verification"] = {"status": "inconclusive", "reason": "post-execution checks failed"}
            output["error"] = str(error)
            stopped = error if isinstance(error, RunStopped) else RunStopped(status, str(error))
        result = self.calls.finish(record, directory, status, output, started)
        if stopped:
            raise stopped
        return result

    def preflight(self):
        def verify(record, directory):
            require(digest(self.run.state()["task"]) == digest(self.task), "task changed after approval")
            require(controller_identity(self.guard) == self.ticket["controller"], "controller changed after staging")
            self.host.verify_tools()
            require(model_identity(self.task, self.guard) == self.ticket["model"], "model content changed after staging")
            framework.verify_clean(Path(self.locations["source"]), self.task["contract"]["execution"]["framework"]["commit"], self.guard)
            self.host.check_config(Path(self.locations["scratch"]))
            device = self.host.sample(directory, force=True)
            return {"device": device, "model": self.ticket["model"], "tools": self.ticket["tools"]}
        return self.call("completion.preflight", {"ticket_hash": digest(self.ticket)}, verify)

    def execute_phase(self, index):
        command = self.plan["commands"][index]
        phase = command["phase"]
        checked = check_completion_records(self.run.store.root, self.task, self.locations,
                                           build_call=self.built.call_id if phase == "run_once" else None, check=self.guard)
        require(checked["status"] == "records_verified", "record verification failed: " + str(checked["diagnostics"]))
        e1 = validate_execution(self.task, {"kind": "run_once", "build_call": self.built.call_id} if phase == "run_once" else {"kind": "build"}, self.approval)
        require(e1["status"] == "valid", "execution contract changed")
        inputs = {"task_hash": digest(self.task), "plan_hash": self.plan["plan_hash"], "command": command}
        cause = None
        if phase == "build":
            cause = self.configure.call_id
            inputs.update(configure_call=cause, configure_output_hash=digest(self.configure.output))
        elif phase == "run_once":
            cause = self.built.call_id
            inputs.update(build_call=cause, build_output_hash=digest(self.built.output))
        def execute(record, directory):
            self.host.verify_tools()
            framework.verify_clean(Path(self.locations["source"]), self.task["contract"]["execution"]["framework"]["commit"], self.guard)
            self.host.check_config(Path(self.locations["scratch"]))
            if phase == "run_once":
                require(model_identity(self.task, self.guard) == self.ticket["model"], "model changed before inference")
                binary = file_identity(Path(self.locations["build"]) / "bin/llama-completion", self.guard)
                require(binary["sha256"] == self.built.output["artifact"]["sha256"], "binary changed before inference")
            self.host.sample(directory, force=True, check=self.guard)
            self.guard(force=True)
            if phase == "run_once":
                from .execution_contract import metadata
                for identity in (self.ticket["model"], binary):
                    path = Path(identity["path"])
                    require(metadata(str(path.parent), path.name) == identity["stat"], "input replaced after final identity check")
            def began(identity):
                require(identity is not None, "process identity unavailable")
                record.update(process=identity, cleanup_complete=False)
                self.run.save("calls", record["id"], record, "completion_process_started")
            process = run_process(command, directory / "process", self.guard, on_start=began,
                                  monitor=lambda group, poll: self.host.sample(directory, group, check=poll))
            record["cleanup_complete"] = process["cleanup_complete"]
            output = {"process": process, "verification": {"status": "inconclusive", "reason": "process did not complete"}}
            if process["status"] != "succeeded":
                return output
            framework.verify_clean(Path(self.locations["source"]), self.task["contract"]["execution"]["framework"]["commit"], self.guard)
            self.host.check_config(Path(self.locations["scratch"]))
            self.host.verify_tools()
            if phase == "configure":
                require((Path(self.locations["build"]) / "CMakeCache.txt").is_file(), "configure cache missing")
                output["verification"] = {"status": "pass", "verifier": "completion-configure-v1"}
            elif phase == "build":
                identity = file_identity(Path(self.locations["build"]) / "bin/llama-completion", self.guard)
                require(identity["executable"] and identity["bytes"] > 0, "build binary missing or not executable")
                output["artifact"] = {"path": str(Path(identity["path"]).relative_to(self.run.path)), "bytes": identity["bytes"], "sha256": identity["sha256"]}
                output["verification"] = {"status": "pass", "verifier": "completion-build-v1"}
            else:
                require(model_identity(self.task, self.guard) == self.ticket["model"], "model changed during inference")
                current = file_identity(Path(self.locations["build"]) / "bin/llama-completion", self.guard)
                require(current["sha256"] == self.built.output["artifact"]["sha256"], "binary changed during inference")
                stderr = (directory / "process/stderr.log").read_text(encoding="utf-8")
                stdout = (directory / "process/stdout.log").read_text(encoding="utf-8")
                evidence = parse_completion_log(stderr, complete=True)
                device = evidence.get("device") or {}
                matched = (evidence["status"] == "evidence_present" and device.get("backend") == "CUDA0"
                           and bus_id(device["bus_id"]) == self.host.latest["bus_id"] and bool(stdout.strip()))
                output.update(evidence=evidence, response={"sha256": hashlib.sha256(stdout.encode()).hexdigest(), "bytes": len(stdout.encode())})
                output["verification"] = {"status": "pass" if matched else "inconclusive", "verifier": "completion-gpu-smoke-v1",
                                          "reason": "single GPU request evidence matched" if matched else "insufficient request/device evidence", "behavior_verified": False}
            return output
        return self.call("completion." + phase, inputs, execute, cause)


def run_completion(store, run_id, approved_ticket_hash):
    """Explicit invocation by the local operator approves exactly this staged ticket once."""
    require(isinstance(approved_ticket_hash, str) and len(approved_ticket_hash) == 64, "explicit --approve-ticket-hash required")
    with store.exclusive():
        run = store.open_run(run_id)
        with locked(run.path / ".execution.lock", blocking=False):
            state = run.state()
            require(state["status"] == "created" and state.get("phase") == "awaiting_approval", "run is not awaiting approval; no replay/resume")
            ticket = RecordReader(store.root, lambda: None).json(f"runs/{run.id}/completion-ticket.json")
            require(digest(ticket) == state.get("ticket_hash") == approved_ticket_hash, "approval does not match staged ticket")
            require(ticket["version"] == VERSION and ticket["run_id"] == run.id and digest(ticket["task"]) == digest(state["task"]), "ticket/task binding mismatch")
            require(ticket["staging_calls"] == len(run.records("calls")) and all(r["status"] == "succeeded" for r in run.records("calls")), "staging Call ledger changed")
            runner = CompletionRunner(run, ticket)
            run.update("completion_execution_approved", status="running", phase="preflight", owner=process_identity(os.getpid()),
                       approval={"ticket_hash": approved_ticket_hash, "uid": os.getuid(), "user": getpass.getuser(), "time": now(), "scope": ticket["scope"]})
            outcome, reason, result = "failed", "execution not completed", None
            try:
                runner.guard()
                require(not Path(runner.locations["build"]).exists(), "build directory must be fresh")
                runner.preflight()
                runner.configure = runner.execute_phase(0)
                runner.built = runner.execute_phase(1)
                result = runner.execute_phase(2)
                runner.guard()
                outcome, reason = "completed", "GPU smoke passed; behavior and performance unverified"
            except BaseException as error:
                outcome = error.status if isinstance(error, RunStopped) else "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
                reason = str(error)
            finally:
                cost = runner.calls.cost_summary()
                cost["seconds"] += ticket["staging_seconds"]
                run.update("completion_finished", status=outcome, phase="finished", ended_at=now(), active_call=None, cost=cost,
                           stop_reason=reason, result_status="gpu_smoke_passed" if outcome == "completed" else "incomplete",
                           readiness="smoke_verified" if outcome == "completed" else "unverified", installed=False,
                           behavior_verified=False, result_call=result.call_id if result else None)
    return run
