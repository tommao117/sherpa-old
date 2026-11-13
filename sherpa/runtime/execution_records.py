"""Read-only provenance checks for completion preparation and build records.

These checks detect stale/mismatched local records, not malicious store writers.
They never execute code, read model files, or grant execution permission.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat

from ..platform.llama_completion import plan_completion
from .execution_contract import _bounded, _constant, _pairs
from .models import Task, identifier
from .store import digest

VERSION = "completion-records-v1"
BUILD_VERSION = "completion-executor-v1"
JSON_LIMIT = 4 * 1024 * 1024
ARTIFACT_LIMIT = 64 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute(value):
    require(isinstance(value, str) and value.startswith("/") and not value.startswith("//")
            and "\x00" not in value and "\\" not in value and ".." not in PurePosixPath(value).parts
            and str(PurePosixPath(value)) == value, "normalized absolute path required")
    return Path(value)


class RecordReader:
    def __init__(self, root, check):
        self.root, self.check = absolute(str(root)), check
        self.remaining = 2 * ARTIFACT_LIMIT

    @contextmanager
    def open(self, relative):
        path = PurePosixPath(relative)
        require(isinstance(relative, str) and bool(path.parts) and not path.is_absolute()
                and str(path) == relative and ".." not in path.parts and "\x00" not in relative
                and "\\" not in relative, "unsafe record path")
        self.check()
        fd = os.open("/", os.O_PATH | os.O_DIRECTORY)
        child = None
        try:
            for component in (*self.root.parts[1:], *path.parts[:-1]):
                next_fd = os.open(component, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            child = os.open(path.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=fd)
            with os.fdopen(child, "rb") as handle:
                child = None
                require(stat.S_ISREG(os.fstat(handle.fileno()).st_mode), "record/artifact must be a regular file")
                yield handle
        finally:
            os.close(fd)
            if child is not None:
                os.close(child)

    def read(self, relative, limit=JSON_LIMIT, *, hash_only=False):
        chunks, hashed, size = [], hashlib.sha256(), 0
        with self.open(relative) as handle:
            before = os.fstat(handle.fileno())
            require(0 <= before.st_size <= limit, "record/artifact exceeds read limit")
            while True:
                self.check()
                chunk = handle.read(min(65536, limit - size + 1))
                if not chunk:
                    break
                size += len(chunk)
                self.remaining -= len(chunk)
                require(size <= limit and self.remaining >= 0, "record/artifact exceeds read budget")
                hashed.update(chunk)
                if not hash_only:
                    chunks.append(chunk)
            after = os.fstat(handle.fileno())
            identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            require(identity(before) == identity(after) and size == before.st_size, "record/artifact changed during read")
            info = {"bytes": size, "sha256": hashed.hexdigest(), "executable": bool(before.st_mode & 0o111)}
        return info if hash_only else b"".join(chunks)

    def json(self, relative):
        value = json.loads(self.read(relative), object_pairs_hook=_pairs, parse_constant=_constant)
        require(isinstance(value, dict), "record must be a JSON object")
        return value

    def call(self, run_id, call_id, name, version, *, process=False):
        base = f"runs/{identifier(run_id)}/calls/{identifier(call_id)}"
        record = self.json(base + "/record.json")
        require(record.get("id") == call_id and record.get("run_id") == run_id, "Call identity mismatch")
        require(record.get("kind") == "tool" and record.get("name") == name
                and record.get("tool_version") == version, "unexpected Call tool/version")
        require(record.get("status") == "succeeded" and record.get("synthetic") is False,
                "successful non-synthetic Call required")
        require(record.get("cleanup_complete") is not False, "Call cleanup is incomplete")
        for field in ("input", "output"):
            require(record.get(field) == f"calls/{call_id}/{field}.json", "Call artifact path mismatch")
        inputs, output = self.json(base + "/input.json"), self.json(base + "/output.json")
        require(digest(inputs) == record.get("input_hash") and digest(output) == record.get("output_hash"), "Call input/output hash mismatch")
        require(output.get("synthetic", False) is False, "synthetic output cannot prove real execution")
        if process:
            require(record.get("cleanup_complete") is True, "build/configure cleanup must be confirmed")
            p = record.get("process")
            require(isinstance(p, dict) and type(p.get("pid")) is int and p["pid"] > 0
                    and isinstance(p.get("start_ticks"), str), "process identity required")
            actual = output.get("process")
            require(isinstance(actual, dict) and actual.get("cleanup_complete") is True
                    and actual.get("status") == "succeeded" and type(actual.get("exit_code")) is int
                    and actual["exit_code"] == 0, "successful process evidence required")
        return record, inputs, output


def check_completion_records(data_root, task, locations, *, build_call=None, check=lambda: None):
    """Check explicit local records; a future executor must recheck before use.

    Build protocol: completion.configure/build Calls with BUILD_VERSION; input
    binds task_hash, plan_hash and command. Build also binds configure_call and
    configure_output_hash. Output holds process {command,status,exit_code,
    cleanup_complete}, verification {status,verifier}, and build artifact
    {path,bytes,sha256}. Artifact path is run-relative, fixed by the plan.
    """
    results, errors, prepared, artifact = [], [], None, None
    task_hash = None

    def run_check(name, depends, fn):
        blocked = [dep for dep in depends if not any(r["name"] == dep and r["status"] == "passed" for r in results)]
        if blocked:
            results.append({"name": name, "status": "skipped", "blocked_by": blocked})
            return
        try:
            fn()
            results.append({"name": name, "status": "passed", "blocked_by": []})
        except (ValueError, TypeError, KeyError, AttributeError, IndexError, OSError, RecursionError, OverflowError) as error:
            results.append({"name": name, "status": "failed", "blocked_by": []})
            errors.append({"check": name, "code": "record_binding_failed", "message": str(error)[:400]})

    context = {}

    def plan_check():
        nonlocal task_hash
        t = _bounded(task.snapshot() if isinstance(task, Task) else task)
        loc = _bounded(locations)
        plan = plan_completion(t, loc)
        require(plan["status"] == "draft", "completion command plan is invalid")
        reader = RecordReader(data_root, check)
        root = absolute(loc["run_root"])
        run_id = identifier(root.name)
        require(root == reader.root / "runs" / run_id, "run root must match data root and Run ID")
        require(loc["source"] == str(root / "workspace" / "framework"), "prepared workspace must be the fixed run-local path")
        binding = t["contract"]["execution"]["framework"]
        require(binding["prepare_run"] == run_id, "prepare must belong to this run; import old checkpoints into a new run first")
        context.update(task=t, loc=loc, plan=plan, reader=reader, run_id=run_id, root=root, binding=binding)
        task_hash = digest(t)

    def state_check():
        state = context["reader"].json(f"runs/{context['run_id']}/run.json")
        require(state.get("id") == context["run_id"] and state.get("kind") == "environment", "Run identity mismatch")
        require(state.get("synthetic", False) is False and state.get("scope") in ("environment_explore", "environment_execution"), "real environment Run required")
        require(state.get("status") in ("running", "completed", "failed", "timed_out", "cancelled", "budget_exhausted")
                and state.get("active_call") is None, "Run contains active or unsafe work")
        require(digest(state.get("task")) == task_hash, "Run task differs from requested contract")
        context["state"] = state

    def prepare_check():
        nonlocal prepared
        binding, reader, run_id = context["binding"], context["reader"], context["run_id"]
        record, inputs, output = reader.call(run_id, binding["prepare_call"], "framework.prepare", "framework-import-v2")
        require(record["output_hash"] == binding["prepare_output_hash"], "prepare output differs from contract")
        require(digest(output) == digest(context["state"].get("framework")), "Run framework snapshot differs from prepare")
        require(digest(output.get("source")) == digest(context["task"]["target"]) == digest(inputs.get("target")), "prepare source mismatch")
        require(output.get("commit") == binding["commit"] and output.get("import_version") == record["tool_version"], "prepare commit/version mismatch")
        require(output.get("path") == context["loc"]["source"] and output.get("model_weights_copied") is False, "prepare workspace/model-copy mismatch")
        require(output.get("import_mode") in ("pinned_local_copy", "pinned_shallow_fetch")
                and output.get("source_commit") == binding["commit"], "pinned Git source required")
        require(output.get("submodules_not_fetched") == [], "unfetched submodules need separate preparation")
        require(type(inputs.get("max_mib")) is int and inputs["max_mib"] > 0
                and inputs["max_mib"] == output.get("framework_max_mib"), "prepare size limit mismatch")
        require(inputs.get("pinned_commit") in (None, binding["commit"]), "prepare pinned commit mismatch")
        prepared = {"run_id": run_id, "call_id": record["id"], "output_hash": record["output_hash"], "commit": output["commit"]}

    def build_check():
        nonlocal artifact
        reader, run_id, plan = context["reader"], context["run_id"], context["plan"]
        rec, inputs, output = reader.call(run_id, build_call, "completion.build", BUILD_VERSION, process=True)
        configure_id = identifier(inputs["configure_call"])
        require(configure_id != build_call and rec.get("caused_by") == configure_id, "build must reference its configure Call")
        config_rec, config_in, config_out = reader.call(run_id, configure_id, "completion.configure", BUILD_VERSION, process=True)
        require(inputs["configure_output_hash"] == config_rec["output_hash"], "configure output binding mismatch")
        for phase, inp, out, command in (("configure", config_in, config_out, plan["commands"][0]),
                                         ("build", inputs, output, plan["commands"][1])):
            require(inp.get("task_hash") == task_hash and inp.get("plan_hash") == plan["plan_hash"], "build/configure task or plan mismatch")
            require(digest(inp.get("command")) == digest(command) == digest(out["process"].get("command")), "actual command differs from fixed plan")
            require(out.get("verification") == {"status": "pass", "verifier": f"completion-{phase}-v1"}, "independent build/configure verification required")
        descriptor = output.get("artifact")
        require(isinstance(descriptor, dict) and set(descriptor) == {"path", "bytes", "sha256"}, "build artifact descriptor required")
        expected = str((absolute(context["loc"]["build"]) / "bin" / "llama-completion").relative_to(context["root"]))
        require(descriptor["path"] == expected, "unexpected binary path")
        require(type(descriptor["bytes"]) is int and 0 < descriptor["bytes"] <= ARTIFACT_LIMIT, "binary size outside bounded verification limit")
        observed = reader.read(f"runs/{run_id}/{expected}", ARTIFACT_LIMIT, hash_only=True)
        require(observed["bytes"] == descriptor["bytes"] and observed["sha256"] == descriptor["sha256"], "binary content identity mismatch")
        require(observed["executable"], "binary is not executable")
        artifact = {**descriptor, "run_id": run_id, "build_call": build_call, "build_output_hash": rec["output_hash"]}

    run_check("plan", [], plan_check)
    run_check("run", ["plan"], state_check)
    run_check("prepare", ["run"], prepare_check)
    if build_call is not None:
        run_check("build", ["prepare"], build_check)
    return {"version": VERSION, "status": "records_verified" if not errors else "invalid", "task_hash": task_hash,
            "checks": results, "diagnostics": errors, "prepare": prepared, "artifact": artifact,
            "executed": False, "execution_authorized": False, "readiness": "unverified",
            "pending_checks": ["authenticated_execution_grant", "remaining_budget_and_cancellation", "current_clean_source_commit",
                               "current_toolchain_and_config_isolation", "model_identity_and_compatibility", "idle_gpu_and_uuid_mapping"],
            "limitations": ["Local provenance only: record hashes are not signatures or human authorization.",
                            "No source/Git, model, device, or execution compatibility was checked.",
                            "File checks are point-in-time; the executor must prevent or recheck replacement before launch."]}
