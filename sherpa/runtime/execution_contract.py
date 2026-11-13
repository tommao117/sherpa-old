"""Offline execution admission: inspect declarations/metadata, never execute a plan."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat

from ..platform.exploration import parts
from .models import Task, identifier
from .store import digest

VERSION = "execution-contract-v1"
MAX_DOCUMENT_BYTES = 128 * 1024
ACTIONS = ("probe", "build", "run_once")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def _bounded(value):
    def walk(item, depth=0):
        if depth > 24:
            raise ValueError("contract nesting exceeds 24 levels")
        if isinstance(item, dict):
            if len(item) > 128 or any(not isinstance(k, str) or len(k) > 256 for k in item):
                raise ValueError("object must have at most 128 bounded string keys")
            for child in item.values():
                walk(child, depth + 1)
        elif isinstance(item, list):
            if len(item) > 256:
                raise ValueError("array exceeds 256 entries")
            for child in item:
                walk(child, depth + 1)
        elif item is not None and not isinstance(item, (str, bool, int, float)):
            raise ValueError("JSON values required")
    walk(value)
    if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode()) > MAX_DOCUMENT_BYTES:
        raise ValueError("contract document exceeds 128 KiB")
    return copy.deepcopy(value)


def load_document(path):
    """Bounded JSON-only input; reject FIFOs and links before reading."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("contract input must be a regular JSON file")
        raw = handle.read(MAX_DOCUMENT_BYTES + 1)
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise ValueError("contract document exceeds 128 KiB")
    try:
        return _bounded(json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant))
    except (RecursionError, UnicodeError) as error:
        raise ValueError("invalid or excessively nested JSON document") from error


def _absolute(value):
    if not isinstance(value, str) or not value.startswith("/") or "\x00" in value or "\\" in value or len(value) > 4096:
        return False
    p = PurePosixPath(value)
    return ".." not in p.parts and str(p) == value


def metadata(root, relative):
    """O_PATH/fstat only: no file contents, including model weights, are read."""
    if not _absolute(root):
        raise ValueError("absolute normalized root required")
    components = parts(relative)
    if not components or str(PurePosixPath(relative)) != relative:
        raise ValueError("normalized relative file path required")
    current = os.open("/", os.O_PATH | os.O_DIRECTORY)
    try:
        for component in PurePosixPath(root).parts[1:]:
            child = os.open(component, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            os.close(current)
            current = child
        for index, component in enumerate(components):
            flags = os.O_PATH | os.O_NOFOLLOW
            if index < len(components) - 1:
                flags |= os.O_DIRECTORY
            child = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = child
        info = os.fstat(current)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("regular file required; links and special files are not accepted")
        return {"bytes": info.st_size, "device": info.st_dev, "inode": info.st_ino, "mtime_ns": info.st_mtime_ns}
    finally:
        os.close(current)


class Checks:
    def __init__(self):
        self.results = []
        self.diagnostics = []
        self.statuses = {}
        self.current = "document"

    def issue(self, field, code, message, remedy):
        self.diagnostics.append({"check": self.current, "field": field, "code": code,
                                 "message": message, "remedy": remedy})

    def require(self, condition, field, code, message, remedy):
        if not condition:
            self.issue(field, code, message, remedy)
        return bool(condition)

    def object(self, value, field, required, optional=()):
        if not self.require(isinstance(value, dict), field, "object_required", "Expected an object", "Provide the named object"):
            return None
        for key in sorted(set(required) - set(value)):
            self.issue(f"{field}.{key}", "missing_field", "Required field is missing", "Provide an explicit value")
        for key in sorted(set(value) - set(required) - set(optional)):
            self.issue(f"{field}.{key}", "unexpected_field", "Field is not in this contract version", "Remove it or use the supported contract")
        return value

    def text(self, value, field, maximum=512):
        return self.require(isinstance(value, str) and bool(value.strip()) and len(value) <= maximum and "\x00" not in value,
                            field, "invalid_text", "Expected nonempty bounded text", "Provide an explicit text value")

    def number(self, value, field, minimum=0, integer=False):
        valid = type(value) in ((int,) if integer else (int, float))
        try:
            valid = valid and math.isfinite(value) and value >= minimum
        except OverflowError:
            valid = False
        return self.require(valid, field, "invalid_number", f"Expected a finite {'integer' if integer else 'number'} >= {minimum}", "Provide a value within the declared budget")

    def names(self, value, field, allowed=None, nonempty=True):
        valid = isinstance(value, list) and (bool(value) or not nonempty) and all(isinstance(v, str) for v in value)
        if valid:
            valid = len(value) == len(set(value)) and all(v and len(v) <= 100 for v in value)
            if allowed is not None:
                valid = valid and all(v in allowed for v in value)
        return self.require(valid, field, "invalid_list", "Expected unique explicit names", "Use supported names without duplicates")

    def run(self, name, depends, operation):
        blocked = [dep for dep in depends if self.statuses.get(dep) != "passed"]
        if blocked:
            self.statuses[name] = "skipped"
            self.results.append({"name": name, "status": "skipped", "blocked_by": blocked})
            return
        self.current = name
        before = len(self.diagnostics)
        operation()
        status = "passed" if len(self.diagnostics) == before else "failed"
        self.statuses[name] = status
        self.results.append({"name": name, "status": status, "blocked_by": []})


def validate_execution(task, action, approval=None):
    """Validate one requested action. Approval is caller-supplied, not authenticated here.

    A pending action is a frozen declaration, not an execution permit. A real executor must
    authenticate approval, resolve adapters, and recheck live identities/budgets.
    """
    c = Checks()
    try:
        task = _bounded(task.snapshot() if isinstance(task, Task) else task)
        action, approval = _bounded(action), _bounded(approval)
    except (ValueError, TypeError, RecursionError, OverflowError) as error:
        c.issue("document", "invalid_document", str(error), "Provide finite, bounded JSON documents")
        return _report(c, None, None, None)
    task_hash = digest(task)
    t = task if isinstance(task, dict) else {}
    contract = t.get("contract") if isinstance(t.get("contract"), dict) else {}
    e = contract.get("execution") if isinstance(contract.get("execution"), dict) else {}
    kind = action.get("kind") if isinstance(action, dict) else None

    def envelope():
        optional = () if kind == "run_once" else ("model", "workload")
        c.object(task, "task", set(Task.__dataclass_fields__) - set(optional), optional)
        try:
            identifier(t.get("id"))
        except ValueError:
            c.issue("task.id", "invalid_id", "Invalid task identifier", "Use a bounded alphanumeric identifier")
        objective = c.object(t.get("objective"), "task.objective", ("metric", "direction"))
        if objective is not None:
            c.text(objective.get("metric"), "task.objective.metric")
            c.require(objective.get("direction") in ("maximize", "minimize"), "task.objective.direction", "invalid_direction", "Unknown objective direction", "Use maximize or minimize")
        c.object(t.get("contract"), "task.contract", ("preserve_behavior", "execution"))
        c.require(contract.get("preserve_behavior") is True, "task.contract.preserve_behavior", "behavior_required", "Behavior preservation must remain enabled", "Keep the task's behavior requirement")
        optional = ("framework", "adapter", "acceptance") if kind == "probe" else (("acceptance",) if kind == "build" else ())
        c.object(contract.get("execution"), "task.contract.execution",
                 {"version", "framework", "adapter", "permissions", "limits", "acceptance"} - set(optional), optional)
        c.require(type(e.get("version")) is int and e["version"] == 1, "task.contract.execution.version", "unsupported_version", "Execution contract version must be 1", "Create an explicit execution contract; reports are not contracts")

    def action_kind():
        c.require(kind in ACTIONS, "action.kind", "unsupported_action", "Unknown action", "Use probe, build or run_once")

    def action_check():
        obj = c.object(action, "action", ("kind", "targets") if kind == "probe" else (("kind", "build_call") if kind == "run_once" else ("kind",)))
        if kind == "probe" and obj is not None:
            c.names(obj.get("targets"), "action.targets", ("host", "toolchain", "gpu"))
        if kind == "run_once" and obj is not None:
            try:
                identifier(obj.get("build_call"))
            except ValueError:
                c.issue("action.build_call", "build_reference_required", "A build Call ID is required", "Reference a successful build bound to this contract")

    def source_check():
        target = t.get("target")
        if not isinstance(target, dict):
            c.object(target, "task.target", ("kind",))
            return
        source_kind = target.get("kind")
        c.object(target, "task.target", ("kind", "url" if source_kind == "git" else "path"), ("revision",))
        c.require(source_kind in ("git", "local"), "task.target.kind", "invalid_source", "Unknown framework source", "Use the user-selected git or local source")
        if source_kind == "local":
            c.require(_absolute(target.get("path")), "task.target.path", "invalid_path", "Absolute normalized source path required", "Specify the original framework directory")
        elif source_kind == "git":
            from urllib.parse import urlsplit
            try:
                value = target.get("url")
                u = urlsplit(value) if isinstance(value, str) else None
                valid = u is not None and u.scheme == "https" and bool(u.hostname) and not u.username and not u.password and not u.query and not u.fragment
            except ValueError:
                valid = False
            c.require(valid, "task.target.url", "invalid_source", "Expected a credential-free HTTPS Git URL", "Keep the explicitly selected framework source")

    def framework_check():
        f = c.object(e.get("framework"), "execution.framework", ("commit", "prepare_run", "prepare_call", "prepare_output_hash"))
        if f is None:
            return
        c.require(isinstance(f.get("commit"), str) and bool(COMMIT.fullmatch(f["commit"])), "execution.framework.commit", "unpinned_framework", "Full fixed commit required", "Use the verified prepare result, not HEAD or a short commit")
        for key in ("prepare_run", "prepare_call"):
            try:
                identifier(f.get(key))
            except ValueError:
                c.issue(f"execution.framework.{key}", "prepare_reference_required", "Prepare Run/Call IDs required", "Reference the framework.prepare call")
        _sha(c, f.get("prepare_output_hash"), "execution.framework.prepare_output_hash")

    def budgets():
        budget = c.object(t.get("budget"), "task.budget", ("seconds", "max_calls", "model_cost_usd"))
        if budget is not None:
            c.number(budget.get("seconds"), "task.budget.seconds", minimum=0.001)
            c.number(budget.get("max_calls"), "task.budget.max_calls", minimum=1, integer=True)
            c.number(budget.get("model_cost_usd"), "task.budget.model_cost_usd")
        limits = c.object(e.get("limits"), "execution.limits", ("step_seconds", "cpu_threads", "memory_mib", "gpu_memory_mib", "disk_mib", "output_mib"))
        if limits is not None:
            c.number(limits.get("step_seconds"), "execution.limits.step_seconds", minimum=0.001)
            for key in ("cpu_threads", "memory_mib", "disk_mib", "output_mib", "gpu_memory_mib"):
                c.number(limits.get(key), f"execution.limits.{key}", minimum=0 if key == "gpu_memory_mib" else 1, integer=True)
        if budget is not None and limits is not None:
            if type(budget.get("seconds")) in (int, float) and type(limits.get("step_seconds")) in (int, float):
                c.require(limits["step_seconds"] <= budget["seconds"], "execution.limits.step_seconds", "exceeds_budget", "Step timeout exceeds total time", "Lower the step timeout or confirm a new budget")

    def permissions():
        p = c.object(e.get("permissions"), "execution.permissions", ("network", "write_scopes", "tools", "trusted_code"))
        if p is None:
            return
        c.require(p.get("network") == "none", "execution.permissions.network", "unsupported_permission", "E1 supports no-network declarations only", "Obtain a separately scoped design for network access")
        c.names(p.get("write_scopes"), "execution.permissions.write_scopes", ("build", "scratch"), nonempty=False)
        c.names(p.get("tools"), "execution.permissions.tools")
        c.require(type(p.get("trusted_code")) is bool, "execution.permissions.trusted_code", "invalid_trust", "Explicit trust decision required", "Specify whether target code is trusted")
        if kind in ("build", "run_once"):
            c.require(p.get("trusted_code") is True, "execution.permissions.trusted_code", "untrusted_code", "This executor design only handles trusted projects", "Review code before permitting execution")
            scopes = p.get("write_scopes")
            c.require(isinstance(scopes, list) and "scratch" in scopes and (kind != "build" or "build" in scopes), "execution.permissions.write_scopes", "write_scope_required", "Action requires its bounded output/scratch scope", "Explicitly approve the required run-local scopes")

    def resources():
        r = c.object(t.get("resources"), "task.resources", ("device", "gpus"))
        if r is None:
            return
        c.require(r.get("device") in ("cpu", "gpu"), "task.resources.device", "device_required", "Explicit cpu/gpu requirement needed", "Specify the task device")
        names_valid = c.names(r.get("gpus"), "task.resources.gpus", nonempty=False)
        if r.get("device") == "cpu":
            c.require(r.get("gpus") == [], "task.resources.gpus", "device_conflict", "CPU task cannot select GPUs", "Keep device and GPU selection consistent")
        if kind == "run_once" and r.get("device") == "gpu":
            c.require(names_valid and bool(r["gpus"]) and all(v.startswith("GPU-") or v.startswith("MIG-") for v in r["gpus"]), "task.resources.gpus", "gpu_identity_required", "Stable GPU identities required before inference", "Confirm UUIDs after a current machine observation")
            limits = e.get("limits") if isinstance(e.get("limits"), dict) else {}
            c.number(limits.get("gpu_memory_mib"), "execution.limits.gpu_memory_mib", minimum=1, integer=True)

    def authorization():
        a = c.object(approval, "approval", ("task_hash", "approved_by", "actions"))
        if a is None:
            return
        c.text(a.get("approved_by"), "approval.approved_by", 100)
        c.require(a.get("task_hash") == task_hash, "approval.task_hash", "approval_mismatch", "Approval does not bind this exact task", "Obtain approval for the unchanged task snapshot")
        valid = c.names(a.get("actions"), "approval.actions", ACTIONS)
        c.require(valid and kind in a["actions"], "approval.actions", "action_not_approved", "Requested action is not approved", "Obtain an explicit grant for this action")

    def adapter():
        a = c.object(e.get("adapter"), "execution.adapter", ("name", "version", "build_parameters"))
        if a is not None:
            c.text(a.get("name"), "execution.adapter.name", 100)
            c.text(a.get("version"), "execution.adapter.version", 100)
            c.require(isinstance(a.get("build_parameters"), dict), "execution.adapter.build_parameters", "object_required", "Explicit build parameter object required", "Provide adapter parameters; not a shell command")

    selected = {}

    def model_selection():
        model = c.object(t.get("model"), "task.model", ("kind", "path", "selected_set", "sets"))
        if model is None:
            return
        c.require(model.get("kind") == "local", "task.model.kind", "unsupported_model", "Local model directory required", "Use a separate local model root")
        c.require(_absolute(model.get("path")), "task.model.path", "invalid_path", "Absolute normalized model root required", "Provide the model directory without traversal")
        sets = model.get("sets")
        if not c.require(isinstance(sets, list) and 1 <= len(sets) <= 16, "task.model.sets", "sets_required", "Provide 1..16 explicit alternative file sets", "Group only jointly loaded files in each set"):
            return
        ids = []
        for index, value in enumerate(sets):
            field = f"task.model.sets[{index}]"
            item = c.object(value, field, ("id", "layout", "weights", "support"))
            if item is None:
                continue
            if c.text(item.get("id"), field + ".id", 100):
                ids.append(item["id"])
            c.require(item.get("layout") in ("single", "sharded"), field + ".layout", "invalid_layout", "Declare single or sharded weights", "Keep alternative quantizations in separate sets")
            weights, support = item.get("weights"), item.get("support")
            valid_weights = c.require(isinstance(weights, list) and 1 <= len(weights) <= 128, field + ".weights", "weights_required", "Provide weight file descriptors", "Declare one primary weight or all ordered shards")
            valid_support = c.require(isinstance(support, list) and len(support) <= 128, field + ".support", "invalid_support", "Provide a support-file list, possibly empty", "Declare tokenizer/config files separately")
            if valid_weights:
                count = len(weights)
                c.require((item.get("layout") == "single" and count == 1) or (item.get("layout") == "sharded" and count >= 2), field + ".weights", "ambiguous_weights", "Weight count conflicts with declared layout", "Split alternatives or explicitly declare joint ordered shards")
            paths = []
            for group, files in (("weights", weights if valid_weights else []), ("support", support if valid_support else [])):
                for j, file in enumerate(files):
                    location = f"{field}.{group}[{j}]"
                    if _file(c, file, location):
                        paths.append(file["path"])
            c.require(len(paths) == len(set(paths)), field, "duplicate_file", "A file appears more than once in this set", "List each loaded file once")
        c.require(len(ids) == len(set(ids)), "task.model.sets", "duplicate_set", "Set IDs must be unique", "Assign unique alternative IDs")
        choice = model.get("selected_set")
        if c.require(isinstance(choice, str) and choice in ids and ids.count(choice) == 1, "task.model.selected_set", "selection_required", "Exactly one file set must be selected", "Select one set explicitly; no default selection is inferred"):
            selected.update(next(s for s in sets if isinstance(s, dict) and s.get("id") == choice))

    def model_files():
        root = t["model"]["path"]
        for group in ("weights", "support"):
            for index, file in enumerate(selected[group]):
                field = f"task.model.selected.{group}[{index}]"
                try:
                    observed = metadata(root, file["path"])
                    c.require(observed["bytes"] == file["bytes"], field, "file_changed", "File size differs from the declared identity", "Refresh identity and confirm the changed task")
                except (OSError, ValueError) as error:
                    c.issue(field, "file_unavailable", str(error)[:240], "Use an existing regular file inside the model root; no links or special files")

    def request():
        w = c.object(t.get("workload"), "task.workload", ("request",))
        if w is None:
            return
        q = c.object(w.get("request"), "task.workload.request", ("interface", "input", "parameters"))
        if q is not None:
            c.require(q.get("interface") == "cli", "task.workload.request.interface", "unsupported_interface", "Initial contract supports single CLI requests", "Keep HTTP service preparation separate")
            c.text(q.get("input"), "task.workload.request.input", 16000)
            c.require(isinstance(q.get("parameters"), dict), "task.workload.request.parameters", "object_required", "Explicit inference parameters required", "Provide parameters for the selected adapter")

    def acceptance():
        a = c.object(e.get("acceptance"), "execution.acceptance", ("scope", "parser", "verifier", "reference", "tolerances"))
        if a is None:
            return
        c.require(a.get("scope") in ("smoke", "behavior"), "execution.acceptance.scope", "invalid_scope", "Choose smoke or behavior validation", "State precisely which result this request should establish")
        for key in ("parser", "verifier"):
            versioned = c.object(a.get(key), f"execution.acceptance.{key}", ("name", "version"))
            if versioned is not None:
                c.text(versioned.get("name"), f"execution.acceptance.{key}.name", 100)
                c.text(versioned.get("version"), f"execution.acceptance.{key}.version", 100)
        if a.get("scope") == "behavior":
            ref = c.object(a.get("reference"), "execution.acceptance.reference", ("path", "sha256"))
            if ref is not None:
                c.require(_absolute(ref.get("path")), "execution.acceptance.reference.path", "invalid_path", "Absolute reference file path required", "Supply a frozen reference fixture")
                _sha(c, ref.get("sha256"), "execution.acceptance.reference.sha256")
            tolerances = c.object(a.get("tolerances"), "execution.acceptance.tolerances", ("absolute", "relative"))
            if tolerances is not None:
                for key in ("absolute", "relative"):
                    c.number(tolerances.get(key), f"execution.acceptance.tolerances.{key}")
        elif a.get("scope") == "smoke":
            c.require(a.get("reference") is None and a.get("tolerances") is None, "execution.acceptance", "scope_conflict", "Smoke scope must not imply behavior comparison", "Use null reference/tolerances or explicitly select behavior scope")

    def reference_file():
        path = Path(e["acceptance"]["reference"]["path"])
        try:
            metadata(str(path.parent), path.name)
        except (OSError, ValueError) as error:
            c.issue("execution.acceptance.reference.path", "reference_unavailable", str(error)[:240],
                    "Provide an existing regular reference fixture; no links or special files")

    c.run("envelope", [], envelope)
    c.run("action_kind", [], action_kind)
    c.run("action", ["action_kind"], action_check)
    c.run("source", [], source_check)
    c.run("budget", [], budgets)
    c.run("permissions", ["action_kind"], permissions)
    c.run("resources", ["action_kind"], resources)
    c.run("approval", ["action_kind"], authorization)
    if kind in ("build", "run_once"):
        c.run("framework", [], framework_check)
        c.run("adapter", [], adapter)
    if kind == "run_once":
        c.run("model_selection", [], model_selection)
        c.run("model_files", ["model_selection"], model_files)
        c.run("request", [], request)
        c.run("acceptance", [], acceptance)
        if isinstance(e.get("acceptance"), dict) and e["acceptance"].get("scope") == "behavior":
            c.run("reference_file", ["acceptance"], reference_file)
    return _report(c, task_hash, task, action, approval)


def _sha(c, value, field):
    return c.require(isinstance(value, str) and bool(SHA256.fullmatch(value)), field, "invalid_sha256", "Lowercase SHA-256 content identity required", "Supply a recorded content digest; offline validation does not hash the file")


def _file(c, value, field):
    item = c.object(value, field, ("path", "sha256", "bytes"))
    if item is None:
        return False
    valid_path = True
    try:
        path = item.get("path")
        if not parts(path) or str(PurePosixPath(path)) != path:
            raise ValueError("normalized relative path required")
    except (ValueError, TypeError):
        valid_path = False
        c.issue(field + ".path", "invalid_path", "Unsafe or noncanonical model-relative path", "Use an explicit path inside the model root")
    _sha(c, item.get("sha256"), field + ".sha256")
    c.number(item.get("bytes"), field + ".bytes", minimum=1, integer=True)
    return valid_path


def _report(c, task_hash, task, action, approval=None):
    valid = not c.diagnostics and all(r["status"] == "passed" for r in c.results)
    pending = None
    if valid and action is not None:
        kind = action["kind"]
        runtime_checks = ["authenticate_approval", "remaining_budget", "resolve_registered_adapter", "current_tools_and_device"]
        if kind in ("build", "run_once"):
            runtime_checks += ["prepare_record_and_clean_commit", "concrete_command_and_write_scope"]
        if kind == "run_once":
            runtime_checks += ["selected_file_content_and_shard_identity", "successful_build_call_and_artifact_binding", "parser_and_verifier_identity"]
            if task["contract"]["execution"]["acceptance"]["scope"] == "behavior":
                runtime_checks.append("reference_content_identity_and_comparison_rules")
        pending = {"kind": kind, "task_hash": task_hash, "task": copy.deepcopy(task), "action": copy.deepcopy(action),
                   "approval_hash": digest(approval), "approval": copy.deepcopy(approval),
                   "status": "requires_runtime_checks", "required_runtime_checks": runtime_checks}
    return {"version": VERSION, "status": "valid" if valid else "invalid", "scope": "offline_action_contract",
            "task_hash": task_hash, "checks": c.results, "diagnostics": c.diagnostics, "pending_action": pending,
            "executed": False, "execution_authorized": False, "readiness": "unverified",
            "limits": ["Approval is caller-supplied data, not authenticated human consent.",
                       "Only selected model/reference file metadata is inspected; content hashes, shard semantics, Git state and hardware are not verified.",
                       "No adapter commands, runtime processes or provider requests are executed."]}
