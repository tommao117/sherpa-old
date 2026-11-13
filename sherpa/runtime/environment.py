"""Observable environment planning; installation and model loading are deliberately absent."""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

from ..platform import environment, machine
from . import workspace
from .calls import Calls, RunStopped
from .models import Task
from .store import atomic_json, digest, locked, now, process_identity


def controller_version(store, run, check):
    inventory = []
    sources = sorted((store.project / "sherpa").rglob("*.py")) + [store.project / "pyproject.toml"]
    total = 0
    for path in sources:
        check()
        if not path.is_file() or path.is_symlink():
            continue
        data = environment.read_bounded(path)
        total += len(data)
        if total > 2 * 1024 * 1024 or len(inventory) >= 128:
            raise ValueError("controller snapshot exceeds environment-plan limit")
        relative = path.relative_to(store.project)
        output = run.path / "controller-snapshot" / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
        inventory.append({"path": relative.as_posix(), "sha256": hashlib.sha256(data).hexdigest()})
    with workspace.interruptible(check, scratch=run.path):
        commit = workspace.git(store.project, "rev-parse", "HEAD", check=False) or None
    return {"commit": commit, "source_id": digest(inventory), "source_files": inventory,
            "snapshot": "controller-snapshot", "dirty": "not_assessed; snapshot identifies actual code"}


def build_plan(declarations, observed, selectors=(), count=1, intent="optimize"):
    selection = machine.select_gpus(observed, selectors, count)
    issues = declarations["issues"] + selection["issues"]
    if observed["gpu_status"] != "observed":
        issues.append("GPU inventory unavailable or incomplete")
    if not observed["tools"].get("uv"):
        issues.append("uv not found; installation executor prerequisite is missing")
    desired_tools = ["uv", "git"] + (["cc", "cmake", "nsys", "py-spy"] if intent == "optimize" else [])
    capabilities = [{"tool": tool, "path": observed["tools"].get(tool),
                     "availability": "found_not_validated" if observed["tools"].get(tool) else "not_found"}
                    for tool in desired_tools]
    metadata = [{k: s[k] for k in ("path", "kind", "sha256", "bytes")} for s in declarations["sources"]]
    return {"schema_version": 1, "plan_version": "environment-plan-v1", "intent": intent,
            "status": "needs_review" if issues else "declarations_collected",
            "readiness": "unverified", "installed": False, "source": declarations["source"],
            "profiles": declarations["profiles"], "selected_profile": declarations["selected_profile"],
            "included_sources": declarations["included_sources"], "python": declarations["python"],
            "packages": declarations["packages"], "declaration_fingerprint": declarations["source_fingerprint"],
            "declaration_files": metadata, "machine": observed, "gpu_selection": selection,
            "capabilities": capabilities, "issues": issues, "warnings": declarations["warnings"],
            "unresolved": declarations["unresolved"],
            "next_steps": [
                {"step": "review", "description": "Review selected profile, remaining declarations and system prerequisites"},
                {"step": "resolve", "description": "Resolve compatible interpreter, package sources and transitive dependencies"},
                {"step": "lock", "description": "Create and verify an immutable lock; a declaration fingerprint is not an environment ID"},
                {"step": "materialize", "description": "Create isolated environment under the data root, never inside target source"},
                {"step": "validate", "description": "Check actual versions, GPU execution and explicitly approved self-tests"}],
            "limitations": ["No source script or target module was executed; no LLM was called.",
                            "Package availability, dependency compatibility and GPU kernels have not been verified.",
                            "GPU list is an observation, not a reservation; recheck before executing.",
                            "Driver version alone does not prove a given CUDA wheel works.",
                            "Only bounded declaration files were hashed; this is not a complete target Git snapshot."]}


def plan_environment(store, source: Path, profile=None, selectors=(), count=1, intent="optimize", announce=None):
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if intent not in {"optimize", "serve"}:
        raise ValueError("intent must be optimize or serve")
    if count < 1 or (selectors and count != 1):
        raise ValueError("use explicit GPUs or a positive automatic GPU count, not both")
    # This task identifies the input package, not an assertion about its model architecture.
    task = Task("environment-plan", {"kind": "local", "path": str(source)},
                {"source": "target-package", "resolved": False},
                resources={"gpus": list(selectors), "selection": "explicit" if selectors else "auto", "count": count},
                budget={"seconds": 60, "max_calls": 4, "model_cost_usd": 0})
    run = store.create_run(task, kind="environment")
    if announce:
        announce(run.id)
    calls = Calls(run, seconds=60, max_calls=4)
    with store.exclusive(), locked(run.path / ".execution.lock", blocking=False):
        run.update("run_started", status="running", phase="inspecting_environment", scope="environment_plan",
                   owner=process_identity(os.getpid()))
        outcome, reason, plan = "completed", "planning completed; environment is not installed or validated", None
        try:
            version = controller_version(store, run, calls.check_interrupt)
            run.update("controller_snapshotted", version=version)
            def invoke(name, inputs, operation):
                started = time.monotonic()
                record, directory = calls.new("tool", name, inputs)
                record.update(tool_version="read-only-v1", synthetic=False, side_effects="read_only_inspection")
                run.save("calls", record["id"], record, "inspection_started")
                try:
                    result = operation()
                    calls.check_interrupt()
                except BaseException as error:
                    status = error.status if isinstance(error, RunStopped) else "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
                    calls.finish(record, directory, status, {"error": str(error)}, started)
                    raise
                calls.finish(record, directory, "succeeded", result, started)
                return result
            declarations = invoke("environment.declarations", {"source": str(source), "profile": profile},
                                  lambda: environment.inspect(source, profile, calls.check_interrupt))
            run.update("phase_changed", phase="inspecting_machine")
            observed = invoke("environment.machine", {"gpu_probe": "nvidia-smi read-only; 3s per query"},
                              lambda: machine.probe(run.path, calls.check_interrupt))
            run.update("phase_changed", phase="planning")
            plan = build_plan(declarations, observed, selectors, count, intent)
            calls.check_interrupt()
            plan["plan_id"] = digest({"declarations": plan["declaration_fingerprint"], "profile": profile,
                                      "selected": plan["gpu_selection"], "intent": intent,
                                      "version": version["source_id"]})
            plan["raw_evidence_calls"] = [c["id"] for c in run.records("calls")]
            atomic_json(run.path / "environment" / "plan.json", plan)
            run.update("environment_planned", environment_plan="environment/plan.json", plan_status=plan["status"],
                       readiness=plan["readiness"])
        except RunStopped as error:
            outcome, reason = error.status, error.reason
        except KeyboardInterrupt:
            outcome, reason = "cancelled", "keyboard interrupt"
        except Exception as error:
            outcome, reason = "failed", f"{type(error).__name__}: {error}"
        finally:
            if outcome != "completed":
                plan = None
            run.update("run_finished", status=outcome, phase="finished", stop_reason=reason, ended_at=now(),
                       active_call=None, cost=calls.cost_summary())
    return run, plan
