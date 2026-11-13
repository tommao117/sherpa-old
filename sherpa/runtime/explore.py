"""Model-directed read-only investigation. Provider tools and target execution are never enabled."""
from __future__ import annotations

import copy
import json
import os
import time
import tomllib
from pathlib import Path

from ..agent import AgentError, Config, Request, Router
from ..platform.exploration import CATALOG, EvidenceTools, bounded_view, validate as validate_tool
from .calls import Calls, RunStopped
from .environment import controller_version
from .evidence_windows import CONTENT_TOOLS, window_views
from .models import Task
from .store import atomic_json, digest, locked, now, process_identity

POLICY = Path(__file__).parents[1] / "agent" / "environment_policy.toml"
CONTEXT_BYTES = 28000
ACTION_FORMS = {
    "tool": {"action": "tool", "tool": "list|read|search|outline|declarations|machine", "parameters": "tool-specific object", "reason": "brief reason"},
    "recall": {"action": "recall", "evidence_id": "previous successful tool call ID", "reason": "why needed"},
    "catalog": {"action": "catalog", "offset": "catalog_next_offset or 0", "reason": "page through older evidence IDs"},
    "delegate": {"action": "delegate", "question": "one bounded question", "reason": "why daily/tools are insufficient",
                 "completion_criteria": "what expert must return", "evidence_ids": ["tool-call-ID"]},
    "finish": {"action": "finish", "summary": "input classification", "routes": [{"name": "unique route name",
               "runtime": "framework/program", "framework_source": "source or missing/unknown", "devices": ["cpu|gpu|unknown"],
               "python_requirement": "needed/version, not required, or unknown", "prerequisites": ["requirements"],
               "steps": ["proposed preparation steps, not executed"], "verification": ["real request check to perform later"],
               "evidence": [{"id": "tool-call-ID", "quote": "exact text from observed contents"}],
               "limitations": ["unverified assumptions"]}], "selected_route": "name or null", "unknowns": ["remaining uncertainty"]},
    "stop": {"action": "stop", "reason": "why cannot produce a supported plan", "unknowns": ["missing evidence"]},
}
EXPERT_FORM = {"advice": "answer to the bounded question", "evidence": [{"id": "tool-call-ID", "quote": "exact quote"}],
               "unknowns": ["unanswered points"]}


def text(value, maximum=1500):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"nonempty string of at most {maximum} characters required")


def strings(value, maximum=12):
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError("bounded string list required")
    for item in value:
        text(item)


def exact_keys(value, keys):
    if not isinstance(value, dict):
        raise ValueError(f"expected an object with exactly these fields: {sorted(keys)}")
    missing, unexpected = set(keys) - set(value), set(value) - set(keys)
    if missing or unexpected:
        raise ValueError(f"missing fields: {sorted(missing)}; unexpected fields: {sorted(unexpected)}; expected exactly these fields: {sorted(keys)}")


def literal_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(literal_text(v) for v in value.values())
    if isinstance(value, list):
        return "\n".join(literal_text(v) for v in value)
    return str(value)


class Investigation:
    def __init__(self, run, task, router, policies, max_rounds=12, max_delegations=1, candidate_timeout=90):
        self.run, self.task, self.router, self.policies = run, task, router, policies
        self.calls = Calls(run, seconds=task.budget["seconds"], max_calls=task.budget["max_calls"])
        self.max_rounds, self.max_delegations, self.candidate_timeout = max_rounds, max_delegations, candidate_timeout
        self.tools = None
        self.evidence, self.seen, self.advice, self.history = {}, set(), [], []
        self.focus, self.previous, self.delegations = None, None, 0
        self.device = task.resources["device"]
        self.final = None
        self.resume_from, self.continuation = None, None
        self.prior_cost = None
        self.catalog_offset = 0
        self.bound_inputs = task.model.get("kind") == "local"
        self.framework_ref = None
        self.framework_max_mib = 512
        self.prepare_only = False
        self.action_forms = copy.deepcopy(ACTION_FORMS)
        self.visible_lines = {}
        self.visible_machine_id = None
        self.pinned_windows = []
        self.report_draft = None
        self.report_forms = copy.deepcopy(ACTION_FORMS)
        if self.bound_inputs:
            from .report_drafts import REPAIR_FORM
            self.action_forms["repair_finish"] = copy.deepcopy(REPAIR_FORM)
            self.report_forms["finish"]["routes"][0].update(framework_commit="runtime-owned commit", model_files=[])
            route = self.action_forms["finish"]["routes"][0]
            for key in ("runtime", "framework_source"):
                route.pop(key)
            route["model_files"] = ["observed path relative to model directory; empty if undecided"]
            route["evidence"] = [{"id": "read/search/outline observation ID", "start": "first displayed line number", "end": "last displayed line number (inclusive, at most 8 lines)"}]
            self.action_forms["pin"] = {"action": "pin", "evidence": route["evidence"],
                                        "reason": "replace retained windows with these currently visible lines; [] releases all, max 6 windows/4500 bytes"}

    def context(self, round_number):
        if self.bound_inputs:
            from .decision_support import context
            return context(self, round_number)
        entries = list(self.evidence.values())
        context = {"goal": "identify runnable implementations and requirements; do not install or launch anything",
                   "requested_device": self.device, "tools": self.tools.catalog if self.bound_inputs else CATALOG, "round": round_number,
                   "remaining": {"calls": self.calls.max_calls - self.calls.count,
                                 "decision_rounds_including_this": self.max_rounds - round_number + 1,
                                 "seconds": max(0, int(self.calls.deadline - time.monotonic())),
                                 "delegations": self.max_delegations - self.delegations},
                   "history": self.history[-4:], "expert_advice": self.advice[-1:],
                   "catalog": [], "catalog_offset": self.catalog_offset, "catalog_total": len(entries),
                   "observations": [], "omitted_evidence_ids": [], "omitted_evidence_count": 0}
        if self.bound_inputs:
            context["goal"] = "determine requirements and compatibility for the USER-SPECIFIED framework and model; never select a replacement framework"
            context["inputs"] = self.tools.binding
            context["framework_unresolved"] = self.framework_ref.get("submodules_not_fetched", [])
        else:
            context["input_mode"] = "legacy_artifact_inspection; not a bound framework optimization task"
        def size(value):
            return len(json.dumps(value, ensure_ascii=False).encode())
        for entry in entries[self.catalog_offset:self.catalog_offset + 20]:
            item = {"id": entry["id"], "tool": entry["tool"], "parameters": entry["parameters"]}
            if context["catalog"] and size(context["catalog"] + [item]) > 5000:
                break
            context["catalog"].append(item)
        next_offset = self.catalog_offset + len(context["catalog"])
        context["catalog_next_offset"] = next_offset if next_offset < len(entries) else None
        if self.continuation:
            context["continuation"] = {k: self.continuation[k] for k in ("source_run", "budget_semantics", "draft", "instructions")}
            latest_machine = next((e for e in reversed(entries) if e["tool"] == "machine"
                                   and e.get("origin_run", self.run.id) == self.run.id), None)
            context["continuation"]["machine_refresh_required"] = latest_machine is None
            context["continuation"]["current_machine_evidence_id"] = latest_machine["id"] if latest_machine else None
            if latest_machine:
                context["continuation"]["instructions"] = (
                    "Machine was refreshed in this segment; use current_machine_evidence_id or recall it. "
                    "Old invalidated machine IDs do not require another refresh. Recheck only if new evidence "
                    "or elapsed time justifies it. This is an observation, not a resource reservation. "
                    "Reuse verified file evidence; correct any historical draft without treating it as verified.")
            invalidated = self.continuation["discarded_evidence"]
            context["continuation"]["invalidated_evidence_ids"] = [e["id"] for e in invalidated[:20]]
            context["continuation"]["invalidated_evidence_count"] = len(invalidated)
            if self.bound_inputs and context["continuation"]["draft"]:
                draft = context["continuation"]["draft"]
                context["continuation"]["draft"] = {**draft, "view": bounded_view(draft["view"], 2500)}
        # Bound fixed context too; reserve room for at least the focused evidence body.
        if size(context) > 14000:
            context["history"] = bounded_view(context["history"], 1500)
            context["expert_advice"] = bounded_view(context["expert_advice"], 2000)
            if self.continuation:
                context["continuation"]["draft"] = bounded_view(context["continuation"]["draft"], 2500)
        priority = [self.focus]
        if self.bound_inputs:
            machine_entry = next((e for e in reversed(entries) if e["tool"] == "machine"), None)
            if machine_entry:
                priority.append(machine_entry["id"])
            model_inventory = next((e for e in reversed(entries) if e["tool"] == "list"
                                    and e["parameters"].get("scope") == "model"
                                    and e["parameters"].get("path", ".") == "."), None)
            if model_inventory:
                priority.append(model_inventory["id"])
            for scope in ("model", "framework"):
                anchor = next((e for e in entries if e["parameters"].get("scope") == scope
                               and e["tool"] in {"read", "declarations"}), None)
                if anchor:
                    priority.append(anchor["id"])
            for item in reversed(self.history[-4:]):
                if item.get("action") == "recall":
                    priority.append(item.get("result", {}).get("evidence_id"))
        ordered = sorted(reversed(entries), key=lambda e: priority.index(e["id"]) if e["id"] in priority else len(priority))
        for entry in ordered:
            context["observations"].append(entry)
            if size(context) > CONTEXT_BYTES - 1800:
                context["observations"].pop()
                context["omitted_evidence_count"] += 1
                if len(context["omitted_evidence_ids"]) < 20:
                    context["omitted_evidence_ids"].append(entry["id"])
        if len(json.dumps(context, ensure_ascii=False).encode()) > CONTEXT_BYTES:
            raise RunStopped("budget_exhausted", "decision context budget exhausted")
        self.seen.update(e["id"] for e in context["observations"])
        return context

    def citations(self, refs, allowed=None):
        if not isinstance(refs, list) or not 1 <= len(refs) <= 8:
            raise ValueError("use 1..8 citations per route/advice; at least one must quote read/declarations contents")
        content_cited = False
        for ref in refs:
            exact_keys(ref, {"id", "quote"})
            text(ref["id"], 100)
            text(ref["quote"], 2000 if self.bound_inputs else 240)
            entry = self.evidence.get(ref["id"])
            if entry is None or ref["id"] not in self.seen or (allowed is not None and ref["id"] not in allowed):
                raise ValueError(f"citation {ref['id']}: evidence not supplied or no longer valid; use a current observation ID")
            view = entry["view"]
            if view.get("truncated"):
                raise ValueError(f"citation {ref['id']}: result truncated; read a narrower fragment")
            data = view.get("data", {})
            if entry["tool"] in CONTENT_TOOLS:
                from .evidence_windows import continuous_texts
                bodies = continuous_texts(data.get("lines", []))
                body = next((block for block in bodies if ref["quote"] in block), "")
                content_cited = True
            elif entry["tool"] == "declarations":
                parsed = data.get("parsed", {})
                body = literal_text({key: parsed.get(key, []) for key in ("packages", "python", "includes", "index")})
                content_cited = True
            elif entry["tool"] in {"list", "machine"}:
                # Supplemental existence/host evidence cannot by itself establish runtime support.
                body = literal_text(data) + "\n" + json.dumps(data, ensure_ascii=False)
            else:
                raise ValueError(f"citation {ref['id']}: unsupported evidence type")
            if len(ref["quote"]) < 5 or ref["quote"] not in body:
                raise ValueError(f"citation {ref['id']} ({entry['tool']}): quote absent; copy the exact visible text or remove this citation")
        if not content_cited:
            raise ValueError("runtime claims require at least one read/declarations quote; list/machine evidence is supplemental only")

    def validate(self, action):
        kind = action.get("action")
        if kind not in self.action_forms:
            raise ValueError("unknown decision action")
        exact_keys(action, self.action_forms[kind])
        if kind == "repair_finish":
            from .report_drafts import expand
            candidate = expand(action, self.report_draft, self.action_forms["finish"])
            return self.validate(candidate)
        if self.bound_inputs and kind == "finish":
            from .decision_support import assemble
            from .report_drafts import candidate_shape
            candidate_shape(action)
            for route in action.get("routes", []) if isinstance(action.get("routes"), list) else []:
                exact_keys(route, self.action_forms["finish"]["routes"][0])
            action = assemble(action, self)
        if kind == "tool":
            text(action["reason"])
            if self.bound_inputs:
                self.tools.validate(action["tool"], action["parameters"])
            else:
                validate_tool(action["tool"], action["parameters"])
        elif kind == "pin":
            text(action["reason"])
            window_views(action["evidence"], self.evidence, self.visible_lines)
        elif kind == "catalog":
            text(action["reason"])
            if isinstance(action["offset"], bool) or not isinstance(action["offset"], int) or not 0 <= action["offset"] < max(1, len(self.evidence)):
                raise ValueError("catalog offset must be within current evidence inventory")
        elif kind == "recall":
            text(action["reason"])
            if action["evidence_id"] not in self.evidence:
                raise ValueError("unknown evidence ID")
        elif kind == "delegate":
            for key in ("reason", "question", "completion_criteria"):
                text(action[key])
            ids = action["evidence_ids"]
            strings(ids, 4)
            if not ids or len(ids) != len(set(ids)) or any(i not in self.seen for i in ids):
                raise ValueError("delegate needs 1..4 previously seen evidence IDs")
        elif kind == "stop":
            text(action["reason"])
            strings(action["unknowns"])
        else:
            text(action["summary"])
            strings(action["unknowns"])
            if (self.bound_inputs and not self.visible_machine_id) or not any(
                    e["tool"] == "machine" and e["id"] in self.seen for e in self.evidence.values()):
                raise ValueError("query machine before finalizing a runtime plan; a complete current observation is required")
            if not isinstance(action["routes"], list) or not 1 <= len(action["routes"]) <= 4:
                raise ValueError("provide 1..4 candidate routes, or stop as blocked")
            names = []
            for route in action["routes"]:
                exact_keys(route, self.report_forms["finish"]["routes"][0])
                if self.bound_inputs:
                    if route["framework_commit"] != self.framework_ref["commit"]:
                        raise ValueError("route must bind to the user-specified framework commit")
                    strings(route["model_files"])
                    self.tools.validate_model_files(route["model_files"])
                for key in ("name", "runtime", "framework_source", "python_requirement"):
                    text(route[key], 1000)
                names.append(route["name"])
                for key in ("prerequisites", "steps", "verification", "limitations"):
                    strings(route[key])
                if not route["steps"] or not route["verification"]:
                    raise ValueError("each route needs preparation and verification steps")
                if not isinstance(route["devices"], list) or not route["devices"] or any(d not in {"cpu", "gpu", "unknown"} for d in route["devices"]):
                    raise ValueError("invalid supported device list")
                self.citations(route["evidence"])
                if self.bound_inputs:
                    scopes = {self.evidence[r["id"]]["parameters"].get("scope") for r in route["evidence"]
                              if self.evidence[r["id"]]["tool"] in CONTENT_TOOLS | {"declarations"}}
                    if scopes != {"framework", "model"}:
                        raise ValueError("each route needs both framework and model content evidence; neither substitutes for the other")
            if len(names) != len(set(names)):
                raise ValueError("route names must be unique")
            selected = action["selected_route"]
            if selected is not None:
                if selected not in names:
                    raise ValueError("selected_route must be a route name or null")
                route = action["routes"][names.index(selected)]
                if self.bound_inputs and not route["model_files"]:
                    raise ValueError("selected route must specify which model files to use; otherwise leave selection null")
                if self.device != "unspecified" and self.device not in route["devices"]:
                    raise ValueError("selected route does not match requested compute device; leave selection null")

    def invoke(self, action, decision):
        started = time.monotonic()
        record, directory = self.calls.new("tool", "environment." + action["tool"], action["parameters"], caused_by=decision)
        record.update(tool_version=self.tools.version, synthetic=False, side_effects="read_only")
        self.run.save("calls", record["id"], record, "evidence_tool_started")
        try:
            result = self.tools.execute(action["tool"], action["parameters"])
            self.calls.check_interrupt()
            self.calls.finish(record, directory, "succeeded", result, started)
            entry = {"id": record["id"], "tool": action["tool"], "parameters": action["parameters"], "view": bounded_view(result)}
            self.evidence[entry["id"]] = entry
            self.focus = entry["id"]
            return {"tool": action["tool"], "status": "succeeded", "evidence_id": entry["id"]}
        except (RunStopped, KeyboardInterrupt) as error:
            status = error.status if isinstance(error, RunStopped) else "cancelled"
            self.calls.finish(record, directory, status, {"error": str(error)}, started)
            raise
        except (OSError, ValueError, TypeError) as error:
            self.calls.finish(record, directory, "failed", {"error": str(error)}, started)
            return {"tool": action["tool"], "status": "failed", "error": str(error)[:500], "call_id": record["id"]}
        except Exception as error:
            self.calls.finish(record, directory, "failed", {"error": str(error)}, started)
            raise

    def delegate(self, action, decision):
        # Reserve at least two calls and time for daily to regain control.
        available = self.calls.max_calls - self.calls.count - 2
        seconds = min(120, self.calls.deadline - time.monotonic() - 15)
        if self.delegations >= self.max_delegations or available <= 0 or seconds <= 0:
            return {"status": "not_started", "reason": "delegation budget exhausted; continue or stop with current evidence"}
        self.delegations += 1
        ids = action["evidence_ids"]
        context = {"question": action["question"], "requested_device": self.device,
                   "evidence": [self.evidence[i] for i in ids]}
        if self.bound_inputs:
            context["inputs"] = self.tools.binding
        if len(json.dumps(context, ensure_ascii=False).encode()) > CONTEXT_BYTES:
            return {"status": "not_started", "reason": "delegated evidence exceeds context budget; choose fewer fragments"}
        def validate_advice(output):
            exact_keys(output, EXPERT_FORM)
            text(output["advice"], 3000)
            strings(output["unknowns"])
            self.citations(output["evidence"], allowed=ids)
            if len(json.dumps(output, ensure_ascii=False).encode()) > 7000:
                raise ValueError("expert advice exceeds 7000-byte return budget")
        request = Request("expert", "bounded environment consultation", self.policies["expert"], context, EXPERT_FORM,
                          intervention_reason=action["reason"], completion_criteria=action["completion_criteria"], caused_by=decision)
        self.run.event("expert_delegated", caused_by=decision, question=action["question"], evidence_ids=ids)
        try:
            answer = self.router.call(request, self.calls, validate_advice, allow_unpriced=True,
                                      timeout_s=self.candidate_timeout, request_timeout_s=seconds,
                                      max_attempts=min(available, 4))
        except AgentError as error:
            if error.category not in {"pool_exhausted", "request_timeout", "attempt_limit", "invalid_response", "capability", "configuration"}:
                raise
            result = {"status": "failed", "reason": str(error), "category": error.category}
        else:
            result = {"status": "answered", "call_id": answer.call_id, "advice": answer.output,
                      "scope": "expert opinion, not an independent observation or a verified environment"}
            self.previous = answer.call_id
        self.advice.append(result)
        self.run.event("expert_returned_to_daily", caused_by=decision, result=result)
        return result

    def execute(self):
        outcome, reason = "completed", "read-only investigation finished"
        with self.run.store.exclusive(), locked(self.run.path / ".execution.lock", blocking=False):
            self.run.update("run_started", status="running", phase="environment_exploration", scope="environment_explore",
                            owner=process_identity(os.getpid()), resume_from=self.resume_from,
                            input_mode="framework_model_v1" if self.bound_inputs else "legacy_artifact_v1",
                            exploration_limits={"max_rounds": self.max_rounds, "max_delegations": self.max_delegations,
                                                "candidate_timeout": self.candidate_timeout})
            try:
                version = controller_version(self.run.store, self.run, self.calls.check_interrupt)
                atomic_json(self.run.path / "agent-config.json", self.router.config.data)
                atomic_json(self.run.path / "environment-policy.json", self.policies)
                from .decision_support import VERSION
                self.run.update("exploration_versions", version=version, agent_config=self.router.config.version,
                                policy_version=digest(self.policies), decision_view=VERSION if self.bound_inputs else "legacy")
                if self.bound_inputs:
                    from .framework import prepare
                    from ..platform.scoped_evidence import ScopedEvidence
                    pinned = None
                    if self.resume_from:
                        from .continuation import cost_total, lineage
                        chain = lineage(self.run.store, self.resume_from)
                        self.prior_cost = cost_total([state["cost"] for _, state, _ in chain])
                        pinned = next((s["framework"] for _, s, _ in reversed(chain) if s.get("framework")), None)
                    self.run.update("phase_changed", phase="preparing_framework")
                    self.framework_ref = prepare(self.run, self.calls, max_mib=self.framework_max_mib, pinned=pinned)
                    self.tools = ScopedEvidence(self.framework_ref, self.task.model, self.run.path, self.calls.check_interrupt)
                else:
                    self.tools = EvidenceTools(Path(self.task.target["path"]), self.run.path, self.calls.check_interrupt)
                if self.resume_from:
                    from .continuation import restore
                    self.run.update("phase_changed", phase="restoring_evidence")
                    restore(self, self.resume_from)
                if self.prepare_only:
                    reason = "framework commit prepared; no model calls, builds or target execution"
                    self.run.update("code_prepared", result_status="code_prepared")
                    return self.run, None
                for number in range(1, self.max_rounds + 1):
                    self.calls.check()
                    self.run.update("phase_changed", phase="daily_decision", decision_round=number)
                    context = self.context(number)
                    request = Request("daily", "choose next read-only environment investigation action", self.policies["daily"],
                                      context, self.action_forms, caused_by=self.previous)
                    try:
                        answer = self.router.call(request, self.calls, self.validate, timeout_s=self.candidate_timeout,
                                                  allow_unpriced=True, request_timeout_s=min(180, self.calls.deadline - time.monotonic()))
                    except AgentError as error:
                        if error.category != "invalid_response":
                            raise
                        self.history.append({"status": "invalid_decision", "reason": str(error)[:400]})
                        record = self.run.records("calls")[-1]
                        self.previous = record["id"]
                        if self.bound_inputs:
                            from .report_drafts import capture
                            capture(self, record)
                        continue
                    action = answer.output
                    self.previous = answer.call_id
                    kind = action["action"]
                    if kind in {"finish", "repair_finish"}:
                        self.calls.check_interrupt()
                        if self.bound_inputs:
                            from .report_drafts import capture, expand
                            from .decision_support import assemble
                            candidate = expand(action, self.report_draft, self.action_forms["finish"])
                            capture(self, self.run.records("calls")[-1])
                            if not self.report_draft or self.report_draft["call_id"] != answer.call_id or self.report_draft["candidate_hash"] != digest(candidate):
                                raise ValueError("validated candidate was not captured consistently; proposal not committed")
                            action = assemble(candidate, self)
                        self.final = {"status": "proposed", "readiness": "unverified", "installed": False,
                                      "source": self.task.target, "requested_device": self.device, "plan": action,
                                      "evidence": list(self.evidence.values()), "decision_call": answer.call_id,
                                      "agent_config": self.router.config.version, "policy_version": digest(self.policies),
                                      "limits": ["Citations checked for existence and literal quotes, not semantic entailment.",
                                                 "No target scripts, installs, builds, URL fetches or target-model inference were executed; remote decision models were called.",
                                                 "Evidence is a snapshot, not a committed target; runtime compatibility remains untested."]}
                        if self.bound_inputs:
                            from .framework import verify_clean
                            verify_clean(Path(self.framework_ref["path"]), self.framework_ref["commit"], self.calls.check_interrupt)
                            self.final["inputs"] = self.tools.binding
                            self.final["framework"] = self.framework_ref
                            from .decision_support import VERSION
                            self.final["assembly"] = {"version": VERSION, "decision_call": answer.call_id,
                                                      "line_selections": [r["evidence"] for r in candidate["routes"]],
                                                      "candidate_id": self.report_draft["id"],
                                                      "candidate_hash": self.report_draft["candidate_hash"]}
                            self.final["limits"] = ["Only the user-specified framework commit was prepared; no compilation, installation or target inference.",
                                                    "Model weights are referenced, not hashed or loaded; compatibility and behavior remain unverified.",
                                                    "Citations prove visible source text, not semantic correctness of the proposal."]
                        self.final["plan_id"] = digest(self.final)
                        atomic_json(self.run.path / "environment" / "proposal.json", self.final)
                        self.run.update("environment_proposed", environment_plan="environment/proposal.json", readiness="unverified")
                        break
                    if kind == "stop":
                        reason = action["reason"]
                        self.run.update("environment_blocked", result_status="blocked", unknowns=action["unknowns"])
                        break
                    if kind == "tool":
                        self.run.update("phase_changed", phase="evidence_collection")
                        result = self.invoke(action, answer.call_id)
                        self.previous = result.get("evidence_id", result.get("call_id", answer.call_id))
                    elif kind == "pin":
                        self.calls.check_interrupt()
                        self.pinned_windows = copy.deepcopy(action["evidence"])
                        result = {"status": "windows_updated", "windows": self.pinned_windows}
                        self.run.update("evidence_windows_changed", evidence_windows=self.pinned_windows,
                                        windows_caused_by=answer.call_id)
                    elif kind == "catalog":
                        self.catalog_offset = action["offset"]
                        result = {"status": "catalog_page", "offset": self.catalog_offset}
                        self.run.event("evidence_catalog_paged", caused_by=answer.call_id, offset=self.catalog_offset)
                    elif kind == "recall":
                        self.focus = action["evidence_id"]
                        result = {"status": "recalled", "evidence_id": self.focus}
                        self.run.event("evidence_recalled", caused_by=answer.call_id, evidence_id=self.focus)
                    else:
                        self.run.update("phase_changed", phase="expert_consultation")
                        result = self.delegate(action, answer.call_id)
                        result = {k: v for k, v in result.items() if k != "advice"}
                    self.history.append({"action": kind, "result": result})
                else:
                    raise RunStopped("budget_exhausted", "decision round limit reached; partial evidence retained")
            except RunStopped as error:
                outcome, reason = error.status, error.reason
            except KeyboardInterrupt:
                outcome, reason = "cancelled", "keyboard interrupt"
            except Exception as error:
                outcome, reason = "failed", f"{type(error).__name__}: {error}"
            finally:
                if self.tools:
                    self.tools.close()
                if outcome != "completed":
                    self.final = None
                segment_cost = self.calls.cost_summary()
                from .continuation import cost_total
                cumulative_cost = (None if self.resume_from and self.prior_cost is None else
                                   cost_total([segment_cost, self.prior_cost] if self.prior_cost is not None else [segment_cost]))
                self.run.update("run_finished", status=outcome, phase="finished", stop_reason=reason,
                                ended_at=now(), active_call=None, cost=segment_cost, cumulative_cost=cumulative_cost)
        return self.run, self.final


def explore_environment(store, source=None, *, allow_model_calls=False, allow_unpriced=False, model_budget=0,
                        seconds=300, max_calls=24, max_rounds=12, max_delegations=1, device="unspecified",
                        candidate_timeout=90, config_path=None, router=None, announce=None, resume_from=None,
                        framework=None, model=None, revision=None, framework_max_mib=512, prepare_only=False):
    if not prepare_only and (not allow_model_calls or not allow_unpriced):
        raise ValueError("explore sends selected evidence to the configured provider; explicit allow_model_calls and allow_unpriced required")
    if not prepare_only and model_budget <= 0:
        raise ValueError("positive model budget required; unknown provider billing cannot be hard-capped")
    previous_task = None
    if resume_from:
        from .continuation import lineage
        previous_task = lineage(store, resume_from)[-1][1]["task"]
        if source is not None or framework is not None or model is not None or revision is not None or device != "unspecified":
            raise ValueError("resume inherits framework, model and requested device; do not override them")
        device = previous_task["resources"]["device"]
    elif framework is not None or model is not None:
        if source is not None or framework is None or model is None:
            raise ValueError("framework and model must both be explicit, without a legacy source argument")
    elif source is None or prepare_only:
        raise ValueError("provide both framework and model; artifact-only input is not a framework optimization task")
    if isinstance(framework_max_mib, bool) or not isinstance(framework_max_mib, int) or not 1 <= framework_max_mib <= 8192:
        raise ValueError("framework_max_mib must be in [1, 8192]")
    if device not in {"cpu", "gpu", "unspecified"}:
        raise ValueError("device must be cpu, gpu or unspecified")
    for value, lower, upper in ((max_calls, 2, 60), (max_rounds, 1, 30), (max_delegations, 0, 4)):
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError("invalid exploration bounds")
    if not 0 < seconds <= 900 or not 0 < candidate_timeout <= 300:
        raise ValueError("invalid timeout")
    budget = {"seconds": seconds, "max_calls": max_calls, "model_cost_usd": 0 if prepare_only else model_budget}
    if previous_task:
        task = Task(**{**previous_task, "budget": budget})
    elif framework is not None:
        from .framework import target
        model_path = Path(model).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        task = Task("environment-explore", target(str(framework), revision),
                    {"kind": "local", "path": str(model_path)}, resources={"gpus": [], "device": device}, budget=budget)
    else:
        source = Path(source).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        task = Task("environment-explore", {"kind": "local", "path": str(source)},
                    {"source": "input artifacts", "resolved": False}, resources={"gpus": [], "device": device}, budget=budget)
    if task.model.get("kind") == "local" and not Path(task.model["path"]).is_dir():
        raise FileNotFoundError(task.model["path"])
    policies = tomllib.loads(POLICY.read_text())
    router = router or Router(Config.load(config_path))
    run = store.create_run(task, kind="environment", sources=[resume_from] if resume_from else [])
    if announce:
        announce(run.id)
    investigation = Investigation(run, task, router, policies, max_rounds, max_delegations, candidate_timeout)
    investigation.resume_from = resume_from
    investigation.framework_max_mib = framework_max_mib
    investigation.prepare_only = prepare_only
    return investigation.execute()
