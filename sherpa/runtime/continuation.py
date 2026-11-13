"""Reconstruct read-only exploration state without reopening a terminal run or replaying actions."""
from __future__ import annotations

import json
from pathlib import Path

from ..platform.exploration import bounded_view
from .store import digest, identifier

MAX_ARTIFACT_BYTES = 4 * 1024 * 1024


def artifact(run, relative):
    path = run.path / relative
    if path.is_symlink() or run.path.resolve() not in path.resolve().parents:
        raise ValueError("unsafe continuation artifact path")
    with path.open("rb") as handle:
        raw = handle.read(MAX_ARTIFACT_BYTES + 1)
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ValueError("continuation artifact exceeds 4 MiB")
    return json.loads(raw)


def lineage(store, run_id):
    chain, visited = [], set()
    while run_id:
        identifier(run_id)
        if run_id in visited or len(chain) >= 16:
            raise ValueError("invalid or overly long continuation chain")
        visited.add(run_id)
        run = store.open_run(run_id)
        state = run.state()
        if state.get("kind") != "environment" or state.get("scope") != "environment_explore":
            raise ValueError("only environment exploration runs can be resumed")
        code_prepared = state["status"] == "completed" and state.get("result_status") == "code_prepared" and state.get("framework")
        if state["status"] not in {"budget_exhausted", "timed_out", "cancelled", "failed"} and not code_prepared:
            raise ValueError("resume requires a stopped incomplete exploration; running/cleanup_failed/completed runs are not resumable")
        records = run.records("calls")
        if len(records) > 60 or any(r["status"] not in {"succeeded", "failed", "cancelled", "timed_out"} for r in records):
            raise ValueError("source run contains unfinished or unsafe calls")
        if any(r.get("cleanup_complete") is False or r.get("provider_metadata", {}).get("cleanup_complete") is False for r in records):
            raise ValueError("source run has incomplete process cleanup")
        if state.get("framework"):
            prepares = [r for r in records if r["name"] == "framework.prepare" and r["status"] == "succeeded"]
            if len(prepares) != 1:
                raise ValueError("pinned framework lacks a unique successful prepare record")
            record = prepares[0]
            call_id = identifier(record["id"])
            prepared = artifact(run, f"calls/{call_id}/output.json")
            if digest(prepared) != record.get("output_hash") or prepared != state["framework"]:
                raise ValueError("pinned framework state differs from verified prepare output")
            expected_path = run.path / "workspace" / "framework"
            if prepared["source"] != state["task"]["target"] or Path(prepared["path"]).resolve() != expected_path.resolve():
                raise ValueError("pinned framework source/workspace mismatch")
        chain.append((run, state, records))
        run_id = state.get("resume_from")
    chain.reverse()
    target = chain[-1][1]["task"]
    if any(any(s["task"][key] != target[key] for key in ("target", "model", "resources")) for _, s, _ in chain):
        raise ValueError("continuation task/source/model mismatch")
    bindings = {(s["framework"]["commit"], digest(s["framework"]["source"])) for _, s, _ in chain if s.get("framework")}
    if len(bindings) > 1:
        raise ValueError("continuation framework commit/source mismatch")
    return chain


def cost_total(costs):
    known = sum(c.get("known_model_cost_usd", c.get("model_cost_usd") or 0) for c in costs)
    unknown = sum(c.get("unpriced_model_calls", int(c.get("model_cost_usd") is None)) for c in costs)
    return {"seconds": round(sum(c.get("seconds", 0) for c in costs), 6),
            "calls": sum(c.get("calls", 0) for c in costs), "known_model_cost_usd": known,
            "unpriced_model_calls": unknown, "model_cost_usd": None if unknown else known}


def restore(investigation, source_run):
    investigation.calls.check_interrupt()
    chain = lineage(investigation.run.store, source_run)
    investigation.prior_cost = cost_total([state["cost"] for _, state, _ in chain])
    previous_task = chain[-1][1]["task"]
    if any(previous_task[key] != getattr(investigation.task, key) for key in ("target", "model", "resources")):
        raise ValueError("resume target/model/resources changed before execution")
    pinned = next((s["framework"] for _, s, _ in reversed(chain) if s.get("framework")), None)
    if pinned and (not investigation.framework_ref or investigation.framework_ref["commit"] != pinned["commit"]):
        raise ValueError("resume must use the previously pinned framework commit")
    inventory, history, experts, discarded, snapshots = {}, [], [], [], []
    draft = None
    ids = set()
    for run, state, records in chain:
        fingerprints = []
        for record in records:
            investigation.calls.check_interrupt()
            call_id = identifier(record["id"])
            if call_id in ids:
                raise ValueError("duplicate call ID across continuation chain")
            ids.add(call_id)
            expected_input = f"calls/{call_id}/input.json"
            expected_output = f"calls/{call_id}/output.json"
            if record.get("input") != expected_input or record.get("output") != expected_output:
                raise ValueError("call artifact location mismatch")
            inputs, output = artifact(run, expected_input), artifact(run, expected_output)
            if digest(inputs) != record.get("input_hash") or digest(output) != record.get("output_hash"):
                raise ValueError(f"source call artifact checksum mismatch: {call_id}")
            fingerprints.append({"id": call_id, "record_hash": digest(record),
                                 "input_hash": record["input_hash"], "output_hash": record["output_hash"]})
            status = record["status"]
            if record["kind"] == "tool" and record["name"].startswith("environment."):
                name = record["name"].removeprefix("environment.")
                if status == "succeeded" and name in {"list", "read", "search", "outline", "declarations", "machine"}:
                    inventory[call_id] = {"id": call_id, "tool": name, "parameters": inputs,
                                         "view": bounded_view(output), "origin_run": run.id,
                                         "origin_output_hash": record["output_hash"]}
                history.append({"action": "tool", "result": {"tool": name, "status": status, "call_id": call_id,
                                                               "reason": str(output.get("error", ""))[:300]}})
            elif record["kind"] == "agent":
                if record.get("role") == "expert":
                    if status == "succeeded":
                        experts.append({"status": "answered", "call_id": call_id, "origin_run": run.id,
                                        "advice": output, "scope": "historical expert opinion; not an observation"})
                    else:
                        experts.append({"status": "failed", "origin_run": run.id, "call_id": call_id,
                                        "reason": str(output.get("error", ""))[:400]})
                else:
                    history.append({"action": output.get("action", "invalid_decision"), "call_id": call_id,
                                    "status": status, "reason": str(output.get("error", output.get("reason", "")))[:400]})
                    # Preserve an unfinished proposed plan as a draft, never as accepted evidence or an action to replay.
                    if status == "succeeded" and output.get("action") == "finish":
                        draft = {"call_id": call_id, "origin_run": run.id, "status": "uncommitted_draft",
                                 "error": state.get("stop_reason", "plan was not committed"), "view": bounded_view(output, 6500)}
                    if record.get("error_category") == "invalid_response":
                        raw_path = f"calls/{call_id}/raw.json"
                        if (run.path / raw_path).exists():
                            raw = artifact(run, raw_path)
                            try:
                                proposed = json.loads(raw.get("text", ""))
                            except (ValueError, TypeError):
                                proposed = None
                            if isinstance(proposed, dict) and proposed.get("action") == "finish":
                                draft = {"call_id": call_id, "origin_run": run.id, "status": "rejected_draft",
                                         "error": str(output.get("error", ""))[:400], "view": bounded_view(proposed, 6500)}
                                fingerprints[-1]["raw_hash"] = digest(raw)
                    if investigation.bound_inputs:
                        from .report_drafts import restore_record
                        investigation.report_draft = restore_record(run, record, output, investigation.report_draft,
                                                                    investigation.action_forms["finish"], artifact)
                        if record.get("report_draft"):
                            fingerprints[-1]["report_draft"] = record["report_draft"]
                        if investigation.report_draft:
                            current = investigation.report_draft
                            draft = {k: current[k] for k in ("call_id", "origin_run", "status", "error")}
                            draft["view"] = bounded_view(current["candidate"], 6500)
        snapshots.append({"run_id": run.id, "state_hash": digest(state), "calls": fingerprints,
                          "agent_config": state.get("agent_config"), "policy_version": state.get("policy_version"),
                          "framework_version": state.get("version"), "cost": state["cost"]})
    for call_id, entry in inventory.items():
        investigation.calls.check_interrupt()
        if entry["tool"] == "machine":
            discarded.append({"id": call_id, "reason": "machine observation expired; query machine again"})
            continue
        try:
            current = investigation.tools.execute(entry["tool"], entry["parameters"])
            if digest(current) != entry["origin_output_hash"]:
                raise ValueError("source contents/listing or tool interpretation changed")
        except (OSError, ValueError, TypeError) as error:
            discarded.append({"id": call_id, "reason": str(error)[:300]})
            continue
        investigation.evidence[call_id] = entry
    retained = set(investigation.evidence)
    investigation.seen.clear()
    investigation.focus = next(reversed(investigation.evidence), None)
    investigation.history = history[-4:]
    if investigation.bound_inputs:
        from .evidence_windows import window_views
        latest = next((s for _, s, _ in reversed(chain) if "evidence_windows" in s), {})
        refs = latest.get("evidence_windows", [])
        if not isinstance(refs, list) or len(refs) > 6 or any(not isinstance(r, dict) for r in refs):
            raise ValueError("invalid stored evidence windows")
        surviving = [r for r in refs if r.get("id") in retained]
        window_views(surviving, investigation.evidence)
        investigation.pinned_windows = surviving
        investigation.run.update("evidence_windows_restored", evidence_windows=surviving,
                                 windows_source=latest.get("id"))
    # Failed local consultations remain informative; successful advice requires all its sources to still be valid.
    investigation.advice = [e for e in experts if e["status"] != "answered" or (
        e["advice"].get("evidence") and all(r.get("id") in retained for r in e["advice"]["evidence"]))][-1:]
    investigation.continuation = {"source_run": source_run, "budget_semantics": "new additional budget for this segment",
                                 "reused_evidence_ids": sorted(retained), "discarded_evidence": discarded,
                                 "source_runs": [run.id for run, _, _ in chain],
                                 "prior_cost": cost_total([state["cost"] for _, state, _ in chain]),
                                 "draft": draft, "instructions": "reuse verified file evidence; refresh machine; correct the rejected draft, do not execute it"}
    atomic = {"continuation": investigation.continuation, "source_snapshots": snapshots,
              "revalidated_evidence": list(investigation.evidence.values()), "history": investigation.history,
              "expert_advice": investigation.advice}
    from .store import atomic_json
    atomic_json(investigation.run.path / "environment" / "continuation.json", atomic)
    investigation.run.update("exploration_continued", resume_from=source_run, continuation="environment/continuation.json",
                             source_chain=investigation.continuation["source_runs"],
                             prior_cost=investigation.continuation["prior_cost"])
