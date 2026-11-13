"""Immutable report candidates and explicit field repairs, never partial acceptance."""
from __future__ import annotations

import copy
import json

from ..platform.exploration import bounded_view
from .store import atomic_json, digest, identifier

VERSION = "report-draft-v1"
MAX_BYTES = 48000
REPAIR_FORM = {"action": "repair_finish", "draft_id": "current report_draft.id", "changes": [
    {"route": "zero-based route index, or null for a top-level field", "field": "exact field name from finish contract",
     "value": "replacement value for the entire field; no automatic defaults"}]}


def finite_json(text):
    return json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def candidate_shape(candidate):
    if not isinstance(candidate, dict) or candidate.get("action") != "finish":
        raise ValueError("draft must be a finish object")
    routes = candidate.get("routes")
    if not isinstance(routes, list) or not 1 <= len(routes) <= 4 or any(not isinstance(r, dict) for r in routes):
        raise ValueError("draft needs 1..4 route objects; send a new finish otherwise")
    if len(json.dumps(candidate, ensure_ascii=False, allow_nan=False).encode()) > MAX_BYTES:
        raise ValueError("draft exceeds 48000-byte limit; send a smaller finish")


def expand(action, draft, finish_form):
    if action.get("action") != "repair_finish":
        candidate_shape(action)
        return copy.deepcopy(action)
    if set(action) != set(REPAIR_FORM) or not draft or action.get("draft_id") != draft["id"]:
        raise ValueError("repair requires the current report_draft.id and exactly action/draft_id/changes")
    changes = action["changes"]
    if not isinstance(changes, list) or not 1 <= len(changes) <= 12:
        raise ValueError("repair requires 1..12 explicit field replacements")
    candidate = copy.deepcopy(draft["candidate"])
    candidate_shape(candidate)
    used = set()
    for change in changes:
        if not isinstance(change, dict) or set(change) != {"route", "field", "value"}:
            raise ValueError("each replacement requires exactly route/field/value")
        index, field = change["route"], change["field"]
        if not isinstance(field, str):
            raise ValueError("repair field must be a string")
        if index is None:
            target, allowed = candidate, set(finish_form) - {"action", "routes"}
        else:
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(candidate["routes"]):
                raise ValueError("repair route index is out of range")
            target, allowed = candidate["routes"][index], set(finish_form["routes"][0])
        if field not in allowed:
            raise ValueError("field is not repairable; identity/action/route structure cannot be changed by field repair")
        if (index, field) in used:
            raise ValueError("duplicate replacement target")
        used.add((index, field))
        target[field] = copy.deepcopy(change["value"])
    candidate_shape(candidate)
    return candidate


def make(candidate, run_id, call_id, status, error, base=None):
    candidate_shape(candidate)
    return {"version": VERSION, "id": f"{run_id}/{call_id}", "origin_run": run_id, "call_id": call_id,
            "candidate": copy.deepcopy(candidate), "candidate_hash": digest(candidate), "status": status,
            "error": str(error)[:1000], "base": ({"id": base["id"], "candidate_hash": base["candidate_hash"]} if base else None)}


def capture(investigation, record):
    """Called only after the current call settled. Leave raw/output unchanged."""
    from .continuation import artifact
    if record.get("role") != "daily" or not (record["status"] == "succeeded" or record.get("error_category") == "invalid_response"):
        return
    call_id = identifier(record["id"])
    raw = artifact(investigation.run, f"calls/{call_id}/raw.json")
    try:
        action = finite_json(raw["text"])
        if not isinstance(action, dict) or action.get("action") not in {"finish", "repair_finish"}:
            return
        candidate = expand(action, investigation.report_draft, investigation.action_forms["finish"])
        output = artifact(investigation.run, record["output"])
        draft = make(candidate, investigation.run.id, call_id,
                     "uncommitted_draft" if record["status"] == "succeeded" else "rejected_draft",
                     output.get("error", "candidate validated but not yet committed"),
                     investigation.report_draft if action["action"] == "repair_finish" else None)
    except (ValueError, TypeError, KeyError):
        return
    relative = f"calls/{call_id}/report-draft.json"
    atomic_json(investigation.run.path / relative, draft)
    record["report_draft"] = {"path": relative, "hash": digest(draft), "raw_hash": digest(raw)}
    investigation.run.save("calls", call_id, record, "report_draft_recorded")
    investigation.report_draft = draft


def restore_record(run, record, output, previous, finish_form, artifact):
    """Check derived candidates against the recorded model action and base, without executing it."""
    ref = record.get("report_draft")
    relevant = record["status"] == "succeeded" or record.get("error_category") == "invalid_response"
    if not relevant:
        if ref:
            raise ValueError("report draft attached to an unsettled/unusable decision")
        return previous
    call_id = identifier(record["id"])
    raw_path = f"calls/{call_id}/raw.json"
    if not ref:
        candidate = output if record["status"] == "succeeded" else None
        if record.get("error_category") == "invalid_response" and (run.path / raw_path).exists():
            try:
                candidate = finite_json(artifact(run, raw_path).get("text", ""))
            except (ValueError, TypeError):
                return previous
        try:
            if not isinstance(candidate, dict) or candidate.get("action") not in {"finish", "repair_finish"}:
                return previous
            repairing = candidate["action"] == "repair_finish"
            candidate = expand(candidate, previous, finish_form)
            return make(candidate, run.id, call_id,
                        "uncommitted_draft" if record["status"] == "succeeded" else "rejected_draft",
                        output.get("error", "candidate validated but not yet committed"), previous if repairing else None)
        except (ValueError, TypeError):
            return previous
    if not isinstance(ref, dict) or ref.get("path") != f"calls/{call_id}/report-draft.json":
        raise ValueError("report draft artifact location mismatch")
    raw, stored = artifact(run, raw_path), artifact(run, ref["path"])
    if digest(stored) != ref.get("hash") or digest(raw) != ref.get("raw_hash"):
        raise ValueError("report draft artifact checksum mismatch")
    action = finite_json(raw.get("text", ""))
    if not isinstance(action, dict) or action.get("action") not in {"finish", "repair_finish"}:
        raise ValueError("report draft does not originate in a report decision")
    if record["status"] == "succeeded" and action != output:
        raise ValueError("report draft action differs from successful call output")
    candidate = expand(action, previous, finish_form)
    expected = make(candidate, run.id, call_id,
                    "uncommitted_draft" if record["status"] == "succeeded" else "rejected_draft",
                    output.get("error", "candidate validated but not yet committed"),
                    previous if action["action"] == "repair_finish" else None)
    if stored != expected:
        raise ValueError("report draft differs from its verified action/base")
    return stored


def evidence_views(draft, evidence):
    """Resurface the model's cited source lines, not its claims or invalidated evidence."""
    from .evidence_windows import select_lines
    views, unavailable, seen = [], [], set()
    if draft is None:
        return views, unavailable
    for route in draft["candidate"]["routes"]:
        refs = route.get("evidence", [])
        if not isinstance(refs, list):
            continue
        for ref in refs[:8]:
            try:
                lines = select_lines(ref, evidence)
                key = (ref["id"], ref["start"], ref["end"])
                if key in seen:
                    continue
                seen.add(key)
                entry = evidence[ref["id"]]
                item = {"id": entry["id"], "tool": entry["tool"], "parameters": entry["parameters"],
                        "requested_by_draft": draft["id"], "displayed_ranges": [[ref["start"], ref["end"]]],
                        "view": {"data": {"path": entry["parameters"]["path"], "lines": lines}, "truncated": False}}
                if len(views) >= 6 or len(json.dumps(views + [item], ensure_ascii=False).encode()) > 3000:
                    raise ValueError("draft evidence window budget reached")
                views.append(item)
            except (ValueError, TypeError, KeyError) as error:
                if len(unavailable) < 8:
                    unavailable.append({"reference": bounded_view(ref, 180), "reason": str(error)[:140]})
    return views, unavailable


def view(draft, finish_form):
    if draft is None:
        return None
    candidate = draft["candidate"]
    routes = []
    repairable = not (set(candidate) - set(finish_form))
    for index, route in enumerate(candidate["routes"]):
        missing = sorted(set(finish_form["routes"][0]) - set(route))
        extra = sorted(set(route) - set(finish_form["routes"][0]))
        repairable = repairable and not extra
        routes.append({"index": index, "name": str(route.get("name", ""))[:120], "missing_fields": missing,
                       "unexpected_fields": bounded_view(extra, 150), "evidence": bounded_view(route.get("evidence"), 500),
                       "model_files": bounded_view(route.get("model_files"), 200)})
    result = {"id": draft["id"], "status": draft["status"], "error": draft["error"][:500], "repairable": repairable,
            "missing_fields": sorted(set(finish_form) - set(candidate)), "routes": routes,
            "candidate": bounded_view(candidate, 1800),
            "instructions": "Draft is not evidence. repair_finish replaces only explicit fields, then revalidates the ENTIRE candidate with current visible evidence and machine facts. Use a new finish for route structure/legacy schema changes. No field is automatically selected."}
    if len(json.dumps(result, ensure_ascii=False).encode()) > 4500:
        result["candidate"] = bounded_view(candidate, 200)
        result["error"] = draft["error"][:120]
        for item, route in zip(routes, candidate["routes"]):
            item["name"] = item["name"][:40]
            item["evidence"] = bounded_view(route.get("evidence"), 128)
            item["model_files"] = bounded_view(route.get("model_files"), 128)
    return result
