"""Deterministic decision views and report assembly; no inferred compatibility facts."""
from __future__ import annotations

import copy
import json

from ..platform.exploration import bounded_view
from .evidence_windows import CONTENT_TOOLS, window_views

VERSION = "decision-support-v3"
MAX_CONTEXT_BYTES = 24000


def size(value):
    return len(json.dumps(value, ensure_ascii=False).encode())


def ranges(numbers):
    result = []
    for number in sorted(set(numbers)):
        if result and number == result[-1][1] + 1:
            result[-1][1] = number
        else:
            result.append([number, number])
    return result


def read_view(entry, budget, focused=False):
    data = entry["view"].get("data", {})
    lines = []
    for line in data.get("lines", []):
        if size(lines + [line]) > budget:
            break
        lines.append(line)
    return {"id": entry["id"], "tool": entry["tool"], "parameters": entry["parameters"],
            "view": {"data": {"path": data.get("path"), "lines": lines},
                     "truncated": len(lines) < len(data.get("lines", []))},
            "displayed_ranges": ranges(line["line"] for line in lines),
            "available_ranges": ranges(line["line"] for line in data.get("lines", [])),
            "more": (("focused view is still partial; use read at the first omitted line, do not repeatedly recall this ID"
                      if focused else "recall this ID for a larger view; the file was already read, do not reread the same range")
                     if len(lines) < len(data.get("lines", [])) else None),
            "first_omitted_line": (data["lines"][len(lines)]["line"] if len(lines) < len(data.get("lines", [])) else None),
            "navigation": {key: data[key] for key in ("matched_lines", "skipped_lines", "scope") if key in data},
            "next_file_line": data.get("next_line")}


def task_facts(investigation):
    entries = list(investigation.evidence.values())
    latest_machine = next((e for e in reversed(entries) if e["tool"] == "machine"
                           and e.get("origin_run", investigation.run.id) == investigation.run.id), None)
    facts = {"inputs": investigation.tools.binding, "requested_device": investigation.device,
             "machine": None, "model_inventory": [], "inventory_pages": [],
             "scope": "observations only; no runtime or model compatibility has been tested"}
    if latest_machine and not latest_machine["view"].get("truncated"):
        facts["machine"] = {"evidence_id": latest_machine["id"], "observation": latest_machine["view"]["data"],
                            "reserved": False}
    facts["machine_observation_issue"] = ("machine result was truncated; a complete observation is required"
                                           if latest_machine and latest_machine["view"].get("truncated") else None)
    # Prefer the latest version of each observed listing page. Do not claim directory-wide completeness.
    pages = {}
    for entry in entries:
        if entry["tool"] == "list" and entry["parameters"].get("scope") == "model" and not entry["view"].get("truncated"):
            path, offset = entry["parameters"].get("path", "."), entry["parameters"].get("offset", 0)
            if offset == 0:
                pages = {key: value for key, value in pages.items() if key[0] != path}
            pages.pop((path, offset), None)
            pages[(path, offset)] = entry
    for entry in reversed(list(pages.values())):
        data = entry["view"]["data"]
        page = {"evidence_id": entry["id"], "path": data["path"], "offset": entry["parameters"].get("offset", 0),
                "next_offset": data.get("next_offset")}
        facts["inventory_pages"].append(page)
        for item in data.get("entries", []):
            path = item["name"] if data["path"] == "." else data["path"] + "/" + item["name"]
            facts["model_inventory"].append({"path": path, "kind": item["kind"], "bytes": item.get("bytes"),
                                             "evidence_id": entry["id"]})
            if len(facts["model_inventory"]) >= 40:
                facts["inventory_limited"] = True
                return facts
    return facts


def context(investigation, round_number):
    entries = list(investigation.evidence.values())
    facts = task_facts(investigation)
    remaining_calls = max(0, investigation.calls.max_calls - investigation.calls.count)
    result = {"view_version": VERSION, "goal": "inspect the specified framework/model; produce a supported plan for the NEXT validation, not proof of runtime compatibility",
              "inputs": investigation.tools.binding, "requested_device": investigation.device,
              "facts": facts, "tools": investigation.tools.catalog, "round": round_number,
              "remaining": {"calls": remaining_calls, "decision_rounds_including_this": investigation.max_rounds - round_number + 1,
                            "delegations": investigation.max_delegations - investigation.delegations},
              "history": investigation.history[-4:], "expert_advice": investigation.advice[-1:],
              "catalog": [], "catalog_offset": investigation.catalog_offset, "catalog_total": len(entries),
              "observations": [], "omitted_evidence_ids": [], "omitted_evidence_count": 0,
              "completion": {"runtime_verified": False,
                             "machine_refresh_required": facts["machine"] is None,
                             "allowed_outcomes": ["finish: proposed next validation with evidence and explicit unknowns", "stop: blocked or insufficient evidence, with reason"],
                             "budget_note": "a tool normally needs two calls (decision + execution); recall needs a later decision to finish. Never invent evidence to save calls."}}
    import time
    result["remaining"]["seconds"] = max(0, int(investigation.calls.deadline - time.monotonic()))
    page = entries[investigation.catalog_offset:investigation.catalog_offset + 20]
    for entry in page:
        item = {"id": entry["id"], "tool": entry["tool"], "parameters": entry["parameters"]}
        if entry["tool"] in CONTENT_TOOLS and not entry["view"].get("truncated"):
            item["available_ranges"] = ranges(line["line"] for line in entry["view"].get("data", {}).get("lines", []))
        if result["catalog"] and size(result["catalog"] + [item]) > 5000:
            break
        result["catalog"].append(item)
    next_offset = investigation.catalog_offset + len(result["catalog"])
    result["catalog_next_offset"] = next_offset if next_offset < len(entries) else None
    if investigation.continuation:
        old = investigation.continuation
        draft = old.get("draft")
        result["continuation"] = {"source_run": old["source_run"], "budget_semantics": old["budget_semantics"],
                                  "machine_refresh_required": facts["machine"] is None,
                                  "current_machine_evidence_id": facts["machine"]["evidence_id"] if facts["machine"] else None,
                                  "instructions": "Use current facts. Old invalidated machine IDs do not require another refresh after a successful query. Drafts are not evidence; use the current output form.",
                                  "invalidated_evidence_ids": [e["id"] for e in old["discarded_evidence"][:20]],
                                  "draft": ({"call_id": draft["call_id"], "status": draft["status"], "error": draft.get("error"),
                                             "view": bounded_view(draft["view"], 1800)} if draft else None)}
    from .report_drafts import view as draft_view
    report_draft = getattr(investigation, "report_draft", None)
    result["report_draft"] = draft_view(report_draft, investigation.action_forms["finish"]) if report_draft else None
    if report_draft and "continuation" in result:
        result["continuation"]["draft"] = {"id": report_draft["id"], "call_id": report_draft["call_id"],
                                           "status": report_draft["status"], "error": report_draft["error"],
                                           "instructions": "see report_draft for the latest candidate; do not reuse an older snapshot"}
    # Keep fact field types stable and preserve the complete machine observation.
    for key in ("model_inventory", "inventory_pages"):
        while facts[key] and size(facts[key]) > 2200:
            facts[key].pop()
            facts["inventory_limited"] = True
    for key, limit in (("history", 1800), ("expert_advice", 2000)):
        if size(result[key]) > limit:
            result[key] = bounded_view(result[key], limit)
    priority = [investigation.focus]
    for scope in ("model", "framework"):
        anchor = next((e for e in entries if e["tool"] in CONTENT_TOOLS and e["parameters"].get("scope") == scope), None)
        if anchor:
            priority.append(anchor["id"])
    ordered = sorted(reversed(entries), key=lambda e: priority.index(e["id"]) if e["id"] in priority else len(priority))
    visible_lines = {}
    retained = getattr(investigation, "pinned_windows", [])
    pinned_views = window_views(retained, investigation.evidence)
    result["retained_windows"] = retained
    while result["catalog"] and size(result) + size(pinned_views) > MAX_CONTEXT_BYTES - 3000:
        result["catalog"].pop()
        result["catalog_next_offset"] = investigation.catalog_offset + len(result["catalog"])
    if size(result) + size(pinned_views) > MAX_CONTEXT_BYTES - 2000:
        raise ValueError("facts plus retained windows exceed context budget; release some windows")
    result["observations"].extend(pinned_views)
    for view in pinned_views:
        investigation.seen.add(view["id"])
        visible_lines.setdefault(view["id"], {}).update({line["line"]: line["text"] for line in view["view"]["data"]["lines"]})
    from .report_drafts import evidence_views
    draft_views, unavailable = evidence_views(report_draft, investigation.evidence)
    result["draft_evidence_unavailable"] = unavailable
    for view in draft_views:
        if size(result) + size(view) >= MAX_CONTEXT_BYTES - 2000:
            result["draft_evidence_unavailable"].append({"id": view["id"], "reason": "context budget; recall before repair"})
            continue
        result["observations"].append(view)
        investigation.seen.add(view["id"])
        visible_lines.setdefault(view["id"], {}).update({line["line"]: line["text"] for line in view["view"]["data"]["lines"]})
    for entry in ordered:
        if entry["id"] in visible_lines and entry["id"] != investigation.focus:
            continue
        if entry["tool"] == "machine":
            if facts["machine"] and entry["id"] == facts["machine"]["evidence_id"]:
                investigation.seen.add(entry["id"])
            continue
        if entry["tool"] in CONTENT_TOOLS and not entry["view"].get("truncated"):
            focused = entry["id"] == investigation.focus
            budget = min(8200 if focused else 2200, max(0, MAX_CONTEXT_BYTES - 3000 - size(result)))
            view = read_view(entry, budget, focused=focused)
        elif entry["tool"] == "declarations":
            view = {"id": entry["id"], "tool": entry["tool"], "parameters": entry["parameters"],
                    "view": bounded_view(entry["view"].get("data", {}), 1200),
                    "citation_instruction": "read the source text before selecting line citations"}
        elif entry["id"] == investigation.focus:
            view = {"id": entry["id"], "tool": entry["tool"], "parameters": entry["parameters"], "view": entry["view"]}
        else:
            view = None
        if view and size(result) + size(view) < MAX_CONTEXT_BYTES - 2000:
            result["observations"].append(view)
            investigation.seen.add(entry["id"])
            if entry["tool"] in CONTENT_TOOLS and not entry["view"].get("truncated") and view["view"]["data"]["lines"]:
                visible_lines.setdefault(entry["id"], {}).update({line["line"]: line["text"] for line in view["view"]["data"]["lines"]})
        else:
            result["omitted_evidence_count"] += 1
            if len(result["omitted_evidence_ids"]) < 20:
                result["omitted_evidence_ids"].append(entry["id"])
    investigation.visible_lines = visible_lines
    investigation.visible_machine_id = facts["machine"]["evidence_id"] if facts["machine"] else None
    result["completion"]["content_scopes_visible"] = sorted({investigation.evidence[i]["parameters"].get("scope") for i in visible_lines})
    if size(result) > MAX_CONTEXT_BYTES:
        raise ValueError("decision view exceeds context budget")
    return result


def assemble(action, investigation):
    """Copy only model-selected, currently displayed line ranges; never decide the claims."""
    result = copy.deepcopy(action)
    if result.get("action") != "finish":
        return result
    routes = result.get("routes")
    if not isinstance(routes, list) or not 1 <= len(routes) <= 4:
        raise ValueError("provide 1..4 proposed routes, or stop with a reason")
    binding = investigation.framework_ref
    for route in routes:
        if not isinstance(route, dict):
            raise ValueError("route must be an object")
        if {"framework_commit", "framework_source", "runtime"} & set(route):
            raise ValueError("framework identity is filled by runtime; omit framework_commit/framework_source/runtime")
        refs = route.get("evidence")
        if not isinstance(refs, list) or not 1 <= len(refs) <= 8:
            raise ValueError("provide 1..8 displayed content line references")
        compiled = []
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) != {"id", "start", "end"}:
                raise ValueError("citation requires id/start/end; runtime copies the literal quote")
            call_id, start, end = ref["id"], ref["start"], ref["end"]
            if not isinstance(call_id, str) or any(isinstance(n, bool) or not isinstance(n, int) for n in (start, end)):
                raise ValueError("citation id must be text and line numbers integers")
            if not 1 <= start <= end or end - start >= 8:
                raise ValueError("citation must select 1..8 consecutive displayed lines")
            lines = investigation.visible_lines.get(call_id, {})
            if any(n not in lines for n in range(start, end + 1)):
                raise ValueError(f"citation {call_id}: range not currently displayed; recall the evidence or read that fragment")
            quote = "\n".join(lines[n] for n in range(start, end + 1))
            if not 5 <= len(quote) <= 2000:
                raise ValueError("selected text must contain 5..2000 characters; select a narrower meaningful range")
            compiled.append({"id": call_id, "quote": quote})
        source = binding["source"]
        route.update(runtime=source.get("url", source.get("path")), framework_source=source.get("url", source.get("path")),
                     framework_commit=binding["commit"], evidence=compiled)
    return result
