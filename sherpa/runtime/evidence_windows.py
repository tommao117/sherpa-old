"""Task-local retention of model-selected literal lines, not a persistent knowledge store."""
from __future__ import annotations

import json

CONTENT_TOOLS = frozenset({"read", "search", "outline"})
MAX_WINDOWS = 6
MAX_WINDOW_BYTES = 4500


def continuous_texts(lines):
    blocks, previous = [], None
    for line in lines:
        if previous is None or line["line"] != previous + 1:
            blocks.append(line["text"])
        else:
            blocks[-1] += "\n" + line["text"]
        previous = line["line"]
    return blocks


def select_lines(ref, evidence, visible=None):
    if not isinstance(ref, dict) or set(ref) != {"id", "start", "end"}:
        raise ValueError("line reference requires id/start/end")
    call_id, start, end = ref["id"], ref["start"], ref["end"]
    if not isinstance(call_id, str) or any(isinstance(n, bool) or not isinstance(n, int) for n in (start, end)):
        raise ValueError("line reference requires a text ID and integer line numbers")
    if not 1 <= start <= end or end - start >= 8:
        raise ValueError("select 1..8 consecutive complete lines")
    entry = evidence.get(call_id)
    if not entry or entry["tool"] not in CONTENT_TOOLS or entry["view"].get("truncated"):
        raise ValueError("window requires complete source evidence, not metadata or a truncated preview")
    original = {line["line"]: line["text"] for line in entry["view"].get("data", {}).get("lines", [])}
    if any(n not in original for n in range(start, end + 1)):
        raise ValueError("window range is missing lines in source evidence")
    if visible is not None and any(visible.get(call_id, {}).get(n) != original[n] for n in range(start, end + 1)):
        raise ValueError("window must select currently displayed complete lines; recall/read the range first")
    lines = [{"line": n, "text": original[n]} for n in range(start, end + 1)]
    if not 5 <= len("\n".join(line["text"] for line in lines)) <= 2000:
        raise ValueError("window text must have 5..2000 characters")
    return lines


def window_views(refs, evidence, visible=None):
    if not isinstance(refs, list) or len(refs) > MAX_WINDOWS:
        raise ValueError(f"keep at most {MAX_WINDOWS} windows, or [] to release all")
    views, keys = [], set()
    for ref in refs:
        lines = select_lines(ref, evidence, visible)
        key = (ref["id"], ref["start"], ref["end"])
        if key in keys:
            raise ValueError("duplicate window reference")
        keys.add(key)
        entry = evidence[ref["id"]]
        views.append({"id": entry["id"], "tool": entry["tool"], "parameters": entry["parameters"], "pinned": True,
                      "displayed_ranges": [[ref["start"], ref["end"]]],
                      "view": {"data": {"path": entry["parameters"]["path"], "lines": lines}, "truncated": False}})
    if len(json.dumps(views, ensure_ascii=False).encode()) > MAX_WINDOW_BYTES:
        raise ValueError("retained windows exceed 4500 bytes; select fewer/shorter ranges")
    return views
