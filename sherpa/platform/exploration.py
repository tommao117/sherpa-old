"""Read-only, bounded evidence tools. Paths are relative to one pinned input directory."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath

from . import environment, machine

CATALOG = {
    "list": {"purpose": "List one directory; names/sizes are not proof of runtime compatibility.",
             "parameters": {"path": "relative directory, default .", "offset": "integer 0..512, default 0"},
             "limits": "40 entries per page, 512 examined; no recursive scan or weight reads"},
    "read": {"purpose": "Read a numbered text fragment from a document/config/source; never execute it.",
             "parameters": {"path": "relative text file", "start": "1-based line, default 1", "lines": "1..80, default 60"},
             "limits": "file <=128 KiB; fragment <=6000 chars; follow next_line for more; links/secrets blocked"},
    "declarations": {"purpose": "Parse ONE chosen dependency declaration. Results are hints, not a solved environment.",
                     "parameters": {"path": "relative declaration file"},
                     "limits": "no includes/scripts executed; unrelated environments are not merged"},
    "search": {"purpose": "Locate a literal keyword in ONE selected text file, returning real line numbers and nearby complete lines.",
               "parameters": {"path": "relative text file", "query": "nonempty literal string (not regex), at most 120 chars",
                              "start": "first line to search, default 1", "around": "context lines each side, 0..4, default 2",
                              "max_matches": "1..8, default 4"},
               "limits": "case-insensitive; bounded UTF-8 excerpts; gaps are not consecutive lines; follow next_line or read a selected section; no match only means this literal is absent in the searched range, NOT that the model/architecture is unsupported"},
    "outline": {"purpose": "Index Markdown ATX headings outside fenced code; heading line numbers locate sections, not proof of their bodies.",
                "parameters": {"path": "relative Markdown file", "start": "first heading line to include, default 1"},
                "limits": "at most 30 headings per page; only actual heading lines are citable; no Setext/HTML heading inference"},
    "machine": {"purpose": "Inspect host, GPU list/occupancy/topology, and installed tools; not reserve GPUs.",
                "parameters": {}, "limits": "read-only system queries; no CUDA initialization; unknown stays unknown"},
}
DENIED = {"node_modules", "venv", "__pycache__", "build", "dist", "logs", "credentials.json",
          "id_rsa", "id_ed25519", "secrets.json", "secrets.toml"}
TEXT_SUFFIXES = {".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".sh", ".py", ".cfg", ".ini", ".cmake",
                 ".c", ".cpp", ".cc", ".h", ".hpp", ".cu", ".cuh", ".rs"}


def integer(value, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"integer must be in [{minimum}, {maximum}]")
    return value


def parts(value):
    if not isinstance(value, str) or len(value) > 512 or "\\" in value or "\x00" in value:
        raise ValueError("invalid relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("path must stay inside the input root")
    for part in path.parts:
        if part.startswith(".") or part.lower() in DENIED or part.endswith((".pem", ".key", ".egg-info")):
            raise ValueError("hidden, sensitive or generated path is excluded")
    return path.parts


def validate(tool, parameters):
    if tool not in CATALOG or not isinstance(parameters, dict):
        raise ValueError("unknown tool or invalid parameters")
    if set(parameters) - set(CATALOG[tool]["parameters"]):
        raise ValueError("unexpected tool parameter")
    if tool != "machine":
        parts(parameters.get("path", "."))
    if tool in {"read", "declarations", "search", "outline"} and not parameters.get("path"):
        raise ValueError("path is required")
    if tool == "list":
        integer(parameters.get("offset", 0), 0, 512)
    if tool in {"read", "search", "outline"}:
        integer(parameters.get("start", 1), 1, 131072)
    if tool == "read":
        integer(parameters.get("lines", 60), 1, 80)
    if tool == "search":
        query = parameters.get("query")
        if not isinstance(query, str) or not query.strip() or len(query) > 120 or "\n" in query or "\x00" in query:
            raise ValueError("query must be a nonempty single-line literal string of at most 120 chars")
        integer(parameters.get("around", 2), 0, 4)
        integer(parameters.get("max_matches", 4), 1, 8)


class EvidenceTools:
    version = "environment-evidence-v1"

    def __init__(self, source: Path, scratch: Path, check, max_text_bytes=environment.MAX_BYTES):
        self.source, self.scratch, self.check = source.resolve(), scratch, check
        self.max_text_bytes = max_text_bytes
        self.root_fd = os.open(self.source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

    def close(self):
        if self.root_fd is not None:
            os.close(self.root_fd)
            self.root_fd = None

    def open_path(self, relative, directory=False):
        components = parts(relative)
        current = os.dup(self.root_fd)
        try:
            for index, component in enumerate(components):
                self.check()
                is_directory = index < len(components) - 1 or directory
                flags = os.O_NOFOLLOW | (os.O_RDONLY | os.O_DIRECTORY if is_directory else os.O_PATH)
                child = os.open(component, flags, dir_fd=current)
                os.close(current)
                current = child
            if directory and not stat.S_ISDIR(os.fstat(current).st_mode):
                raise ValueError("not a directory")
            fd, current = current, None
            return fd
        finally:
            if current is not None:
                os.close(current)

    def text(self, relative):
        name = PurePosixPath(relative).name
        if PurePosixPath(name).suffix.lower() not in TEXT_SUFFIXES and name.lower() not in {"dockerfile", "makefile", "readme", "license"}:
            raise ValueError("file is not an allowed text/document type; weight/binary reads are disabled")
        anchor = self.open_path(relative)
        try:
            info = os.fstat(anchor)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("not a regular file")
            if info.st_size > self.max_text_bytes:
                raise ValueError(f"text file exceeds {self.max_text_bytes // 1024} KiB; choose a smaller source")
            fd = os.open(f"/proc/self/fd/{anchor}", os.O_RDONLY)
            with os.fdopen(fd, "rb") as handle:
                data = handle.read(self.max_text_bytes + 1)
            if len(data) > self.max_text_bytes or b"\x00" in data:
                raise ValueError("oversized or binary content")
            return data.decode("utf-8"), hashlib.sha256(data).hexdigest()
        finally:
            os.close(anchor)

    def list(self, path=".", offset=0):
        fd = self.open_path(path, directory=True)
        entries = []
        try:
            with os.scandir(fd) as scan:
                for i, entry in enumerate(scan):
                    self.check()
                    if i >= 512:
                        raise ValueError("directory exceeds 512 entries; choose a narrower directory")
                    try:
                        parts(entry.name)
                    except ValueError:
                        continue
                    info = entry.stat(follow_symlinks=False)
                    kind = "directory" if stat.S_ISDIR(info.st_mode) else "file" if stat.S_ISREG(info.st_mode) else "blocked"
                    entries.append({"name": entry.name, "kind": kind, "bytes": info.st_size})
        finally:
            os.close(fd)
        entries.sort(key=lambda item: item["name"])
        selected = entries[offset:offset + 40]
        return {"path": path, "entries": selected, "next_offset": offset + 40 if offset + 40 < len(entries) else None,
                "scope": "metadata only; hidden/generated/sensitive names omitted"}

    def read(self, path, start=1, lines=60):
        text, sha = self.text(path)
        source_lines = text.splitlines()
        if start > max(1, len(source_lines)):
            raise ValueError("start line beyond file")
        output, used = [], 0
        for number in range(start, min(len(source_lines) + 1, start + lines)):
            line = source_lines[number - 1]
            if used + len(line) > 6000:
                if not output:
                    raise ValueError("single line exceeds fragment limit; use declarations parser where applicable")
                break
            output.append({"line": number, "text": line})
            used += len(line)
        end = output[-1]["line"] if output else 0
        return {"path": path, "sha256": sha, "lines": output, "total_lines": len(source_lines),
                "next_line": end + 1 if end < len(source_lines) else None,
                "scope": "quoted file contents, not verified runtime behavior"}

    def declarations(self, path):
        text, sha = self.text(path)
        kind = environment.file_kind(PurePosixPath(path).name) or "unparsed"
        parsed = environment.parse({"path": path, "kind": kind, "text": text})
        # No invented line precision: the original parser uses placeholders for structured sources.
        for key in ("packages", "python", "unresolved"):
            for item in parsed[key]:
                item.pop("line", None)
        return {"path": path, "sha256": sha, "parsed": parsed,
                "scope": "single declaration only; includes/conditions not executed, dependencies not resolved"}

    def search(self, path, query, start=1, around=2, max_matches=4):
        text, sha = self.text(path)
        source_lines = text.splitlines()
        matches, selected, next_line = [], set(), None
        needle = query.casefold()
        for number in range(start, len(source_lines) + 1):
            self.check()
            if needle not in source_lines[number - 1].casefold():
                continue
            if len(matches) == max_matches:
                next_line = number
                break
            matches.append(number)
            selected.update(range(max(1, number - around), min(len(source_lines), number + around) + 1))
        output, skipped, used = [], [], 0
        for number in sorted(selected):
            line = {"line": number, "text": source_lines[number - 1]}
            length = len(json.dumps(line, ensure_ascii=False).encode())
            if used + length > 7500:
                skipped.append(number)
                continue
            output.append(line)
            used += length
        return {"path": path, "sha256": sha, "lines": output, "matched_lines": matches,
                "next_line": next_line, "skipped_lines": skipped, "total_lines": len(source_lines),
                "scope": "literal matches and complete source lines; gaps/skipped lines are not displayed; no semantic conclusion"}

    def outline(self, path, start=1):
        if PurePosixPath(path).suffix.lower() != ".md" and PurePosixPath(path).name.lower() != "readme":
            raise ValueError("outline supports Markdown ATX headings only; use search for source code")
        text, sha = self.text(path)
        output, next_line, fence, used = [], None, None, 0
        skipped = []
        for number, text_line in enumerate(text.splitlines(), 1):
            self.check()
            marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", text_line)
            if marker:
                token, suffix = marker.groups()
                if fence is None:
                    fence = token
                elif token[0] == fence[0] and len(token) >= len(fence) and not suffix.strip():
                    fence = None
                continue
            if fence or number < start or not re.match(r"^ {0,3}#{1,6}(?:\s|$)", text_line):
                continue
            if len(output) == 30:
                next_line = number
                break
            line = {"line": number, "text": text_line}
            length = len(json.dumps(line, ensure_ascii=False).encode())
            if length > 7500:
                if len(skipped) < 30:
                    skipped.append(number)
                continue
            if used + length > 7500:
                next_line = number
                break
            output.append(line)
            used += length
        return {"path": path, "sha256": sha, "lines": output, "next_line": next_line,
                "skipped_lines": skipped, "scope": "ATX headings only, not section bodies; read/search at the actual heading line"}

    def execute(self, tool, parameters):
        validate(tool, parameters)
        self.check()
        if tool == "machine":
            result = machine.probe(self.scratch, self.check)
            # Provider sees relevant facts, not duplicated raw command output or local tool paths.
            return {"host": result["host"], "gpus": result["gpus"], "gpu_status": result["gpu_status"],
                    "topology": result["topology"], "tools": {k: bool(v) for k, v in result["tools"].items()},
                    "tool_detection": {"method": "current process PATH lookup only",
                                       "true_means": "executable found; version and functionality not validated",
                                       "false_means": "not found on current PATH; may exist elsewhere; not proof of not installed"},
                    "cuda_runtime_verified": False, "reserved": False}
        return getattr(self, tool)(**parameters)


def bounded_view(result, limit=11000):
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 64:
        raise ValueError("view budget must be an integer of at least 64 bytes")
    encoded = json.dumps(result, ensure_ascii=False)
    full = {"data": result, "truncated": False}
    if len(json.dumps(full, ensure_ascii=False).encode()) <= limit:
        return full
    view = {"preview": "", "truncated": True}
    if limit >= 300:
        view["instruction"] = "result only partly visible; call read/list with narrower scope before making claims"
    low, high = 0, min(len(encoded), limit)
    while low < high:
        middle = (low + high + 1) // 2
        view["preview"] = encoded[:middle]
        if len(json.dumps(view, ensure_ascii=False).encode()) <= limit:
            low = middle
        else:
            high = middle - 1
    view["preview"] = encoded[:low]
    return view
