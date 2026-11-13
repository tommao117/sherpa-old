"""Bounded declaration inspection, not a dependency resolver or script interpreter."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tomllib
from pathlib import Path

VERSION = "declarations-v1"
SKIP = {".git", ".codebuddy", "__pycache__", "node_modules", "build", "dist", "artifacts",
        "models", "weights", "logs", "venv"}
MAX_BYTES = 128 * 1024
MAX_TOTAL = 1024 * 1024
MAX_ENTRIES = 512
MAX_FILES = 64
PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9_.+!-]*)")


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def read_bounded(path: Path) -> bytes:
    anchor = os.open(path, os.O_PATH | os.O_NOFOLLOW)
    try:
        info = os.fstat(anchor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("not a regular file")
        fd = os.open(f"/proc/self/fd/{anchor}", os.O_RDONLY)
    finally:
        os.close(anchor)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if info.st_size > MAX_BYTES:
            raise ValueError("declaration exceeds 128 KiB")
        data = handle.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError("declaration exceeds 128 KiB")
        return data


def file_kind(name):
    lower = name.lower()
    if lower == "environment.json":
        return "snapshot"
    if lower.startswith("requirements") and lower.endswith(".txt"):
        return "requirements"
    if lower == "pyproject.toml":
        return "pyproject"
    if lower.startswith("setup") and lower.endswith(".sh"):
        return "script"
    if lower.startswith("readme") or lower == "optimization.md":
        return "documentation"
    if lower in {"uv.lock", "poetry.lock", "environment.yml", "environment.yaml", "dockerfile"}:
        return "unparsed"
    return None


def discover(root: Path, check=lambda: None):
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    found, warnings = [], []
    queue, entries, total = [(root, 0)], 0, 0
    while queue:
        directory, depth = queue.pop(0)
        check()
        with os.scandir(directory) as listing:
            for entry in listing:
                check()
                entries += 1
                if entries > MAX_ENTRIES:
                    raise ValueError("declaration discovery exceeds 512 entries; narrow the source directory")
                if entry.is_symlink():
                    if file_kind(entry.name):
                        warnings.append(f"skipped symlink: {Path(entry.path).relative_to(root)}")
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if depth < 2 and entry.name not in SKIP and not entry.name.startswith("."):
                        queue.append((Path(entry.path), depth + 1))
                    continue
                kind = file_kind(entry.name)
                if not kind:
                    continue
                relative = Path(entry.path).relative_to(root).as_posix()
                try:
                    data = read_bounded(Path(entry.path))
                    total += len(data)
                    if total > MAX_TOTAL or len(found) >= MAX_FILES:
                        raise OverflowError("declaration discovery exceeds 1 MiB / 64 files")
                    found.append({"path": relative, "kind": kind, "sha256": hashlib.sha256(data).hexdigest(),
                                  "bytes": len(data), "text": data.decode("utf-8")})
                except (OSError, ValueError, UnicodeError) as error:
                    warnings.append(f"cannot read {relative}: {error}")
    return sorted(found, key=lambda item: item["path"]), warnings


def requirement(value):
    match = PIN.fullmatch(value.strip())
    return (normalize(match[1]), match[2]) if match else None


def parse(source):
    result = {"path": source["path"], "kind": source["kind"], "python": [], "packages": [],
              "includes": [], "unresolved": [], "index": None}
    kind, text = source["kind"], source["text"]
    def package(value, line, index=None):
        pin = requirement(value)
        if pin:
            result["packages"].append({"name": pin[0], "version": pin[1], "line": line, "index": index})
        elif value.strip():
            result["unresolved"].append({"line": line, "declaration": value.strip(),
                                         "reason": "not a supported exact pin; retain for dependency resolution"})
    try:
        if kind == "snapshot":
            data = json.loads(text)
            if not isinstance(data, dict) or not isinstance(data.get("packages", {}), dict):
                raise ValueError("unrecognized environment snapshot schema")
            python = re.match(r"^(\d+\.\d+(?:\.\d+)?)\b", str(data.get("python", "")))
            if python:
                result["python"].append({"value": python[1], "role": "observed", "line": 1})
            for name, version in data.get("packages", {}).items():
                package(f"{name}=={version}", 1)
        elif kind == "requirements":
            for number, line in enumerate(text.splitlines(), 1):
                line = re.split(r"\s+#", line, maxsplit=1)[0].strip()
                if not line or line.startswith("#"):
                    continue
                include = re.fullmatch(r"(?:-r|--requirement)\s+(\S+)", line)
                index = re.fullmatch(r"--index-url\s+(https://[^\s]+)", line)
                if include:
                    result["includes"].append(include[1])
                elif index:
                    result["index"] = index[1]
                else:
                    package(line, number)
        elif kind == "pyproject":
            data = tomllib.loads(text)
            project = data.get("project", {})
            if project.get("requires-python"):
                result["python"].append({"value": project["requires-python"], "role": "constraint", "line": 1})
            for value in project.get("dependencies", []):
                package(value, 1)
            if project.get("dynamic"):
                result["unresolved"].append({"reason": "dynamic project metadata is not executed"})
        elif kind == "script":
            python = re.search(r"PYTHON:-python(\d+\.\d+(?:\.\d+)?)", text)
            if python:
                result["python"].append({"value": python[1], "role": "requested", "line": 1})
            for number, line in enumerate(text.splitlines(), 1):
                if not re.search(r"\bpip\s+install\b", line) or line.lstrip().startswith("#"):
                    continue
                index_match = re.search(r"--index-url\s+['\"]?(https://[^\s'\"]+)", line)
                index = index_match[1] if index_match else None
                for match in PIN.finditer(line):
                    package(match[0], number, index)
                for match in re.finditer(r"(?:-r|--requirement)\s+[\"']?([^\s\"']+)", line):
                    result["includes"].append(match[1])
                if not PIN.search(line) and not re.search(r"(?:-r|--requirement)\s", line):
                    if not re.search(r"--upgrade\s+pip\s*$", line):
                        result["unresolved"].append({"line": number, "reason": "install expression not understood"})
            if re.search(r"^\s*(if\b|for\b|case\b)|<<", text, re.MULTILINE):
                result["unresolved"].append({"reason": "conditional/dynamic script; extracted values are candidates only"})
        elif kind in {"unparsed", "documentation"}:
            result["unresolved"].append({"reason": "requires supported parser or document-reading decision"})
    except (ValueError, TypeError, AttributeError) as error:
        result["unresolved"].append({"reason": f"cannot parse declaration: {error}"})
    return result


def inspect(root: Path, profile=None, check=lambda: None):
    sources, warnings = discover(root, check)
    parsed = {s["path"]: parse(s) for s in sources}
    scripts = {Path(s["path"]).stem.removeprefix("setup_"): s["path"] for s in sources
               if s["kind"] == "script" and len(Path(s["path"]).parts) == 1}
    profiles = scripts or {"default": None}
    selected = profile or (next(iter(profiles)) if len(profiles) == 1 else None)
    issues = []
    if selected is not None and selected not in profiles:
        raise ValueError(f"unknown profile {selected!r}; choose from {sorted(profiles)}")
    if selected is None:
        issues.append("multiple installation profiles found; select one explicitly, do not merge them")
    included = set()
    if selected:
        if profiles[selected]:
            pending = [profiles[selected]]
        else:
            pending = [s["path"] for s in sources if s["kind"] in {"requirements", "pyproject", "snapshot"}
                       and len(Path(s["path"]).parts) == 1]
        # Profile-local snapshots supplement, but never silently override install declarations.
        pending += [s["path"] for s in sources if s["kind"] == "snapshot"
                    and Path(s["path"]).parent.as_posix() == selected]
        while pending:
            check()
            relative = pending.pop()
            if relative in included:
                continue
            if relative not in parsed:
                issues.append(f"referenced declaration not discovered: {relative}")
                continue
            included.add(relative)
            item = parsed[relative]
            for include in item["includes"]:
                prefix = "$ROOT_DIR/"
                if include.startswith(prefix):
                    reference = Path(include[len(prefix):])
                else:
                    reference = Path(relative).parent / include
                if "$" in str(reference) or ".." in reference.parts or reference.is_absolute():
                    issues.append(f"unsafe or dynamic include is not followed: {relative}")
                else:
                    pending.append(reference.as_posix())
    declarations = []
    python = []
    unresolved = [{"source": s["path"], "reason": "snapshot is outside selected profile; not merged"}
                  for s in sources if s["kind"] == "snapshot" and s["path"] not in included]
    for relative in sorted(included):
        item = parsed[relative]
        python += [{**p, "source": relative} for p in item["python"]]
        declarations += [{**p, "source": relative, "kind": item["kind"], "index": p["index"] or item["index"]}
                         for p in item["packages"]]
        unresolved += [{**u, "source": relative} for u in item["unresolved"]]
    packages = []
    for name in sorted({p["name"] for p in declarations}):
        evidence = [p for p in declarations if p["name"] == name]
        pins = sorted({p["version"] for p in evidence})
        indexes = sorted({p["index"] for p in evidence if p["index"]})
        public = {pin.split("+", 1)[0] for pin in pins}
        local = {pin for pin in pins if "+" in pin}
        conflict = len(public) > 1 or len(local) > 1 or len(indexes) > 1
        if conflict:
            issues.append(f"conflicting declarations for {name}; no version selected")
        packages.append({"name": name, "versions": pins, "indexes": indexes, "conflict": conflict,
                         "evidence": evidence, "only_snapshot": all(p["kind"] == "snapshot" for p in evidence)})
    majors = {tuple(p["value"].split(".")[:2]) for p in python if p["role"] != "constraint"}
    if len(majors) > 1:
        issues.append("Python interpreter declarations conflict")
    if not python:
        issues.append("Python requirement not established")
    if not packages:
        issues.append("no supported dependency pins found")
    docs = [s["path"] for s in sources if s["kind"] == "documentation"]
    if docs:
        unresolved.append({"reason": "prose may add prerequisites; no LLM has read these documents", "sources": docs})
    for s in sources:
        if s["kind"] == "unparsed":
            unresolved.append({"source": s["path"], "reason": "lock/system declaration not interpreted"})
    if unresolved:
        issues.append("uninterpreted declarations remain; review required before installation")
    if warnings:
        issues.append("some declarations were skipped or unreadable")
    return {"inspector_version": VERSION, "source": str(root.resolve()), "profiles": profiles,
            "selected_profile": selected, "included_sources": sorted(included), "python": python,
            "packages": packages, "sources": sources, "warnings": warnings, "unresolved": unresolved,
            "issues": issues, "source_fingerprint": hashlib.sha256(json.dumps(
                [(s["path"], s["sha256"]) for s in sources], sort_keys=True).encode()).hexdigest()}
