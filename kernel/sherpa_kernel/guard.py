"""Guard: the judge cannot be modified by the thing it judges.

- check_paths(): a proposed change to Sherpa (list of repo-relative paths) must not touch the judge
- bench manifest: every file under bench/ is hashed into bench/MANIFEST.sha256; the bench version is
  the hash of that manifest, and scoring refuses to run on a bench that does not match it
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

from .util import BENCH, REPO, sha256_bytes, sha256_file

PROTECTED = ("kernel/", "bench/")
MANIFEST = "MANIFEST.sha256"
_SKIP_DIRS = {"__pycache__", ".pytest_cache"}
# What the bench version covers: everything that can change a score. Documentation, reports, the
# human-run tools and retired tasks are still guard-protected but do not change the version.
SCORING = ("tasks/", "deliverables/", "baselines/", "oracle/", "doctor.yaml")


class GuardViolation(PermissionError):
    pass


def check_paths(paths: list[str]) -> None:
    bad = []
    for p in paths:
        norm = str(PurePosixPath(p.replace("\\", "/")))
        if norm.startswith("/") or ".." in PurePosixPath(norm).parts:
            bad.append(p)
            continue
        if any(norm == x.rstrip("/") or norm.startswith(x) for x in PROTECTED):
            bad.append(p)
    if bad:
        raise GuardViolation(f"changes to the judge are human-only: {bad}")


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name != MANIFEST
                  and not (_SKIP_DIRS & set(p.relative_to(root).parts))
                  and p.relative_to(root).as_posix().startswith(SCORING))


def compute_manifest(root: Path = BENCH) -> str:
    return "".join(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in _files(root))


def seal(root: Path = BENCH) -> str:
    """Human action: freeze the current bench content. Returns the bench version."""
    m = compute_manifest(root)
    (root / MANIFEST).write_text(m)
    return bench_version(root)


def bench_version(root: Path = BENCH) -> str:
    p = root / MANIFEST
    return sha256_bytes(p.read_bytes())[:16] if p.exists() else "unsealed"


def verify(root: Path = BENCH) -> list[str]:
    """Differences between bench content and its manifest (empty list = intact)."""
    p = root / MANIFEST
    if not p.exists():
        return ["bench is not sealed (no MANIFEST.sha256)"]
    want = dict(reversed(line.split("  ", 1)) for line in p.read_text().splitlines() if line.strip())
    have = dict(reversed(line.split("  ", 1)) for line in compute_manifest(root).splitlines() if line.strip())
    diffs = [f"modified: {k}" for k in want if k in have and want[k] != have[k]]
    diffs += [f"missing: {k}" for k in want if k not in have]
    diffs += [f"unsealed: {k}" for k in have if k not in want]
    return diffs


def repo_relative(p: Path) -> str:
    return p.resolve().relative_to(REPO).as_posix()
