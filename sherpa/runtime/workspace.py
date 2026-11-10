"""Framework source management: repo cache -> per-task worktree -> branch sherpa/<task>."""
from __future__ import annotations

from pathlib import Path

from sherpa.util import DATA, run


def repo_cache(name: str, url: str) -> Path:
    p = DATA / "repos" / f"{name}.git"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--bare", "--filter=blob:none", url, str(p)], timeout=1800)
    return p


def fetch(cache: Path) -> None:
    run(["git", "-C", str(cache), "fetch", "--tags", "origin", "+refs/heads/*:refs/heads/*"], timeout=1800)


def resolve(cache: Path, ref: str) -> str:
    return run(["git", "-C", str(cache), "rev-parse", f"{ref}^{{commit}}"]).stdout.strip()


def create_worktree(cache: Path, ref: str, dest: Path, branch: str) -> dict:
    commit = resolve(cache, ref)
    if not dest.exists():
        run(["git", "-C", str(cache), "worktree", "add", "-B", branch, str(dest), commit])
    return {"ref": ref, "base_commit": commit, "branch": branch, "path": str(dest)}


def head(wt: Path) -> str:
    return run(["git", "-C", str(wt), "rev-parse", "HEAD"]).stdout.strip()


def dirty(wt: Path) -> bool:
    return bool(run(["git", "-C", str(wt), "status", "--porcelain"]).stdout.strip())


def commit_all(wt: Path, message: str) -> str | None:
    if not dirty(wt):
        return None
    run(["git", "-C", str(wt), "add", "-A"])
    run(["git", "-C", str(wt), "-c", "user.name=sherpa", "-c", "user.email=sherpa@localhost",
         "commit", "-q", "-m", message])
    return head(wt)


def reset_to(wt: Path, commit: str) -> None:
    run(["git", "-C", str(wt), "reset", "-q", "--hard", commit])
    run(["git", "-C", str(wt), "clean", "-qfd", "-e", "build/"])


def diff(wt: Path, base: str) -> str:
    return run(["git", "-C", str(wt), "diff", base, "HEAD"]).stdout
