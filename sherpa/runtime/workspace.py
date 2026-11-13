from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path

from .store import atomic_json, digest

_INTERRUPTION = ContextVar("git_interruption", default=lambda: None)


@contextmanager
def interruptible(check, scratch=None):
    token = _INTERRUPTION.set(check)
    scratch_token = _SCRATCH.set(scratch)
    try:
        yield
    finally:
        _INTERRUPTION.reset(token)
        _SCRATCH.reset(scratch_token)


_SCRATCH = ContextVar("git_scratch", default=None)

EXCLUDED = {".git", ".codebuddy", ".venv", ".venv-jev", "venv", "__pycache__", ".pytest_cache",
            "node_modules", "build", "dist", "logs", "models", "weights", ".ssh"}
BINARY_SUFFIXES = {".gguf", ".safetensors", ".bin", ".pt", ".pth", ".so", ".o", ".pyc"}
MAX_FILE = 2 * 1024 * 1024
MAX_TOTAL = 16 * 1024 * 1024
GIT_RESOURCE_SETTINGS = {"pack.threads": "1", "index.threads": "1", "pack.windowMemory": "32m",
                         "checkout.workers": "1"}


def git(path: Path, *args, check=True, timeout_s=30, log_directory=None, file_limit_mib=32, cpu_seconds=15) -> str:
    from .calls import Calls
    inherited = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env = {**inherited, "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_CONFIG_GLOBAL": os.devnull, "GIT_AUTHOR_NAME": "Sherpa", "GIT_AUTHOR_EMAIL": "sherpa@localhost",
           "GIT_COMMITTER_NAME": "Sherpa", "GIT_COMMITTER_EMAIL": "sherpa@localhost",
           "GIT_OPTIONAL_LOCKS": "0", "GIT_LFS_SKIP_SMUDGE": "1",
           "GIT_ALLOW_PROTOCOL": "https:file",
           "SHERPA_GIT_FILE_LIMIT_MIB": str(file_limit_mib), "SHERPA_GIT_CPU_SECONDS": str(cpu_seconds)}
    env["GIT_CONFIG_COUNT"] = str(len(GIT_RESOURCE_SETTINGS))
    for index, (key, value) in enumerate(GIT_RESOURCE_SETTINGS.items()):
        env[f"GIT_CONFIG_KEY_{index}"] = key
        env[f"GIT_CONFIG_VALUE_{index}"] = value
    command = [sys.executable, "-I", str(Path(__file__).with_name("git_worker.py")), str(os.getpid()), "git", *args]
    _INTERRUPTION.get()()
    if log_directory:
        log_directory.mkdir(parents=True, exist_ok=False)
        atomic_json(log_directory / "request.json", {"cwd": str(path), "args": list(args), "timeout_s": timeout_s,
                                                    "resource_settings": GIT_RESOURCE_SETTINGS,
                                                    "memory_limit_mib": 512, "file_limit_mib": file_limit_mib,
                                                    "cpu_seconds": cpu_seconds})
    with ((log_directory / "stdout.log").open("w+b") if log_directory else tempfile.TemporaryFile(dir=_SCRATCH.get())) as stdout, \
         ((log_directory / "stderr.log").open("w+b") if log_directory else tempfile.TemporaryFile(dir=_SCRATCH.get())) as stderr:
        process = subprocess.Popen(command, cwd=path, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            deadline = time.monotonic() + timeout_s
            while process.poll() is None:
                _INTERRUPTION.get()()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Git operation exceeded {timeout_s}-second limit")
                time.sleep(0.01)
            stdout.seek(0)
            stderr.seek(0)
            out = stdout.read(MAX_TOTAL + 1)
            error = stderr.read(8192).decode(errors="replace")
            if len(out) > MAX_TOTAL:
                raise ValueError("Git output exceeds prototype limit")
            if check and process.returncode:
                raise RuntimeError(f"git {args[0]} failed: {error[-800:]}")
            return out.decode().strip() if process.returncode == 0 else ""
        finally:
            cleaned = Calls.cleanup(process)
            if log_directory:
                atomic_json(log_directory / "result.json", {"returncode": process.returncode, "cleanup_complete": cleaned})
            if not cleaned:
                from .calls import RunStopped
                raise RunStopped("cleanup_failed", "Git process group failed to stop")


def source_files(root: Path) -> list[Path]:
    files, total = [], 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not d.startswith(".venv")
                         and not d.endswith(".egg-info") and not (Path(directory) / d).is_symlink())
        for name in sorted(names):
            path = Path(directory) / name
            if (path.is_symlink() or path.suffix in BINARY_SUFFIXES or name.startswith(".env")
                    or name in {"credentials.json", ".credentials.json", "id_rsa", "id_ed25519"}
                    or path.suffix in {".pem", ".key"}):
                continue
            size = path.stat().st_size
            if size > MAX_FILE:
                raise ValueError(f"source file exceeds snapshot limit; separate large artifacts first: {path}")
            total += size
            if total > MAX_TOTAL or len(files) >= 2000:
                raise ValueError("source snapshot exceeds prototype limit (16 MiB / 2000 files)")
            files.append(path)
    return files


def snapshot(root: Path, destination: Path) -> dict:
    files = source_files(root)
    inventory = []
    for source in files:
        relative = source.relative_to(root)
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output)
        inventory.append({"path": relative.as_posix(), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                          "executable": bool(source.stat().st_mode & 0o111)})
    return {"files": inventory, "content_id": digest(inventory)}


def capture_framework(project: Path, destination: Path) -> dict:
    inventory = snapshot(project, destination)
    commit = git(project, "rev-parse", "HEAD", check=False) or None
    dirty = bool(git(project, "status", "--porcelain", "--untracked-files=normal", check=False))
    return {"commit": commit, "dirty": dirty, "snapshot": str(destination), **inventory}


def commit_all(path: Path, message: str) -> str:
    git(path, "add", "--all", "--", ".")
    git(path, "commit", "--allow-empty", "-m", message)
    return git(path, "rev-parse", "HEAD")


def prepare(target: dict, destination: Path) -> dict:
    """Create an independent target repository; never mutate the supplied source."""
    if destination.exists():
        raise FileExistsError(destination)
    location = Path(target.get("path", target.get("url", ""))).expanduser().resolve()
    if not location.is_dir():
        raise ValueError("this lightweight prototype accepts local directories/Git repositories only; remote fetch is deferred")
    if location == destination.resolve() or location in destination.resolve().parents:
        raise ValueError("workspace cannot be nested inside its source")
    if (location / ".git").is_file():
        raise ValueError("linked worktree input requires a future explicit import adapter")
    # Bound repository input before cloning; compressed history can be much larger than source.
    total, count = 0, 0
    if (location / ".git").is_dir() or target["kind"] == "git":
        for directory, dirs, names in os.walk(location, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink()]
            for name in names:
                file = Path(directory) / name
                if file.is_symlink():
                    raise ValueError("Git source containing symlinks is not supported by this prototype")
                total += file.stat().st_size
                count += 1
                if total > 32 * 1024 * 1024 or count > 5000:
                    raise ValueError("Git input exceeds prototype limit (32 MiB / 5000 files)")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if target["kind"] == "git":
        url = str(location)
        git(destination.parent, "clone", "--no-hardlinks", "--no-checkout", "--", url, str(destination))
        revision = target.get("revision", "HEAD")
        commit = git(destination, "rev-parse", "--verify", "--end-of-options", revision + "^{commit}")
        git(destination, "checkout", "--detach", commit)
        result = {"source": target, "source_commit": commit, "commit": commit, "dirty_source": False}
    else:
        root = Path(target["path"]).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        if root == destination.resolve() or root in destination.resolve().parents:
            raise ValueError("workspace cannot be nested inside its source")
        source_commit = git(root, "rev-parse", "HEAD", check=False) if (root / ".git").exists() else None
        inventory = snapshot(root, destination)
        if not inventory["files"]:
            raise ValueError("source contains no eligible code files")
        if source_commit:
            # Clone history separately, then make the captured working tree a commit in our clone.
            staging = destination.with_name(destination.name + "-history")
            git(destination.parent, "clone", "--no-hardlinks", "--no-checkout", "--", str(root), str(staging))
            shutil.move(str(staging / ".git"), str(destination / ".git"))
            staging.rmdir()
        else:
            git(destination, "init")
        git(destination, "add", "--all", "--", ".")
        included = [item["path"] for item in inventory["files"]]
        git(destination, "add", "-f", "--", *included)
        git(destination, "commit", "--allow-empty", "-m", "Snapshot target input for isolated experiment")
        commit = git(destination, "rev-parse", "HEAD")
        tracked = git(destination, "ls-tree", "-rz", "--name-only", commit).rstrip("\0").split("\0")
        if set(tracked) != set(included):
            raise ValueError("snapshot files differ from committed tree")
        result = {"source": target, "source_commit": source_commit, "commit": commit,
                  "dirty_source": bool(source_commit and git(root, "status", "--porcelain")), **inventory}
    result["path"] = str(destination)
    atomic_json(destination.parent / "source.json", result)
    return result
