"""Import explicitly supplied framework code into a private, commit-pinned repository."""
from __future__ import annotations

import os
import re
import shutil
import stat
import time
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from . import workspace
from .calls import RunStopped
from .store import atomic_json

VERSION = "framework-import-v2"
DEFAULT_MAX_MIB = 512
MAX_ENTRIES = 20000


def target(value, revision=None):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("framework Git HTTPS URL or local directory is required")
    if revision is not None and (not isinstance(revision, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,200}", revision)
                                 or ".." in revision or "@{" in revision):
        raise ValueError("revision must be HEAD, a branch/tag/ref or a full commit ID")
    if value.startswith("https://"):
        parsed = urlsplit(value)
        if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.path.strip("/"):
            raise ValueError("use a credential-free HTTPS repository URL")
        return {"kind": "git", "url": value, "revision": revision or "HEAD"}
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError("framework must be an existing directory or an HTTPS Git URL; other protocols are not supported")
    if revision:
        if not (path / ".git").exists():
            raise ValueError("revision requires a Git repository")
        return {"kind": "git", "url": str(path), "revision": revision}
    return {"kind": "local", "path": str(path)}


def check_tree(path, maximum, check):
    total, count = 0, 0
    if not path.exists():
        return
    for directory, dirs, names in os.walk(path, followlinks=False):
        check()
        count += len(dirs) + len(names)
        if count > MAX_ENTRIES * 4:
            raise ValueError("framework workspace exceeds entry limit")
        dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
        for name in names:
            try:
                info = (Path(directory) / name).lstat()
            except FileNotFoundError:
                continue
            total += info.st_size
            if total > maximum:
                raise ValueError("framework workspace exceeded configured disk budget")


def copy_pinned_objects(source, destination, maximum, check):
    source_git, target_git = source / ".git", destination / ".git"
    objects = source_git / "objects"
    if source_git.is_symlink() or objects.is_symlink() or (objects / "info").is_symlink():
        raise ValueError("pinned object storage must not be symlinked")
    if (objects / "info/alternates").exists():
        raise ValueError("pinned repository depends on external object storage")
    total, count = 0, 0
    def copy_file(path):
        nonlocal total, count
        check()
        count += 1
        if count > MAX_ENTRIES * 4:
            raise ValueError("pinned object file limit exceeded")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("pinned object must be a regular file")
            target = target_git / path.relative_to(source_git)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                while True:
                    check()
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > maximum:
                        raise ValueError("pinned objects exceeded disk budget")
                    output.write(chunk)
    for directory, dirs, names in os.walk(objects, followlinks=False):
        check()
        if any((Path(directory) / name).is_symlink() for name in dirs):
            raise ValueError("pinned object directory must not be symlinked")
        if Path(directory) == objects:
            dirs[:] = [name for name in dirs if name != "info"]
        for name in names:
            if name.endswith((".lock", ".promisor")):
                raise ValueError("pinned objects are incomplete or being modified")
            copy_file(Path(directory) / name)
    shallow = source_git / "shallow"
    if shallow.exists() or shallow.is_symlink():
        copy_file(shallow)


def import_code(source, destination, logs, check, *, max_mib=DEFAULT_MAX_MIB, timeout_s=120, pinned=None, progress=None):
    if isinstance(max_mib, bool) or not isinstance(max_mib, int) or not 1 <= max_mib <= 8192:
        raise ValueError("framework_max_mib must be in [1, 8192]")
    if destination.exists():
        raise FileExistsError(destination)
    if source["kind"] == "local":
        root = Path(source["path"]).resolve()
        if root == destination.resolve() or root in destination.resolve().parents:
            raise ValueError("framework workspace cannot be inside input directory")
    maximum = max_mib * 1024 * 1024
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < maximum + 256 * 1024 * 1024:
        raise ValueError("insufficient disk space for framework workspace and reserve")
    count, next_disk_check = 0, 0
    def guard():
        nonlocal next_disk_check
        check()
        if time.monotonic() >= next_disk_check:
            check_tree(destination, maximum, check)
            next_disk_check = time.monotonic() + 0.5
    def git(path, *args, check_result=True):
        nonlocal count
        count += 1
        if progress:
            progress(args[0])
        with workspace.interruptible(guard, scratch=logs):
            return workspace.git(path, *args, check=check_result, timeout_s=timeout_s,
                                 log_directory=logs / f"git-{count:03d}", file_limit_mib=max_mib, cpu_seconds=60)
    logs.mkdir(parents=True, exist_ok=True)
    origin_commit, excluded = None, []
    if pinned:
        repository, revision = str(pinned["path"]), pinned["commit"]
        verify_clean(Path(repository), revision, check)
    elif source["kind"] == "git":
        repository, revision = source["url"], source.get("revision", "HEAD")
    else:
        repository, revision = None, None
    empty_template = logs / "empty-git-template"
    empty_template.mkdir()
    destination.mkdir()
    git(destination, "init", f"--template={empty_template}")
    if pinned:
        check_tree(Path(repository), maximum, check)
        if progress:
            progress("copy_pinned_objects")
        copy_pinned_objects(Path(repository), destination, maximum, guard)
    if repository is not None:
        if pinned:
            commit = git(destination, "rev-parse", "--verify", "--end-of-options", revision + "^{commit}")
        else:
            git(destination, "fetch", "--depth=1", "--no-tags", "--no-recurse-submodules", "--", repository, revision)
            commit = git(destination, "rev-parse", "--verify", "FETCH_HEAD^{commit}")
        if pinned and commit != pinned["commit"]:
            raise ValueError("fetched framework differs from pinned commit")
        listing = git(destination, "ls-tree", "-rlz", "--full-tree", commit)
        total, entries, submodules, symlinks = 0, 0, [], []
        for row in listing.rstrip("\0").split("\0"):
            if not row:
                continue
            header, name = row.split("\t", 1)
            mode, kind, oid, size = header.split()
            parts = PurePosixPath(name)
            if parts.is_absolute() or ".." in parts.parts:
                raise ValueError("unsafe framework tree path")
            entries += 1
            if mode == "160000":
                submodules.append({"path": name, "commit": oid})
                continue
            if mode == "120000":
                symlinks.append(name)
            total += int(size)
            if total > maximum or entries > MAX_ENTRIES:
                raise ValueError("framework checkout exceeds size/entry limits")
        git(destination, "checkout", "--detach", commit)
        result = {"commit": commit, "source_commit": commit, "submodules_not_fetched": submodules,
                  "symlinks_not_read_by_tools": symlinks,
                  "import_mode": "pinned_local_copy" if pinned else "pinned_shallow_fetch"}
    else:
        root = Path(source["path"])
        if root == destination.resolve() or root in destination.resolve().parents:
            raise ValueError("framework workspace cannot be inside input directory")
        if (root / ".git").exists():
            origin_commit = git(root, "rev-parse", "HEAD", check_result=False) or None
        inventory, total = [], 0
        for directory, dirs, names in os.walk(root, followlinks=False):
            check()
            dirs[:] = sorted(d for d in dirs if d not in workspace.EXCLUDED and not d.startswith(".")
                             and not (Path(directory) / d).is_symlink() and not d.endswith(".egg-info"))
            for name in sorted(names):
                check()
                path = Path(directory) / name
                relative = path.relative_to(root)
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or name in workspace.EXCLUDED or path.suffix in workspace.BINARY_SUFFIXES or name.startswith(".env") or name in {
                    "credentials.json", "id_rsa", "id_ed25519"} or path.suffix in {".pem", ".key"}:
                    excluded.append(relative.as_posix())
                    continue
                total += info.st_size
                if total > maximum // 2 or len(inventory) >= MAX_ENTRIES:
                    raise ValueError("local source snapshot exceeds configured size/entry budget")
                output = destination / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, output, follow_symlinks=False)
                inventory.append(relative.as_posix())
        if not inventory:
            raise ValueError("no eligible framework files")
        # Explicit inventory preserves files even if the user's .gitignore excludes source files.
        for start in range(0, len(inventory), 100):
            git(destination, "add", "-f", "--", *inventory[start:start + 100])
        git(destination, "commit", "-m", "Snapshot user-supplied framework for isolated investigation")
        commit = git(destination, "rev-parse", "HEAD")
        result = {"commit": commit, "source_commit": origin_commit, "import_mode": "working_tree_snapshot",
                  "excluded_files": excluded, "submodules_not_fetched": []}
    check_tree(destination, maximum, check)
    verify_clean(destination, commit, check)
    result.update(source=source, path=str(destination), framework_max_mib=max_mib,
                  import_version=VERSION, model_weights_copied=False)
    atomic_json(destination.parent / "framework.json", result)
    return result


def verify_clean(path, commit, check):
    with workspace.interruptible(check, scratch=path.parent):
        if workspace.git(path, "rev-parse", "HEAD") != commit or workspace.git(path, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError("framework workspace changed from pinned commit")
        if workspace.git(path, "ls-files", "--others", "--ignored", "--exclude-standard"):
            raise ValueError("unversioned ignored files present in framework workspace")


def prepare(run, calls, *, max_mib=DEFAULT_MAX_MIB, pinned=None):
    started = time.monotonic()
    record, directory = calls.new("tool", "framework.prepare", {"target": run.state()["task"]["target"],
                                    "max_mib": max_mib, "pinned_commit": pinned["commit"] if pinned else None})
    record.update(synthetic=False, tool_version=VERSION)
    run.save("calls", record["id"], record, "framework_prepare_started")
    try:
        result = import_code(run.state()["task"]["target"], run.path / "workspace" / "framework", directory,
                             calls.check_interrupt, max_mib=max_mib, pinned=pinned,
                             progress=lambda phase: run.event("framework_prepare_progress", call_id=record["id"], phase=phase))
        calls.check_interrupt()
        calls.finish(record, directory, "succeeded", result, started)
        run.update("framework_pinned", framework=result, input_mode="framework_model_v1")
        return result
    except BaseException as error:
        status = error.status if isinstance(error, RunStopped) else "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        calls.finish(record, directory, status, {"error": str(error)}, started)
        raise
