"""Small shared helpers: paths, yaml/json io, hashing, subprocess."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "assets"
DATA = Path(os.environ.get("SHERPA_DATA", REPO.parent / "sherpa-data")).resolve()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read_yaml(p: Path) -> Any:
    with open(p) as f:
        return yaml.safe_load(f)


def write_yaml(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def read_json(p: Path) -> Any:
    with open(p) as f:
        return json.load(f)


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(p)


def append_jsonl(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(p: Path) -> list[Any]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def tree_hash(root: Path) -> str:
    """Content hash of a directory (used to verify assets are frozen)."""
    h = hashlib.sha256()
    for f in sorted(root.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            h.update(str(f.relative_to(root)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def run(cmd: list[str] | str, cwd: Path | None = None, env: dict | None = None,
        log: Path | None = None, timeout: float | None = None, check: bool = True) -> subprocess.CompletedProcess:
    full_env = {**os.environ, **(env or {})}
    shell = isinstance(cmd, str)
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            f.write(f"\n$ {cmd if shell else ' '.join(map(str, cmd))}\n")
            f.flush()
            r = subprocess.run(cmd, cwd=cwd, env=full_env, stdout=f, stderr=subprocess.STDOUT,
                               timeout=timeout, shell=shell, text=True)
    else:
        r = subprocess.run(cmd, cwd=cwd, env=full_env, capture_output=True, timeout=timeout,
                           shell=shell, text=True)
    if check and r.returncode != 0:
        tail = ""
        if log and log.exists():
            tail = "\n".join(log.read_text().splitlines()[-30:])
        else:
            tail = (r.stdout or "")[-2000:] + (r.stderr or "")[-2000:]
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}\n{tail}")
    return r
