"""Paths, IO, hashing and subprocess helpers shared by the judge."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

KERNEL = Path(__file__).resolve().parent.parent          # <repo>/kernel
REPO = KERNEL.parent                                      # <repo>
BENCH = Path(os.environ.get("SHERPA_BENCH", REPO / "bench")).resolve()
DATA = Path(os.environ.get("SHERPA_DATA", REPO.parent / "sherpa-data")).resolve()
JUDGE_DATA = DATA / "judge"                               # judge-owned runtime data (builds, records, artifacts)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read_yaml(p: Path) -> Any:
    with open(p) as f:
        return yaml.safe_load(f)


def write_yaml(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)
    tmp.replace(p)


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
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(p: Path) -> list[Any]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    """Hash of a JSON-serialisable object, independent of key order."""
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None, log: Path | None = None,
        timeout: float | None = None, check: bool = True, base_env: dict | None = None) -> subprocess.CompletedProcess:
    """`base_env` replaces os.environ entirely (used for hermetic builds); `env` is layered on top."""
    full_env = {**(os.environ if base_env is None else base_env), **(env or {})}
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            f.write(f"\n$ {' '.join(map(str, cmd))}\n")
            f.flush()
            r = subprocess.run(cmd, cwd=cwd, env=full_env, stdout=f, stderr=subprocess.STDOUT,
                               timeout=timeout, text=True)
    else:
        r = subprocess.run(cmd, cwd=cwd, env=full_env, capture_output=True, timeout=timeout, text=True)
    if check and r.returncode != 0:
        tail = log.read_text()[-3000:] if log else (r.stdout or "")[-2000:] + (r.stderr or "")[-2000:]
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}\n{tail}")
    return r
