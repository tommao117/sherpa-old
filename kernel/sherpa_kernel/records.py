"""Append-only records of everything the judge measured.

Rules (v0 DES-18, design §3.4):
  - records keep raw observations; derived values (metrics, verdicts, scores) are recomputed on read,
    so a refined definition re-scores history without migration
  - large raw data (per-request timings, oracle trajectories) lives in an artifact file referenced
    by path + sha256; the record itself stays small
  - every record carries judge version, bench version and machine fingerprint
  - records are never shown to an LLM directly; Sherpa reads them through queries
"""
from __future__ import annotations

import gzip
import json
import time
import uuid
from pathlib import Path

from .util import JUDGE_DATA, KERNEL, append_jsonl, now_iso, read_jsonl, sha256_bytes, sha256_file

SCHEMA = 1
KINDS = {"score", "probe", "calibration", "doctor"}   # probe = quick, non-final measurement (no oracle)


def root() -> Path:
    return JUDGE_DATA / "records"


def judge_version() -> str:
    """Content hash of the judge's own code: a record is only comparable to records of the same judge."""
    h = []
    for p in sorted((KERNEL / "sherpa_kernel").rglob("*.py")):
        h.append(f"{p.relative_to(KERNEL)}:{sha256_file(p)}")
    return sha256_bytes("\n".join(h).encode())[:16]


def new_id(kind: str) -> str:
    return f"{kind[0]}{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def save_artifact(rid: str, name: str, obj) -> dict:
    p = root() / "artifacts" / rid / f"{name}.json.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, separators=(",", ":")).encode()
    with gzip.open(p, "wb") as f:
        f.write(data)
    return {"path": str(p.relative_to(root())), "sha256": sha256_bytes(data)}


def load_artifact(ref: dict):
    p = root() / ref["path"]
    data = gzip.open(p).read()
    if sha256_bytes(data) != ref["sha256"]:
        raise ValueError(f"artifact {p} was modified (sha256 mismatch)")
    return json.loads(data)


def write(kind: str, rec: dict, rid: str | None = None) -> str:
    if kind not in KINDS:
        raise ValueError(f"unknown record kind {kind}")
    rid = rid or new_id(kind)
    append_jsonl(root() / f"{kind}.jsonl", {"id": rid, "schema": SCHEMA, "kind": kind, "time": now_iso(),
                                            "judge": judge_version(), **rec})
    return rid


def read(kind: str) -> list[dict]:
    return read_jsonl(root() / f"{kind}.jsonl")


def get(rid: str) -> dict:
    for kind in KINDS:
        for r in read(kind):
            if r["id"] == rid:
                return r
    raise KeyError(rid)
