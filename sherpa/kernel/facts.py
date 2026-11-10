"""Facts: immutable observations. Never injected into LLM context directly;
read by deterministic code (metabench replay, aggregate queries).

v0 storage: one JSONL per task under sherpa-data/facts/. OQ-1 (retention / hot set) still open.
"""
from __future__ import annotations

from sherpa.util import DATA, append_jsonl, now_iso, read_jsonl

SCHEMA = 1
REQUIRED = ["task", "experiment", "context", "features", "action", "predicted", "observed", "gates", "outcome"]
OUTCOMES = {"accepted", "rejected", "killed", "failed"}


def fact_dir():
    return DATA / "facts"


def write(fact: dict) -> str:
    missing = [k for k in REQUIRED if k not in fact]
    if missing:
        raise ValueError(f"fact missing fields: {missing}")
    if fact["outcome"] not in OUTCOMES:
        raise ValueError(f"bad outcome {fact['outcome']}")
    p = fact_dir() / f"{fact['task']}.jsonl"
    existing = read_jsonl(p)
    fid = f"{fact['task']}/f{len(existing) + 1:03d}"
    rec = {"id": fid, "schema": SCHEMA, "time": now_iso(), **fact}
    append_jsonl(p, rec)
    return fid


def all_facts() -> list[dict]:
    out = []
    for p in sorted(fact_dir().glob("*.jsonl")):
        out.extend(read_jsonl(p))
    return out


def by_task(task: str) -> list[dict]:
    return read_jsonl(fact_dir() / f"{task}.jsonl")


def aggregate(action_sig: str | None = None) -> list[dict]:
    """The only view LLMs get: aggregated stats, never raw facts."""
    from sherpa.kernel import knowledge as K
    groups: dict = {}
    for f in all_facts():
        sig = K.fact_signature(f)

        if action_sig and sig != action_sig:
            continue
        g = groups.setdefault(sig, {"signature": sig, "n": 0, "deltas": [], "outcomes": {}})
        g["n"] += 1
        d = f["observed"].get("delta_pct")
        if d is not None:
            g["deltas"].append(d)
        g["outcomes"][f["outcome"]] = g["outcomes"].get(f["outcome"], 0) + 1
    return list(groups.values())
