"""Invariants (design §5.4). No policy can bypass these; the engine calls them at every transition."""
from __future__ import annotations

from sherpa.util import ASSETS, tree_hash


class InvariantViolation(RuntimeError):
    pass


def assets_fingerprint() -> str:
    return tree_hash(ASSETS)


def check_assets_frozen(frozen: str) -> None:
    now = assets_fingerprint()
    if now != frozen:
        raise InvariantViolation(f"assets changed during task ({frozen} -> {now}); evolution only between tasks")


def check_prediction_declared(exp: dict) -> None:
    p = exp.get("predicted") or {}
    if p.get("delta_pct") is None:
        raise InvariantViolation(f"experiment {exp.get('id')} has no declared prediction")


def check_accept(exp: dict) -> None:
    """Accepting a change requires a trustworthy measurement and all gates passed."""
    gates = exp.get("gates") or {}
    if not exp.get("measurement", {}).get("trustworthy"):
        raise InvariantViolation(f"{exp['id']}: cannot accept without trustworthy measurement")
    for g in ("correctness", "reproducible"):
        if not gates.get(g, {}).get("pass"):
            raise InvariantViolation(f"{exp['id']}: cannot accept, gate {g} not passed")


def check_fact_written(exp: dict) -> None:
    if exp.get("state") in ("concluded",) and not exp.get("fact_id"):
        raise InvariantViolation(f"{exp['id']}: concluded without a fact")


def check_budget(n_experiments: int, max_experiments: int, elapsed_s: float, max_seconds: float) -> None:
    if n_experiments > max_experiments:
        raise InvariantViolation(f"experiment budget exceeded ({n_experiments} > {max_experiments})")
    if elapsed_s > max_seconds:
        raise InvariantViolation(f"time budget exceeded ({elapsed_s:.0f}s > {max_seconds:.0f}s)")
