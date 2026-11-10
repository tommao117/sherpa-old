"""Knowledge interpreter: how `when` / `effect` are evaluated.

This lives in kernel/ (not assets/) on purpose: the knowledge *content* evolves, the *semantics*
used to judge it must not, otherwise evolution could redefine what "correct prediction" means.

Entry format (assets/knowledge/actions/*.yaml):
    id, node, kind: close_gap|move_bound,
    when:   {feature: "<op><value>" | value}       ops: > >= < <= == !=, or "in [a, b]"
    effect: {metric, change_pct: [lo, hi]}          static range
          | {metric, recover_gap: [lo, hi]}         fraction of the node's measured gap removed
    apply:  {<framework>: {knobs: {...}} | {code: "..."}}
    evidence: [fact ids], counter: [fact ids]
"""
from __future__ import annotations

import re
from pathlib import Path

from sherpa.util import ASSETS, read_yaml

_OP = re.compile(r"^\s*(>=|<=|==|!=|>|<)\s*(.+)$")


def _coerce(v: str):
    v = v.strip()
    try:
        return float(v)
    except ValueError:
        return v.strip("'\"")


def cond_ok(expr, value) -> bool:
    if value is None:
        return False
    if isinstance(expr, str):
        m = _OP.match(expr)
        if m:
            op, rhs = m.group(1), _coerce(m.group(2))
            try:
                return {">": value > rhs, ">=": value >= rhs, "<": value < rhs, "<=": value <= rhs,
                        "==": value == rhs, "!=": value != rhs}[op]
            except TypeError:
                return False
        if expr.startswith("in "):
            items = [_coerce(x) for x in expr[3:].strip("[] ").split(",")]
            return value in items
    return value == expr


def load(root: Path | None = None) -> list[dict]:
    root = root or ASSETS / "knowledge"
    out = []
    for f in sorted((root / "actions").glob("*.yaml")):
        d = read_yaml(f) or {}
        d["_file"] = str(f.relative_to(root))
        out.append(d)
    return out


def load_priors(root: Path | None = None) -> list[dict]:
    root = root or ASSETS / "knowledge"
    return [read_yaml(f) | {"_file": f.name} for f in sorted((root / "priors").glob("*.yaml"))]


def matches(entry: dict, features: dict) -> bool:
    return all(cond_ok(expr, features.get(k)) for k, expr in (entry.get("when") or {}).items())


def signature(framework: str, apply: dict, base_config: dict | None = None) -> str:
    """Identity of an action for matching facts <-> knowledge.

    Only knobs that actually *differ from the base config* count: an experiment run on top of
    `{parallel: 32}` that adds `backend_sampling` is the action "backend_sampling", not
    "backend_sampling + parallel" (devlog DES-18).
    """
    if "knobs" in apply:
        knobs = apply["knobs"] or {}
        if base_config is not None:
            knobs = {k: v for k, v in knobs.items() if base_config.get(k) != v}
        return f"{framework}:knobs:" + ",".join(sorted(knobs))
    if "code" in apply:
        return f"{framework}:code:{apply.get('id', 'anon')}"
    return f"{framework}:unknown"


def fact_signature(fact: dict) -> str:
    """Recompute an action signature from a fact's raw fields.

    Never trust the `signature` stored inside a fact: it is a *derived* value, and derived values
    go stale when their definition evolves. Facts keep raw context so derivations can be redone.
    """
    fw = fact["context"]["framework"]["name"]
    a = fact["action"]
    if a.get("type") == "config":
        return signature(fw, {"knobs": a.get("knobs") or {}}, a.get("base_config") or {})
    return signature(fw, {"code": True, "id": a.get("id", "anon")})


def predict(entry: dict, features: dict, diagnosis: list[dict] | None = None) -> float | None:
    """Predicted primary-metric change in %, or None if not applicable."""
    if not matches(entry, features):
        return None
    eff = entry.get("effect") or {}
    if "change_pct" in eff:
        lo, hi = eff["change_pct"]
        return (lo + hi) / 2
    if "recover_gap" in eff and diagnosis:
        lo, hi = eff["recover_gap"]
        node = entry["node"]
        row = next((d for d in diagnosis if d["node"] == node), None)
        if not row:
            return None
        total = sum(d["measured"] for d in diagnosis)
        removed = row["gap"] * (lo + hi) / 2
        if total - removed <= 0:
            return None
        return (total / (total - removed) - 1) * 100
    return None


def resolve_apply(apply: dict, context: dict) -> dict:
    """Resolve symbolic knob values against the current context.

    Knowledge must generalize, so an entry says `parallel: concurrency` rather than `parallel: 32`.
    Symbols are resolved from features + capacity here, in kernel/, so the *semantics* of a symbol
    cannot be redefined by evolution (devlog DES-22).
    """
    if "knobs" not in apply:
        return apply
    out = {}
    for k, v in (apply["knobs"] or {}).items():
        if isinstance(v, str) and v in context:
            out[k] = context[v]
        elif isinstance(v, str) and not v.replace(".", "").replace("-", "").isdigit() and v not in ("on", "off", "auto") \
                and not isinstance(v, bool):
            raise ValueError(f"knob '{k}': unresolved symbol '{v}' (known symbols: {sorted(context)})")
        else:
            out[k] = v
    return {**apply, "knobs": out}


def query(features: dict, framework: str, diagnosis: list[dict] | None = None,
          root: Path | None = None, context: dict | None = None) -> list[dict]:
    """Applicable actions for this context, ranked by predicted gain.

    `context` supplies values for symbolic knob values (e.g. concurrency, max_useful_batch).
    """
    ctx = {**(features or {}), **(context or {})}
    res = []
    for e in load(root):
        if framework not in (e.get("apply") or {}):
            continue
        p = predict(e, features, diagnosis)
        if p is None:
            continue
        try:
            ap = resolve_apply(e["apply"][framework], ctx)
        except ValueError as err:
            res.append({"id": e["id"], "node": e.get("node"), "kind": e.get("kind"),
                        "predicted_pct": round(p, 2), "apply": None, "error": str(err)})
            continue
        res.append({"id": e["id"], "node": e.get("node"), "kind": e.get("kind"), "predicted_pct": round(p, 2),
                    "apply": ap, "signature": signature(framework, ap),
                    "evidence_n": len(e.get("evidence") or []), "counter_n": len(e.get("counter") or [])})
    return sorted(res, key=lambda x: -x["predicted_pct"])
