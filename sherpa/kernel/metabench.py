"""Metabench (outer-loop judge). v0: offline replay only.

For every fact, ask a knowledge version "what would you have predicted for this action in this
context?" and compare with what was actually observed.
"""
from __future__ import annotations

from pathlib import Path

from sherpa.kernel import facts as F
from sherpa.kernel import knowledge as K


def replay(knowledge_root: Path | None, fact_list: list[dict] | None = None) -> dict:
    fact_list = fact_list if fact_list is not None else F.all_facts()
    entries = K.load(knowledge_root)
    rows = []
    for f in fact_list:
        obs = f["observed"].get("delta_pct")
        if obs is None:
            continue
        fw = f["context"]["framework"]["name"]
        sig = K.fact_signature(f)          # recomputed, never the stored value
        diag = (f["observed"].get("ledger_before") or {}).get("diagnosis")
        preds = []
        for e in entries:
            ap = (e.get("apply") or {}).get(fw)
            if ap and K.signature(fw, ap) == sig:
                p = K.predict(e, f["features"], diag)
                if p is not None:
                    preds.append((e["id"], p))
        pred = preds[0][1] if preds else None
        rows.append({"fact": f["id"], "signature": sig, "observed": obs, "predicted": pred,
                     "by": preds[0][0] if preds else None,
                     "abs_err": abs(pred - obs) if pred is not None else None,
                     "sign_ok": (pred > 0) == (obs > 0) if pred is not None else None})
    covered = [r for r in rows if r["predicted"] is not None]
    return {
        "n_facts": len(rows),
        "coverage": len(covered) / len(rows) if rows else 0.0,
        "mae_pct": sum(r["abs_err"] for r in covered) / len(covered) if covered else None,
        "sign_acc": sum(r["sign_ok"] for r in covered) / len(covered) if covered else None,
        "rows": rows,
    }


def compare(old: dict, new: dict) -> dict:
    reasons = []
    if new["coverage"] < old["coverage"]:
        reasons.append(f"coverage drops {old['coverage']:.2f} -> {new['coverage']:.2f}")
    if old["mae_pct"] is not None and (new["mae_pct"] is None or new["mae_pct"] > old["mae_pct"] + 1e-9):
        reasons.append(f"mae worse {old['mae_pct']} -> {new['mae_pct']}")
    if old["sign_acc"] is not None and (new["sign_acc"] is None or new["sign_acc"] < old["sign_acc"]):
        reasons.append(f"sign accuracy worse {old['sign_acc']} -> {new['sign_acc']}")
    improved = (new["coverage"] > old["coverage"]) or (
        old["mae_pct"] is not None and new["mae_pct"] is not None and new["mae_pct"] < old["mae_pct"])
    return {"pass": not reasons, "improved": improved, "reasons": reasons,
            "old": {k: old[k] for k in ("n_facts", "coverage", "mae_pct", "sign_acc")},
            "new": {k: new[k] for k in ("n_facts", "coverage", "mae_pct", "sign_acc")}}
