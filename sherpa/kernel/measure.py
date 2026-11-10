"""Measurement contract: unified result format, repeats, variance check, comparison.

Adapters produce one dict of metrics per repeat; everything else is decided here.
"""
from __future__ import annotations

import statistics

SCHEMA = 1
CV_MAX = 0.05          # 主指标变异系数上限，超过则测量不可信
MIN_REPEATS = 3
MIN_EFFECT_PCT = 1.0   # 小于 1% 的变化一律视为噪声


def summarize(repeats: list[dict]) -> dict:
    keys = sorted({k for r in repeats for k, v in r.items() if isinstance(v, (int, float))})
    out = {}
    for k in keys:
        xs = [float(r[k]) for r in repeats if k in r]
        mean = statistics.fmean(xs)
        std = statistics.stdev(xs) if len(xs) > 1 else 0.0
        out[k] = {"mean": mean, "std": std, "cv": (std / mean) if mean else 0.0, "n": len(xs), "samples": xs}
    return out


def make(repeats: list[dict], primary: str, meta: dict | None = None) -> dict:
    s = summarize(repeats)
    p = s.get(primary)
    problems = []
    if p is None:
        problems.append(f"primary metric '{primary}' missing")
    else:
        if p["n"] < MIN_REPEATS:
            problems.append(f"only {p['n']} repeats (< {MIN_REPEATS})")
        if p["cv"] > CV_MAX:
            problems.append(f"cv {p['cv']:.3f} > {CV_MAX}")
    return {"schema": SCHEMA, "primary": primary, "summary": s, "trustworthy": not problems,
            "problems": problems, "meta": meta or {}}


def compare(base: dict, cand: dict, higher_is_better: bool = True) -> dict:
    k = base["primary"]
    b, c = base["summary"][k], cand["summary"][k]
    delta_pct = (c["mean"] - b["mean"]) / b["mean"] * 100
    noise_pct = 2 * max(b["cv"], c["cv"]) * 100
    threshold = max(noise_pct, MIN_EFFECT_PCT)
    improved = (delta_pct > threshold) if higher_is_better else (delta_pct < -threshold)
    regressed = (delta_pct < -threshold) if higher_is_better else (delta_pct > threshold)
    return {"metric": k, "base": b["mean"], "cand": c["mean"], "delta_pct": round(delta_pct, 2),
            "noise_pct": round(noise_pct, 2), "significant": abs(delta_pct) > threshold,
            "improved": improved, "regressed": regressed}
