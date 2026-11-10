"""Statistics of the measurement contract: summaries, paired comparison, significance.

What has to be controlled is the uncertainty of the *score*, not the spread of single runs.
Interleaved candidate/reference pairs cancel slow drift (thermals, clocks, background); the
remaining pair-to-pair noise is averaged down by adding pairs until the standard error of the
mean ratio is small enough (adaptive, see bench.score). Observed on WSL2 / RTX 5070 Ti: GPU clocks
at segment start vary 1.4-2.9 GHz, single-pair ratio CV ~6%.
"""
from __future__ import annotations

import math
import statistics

MIN_EFFECT = 0.01      # changes below 1% are never called significant
SE_MAX = 0.02          # a ratio is trustworthy when its standard error is <= 2%
MIN_PAIRS = 3


def summarize(xs: list[float]) -> dict:
    xs = [float(x) for x in xs]
    mean = statistics.fmean(xs) if xs else float("nan")
    std = statistics.stdev(xs) if len(xs) > 1 else 0.0
    n = len(xs)
    return {"mean": mean, "std": std, "cv": std / mean if mean else 0.0, "n": n,
            "se": (std / mean / math.sqrt(n)) if mean and n > 1 else float("inf"),
            "min": min(xs) if xs else None, "max": max(xs) if xs else None, "samples": xs}


def paired_ratio(cand: list[float], ref: list[float], higher_is_better: bool) -> dict:
    """Per-pair performance ratio (>1 means candidate better), its spread and standard error."""
    if len(cand) != len(ref) or not cand:
        raise ValueError("paired comparison needs equally many, non-zero samples")
    rs = [(c / r) if higher_is_better else (r / c) for c, r in zip(cand, ref)]
    s = summarize(rs)
    s["gmean"] = math.exp(statistics.fmean(math.log(x) for x in rs))   # right average for ratios
    return s


def significant(ratio: dict) -> dict:
    """Is the mean ratio distinguishable from 1? (2 standard errors, never below MIN_EFFECT)"""
    threshold = max(2 * ratio["se"], MIN_EFFECT) if math.isfinite(ratio["se"]) else math.inf
    delta = ratio["gmean"] - 1.0
    return {"delta": delta, "threshold": threshold, "better": delta > threshold, "worse": delta < -threshold,
            "significant": abs(delta) > threshold}


def trustworthy(s: dict, min_n: int = MIN_PAIRS, se_max: float = SE_MAX) -> list[str]:
    problems = []
    if s["n"] < min_n:
        problems.append(f"only {s['n']} samples (< {min_n})")
    elif s["se"] > se_max:
        problems.append(f"standard error {s['se']:.3f} > {se_max} after {s['n']} samples")
    return problems
