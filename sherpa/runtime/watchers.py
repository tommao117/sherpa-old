"""Deterministic watchers: decide which intermediate observations are worth an LLM's attention."""
from __future__ import annotations

REGRESSION_ALERT = 0.70   # 中间结果低于 incumbent 的 70% 时上报 Monitor


def after_repeat(primary: str, repeat_metrics: dict, incumbent_mean: float | None,
                 higher_is_better: bool, server_alive: bool) -> dict | None:
    if not server_alive:
        return {"rule": "server_crashed", "auto": "kill"}          # 规则直接处理，不上报 LLM
    v = repeat_metrics.get(primary)
    if v is None:
        return {"rule": "metric_missing", "auto": "kill"}
    if incumbent_mean:
        ratio = v / incumbent_mean if higher_is_better else incumbent_mean / v
        if ratio < REGRESSION_ALERT:
            return {"rule": "large_regression", "ratio": round(ratio, 3), "value": v, "incumbent": incumbent_mean}
    return None
