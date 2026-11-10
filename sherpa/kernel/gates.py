"""Gates: the only boundary of an unbounded action space.

correctness  greedy outputs vs baseline (lossless) / tolerance (lossy, OQ-9: not implemented in v0)
reproducible measurement trustworthy (cv within threshold, enough repeats)
deliverable  deliverable launch script brings up a healthy server that answers a request
"""
from __future__ import annotations

# 无损动作理论上应逐 token 一致，但 batch 组成不同会导致浮点累加顺序不同（batch-variance），
# greedy 输出在若干 token 后可能分叉。v0 用"公共前缀长度"作为判据，阈值见下。
MIN_COMMON_PREFIX = 16
GREEDY_TOKENS = 48

# 正确性门禁的固定 prompt 集（锚点的一部分，不随资产进化）
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Explain in one paragraph why the sky is blue.",
    "1, 1, 2, 3, 5, 8, 13,",
    "Translate to French: The weather is nice today.",
    "Q: What is 17 * 23?\nA:",
]


def correctness(baseline: list[list[int]], cand: list[list[int]], lossy: bool) -> dict:
    if lossy:
        return {"gate": "correctness", "pass": False, "reason": "lossy actions need an eval set (OQ-9), not supported in v0"}
    details = []
    ok = True
    for i, (a, b) in enumerate(zip(baseline, cand)):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        exact = a == b
        passed = exact or n >= min(MIN_COMMON_PREFIX, len(a))
        ok &= passed
        details.append({"prompt": i, "exact": exact, "common_prefix": n, "len": len(a), "pass": passed})
    if len(baseline) != len(cand):
        ok = False
    return {"gate": "correctness", "pass": ok, "exact_all": all(d["exact"] for d in details), "details": details}


def reproducible(measurement: dict) -> dict:
    return {"gate": "reproducible", "pass": measurement["trustworthy"], "problems": measurement["problems"]}


def deliverable(healthy: bool, answered: bool, log: str = "") -> dict:
    return {"gate": "deliverable", "pass": healthy and answered, "log": log}
