"""Localise the biggest performance jump inside a span of upstream history (OQ-20).

Splits the first-parent history between <old> and <new> in half, measures the midpoint, and
recurses into the half that carries the larger part of the gap, until the range is one commit or
the gap left in the range is below the noise floor. The commit found becomes a code-task candidate
(screen it with screen_code_tasks.py, then make_code_task.py).

Usage: kernel/.venv/bin/python bench/tools/bisect_span.py <out.jsonl> <old> <new> [server_c32|tg128|pp2048]
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

from screen_code_tasks import llama_bench, server_tput
from sherpa_kernel import llamacpp as L
from sherpa_kernel.util import append_jsonl, run

NOISE = 0.06


def history(old: str, new: str) -> list[str]:
    out = run(["git", "--git-dir", str(L.BARE), "rev-list", "--first-parent", "--reverse", f"{old}..{new}"]).stdout
    return [L.resolve(old)] + out.split()


def measure(sha: str, metric: str, cache: dict) -> float | None:
    if sha in cache:
        return cache[sha]
    try:
        b = L.build(sha)
    except Exception:  # noqa: BLE001
        cache[sha] = None
        return None
    if metric == "server_c32":
        xs = [x for x in (server_tput(b, "bis", []) for _ in range(3)) if x]
        v = statistics.median(xs) if xs else None
    else:
        v = (llama_bench(b).get(metric) or {}).get("mean")
    cache[sha] = v
    return v


def main(out: str, old: str, new: str, metric: str = "server_c32") -> None:
    commits = history(old, new)
    cache: dict = {}
    lo, hi = 0, len(commits) - 1
    diffuse = False
    while hi - lo > 1:
        vlo, vhi = measure(commits[lo], metric, cache), measure(commits[hi], metric, cache)
        mid = (lo + hi) // 2
        vm = measure(commits[mid], metric, cache)
        while vm is None and mid < hi - 1:          # unbuildable commit: step forward
            mid += 1
            vm = measure(commits[mid], metric, cache)
        rec = {"lo": commits[lo], "mid": commits[mid], "hi": commits[hi], "metric": metric,
               "v_lo": vlo, "v_mid": vm, "v_hi": vhi, "range": hi - lo}
        append_jsonl(Path(out), rec)
        print(f"[{hi - lo:5d}] lo {vlo} mid {vm} hi {vhi}", flush=True)
        if None in (vlo, vm, vhi):
            break
        left, right = vm / vlo - 1, vhi / vm - 1
        if max(left, right) < NOISE:
            diffuse = True                           # the gap is spread over many small commits
            break
        lo, hi = (lo, mid) if left >= right else (mid, hi)
    gain = (cache.get(commits[hi]) or 0) / (cache.get(commits[lo]) or 1) - 1
    if diffuse:
        append_jsonl(Path(out), {"diffuse": True, "lo": commits[lo], "hi": commits[hi], "range": hi - lo,
                                 "metric": metric, "gain": gain})
        print(f"diffuse: {gain:+.1%} left over {hi - lo} commits, no single step above {NOISE:.0%}", flush=True)
        return
    info = L.commit_info(commits[hi])
    append_jsonl(Path(out), {"found": commits[hi], "parent": commits[lo], "subject": info["subject"],
                             "date": info["date"], "range": hi - lo, "metric": metric, "gain": gain})
    print("found", commits[hi][:10], info["subject"], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
