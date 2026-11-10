"""How much faster is the newest base than older tags on this machine? (OQ-20, version spans)

Single upstream PRs mostly claim 1-6% end-to-end, at or below this machine's measurement noise.
A span of upstream work (old tag -> new tag) accumulates many of them; a large gap can then be
bisected into a sharper code task.

Usage: kernel/.venv/bin/python bench/tools/span_sweep.py <out.jsonl> <new-tag> <old-tag> [<old-tag> ...]
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

from screen_code_tasks import llama_bench, server_tput  # same measurement protocol as the PR screen
from sherpa_kernel import llamacpp as L
from sherpa_kernel.util import append_jsonl


def main(out_path: str, new: str, olds: list[str]) -> None:
    out = Path(out_path)
    bn = L.build(new)
    for old in olds:
        rec = {"new": new, "old": old, "old_date": L.commit_info(old)["date"]}
        try:
            bo = L.build(old)
        except Exception as e:  # noqa: BLE001
            rec.update(status="build_failed", error=str(e)[-400:])
            append_jsonl(out, rec)
            print(old, "build_failed", flush=True)
            continue
        rec["bench"] = {"old": llama_bench(bo), "new": llama_bench(bn)}
        tp = {"old": [], "new": []}
        for _ in range(3):
            tp["old"].append(server_tput(bo, "o", []))
            tp["new"].append(server_tput(bn, "n", []))
        rec["server"] = tp
        gains = {}
        for k in ("pp2048", "tg128"):
            o, n = rec["bench"]["old"].get(k), rec["bench"]["new"].get(k)
            if o and n:
                gains[k] = n["mean"] / o["mean"] - 1
        if all(x for x in tp["old"] + tp["new"]):
            gains["server_c32"] = statistics.fmean(n / o for n, o in zip(tp["new"], tp["old"])) - 1
        rec["gains_new_over_old"] = gains
        rec["status"] = "ok"
        append_jsonl(out, rec)
        print(old, {k: f"{v:+.1%}" for k, v in gains.items()}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
