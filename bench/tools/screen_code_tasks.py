"""Screen llama.cpp performance commits for use as code tasks (OQ-20).

For each candidate commit C (parent P = C^): build both with the judge's driver, then measure
  - llama-bench pp2048 / tg128 (FA on), 8 repetitions each
  - llama-server throughput at C=32 (-np 32), alternating P,C x3
A candidate is reproducible if C beats P beyond max(2*CV, 3%) on at least one workload.

Usage: kernel/.venv/bin/python bench/tools/screen_code_tasks.py <out.jsonl> <sha> [<sha> ...]
"""
from __future__ import annotations

import json
import statistics
import sys
import time
import traceback
from pathlib import Path

from sherpa_kernel import llamacpp as L
from sherpa_kernel import loadgen as G
from sherpa_kernel import machine
from sherpa_kernel.util import append_jsonl, run

MODEL = "/home/tommao/projects/sherpa-data/judge/models/qwen3-0.6b--conv-3b4d6ab3f7/bf16.gguf"
WL = G.Workload(prompt_tokens=256, output_tokens=128, concurrency=32, num_requests=96, warmup_requests=16,
                sampling={"temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05})
# Must fit in 16 GB: `-c 131072` without -kvu spills into WSL shared GPU memory (3-8x slower, noisy).
SERVER_ARGS = ["-np", "32", "-kvu", "-c", "16384", "-fit", "off", "-fa", "on"]


def llama_bench(b: L.Build) -> dict:
    with machine.exclusive("screen llama-bench"):
        machine.wait_quiet()
        # pp512 on a 0.6B model lasts ~10 ms and is dominated by clock ramp-up noise: use pp2048, 8 reps
        r = run([str(b.exe("llama-bench")), "-m", MODEL, "-ngl", "99", "-fa", "1", "-p", "2048", "-n", "128",
                 "-r", "8", "-o", "json"], timeout=900, check=False)
    if r.returncode != 0:
        return {"error": (r.stderr or r.stdout)[-400:]}
    out = {}
    for row in json.loads(r.stdout):
        k = "pp2048" if row["n_prompt"] else "tg128"
        out[k] = {"mean": row["avg_ts"], "std": row["stddev_ts"]}
    return out


def server_tput(b: L.Build, label: str, extra: list[str]) -> float | None:
    try:
        with L.start(b, MODEL, SERVER_ARGS + extra, {}, Path(f"/tmp/sherpa-scratch/screen-{label}.log")) as s:
            raw = G.run_load(s.url, WL, seed=11)
            if G.contract_violations(raw, WL):
                return None
            return G.metrics(raw, WL)["output_tok_per_s"]
    except Exception:  # noqa: BLE001
        return None


def screen(spec: str, extra: list[str]) -> dict:
    """spec = <sha> or <sha>+<fix-sha> (a later commit that only fixes the PR's build)."""
    sha, _, fix = spec.partition("+")
    info = L.commit_info(sha)
    parent = L.resolve(f"{sha}^")
    rec = {"sha": info["sha"], "parent": parent, "date": info["date"], "subject": info["subject"], "server_extra": extra,
           "fix": L.resolve(fix) if fix else None}
    t0 = time.time()
    try:
        bp = L.build(parent)
        bc = L.build(parent, L.diff(parent, sha) + L.diff(f"{fix}^", fix)) if fix else L.build(sha)
    except Exception as e:  # noqa: BLE001
        rec.update(status="build_failed", error=str(e)[-600:])
        return rec
    rec["build_s"] = round(time.time() - t0, 1)
    rec["bench"] = {"parent": llama_bench(bp), "commit": llama_bench(bc)}
    tp = {"parent": [], "commit": []}
    for _ in range(3):
        tp["parent"].append(server_tput(bp, "p", extra))
        tp["commit"].append(server_tput(bc, "c", extra))
    rec["server"] = tp
    gains = {}
    for k in ("pp2048", "tg128"):
        p, c = rec["bench"]["parent"].get(k), rec["bench"]["commit"].get(k)
        if p and c:
            noise = 2 * max(p["std"] / p["mean"], c["std"] / c["mean"])
            gains[k] = {"gain": c["mean"] / p["mean"] - 1, "noise": noise}
    if all(x is not None for x in tp["parent"] + tp["commit"]):
        p, c = tp["parent"], tp["commit"]
        ratios = [ci / pi for ci, pi in zip(c, p)]
        noise = 2 * max(statistics.stdev(p) / statistics.fmean(p), statistics.stdev(c) / statistics.fmean(c))
        gains["server_c32"] = {"gain": statistics.fmean(ratios) - 1, "noise": noise}
    rec["gains"] = gains
    rec["reproducible"] = any(g["gain"] > max(g["noise"], 0.03) for g in gains.values())
    rec["status"] = "ok" if gains else "measure_failed"
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


if __name__ == "__main__":
    out = Path(sys.argv[1])
    for arg in sys.argv[2:]:
        sha, _, flag = arg.partition(":")
        extra = ["-bs"] if flag == "bs" else []
        try:
            rec = screen(sha, extra)
        except Exception:  # noqa: BLE001
            rec = {"sha": sha, "status": "crashed", "error": traceback.format_exc()[-800:]}
        append_jsonl(out, rec)
        g = {k: f"{v['gain']:+.1%} (noise {v['noise']:.1%})" for k, v in (rec.get("gains") or {}).items()}
        print(sha, rec.get("status"), rec.get("reproducible"), g, rec.get("subject", "")[:60], flush=True)
