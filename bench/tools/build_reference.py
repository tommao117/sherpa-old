"""Build the expert reference deliverable of a config task (human-run tool, writes into bench/).

Reference = the best configuration among a small grid of knobs that llama.cpp documents for serving
(tools/server/README.md at the task base), measured on this machine, and passing the oracle.
It is a transparent stand-in for "what a careful engineer following the docs would deploy".

Grid (all documented in tools/server/README.md @ b12242), every point honours the task contract
(any request up to max_context tokens must be served):
  unified      -np C -kvu -c <pool>       one shared KV pool: 1.1 x (full workload + one max-context
                                          request); when a shared pool overflows llama.cpp fails every
                                          in-flight request, so it must cover the contract case
  per-slot     -np n -c n*max_context     n in {C, C/2, C/4}: private KV per slot (attention only reads
                                          its own sequence) but fewer slots if memory is short
  always       -fit off                   never silently move layers to CPU (observed: FA disabled)
               -fa on, -ub {512, 1024, 2048} (-b = max(2048, ub))
Points that do not fit in memory fail to start and are skipped.

Usage: kernel/.venv/bin/python bench/tools/build_reference.py <task-id>
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import yaml

from sherpa_kernel import bench as B, llamacpp as L, loadgen as G, oracle as O
from sherpa_kernel.util import BENCH, read_yaml

SOURCE = "tools/server/README.md @ b12242: -np, -kvu/-c, -fit, -fa, -ub/-b"


def grid(task: B.Task) -> list[list[str]]:
    wl = task.raw["workload"]
    c = wl["concurrency"]
    need = c * (wl["prompt_tokens"] + wl["output_tokens"])
    pool = int(1.1 * (task.contract["max_context"] + need))   # full workload + one max-context request
    pool = int(math.ceil(pool / 1024) * 1024)
    mc = task.contract["max_context"]
    bases = [["-np", str(c), "-kvu", "-c", str(pool)]]
    bases += [["-np", str(n), "-c", str(n * mc)] for n in sorted({c, max(1, c // 2), max(1, c // 4)}, reverse=True)]
    return [b + ["-fit", "off", "-fa", "on", "-ub", str(ub), "-b", str(max(2048, ub))]
            for b in bases for ub in (512, 1024, 2048)]


def measure(p: B.Prepared, args: list[str], wl: G.Workload, seed: int, log: Path, metric: str) -> float | None:
    try:
        with L.start(p.build, p.model_path, args, {}, log) as s:
            raw = G.run_load(s.url, wl, seed)
    except Exception as e:  # noqa: BLE001
        print("   failed:", str(e).splitlines()[-1][:160], flush=True)
        return None
    v = G.contract_violations(raw, wl)
    if v:
        print("   violation:", v[0], flush=True)
        return None
    return G.metrics(raw, wl)[metric]


def main(task_id: str) -> None:
    task = B.load_task(task_id)
    wl = task.workload
    log_dir = Path("/tmp/sherpa-scratch/refgrid") / task_id
    log_dir.mkdir(parents=True, exist_ok=True)
    floor = task.deliverable("floor")
    p = B.prepare(task, floor, log_dir)
    cache_p = log_dir / "grid-cache.json"          # first-pass results survive a restart of this tool
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    rows = []
    for i, args in enumerate(grid(task)):
        key = " ".join(args)
        v = cache[key] if key in cache else measure(p, args, wl, 500, log_dir / f"g{i}.log", task.metric)
        cache[key] = v
        cache_p.write_text(json.dumps(cache, indent=1))
        print(f"grid {i}: {key} -> {v}", flush=True)
        rows.append({"args": args, "first": v})
    ok = sorted([r for r in rows if r["first"]], key=lambda r: -r["first"])[:3]
    for rep in range(2):  # re-measure the top 3 in rotation
        for r in ok:
            r.setdefault("more", []).append(measure(p, r["args"], wl, 600 + rep, log_dir / "top.log", task.metric))
    for r in ok:
        xs = [x for x in [r["first"], *r.get("more", [])] if x]
        r["mean"], r["cv"] = statistics.fmean(xs), (statistics.stdev(xs) / statistics.fmean(xs) if len(xs) > 1 else 0)
    ok.sort(key=lambda r: -r["mean"])
    pol = read_yaml(BENCH / "oracle" / "policy.yaml")["lossless"]
    chosen = None
    for r in ok:
        with L.start(p.build, p.model_path, r["args"], {}, log_dir / "oracle.log") as s:
            ev = O.evaluate(O.run(s.url, task.reference_data(), wl, task.contract, n_samples=pol["n_samples"]),
                            task.reference_data(), pol, task.contract)
        r["oracle_pass"] = ev["pass"]
        print(f"oracle {' '.join(r['args'])}: {ev['pass']}", flush=True)
        if ev["pass"]:
            chosen = r
            break
    if not chosen:
        raise SystemExit("no grid configuration passes the oracle")
    out = {"framework": "llama.cpp", "base": task.raw["framework"]["base"], "model_dtype": "bf16",
           "args": chosen["args"], "env": {},
           "source": f"best of documented-knob grid ({SOURCE}), measured on {task.raw['machine']}",
           "grid": [{"args": " ".join(r["args"]), task.metric: r.get("mean") or r["first"]} for r in rows]}
    path = BENCH / "deliverables" / task_id / "reference.yaml"
    path.write_text(yaml.safe_dump(out, sort_keys=False))
    print("wrote", path, chosen["mean"])


if __name__ == "__main__":
    main(sys.argv[1])
