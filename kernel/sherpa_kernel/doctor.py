"""Machine health check against llama.cpp's public CUDA scoreboard (bench/doctor.yaml)."""
from __future__ import annotations

import json
from pathlib import Path

from . import llamacpp as L
from . import machine, records
from .util import BENCH, DATA, read_yaml, run, sha256_file


def model_path(spec: dict) -> Path:
    p = DATA / "models" / spec["file"]
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        run(["curl", "-sSL", "-o", str(tmp), spec["url"]], timeout=7200)
        tmp.replace(p)
    return p


def check(measured: dict, board: dict, max_slowdown: float) -> list[str]:
    problems = []
    for fa in ("fa0", "fa1"):
        for k, ref in board[fa].items():
            got = (measured.get(fa) or {}).get(k)
            if got is None:
                problems.append(f"{fa}.{k}: not measured")
            elif got < (1 - max_slowdown) * ref:
                problems.append(f"{fa}.{k}: {got:.1f} t/s is {1 - got / ref:.0%} below scoreboard {ref:.1f}")
    return problems


def run_doctor() -> dict:
    cfg = read_yaml(BENCH / "doctor.yaml")
    gpu = machine.gpus()[0]["name"]
    board = cfg["scoreboard"].get(gpu)
    mp = model_path(cfg["model"])
    digest = sha256_file(mp)
    b = L.build(cfg["build"])
    with machine.exclusive("doctor"):
        quiet = machine.wait_quiet()
        before = {**machine.gpu_state(), **quiet}
        r = run([str(b.exe("llama-bench")), "-m", str(mp), "-ngl", "99", "-fa", "0,1", "-o", "json"], timeout=1800)
    measured: dict = {}
    for row in json.loads(r.stdout):
        fa = f"fa{int(bool(row['flash_attn']))}"
        k = "pp512" if row["n_prompt"] else "tg128"
        measured.setdefault(fa, {})[k] = row["avg_ts"]
    problems = ["model sha256 mismatch"] if digest != cfg["model"]["sha256"] else []
    problems += check(measured, board, cfg["max_slowdown"]) if board else [f"no scoreboard entry for '{gpu}'"]
    rec = {"machine": machine.fingerprint(), "gpu_state": before, "build": b.info["commit"],
           "measured": measured, "scoreboard": board, "pass": not problems, "problems": problems}
    rec["id"] = records.write("doctor", rec)
    return rec


def latest_pass(fp: dict | None = None) -> dict | None:
    """Most recent passing doctor record on this machine with the same driver and CUDA."""
    fp = fp or machine.fingerprint()
    keys = ("machine", "driver", "cuda", "gpu_uuid")
    for r in reversed(records.read("doctor")):
        if r.get("pass") and all(r["machine"].get(k) == fp.get(k) for k in keys):
            return r
    return None
