"""Machine identity and toolchain discovery.

The machine id binds bench baselines to hardware: a reference measured on one GPU says nothing
about another. The fingerprint is stored with every record so results can be audited later.
"""
from __future__ import annotations

import fcntl
import os
import platform
import re
import shutil
import subprocess
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

CUDA_ROOTS = ["/usr/local/cuda"]


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


@lru_cache(maxsize=1)
def gpus() -> list[dict]:
    out = _sh(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap,uuid",
               "--format=csv,noheader,nounits"])
    res = []
    for line in out.splitlines():
        i, name, mem, drv, cc, uuid = [x.strip() for x in line.split(",")]
        res.append({"index": int(i), "name": name, "memory_mib": int(mem), "driver": drv,
                    "compute_cap": cc, "uuid": uuid})
    return res


@lru_cache(maxsize=1)
def cuda() -> dict | None:
    roots = ([os.environ["SHERPA_CUDA_ROOT"]] if os.environ.get("SHERPA_CUDA_ROOT") else []) + CUDA_ROOTS
    for r in roots:
        root = Path(r).resolve()
        nvcc = root / "bin" / "nvcc"
        if nvcc.exists():
            v = _sh([str(nvcc), "--version"])
            m = re.search(r"release (\d+\.\d+)", v)
            cupti_inc = next((p for p in [root / "include", root / "extras/CUPTI/include"]
                              if (p / "cupti.h").exists()), None)
            return {"root": str(root), "nvcc": str(nvcc), "version": m.group(1) if m else "?",
                    "cupti_include": str(cupti_inc) if cupti_inc else None}
    return None


def tool(name: str) -> str | None:
    return os.environ.get(f"SHERPA_{name.upper()}") or shutil.which(name)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def machine_id(gpu_index: int = 0) -> str:
    """Stable id used to key baselines: <gpu-model>@<hostname>."""
    g = gpus()[gpu_index]
    name = g["name"].replace("NVIDIA ", "").replace("GeForce ", "")
    return f"{slug(name)}@{slug(platform.node())}"


def fingerprint(gpu_index: int = 0) -> dict:
    g = gpus()[gpu_index] if gpus() else {}
    c = cuda() or {}
    return {"machine": machine_id(gpu_index) if g else None, "gpu": g.get("name"), "gpu_uuid": g.get("uuid"),
            "driver": g.get("driver"), "compute_cap": g.get("compute_cap"), "cuda": c.get("version"),
            "kernel": platform.release(), "wsl": "microsoft" in platform.release().lower(),
            "cpu_count": os.cpu_count()}


@contextmanager
def exclusive(reason: str = ""):
    """Machine-wide lock held around every build and every measurement.

    Builds saturate the CPU and the server's scheduler is CPU-bound for small models, so a build or a
    second benchmark running concurrently silently corrupts a measurement. Processes interleave at
    segment granularity instead of overlapping. Re-entrant within one process.
    """
    global _lock_depth
    if _lock_depth:
        _lock_depth += 1
        try:
            yield
        finally:
            _lock_depth -= 1
        return
    from .util import JUDGE_DATA
    JUDGE_DATA.mkdir(parents=True, exist_ok=True)
    with open(JUDGE_DATA / "machine.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(f"{os.getpid()} {reason}\n")
        f.flush()
        _lock_depth = 1
        try:
            yield
        finally:
            _lock_depth = 0
            fcntl.flock(f, fcntl.LOCK_UN)


_lock_depth = 0

QUIET_CPU = 0.10          # system CPU busy fraction above which the machine is not quiet


def cpu_busy(seconds: float = 1.0) -> float:
    """System-wide CPU busy fraction over a short window (/proc/stat)."""
    import time

    def snap():
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        return sum(v), v[3] + v[4]
    t1, i1 = snap()
    time.sleep(seconds)
    t2, i2 = snap()
    return 1.0 - (i2 - i1) / max(1, t2 - t1)


def wait_quiet(limit: float = QUIET_CPU, timeout: float = 900) -> dict:
    """Wait until nothing else loads the CPU. The lock only coordinates judge processes; any other
    CPU-heavy process (observed: a model download decompressing on many threads) silently cut
    llama-server throughput by ~30%."""
    import time
    t0 = time.time()
    busy = cpu_busy()
    while busy > limit and time.time() - t0 < timeout:
        time.sleep(5)
        busy = cpu_busy()
    return {"cpu_busy": round(busy, 3), "waited_s": round(time.time() - t0, 1), "quiet": busy <= limit}


def gpu_state(gpu_index: int = 0) -> dict:
    """Instantaneous state; recorded around measurements to spot thermal / clock anomalies."""
    out = _sh(["nvidia-smi", f"--id={gpu_index}",
               "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used,utilization.gpu",
               "--format=csv,noheader,nounits"])
    keys = ["temp_c", "sm_clock_mhz", "mem_clock_mhz", "power_w", "mem_used_mib", "util_pct"]
    vals = [x.strip() for x in out.split(",")] if out else []
    res = {}
    for k, v in zip(keys, vals):
        try:
            res[k] = float(v)
        except ValueError:
            res[k] = None
    res["loadavg_1m"] = round(os.getloadavg()[0], 2)
    return res
