"""Execution environment: probe the machine, locate toolchains, declare capabilities.

Result is the machine profile (sherpa-data/env.yaml). Humans can override fields in
sherpa-data/env.override.yaml. The profile is also the environment fingerprint stored in facts.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

from sherpa.util import DATA, REPO, read_yaml, run, write_yaml


def _cuda_candidates() -> list[Path]:
    c = []
    if os.environ.get("SHERPA_CUDA_ROOT"):
        c.append(Path(os.environ["SHERPA_CUDA_ROOT"]))
    c += [Path("/usr/local/cuda")]
    pyroots = {Path(sys.prefix)}
    for pat in ["/root/.workbuddy/binaries/python/envs/*", "/root/envs/venvs/*", "/opt/conda/envs/*"]:
        pyroots.update(Path(p) for p in glob.glob(pat))
    for r in sorted(pyroots):
        c += [Path(p) for p in glob.glob(str(r / "lib/python3*/site-packages/nvidia/cu1*"))]
    return c


def find_cuda() -> dict | None:
    for root in _cuda_candidates():
        nvcc = root / "bin" / "nvcc"
        if nvcc.exists() and (root / "include" / "cuda_runtime.h").exists():
            v = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True).stdout
            ver = v.strip().splitlines()[-1] if v else "?"
            libdir = root / "lib64" if (root / "lib64" / "libcudart.so").exists() else root / "lib"
            return {"root": str(root), "nvcc": str(nvcc), "libdir": str(libdir), "version": ver,
                    "pip_wheel": "site-packages" in str(root),
                    "cupti": (root / "include" / "cupti_activity.h").exists()
                    or (root / "extras/CUPTI/include/cupti_activity.h").exists()}
    return None


def gpus() -> list[dict]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    except FileNotFoundError:
        return []
    res = []
    for line in out.strip().splitlines():
        i, name, mem, drv, cc = [x.strip() for x in line.split(",")]
        res.append({"index": int(i), "name": name, "memory_mib": int(mem), "driver": drv, "compute_cap": cc})
    return res


def _python_with(mods: list[str]) -> str | None:
    for py in [sys.executable, shutil.which("python3")]:
        if py and subprocess.run([py, "-c", "import " + ",".join(mods)], capture_output=True).returncode == 0:
            return py
    return None


def build_tracer(cuda: dict) -> str | None:
    out = DATA / "tools"
    so = out / "libsherpa_cupti.so"
    src = REPO / "crates/cupti-trace/cupti_trace.c"
    if so.exists() and so.stat().st_mtime >= src.stat().st_mtime:
        return str(so)
    try:
        run(["bash", str(REPO / "crates/cupti-trace/build.sh"), cuda["root"], str(out)])
        return str(so)
    except Exception:
        return None


def probe(save: bool = True) -> dict:
    cuda = find_cuda()
    env = {
        "hostname": os.uname().nodename,
        "kernel": os.uname().release,
        "wsl": "microsoft" in os.uname().release.lower(),
        "cpu_count": os.cpu_count(),
        "gpus": gpus(),
        "cuda": cuda,
        "python_convert": _python_with(["gguf", "torch", "safetensors", "transformers"]),
        "tools": {t: shutil.which(t) for t in ["cmake", "ninja", "gcc", "git", "nsys", "ncu"]},
    }
    caps = {"cuda_build": bool(cuda), "kernel_trace": None}
    if env["tools"]["nsys"]:
        caps["kernel_trace"] = "nsys"
    elif cuda and cuda["cupti"]:
        lib = build_tracer(cuda)
        if lib:
            caps["kernel_trace"] = "cupti-inject"
            env["tracer_lib"] = lib
    env["capabilities"] = caps
    ov = DATA / "env.override.yaml"
    if ov.exists():
        env.update(read_yaml(ov) or {})
    if save:
        write_yaml(DATA / "env.yaml", env)
    return env


def fingerprint(env: dict) -> dict:
    g = env["gpus"][0] if env["gpus"] else {}
    return {"gpu": g.get("name"), "driver": g.get("driver"), "compute_cap": g.get("compute_cap"),
            "cuda": (env.get("cuda") or {}).get("version"), "wsl": env.get("wsl")}
