"""Read-only host inventory. NVIDIA queries do not establish runtime CUDA compatibility."""
from __future__ import annotations

import csv
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def query(argv, scratch: Path, check=lambda: None):
    from ..runtime.calls import Calls
    check()
    with tempfile.TemporaryFile(dir=scratch) as stdout, tempfile.TemporaryFile(dir=scratch) as stderr:
        process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True,
                                   env={**os.environ, "LC_ALL": "C"})
        try:
            import time
            deadline = time.monotonic() + 3
            while process.poll() is None:
                check()
                if time.monotonic() >= deadline:
                    return {"status": "unknown", "reason": "inventory query timed out"}
                if os.fstat(stdout.fileno()).st_size + os.fstat(stderr.fileno()).st_size > 65536:
                    return {"status": "unknown", "reason": "inventory output exceeded limit"}
                time.sleep(0.05)
            stdout.seek(0)
            stderr.seek(0)
            out, error = stdout.read(65537), stderr.read(8192)
            if process.returncode or len(out) > 65536:
                return {"status": "unknown", "reason": error.decode(errors="replace")[:1000] or "query failed"}
            return {"status": "observed", "raw": out.decode(errors="replace")}
        finally:
            if not Calls.cleanup(process):
                raise RuntimeError("inventory process failed to stop")


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_gpus(text):
    out = []
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 7 or not row[0].strip().isdigit():
            raise ValueError("unexpected GPU inventory row")
        index, uuid, name, total, used, driver, cap = (value.strip() for value in row)
        out.append({"index": int(index), "uuid": uuid, "name": name,
                    "memory_total_mib": number(total), "memory_used_mib": number(used),
                    "driver": driver, "compute_capability": cap if re.fullmatch(r"\d+\.\d+", cap) else None})
    if len({g["uuid"] for g in out}) != len(out):
        raise ValueError("duplicate GPU UUID in inventory")
    return out


def topology(text):
    matrix, columns = {}, []
    for line in re.sub(r"\x1b\[[0-9;]*m", "", text).splitlines():
        parts = line.split()
        if not parts:
            continue
        if not columns:
            for token in parts:
                if re.fullmatch(r"GPU\d+", token):
                    columns.append(token)
                else:
                    break
            continue
        if parts[0] in columns and len(parts) > len(columns):
            links = parts[1:len(columns) + 1]
            if all(re.fullmatch(r"X|SYS|NODE|PHB|PXB|PIX|NV\d+", link) for link in links):
                matrix[parts[0]] = dict(zip(columns, links))
    return matrix


def probe(scratch: Path, check=lambda: None):
    memory = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                memory[key] = int(value.split()[0]) // 1024
    except (OSError, ValueError):
        pass
    result = {"host": {"system": platform.system(), "architecture": platform.machine(),
                       "kernel": platform.release(), "wsl": "microsoft" in platform.release().lower(),
                       "cpu_count": os.cpu_count(), "libc": list(platform.libc_ver()),
                       "controller_python": platform.python_version(), "memory_mib": memory},
              "tools": {name: shutil.which(name) for name in ("uv", "git", "cc", "cmake", "nvcc", "nsys", "py-spy")},
              "gpu_status": "unknown", "gpus": [], "queries": {}, "topology": {},
              "cuda_runtime_verified": False}
    smi = shutil.which("nvidia-smi")
    if not smi:
        result["gpu_reason"] = "nvidia-smi unavailable; this is not proof that no GPU exists"
        return result
    queries = {
        "gpus": [smi, "--query-gpu=index,uuid,name,memory.total,memory.used,driver_version,compute_cap",
                 "--format=csv,noheader,nounits"],
        "processes": [smi, "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        "topology": [smi, "topo", "-m"],
    }
    for name, argv in queries.items():
        try:
            result["queries"][name] = query(argv, scratch, check)
        except OSError as error:
            result["queries"][name] = {"status": "unknown", "reason": str(error)}
    inventory = result["queries"]["gpus"]
    if inventory["status"] == "observed":
        try:
            result["gpus"] = parse_gpus(inventory["raw"])
            result["gpu_status"] = "observed"
        except ValueError as error:
            result["gpu_reason"] = str(error)
    apps = result["queries"]["processes"]
    held, processes_known = set(), apps["status"] == "observed"
    if processes_known:
        for row in csv.reader(apps["raw"].splitlines(), skipinitialspace=True):
            if len(row) != 2 or not row[1].strip().isdigit():
                processes_known = False
                break
            held.add(row[0].strip())
        if held - {g["uuid"] for g in result["gpus"]}:
            processes_known = False
    for gpu in result["gpus"]:
        used = gpu["memory_used_mib"]
        gpu["occupancy"] = ("busy" if gpu["uuid"] in held or (used is not None and used > 512)
                            else "idle_observed" if processes_known and used is not None else "unknown")
    topo = result["queries"]["topology"]
    if topo["status"] == "observed":
        result["topology"] = topology(topo["raw"])
    return result


def select_gpus(machine, selectors=(), count=1):
    if count < 1:
        raise ValueError("GPU count must be positive")
    inventory = machine["gpus"]
    selected = []
    if selectors:
        for selector in selectors:
            matches = [g for g in inventory if str(g["index"]) == selector or g["uuid"] == selector]
            if len(matches) != 1:
                raise ValueError(f"GPU selector {selector!r} not uniquely found in observed inventory")
            if matches[0] in selected:
                raise ValueError("duplicate GPU selection")
            selected.append(matches[0])
    else:
        selected = sorted((g for g in inventory if g.get("occupancy") == "idle_observed"),
                          key=lambda g: g["index"])[:count]
    issues = []
    if not selected or (not selectors and len(selected) < count):
        issues.append("not enough observably idle GPUs; allocation is unresolved")
    if any(g.get("occupancy") != "idle_observed" for g in selected):
        issues.append("selected GPUs are busy or occupancy is unknown; not reserved or ready")
    return {"requested": list(selectors), "requested_count": len(selectors) if selectors else count,
            "uuids": [g["uuid"] for g in selected], "reserved": False, "issues": issues,
            "policy": "explicit selection" if selectors else "propose first observably idle cards; not topology-optimized"}
