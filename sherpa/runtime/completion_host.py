"""Live host checks used only after explicit completion execution approval."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import re
import shutil
import time

from .calls import RunStopped
from .execution_records import ARTIFACT_LIMIT, RecordReader, absolute, require
from .execution_contract import metadata
from .managed_process import group_pids, run_process


def file_identity(path, check, *, limit=ARTIFACT_LIMIT):
    path = absolute(str(path))
    reader = RecordReader(path.parent, check)
    reader.remaining = limit
    before = metadata(str(path.parent), path.name)
    result = reader.read(path.name, limit, hash_only=True)
    require(before == metadata(str(path.parent), path.name), "file replaced during hashing")
    return {"path": str(path), **result, "stat": before}


def selected_weight(task):
    model = task["model"]
    selected = [s for s in model["sets"] if s["id"] == model["selected_set"]]
    require(len(selected) == 1 and selected[0]["layout"] == "single" and len(selected[0]["weights"]) == 1
            and selected[0]["support"] == [], "one explicit self-contained GGUF is required")
    return selected[0]["weights"][0]


def model_identity(task, check):
    file = selected_weight(task)
    root = absolute(task["model"]["path"])
    path = root / file["path"]
    require(path.parent == root and path.suffix == ".gguf", "initial live runner supports a GGUF directly in model root")
    return file_identity(path, check, limit=64 * 1024 * 1024 * 1024)


def bus_id(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", value) is not None,
            "GPU PCI bus identity unavailable")
    domain, bus, device = value.lower().split(":")
    return f"{int(domain, 16):04x}:{bus}:{device}"


def gpu_snapshot(gpu_csv, processes_csv, uuid, *, own_pids=(), memory_mib):
    rows = list(csv.reader(gpu_csv.splitlines(), skipinitialspace=True))
    require(rows and all(len(row) == 5 for row in rows), "GPU inventory incomplete")
    require(len({r[0].strip() for r in rows}) == len(rows), "duplicate GPU rows")
    selected = [r for r in rows if r[0].strip() == uuid]
    require(len(selected) == 1, "selected GPU UUID not uniquely present")
    _, pci, used, total, utilization = (x.strip() for x in selected[0])
    used, total, utilization = float(used), float(total), float(utilization)
    require(all(math.isfinite(v) for v in (used, total, utilization)) and 0 <= used <= total
            and 0 <= utilization <= 100 and total > 0, "GPU readings invalid")
    apps = list(csv.reader(processes_csv.splitlines(), skipinitialspace=True))
    require(all(len(r) == 2 and r[1].strip().isdigit() for r in apps), "GPU process visibility unavailable")
    require(all(r[0].strip() in {v[0].strip() for v in rows} for r in apps), "unknown GPU in process inventory")
    active = {int(r[1]) for r in apps if r[0].strip() == uuid}
    if active - set(own_pids):
        raise RunStopped("failed", "selected GPU is occupied by another process")
    if not active and (used > 512 or utilization > 0):
        raise RunStopped("failed", "GPU idle state cannot be established")
    if used > memory_mib:
        raise RunStopped("budget_exhausted", "GPU memory budget exceeded")
    if not own_pids and total - used < memory_mib:
        raise RunStopped("failed", "insufficient free GPU memory for requested budget")
    return {"uuid": uuid, "bus_id": bus_id(pci), "memory_used_mib": used, "memory_total_mib": total,
            "utilization": utilization, "processes": sorted(active)}


class CompletionHost:
    def __init__(self, run, calls, task, tools):
        self.run, self.calls, self.task, self.tools = run, calls, task, tools
        self.count, self.last, self.latest = 0, 0.0, None

    def _query(self, argv, root, check):
        self.count += 1
        command = {"argv": argv, "cwd": str(root), "env": {"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
                   "inherit_environment": False, "stdin": "DEVNULL",
                   "limits": {"step_seconds": 3, "cpu_threads": 1, "memory_mib": 256, "disk_mib": 1, "output_mib": 0.0625}}
        directory = root / f"query-{self.count:05d}"
        result = run_process(command, directory, check)
        if result["status"] != "succeeded":
            raise RunStopped(result["status"], "host query: " + result["reason"])
        return (directory / "stdout.log").read_text(encoding="utf-8")

    def sample(self, root, group=None, *, force=False, check=None):
        check = check or self.calls.check_interrupt
        check()
        if not force and time.monotonic() - self.last < 1:
            return self.latest
        memory = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key == "MemAvailable":
                memory[key] = int(value.split()[0]) / 1024
        require("MemAvailable" in memory, "host memory observation unavailable")
        if memory["MemAvailable"] < 1024:
            raise RunStopped("budget_exhausted", "host memory reserve below 1 GiB")
        if group is None and memory["MemAvailable"] < self.task["contract"]["execution"]["limits"]["memory_mib"] + 1024:
            raise RunStopped("failed", "insufficient host memory for the declared budget")
        if shutil.disk_usage(self.run.path).free < 256 * 1024 * 1024:
            raise RunStopped("budget_exhausted", "disk free reserve below 256 MiB")
        smi = self.tools["nvidia_smi"]["path"]
        inventory = self._query([smi, "--query-gpu=uuid,pci.bus_id,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], root, check)
        apps = self._query([smi, "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], root, check)
        self.latest = gpu_snapshot(inventory, apps, self.task["resources"]["gpus"][0],
                                   own_pids=group_pids(group) if group else (),
                                   memory_mib=self.task["contract"]["execution"]["limits"]["gpu_memory_mib"])
        self.last = time.monotonic()
        return self.latest

    def verify_tools(self):
        for expected in self.tools.values():
            actual = file_identity(expected["path"], self.calls.check_interrupt, limit=128 * 1024 * 1024)
            require(actual == expected, "tool content changed: " + expected["path"])

    @staticmethod
    def check_config(scratch):
        for path in (Path("/etc/llama.cpp/config.ini"), scratch / "config/llama.cpp/config.ini",
                     scratch / "home/.config/llama.cpp/config.ini"):
            require(not path.exists() and not path.is_symlink(), "implicit llama.cpp config must be absent: " + str(path))
