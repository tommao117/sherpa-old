"""Bounded, synthetic tools for runtime verification. No models, network or GPU."""
from __future__ import annotations

import ctypes
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path


def write(path: Path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def parent_died(signum, frame):
    os.killpg(os.getpgrp(), signal.SIGKILL)


def main():
    request_path = Path(sys.argv[1])
    owner = int(sys.argv[2])
    signal.signal(signal.SIGTERM, parent_died)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot enable parent-death protection")
    if os.getppid() != owner:
        parent_died(None, None)
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024 * 1024,) * 2)
    request = json.loads(request_path.read_text())
    tool, parameters = request["tool"], request["parameters"]
    directory = request_path.parent
    started = time.monotonic()
    write(directory / "progress.json", {"phase": "executing", "message": "synthetic tool started"})
    if parameters.get("child"):
        child = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(120)"])
        write(directory / "child.json", {"pid": child.pid})
    delay = parameters.get("delay", 0)
    while time.monotonic() - started < delay:
        time.sleep(min(0.05, delay))
        write(directory / "progress.json", {"phase": "executing", "elapsed": round(time.monotonic() - started, 1)})
    if tool == "fail":
        print("synthetic failure: sample unavailable", file=sys.stderr, flush=True)
        return 2
    if tool == "evaluate":
        output = {"synthetic": True, "behavior_passed": parameters.get("behavior_passed", True),
                  "measurement_complete": True, "metric": parameters.get("metric", 110),
                  "limitations": ["Synthetic result; not evidence of inference speedup."]}
    else:
        output = {"synthetic": True, "summary": "mock observation collected", "complete": True}
    write(directory / "result.json", output)
    write(directory / "progress.json", {"phase": "complete"})
    print("synthetic tool completed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
