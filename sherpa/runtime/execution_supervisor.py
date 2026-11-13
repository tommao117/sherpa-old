"""Keep parent-death protection alive until the controller cleans the group."""
import ctypes
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time


def stop(signum, frame):
    os.killpg(os.getpgrp(), signal.SIGKILL)


def main():
    owner = int(sys.argv[1])
    signal.signal(signal.SIGTERM, stop)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "parent-death protection unavailable")
    if os.getppid() != owner:
        stop(None, None)
    request = json.loads(Path(sys.argv[2]).read_text())
    command, destination = request["command"], Path(request["status_path"])
    limits = command["limits"]
    cpu = max(1, math.ceil(limits["step_seconds"] * limits["cpu_threads"]))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits["disk_mib"] * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    def report(value):
        temp = destination.with_suffix(".tmp")
        temp.write_text(json.dumps(value))
        temp.replace(destination)
    child = subprocess.Popen(command["argv"], cwd=command["cwd"], env=command["env"], stdin=subprocess.DEVNULL)
    report({"pid": child.pid, "exit_code": None})
    code = child.wait()
    report({"pid": child.pid, "exit_code": code})
    os.close(1)
    os.close(2)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
