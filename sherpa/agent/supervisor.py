"""Lifetime supervisor for a CLI attempt; never detaches from its owned process group."""
import ctypes
import json
from pathlib import Path
import os
import resource
import signal
import subprocess
import sys


def stop(signum, frame):
    os.killpg(os.getpgrp(), signal.SIGKILL)


def main():
    owner = int(sys.argv[1])
    signal.signal(signal.SIGTERM, stop)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot establish parent-death protection")
    if os.getppid() != owner:
        stop(None, None)
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    status_path = Path(os.environ["SHERPA_AGENT_PROCESS_STATUS"]) if os.environ.get("SHERPA_AGENT_PROCESS_STATUS") else None
    def report(value):
        if status_path:
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(value) + "\n")
            temporary.replace(status_path)
    child = subprocess.Popen(sys.argv[2:])
    report({"child_pid": child.pid, "child_returncode": None, "child_signal": None})
    result = child.wait()
    report({"child_pid": child.pid, "child_returncode": result,
            "child_signal": signal.Signals(-result).name if result < 0 else None})
    return 128 - result if result < 0 else result


if __name__ == "__main__":
    sys.exit(main())
