"""Fixed E2 subprocess fixture. Never compiles or loads user code/models."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    memory = request["memory_mib"] * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    fault = request["fault"]
    if fault["child"]:
        guard = ("import ctypes,os,signal,sys,time; owner=int(sys.argv[1]); "
                 "libc=ctypes.CDLL(None); code=libc.prctl(1,signal.SIGKILL,0,0,0); "
                 "sys.exit(1) if code or os.getppid()!=owner else None; time.sleep(120)")
        child = subprocess.Popen([sys.executable, "-I", "-B", "-c", guard, str(os.getpid())])
        write(Path("child.json"), {"pid": child.pid})
    write(Path("ready.json"), {"pid": os.getpid(), "synthetic": True})
    for stream, count in ((sys.stdout, fault["stdout_bytes"]), (sys.stderr, fault["stderr_bytes"])):
        while count:
            chunk = min(count, 8192)
            stream.buffer.write(b"x" * chunk)
            stream.flush()
            count -= chunk
    time.sleep(fault["delay"])
    if fault["exit_code"]:
        return fault["exit_code"]
    kind = request["action"]["kind"]
    if kind == "probe":
        value = {"synthetic": True, "device": "cpu-fixture", "tools": ["fixture-python"]}
    elif kind == "build":
        value = {"synthetic": True, "task_hash": request["task_hash"], "format": "fixture-artifact-v1"}
        if not fault["missing_artifact"]:
            write(Path("artifact.json"), value)
    else:
        value = {"synthetic": True, "task_hash": request["task_hash"], "request_hash": request["request_hash"],
                 "device": "cpu-fixture", "output": request["text"].upper()}
        if fault["wrong_output"]:
            value["output"] = "deliberately incorrect fixture response"
    if fault["malformed"]:
        Path("result.json").write_text("not json")
    else:
        write(Path("result.json"), value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
