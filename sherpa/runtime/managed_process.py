"""Bounded process transport; only trusted controllers may supply commands."""
from __future__ import annotations

import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

from .calls import Calls, RunStopped
from .execution_contract import load_document
from .store import atomic_json, process_identity

SUPERVISOR = Path(__file__).with_name("execution_supervisor.py")


def group_pids(group):
    result = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group and fields[0] not in ("Z", "X"):
                result.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return result


def group_rss_mib(group):
    total = 0
    for pid in group_pids(group):
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) / 1024
        except FileNotFoundError:
            continue
    return total


def disk_bytes(root):
    total = 0
    for directory, _, files in os.walk(root, followlinks=False):
        for name in files:
            try:
                total += (Path(directory) / name).lstat().st_size
            except FileNotFoundError:
                continue
    return total


def run_process(command, directory, check, *, on_start=lambda identity: None, monitor=lambda group, poll: None):
    """Record and clean one group. Returns outcome; caller records a Call first.

    Output cap bounds captured stdout+stderr, RSS is sampled, and neither is
    a security sandbox. Cleanup failure takes precedence over all other errors.
    """
    limits = command["limits"]
    for key in ("step_seconds", "cpu_threads", "memory_mib", "disk_mib", "output_mib"):
        value = limits[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("invalid process limit: " + key)
    if command.get("inherit_environment") is not False or command.get("stdin") != "DEVNULL":
        raise ValueError("explicit environment and disabled stdin required")
    directory.mkdir(parents=True, exist_ok=False)
    status_path = directory / "child.json"
    atomic_json(directory / "request.json", {"command": command, "status_path": str(status_path)})
    started, proc, cleaned = time.monotonic(), None, True
    status, reason, child = "failed", "process did not start", {}
    captured, observed, peak = {"stdout": 0, "stderr": 0}, {"stdout": 0, "stderr": 0}, 0
    cap = int(limits["output_mib"] * 1024 * 1024)
    errors = []
    try:
        with selectors.DefaultSelector() as sel, (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err:
            try:
                check()
                proc = subprocess.Popen([sys.executable, "-I", "-B", str(SUPERVISOR), str(os.getpid()), str(directory / "request.json")],
                    cwd=directory, env={"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"},
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                cleaned = False
                on_start(process_identity(proc.pid))
                for pipe, name, dest in ((proc.stdout, "stdout", out), (proc.stderr, "stderr", err)):
                    os.set_blocking(pipe.fileno(), False)
                    sel.register(pipe, selectors.EVENT_READ, (name, dest))
                def poll():
                    nonlocal peak
                    check()
                    if time.monotonic() - started >= limits["step_seconds"]:
                        raise RunStopped("timed_out", "step timeout")
                    rss = group_rss_mib(proc.pid)
                    peak = max(peak, rss)
                    if rss > limits["memory_mib"]:
                        raise RunStopped("budget_exhausted", "process group memory limit exceeded")
                while True:
                    poll()
                    monitor(proc.pid, poll)
                    poll()
                    if status_path.exists():
                        child = load_document(status_path)
                    ready = sel.select(0.025)
                    for key, _ in ready:
                        try:
                            block = os.read(key.fileobj.fileno(), 16384)
                        except BlockingIOError:
                            continue
                        if not block:
                            sel.unregister(key.fileobj)
                            continue
                        name, dest = key.data
                        observed[name] += len(block)
                        amount = min(len(block), max(0, cap - sum(captured.values())))
                        dest.write(block[:amount])
                        dest.flush()
                        captured[name] += amount
                        if sum(observed.values()) > cap:
                            raise RunStopped("budget_exhausted", "output limit exceeded")
                    if child.get("exit_code") is not None and (not sel.get_map() or not ready):
                        status = "succeeded" if child["exit_code"] == 0 else "failed"
                        reason = "completed" if status == "succeeded" else "nonzero exit"
                        break
                    if proc.poll() is not None and child.get("exit_code") is None:
                        raise RuntimeError("supervisor exited without child outcome")
            finally:
                if proc is not None:
                    try:
                        cleaned = Calls.cleanup(proc) is True
                    except (Exception, KeyboardInterrupt) as error:
                        errors.append(f"cleanup: {type(error).__name__}: {error}")
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait(timeout=2)
                        except (OSError, subprocess.TimeoutExpired):
                            pass
                    finally:
                        proc.stdout.close()
                        proc.stderr.close()
    except RunStopped as error:
        status, reason = error.status, error.reason
    except KeyboardInterrupt:
        status, reason = "cancelled", "keyboard interrupt"
    except Exception as error:
        status, reason = "failed", f"{type(error).__name__}: {error}"
    if not cleaned:
        status, reason = "cleanup_failed", "owned process group cleanup not confirmed"
    elif status != "cleanup_failed":
        try:
            check()
        except RunStopped as error:
            status, reason = error.status, error.reason
        except KeyboardInterrupt:
            status, reason = "cancelled", "keyboard interrupt during cleanup"
    result = {"command": command, "status": status, "reason": reason, "exit_code": child.get("exit_code"),
              "identity": process_identity(proc.pid) if proc else None, "group": proc.pid if proc else None,
              "cleanup_complete": cleaned, "captured_bytes": captured, "observed_bytes": observed,
              "peak_rss_mib": peak, "seconds": time.monotonic() - started, "errors": errors}
    atomic_json(directory / "process.json", result)
    return result
