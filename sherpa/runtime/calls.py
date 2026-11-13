from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .models import ToolResult
from .store import atomic_json, digest, now, process_identity, read_json

TOOLS = {
    "inspect": {"purpose": "Collect a synthetic observation", "parameters": ["delay", "child"]},
    "fail": {"purpose": "Produce a controlled failure", "parameters": ["delay", "child"]},
    "evaluate": {"purpose": "Return synthetic behavior and performance results",
                 "parameters": ["delay", "child", "metric", "behavior_passed"]},
}


class RunStopped(RuntimeError):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status, self.reason = status, reason


class Calls:
    def __init__(self, run, seconds=60, max_calls=20):
        self.run = run
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.max_calls, self.count = max_calls, 0

    def check(self):
        self.check_interrupt()
        if self.count >= self.max_calls:
            raise RunStopped("budget_exhausted", "call budget exhausted")

    def check_interrupt(self):
        cancel = self.run.cancellation()
        if cancel:
            raise RunStopped("cancelled", cancel["reason"])
        if time.monotonic() >= self.deadline:
            raise RunStopped("timed_out", "wall-clock budget exhausted")

    def new(self, kind, name, inputs, experiment_id=None, caused_by=None, large_model=False, reason=None):
        self.check()
        self.count += 1
        call_id = f"call-{self.count:04d}-{uuid.uuid4().hex[:6]}"
        directory = self.run.path / "calls" / call_id
        atomic_json(directory / "input.json", inputs)
        record = {"id": call_id, "kind": kind, "name": name, "run_id": self.run.id,
                  "experiment_id": experiment_id, "caused_by": caused_by,
                  "status": "running", "started_at": now(), "ended_at": None,
                  "input": f"calls/{call_id}/input.json", "input_hash": digest(inputs),
                  "large_model_intervention": large_model, "intervention_reason": reason,
                  "attempt": 1, "process": None, "seconds": 0}
        self.run.save("calls", call_id, record, "call_started")
        self.run.update("active_call_changed", active_call=call_id)
        return record, directory

    def cost_summary(self):
        records = self.run.records("calls")
        agents = [r for r in records if r["kind"] == "agent" and not r.get("synthetic", False)]
        unknown = sum(r.get("cost_usd") is None for r in agents)
        known = sum(r.get("cost_usd") or 0 for r in agents)
        return {"seconds": round(time.monotonic() - self.started, 6), "calls": self.count,
                "known_model_cost_usd": known, "unpriced_model_calls": unknown,
                "model_cost_usd": None if unknown else known}

    def finish(self, record, directory, status, output, started):
        atomic_json(directory / "output.json", output)
        record.update(status=status, ended_at=now(), seconds=round(time.monotonic() - started, 6),
                      output=f"calls/{record['id']}/output.json", output_hash=digest(output))
        self.run.save("calls", record["id"], record, "call_finished")
        self.run.update("active_call_changed", active_call=None, cost=self.cost_summary())
        return ToolResult(status, output, record["id"])

    def decide(self, agent, context):
        started = time.monotonic()
        record, directory = self.new("agent", agent.name, context)
        record.update(model=agent.model, synthetic=True, usage={"input_tokens": None, "output_tokens": None},
                      cost_usd=0)
        try:
            action = agent.decide(context)
            parsed = action.snapshot()
            atomic_json(directory / "raw.json", {"synthetic": True, "response": parsed})
            self.finish(record, directory, "succeeded", parsed, started)
            return action, record["id"]
        except Exception as error:
            self.finish(record, directory, "failed", {"error": str(error)}, started)
            raise

    def tool(self, name, parameters, timeout_s=10, experiment_id=None, caused_by=None):
        started = time.monotonic()
        inputs = {"tool": name, "parameters": parameters, "timeout_s": timeout_s}
        record, directory = self.new("tool", name, inputs, experiment_id, caused_by)
        record.update(tool_version="synthetic-v1", synthetic=True)
        status, output, proc = "failed", {}, None
        try:
            self.validate(name, parameters, timeout_s)
            env = {"PATH": os.defpath, "HOME": str(directory), "TMPDIR": str(directory),
                   "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
            with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
                proc = subprocess.Popen([sys.executable, "-I", str(Path(__file__).with_name("worker.py")),
                                         str(directory / "input.json"), str(os.getpid())],
                                        cwd=directory, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                record["process"] = process_identity(proc.pid)
                record["logs"] = [f"calls/{record['id']}/{name}.log" for name in ("stdout", "stderr")]
                self.run.save("calls", record["id"], record, "tool_process_started")
                deadline = min(started + timeout_s, self.deadline)
                previous = None
                while proc.poll() is None:
                    if self.run.cancellation():
                        status, output = "cancelled", {"reason": "cancellation requested"}
                        break
                    if time.monotonic() >= deadline:
                        status, output = "timed_out", {"reason": "tool or run deadline exceeded"}
                        break
                    progress_path = directory / "progress.json"
                    if progress_path.exists():
                        progress = read_json(progress_path)
                        if progress != previous:
                            self.run.event("tool_progress", call_id=record["id"], progress=progress)
                            previous = progress
                    time.sleep(0.05)
                else:
                    if proc.returncode == 0:
                        result = directory / "result.json"
                        if result.stat().st_size > 1024 * 1024:
                            raise ValueError("tool output exceeds limit")
                        status, output = "succeeded", read_json(result)
                    else:
                        status, output = "failed", {"exit_code": proc.returncode,
                                                    "reason": "tool failed; inspect stderr.log"}
        except Exception as error:
            output = {"error": f"{type(error).__name__}: {error}"}
        finally:
            if proc:
                cleaned = self.cleanup(proc)
                record["cleanup_complete"] = cleaned
                if not cleaned:
                    status = "cleanup_failed"
            result = self.finish(record, directory, status, output, started)
        if status == "cleanup_failed":
            raise RunStopped("cleanup_failed", "owned process group has not stopped; no further work allowed")
        return result

    @staticmethod
    def validate(name, parameters, timeout_s):
        if name not in TOOLS:
            raise ValueError("unknown tool; arbitrary command execution is not supported")
        if set(parameters) - set(TOOLS[name]["parameters"]):
            raise ValueError("unknown tool parameters")
        for value in (parameters.get("delay", 0), timeout_s):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid duration")
        if timeout_s == 0 or parameters.get("delay", 0) > 120:
            raise ValueError("duration exceeds prototype limits")
        if "child" in parameters and not isinstance(parameters["child"], bool):
            raise ValueError("child must be boolean")
        if "behavior_passed" in parameters and not isinstance(parameters["behavior_passed"], bool):
            raise ValueError("behavior_passed must be boolean")
        metric = parameters.get("metric", 0)
        if isinstance(metric, bool) or not isinstance(metric, (float, int)) or not math.isfinite(metric):
            raise ValueError("metric must be finite")

    @staticmethod
    def cleanup(proc):
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                continue
            if Calls.group_gone(proc.pid):
                return True
            time.sleep(0.05)
        return Calls.group_gone(proc.pid)

    @staticmethod
    def group_gone(group):
        for path in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = path.read_text().rsplit(")", 1)[1].split()
                if int(fields[2]) == group and fields[0] not in {"Z", "X"}:
                    return False
            except (OSError, ValueError, IndexError):
                continue
        return True
