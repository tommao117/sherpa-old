from __future__ import annotations

import os
import time
from typing import Protocol

from .calls import Calls, RunStopped, TOOLS
from .models import Action, Task
from .store import locked, now, process_identity
from . import workspace


class DecisionAgent(Protocol):
    name: str
    model: str

    def decide(self, context: dict) -> Action: ...


class Engine:
    """Serial state machine. Scenario-specific setup and experiment logic are explicit hooks."""

    def __init__(self, run, task: Task, agent: DecisionAgent):
        self.run, self.task, self.agent = run, task, agent
        self.calls = Calls(run, seconds=task.budget["seconds"], max_calls=task.budget["max_calls"])
        self.history = []

    def context(self):
        state = self.run.state()
        return {"task": state["task"], "phase": state["phase"], "best": state["best"],
                "baseline": state["baseline"], "history": self.history[-10:],
                "experiments": self.run.records("experiments"), "tools": TOOLS,
                "budget": {"seconds_left": max(0, self.calls.deadline - time.monotonic()),
                           "calls_left": self.calls.max_calls - self.calls.count}}

    def setup(self):
        raise NotImplementedError("a scenario must establish its baseline")

    def experiment(self, action, decision_id):
        raise NotImplementedError("a scenario must implement candidate verification")

    def finalize(self):
        pass

    def execute(self):
        with self.run.store.exclusive(), locked(self.run.path / ".execution.lock", blocking=False):
            if self.run.state()["status"] != "created":
                raise ValueError("run may only execute once")
            self.run.update("run_started", status="running", phase="preparing", owner=process_identity(os.getpid()))
            outcome, reason = "completed", "finished"
            try:
                with workspace.interruptible(self.calls.check_interrupt, scratch=self.run.path):
                    version = workspace.capture_framework(self.run.store.project, self.run.path / "framework-snapshot")
                    self.run.update("framework_frozen", version=version)
                    self.calls.check()
                    self.setup()
                    while True:
                        self.calls.check()
                        action, decision = self.calls.decide(self.agent, self.context())
                        if action.kind == "stop":
                            reason = action.reason
                            break
                        if action.kind == "experiment":
                            if self.run.state()["baseline"] is None:
                                raise ValueError("cannot experiment before baseline is established")
                            self.experiment(action, decision)
                        else:
                            self.run.update("phase_changed", phase="investigating")
                            result = self.calls.tool(action.tool, action.parameters, timeout_s=30, caused_by=decision)
                            self.history.append({"call_id": result.call_id, "tool": action.tool,
                                                 "status": result.status, "output": result.output})
                            self.run.update("phase_changed", phase="deciding")
            except RunStopped as error:
                outcome, reason = error.status, error.reason
            except KeyboardInterrupt:
                outcome, reason = "cancelled", "keyboard interrupt"
            except Exception as error:
                outcome, reason = "failed", f"{type(error).__name__}: {error}"
            finally:
                self.run.update("phase_changed", phase="finalizing")
                try:
                    self.finalize()
                except Exception as error:
                    outcome, reason = "failed", f"finalization failed: {error}"
                for exp in self.run.records("experiments"):
                    if exp["status"] not in {"completed", "incomplete"}:
                        exp.update(status="incomplete", conclusion="inconclusive", stop_reason=reason)
                        self.run.save("experiments", exp["id"], exp, "experiment_interrupted")
                self.run.update("run_finished", status=outcome, phase="finished", stop_reason=reason, ended_at=now(),
                                active_call=None, current_experiment=None,
                                cost=self.calls.cost_summary())
        return self.run.state()
