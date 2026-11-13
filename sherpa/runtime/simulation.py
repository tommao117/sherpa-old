"""Synthetic scenario only. These values and decisions never represent real inference results."""
from __future__ import annotations

from pathlib import Path

from .engine import Engine
from .models import Action, Experiment, Task
from .store import Store, atomic_json, digest
from . import workspace


class ScriptedAgent:
    name = "scripted"
    model = "mock-small-v1"

    def __init__(self, scenario, delay):
        self.scenario, self.delay = scenario, delay

    def decide(self, context):
        history = context["history"]
        if self.scenario in {"slow", "timeout"}:
            if not history:
                return Action("tool", "inspect", {"delay": self.delay, "child": True}, "observe cancellation/timeout")
            return Action("stop", reason="synthetic observation finished")
        if not history:
            return Action("tool", "fail" if self.scenario == "failure" else "inspect", reason="collect evidence")
        if history[-1]["status"] == "failed":
            return Action("tool", "inspect", reason="previous collection failed; try alternate mock observation")
        if not context["experiments"]:
            return Action("experiment", parameters={"metric": 110, "behavior_passed": self.scenario != "invalid"},
                          reason="test one synthetic candidate")
        return Action("stop", reason="demo plan complete; no real optimization was attempted")


class SimulationEngine(Engine):
    def setup(self):
        self.source = workspace.prepare(self.task.target, self.run.path / "workspace" / "framework")
        self.run.update("target_prepared", source=self.source, phase="baseline")
        self.measurement = {"id": "mock-contract-v1", "synthetic": True,
                            "workload": {"requests": ["fixed-input"], "concurrency": 1},
                            "behavior": "mock exact check", "metric": "synthetic_throughput"}
        atomic_json(self.run.path / "baseline" / "measurement-contract.json", self.measurement)
        baseline = self.calls.tool("evaluate", {"metric": 100})
        self.calls.check_interrupt()
        if baseline.status != "succeeded" or not baseline.output.get("behavior_passed"):
            raise RuntimeError("synthetic baseline could not be established")
        reference = {"commit": self.source["commit"], "environment": "mock-python-stdlib-v1",
                     "configuration": {}, "resources": self.task.resources, "metric": 100,
                     "measurement_contract": digest(self.measurement), "measurement_call": baseline.call_id,
                     "synthetic": True}
        atomic_json(self.run.path / "baseline" / "reference.json", reference)
        self.run.update("baseline_established", baseline=reference, best=reference, phase="deciding")

    def experiment(self, action, decision_id):
        best = self.run.state()["best"]
        exp_id = f"experiment-{len(self.run.records('experiments')) + 1:04d}"
        path = Path(self.source["path"])
        workspace.git(path, "checkout", "--detach", best["commit"])
        if (path / "fixture.py").read_text() != "VALUE = 1\n":
            raise ValueError("demo refuses to modify anything except its exact synthetic fixture")
        (path / "fixture.py").write_text("VALUE = 2\n")
        candidate = workspace.commit_all(path, "Synthetic candidate for runtime test")
        candidate_ref = {"commit": candidate, "environment": "mock-python-stdlib-v1",
                         "configuration": {}, "resources": self.task.resources,
                         "measurement_contract": digest(self.measurement), "synthetic": True}
        exp = Experiment(exp_id, action.reason, decision_id, best, candidate_ref, self.measurement).snapshot()
        self.run.save("experiments", exp_id, exp, "experiment_created")
        difference = workspace.git(path, "diff", best["commit"], candidate)
        (self.run.path / "experiments" / exp_id / "change.patch").write_text(difference)
        exp["status"] = "validating"
        self.run.save("experiments", exp_id, exp, "experiment_phase_changed")
        self.run.update("phase_changed", phase="validating", current_experiment=exp_id)
        behavior = self.calls.tool("evaluate", {"metric": 0, "behavior_passed": action.parameters["behavior_passed"]},
                                   experiment_id=exp_id, caused_by=decision_id)
        exp["calls"].append(behavior.call_id)
        self.run.save("experiments", exp_id, exp, "behavior_call_returned")
        self.calls.check_interrupt()
        if behavior.status != "succeeded":
            exp.update(status="incomplete", conclusion="inconclusive")
        elif not behavior.output.get("behavior_passed"):
            exp.update(status="completed", conclusion="rejected", behavior_passed=False)
        else:
            self.calls.check()
            exp["status"] = "measuring"
            self.run.save("experiments", exp_id, exp, "experiment_phase_changed")
            self.run.update("phase_changed", phase="measuring")
            measured = self.calls.tool("evaluate", {"metric": action.parameters["metric"]},
                                       experiment_id=exp_id, caused_by=behavior.call_id)
            exp["calls"].append(measured.call_id)
            self.run.save("experiments", exp_id, exp, "measurement_call_returned")
            self.calls.check_interrupt()
            clean = not workspace.git(path, "status", "--porcelain")
            if not clean or workspace.git(path, "rev-parse", "HEAD") != candidate:
                raise ValueError("candidate changed during validation/measurement")
            if measured.status == "succeeded" and measured.output.get("measurement_complete"):
                metric = measured.output["metric"]
                accepted = metric > best["metric"]
                exp.update(status="completed", conclusion="accepted" if accepted else "not_improved",
                           behavior_passed=True, metric=metric, synthetic=True)
                if accepted:
                    self.run.update("best_changed", best={**candidate_ref, "metric": metric,
                                                         "experiment_id": exp_id, "measurement_call": measured.call_id})
            else:
                exp.update(status="incomplete", conclusion="inconclusive")
        self.run.save("experiments", exp_id, exp, "experiment_finished")
        workspace.git(path, "checkout", "--detach", self.run.state()["best"]["commit"])
        self.run.update("phase_changed", phase="deciding", current_experiment=None)
        self.history.append({"experiment_id": exp_id, "status": exp["status"], "conclusion": exp["conclusion"]})

    def finalize(self):
        if self.run.state()["best"]:
            atomic_json(self.run.path / "deliverable" / "result.json",
                        {"synthetic": True, "best": self.run.state()["best"],
                         "warning": "Runtime demonstration only; no model or performance optimization ran."})


def demo(store: Store, scenario="normal", delay=0.1, seconds=60, announce=None):
    if scenario not in {"normal", "failure", "invalid", "slow", "timeout"}:
        raise ValueError("unknown demonstration scenario")
    if not 0 <= delay <= 120:
        raise ValueError("delay must be between 0 and 120 seconds")
    source = store.root / "cache" / "mock-source-v1"
    source.mkdir(parents=True, exist_ok=True)
    fixture = source / "fixture.py"
    if fixture.exists() and fixture.read_text() != "VALUE = 1\n":
        raise ValueError("mock source cache changed")
    fixture.write_text("VALUE = 1\n")
    task = Task("mock-task", {"kind": "local", "path": str(source)}, {"kind": "synthetic", "id": "no-model"},
                workload={"source": "synthetic", "resolved": True, "concurrency": 1},
                budget={"seconds": seconds, "max_calls": 20, "model_cost_usd": 0})
    run = store.create_run(task)
    if announce:
        announce(run.id)
    state = SimulationEngine(run, task, ScriptedAgent(scenario, 120 if scenario == "timeout" else delay)).execute()
    return run, state
