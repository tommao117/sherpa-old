"""Resources + experiment lifecycle (v0: one local GPU, serial).

Experiment state machine (owned here; engine only receives transition events):
    pending -> preparing -> running -> measured -> gated -> finished
                   |           |                              \\-> (engine) concluded
                   \\-> failed  \\-> killed
"""
from __future__ import annotations

import time
from pathlib import Path

from sherpa.kernel import gates as G
from sherpa.kernel import measure as M
from sherpa.model import physics
from sherpa.model.ledger import Ledger
from sherpa.runtime import watchers, workspace
from sherpa.runtime.backends import Request
from sherpa.util import ASSETS, now_iso, write_yaml


class Scheduler:
    def __init__(self, task, adapter, env, arch, hw, wl, sampling, model_path: Path, bin_dir: Path, backend,
                 bus=None):
        self.task, self.adapter, self.env = task, adapter, env
        self.arch, self.hw, self.wl, self.sampling = arch, hw, wl, sampling
        self.model_path, self.bin_dir, self.backend, self.bus = model_path, bin_dir, backend, bus
        self.gpus = [g["index"] for g in env["gpus"]][:1]   # v0: concurrency = 1
        self._port = 18100

    def _next_port(self) -> int:
        self._port += 1
        return self._port

    def _transition(self, exp: dict, state: str, **extra) -> None:
        exp["state"] = state
        exp.setdefault("timeline", []).append({"state": state, "time": now_iso(), **extra})
        self.task.save_experiment(exp)

    # ------------------------------------------------------------ measuring
    def theory_ledger(self, eff: dict) -> Ledger:
        B = min(eff.get("slots") or 1, self.wl.concurrency)
        b = physics.bounds(self.arch, self.hw, self.wl, decode_batch=B, kv_dtype=eff.get("kv_dtype") or "f16")
        return Ledger.from_bounds(b, meta={"effective_config": eff})

    def measure(self, label: str, config: dict, incumbent_mean: float | None = None, on_alert=None) -> dict:
        """Deploy -> greedy tokens -> bench xR (watched) -> profile -> stop. Returns raw results."""
        log = self.task.logs / f"{label}-server.log"
        srv = self.adapter.start(self.bin_dir, self.model_path, config, self._next_port(), log, gpu=self.gpus[0])
        out = {"label": label, "config": config, "killed": None}
        try:
            out["effective"] = self.adapter.effective_config(srv)
            out["tokens"] = self.adapter.greedy(srv, G.PROMPTS, G.GREEDY_TOKENS)
            reps = []
            for i in range(self.wl.repeats):
                r = self.adapter.bench_once(srv, self.wl, i, self.sampling)
                reps.append(r)
                alert = watchers.after_repeat(self.wl.primary_metric, r, incumbent_mean, self.wl.higher_is_better,
                                              srv.alive())
                if alert:
                    verdict = alert.get("auto") or (on_alert(alert, reps) if on_alert else "continue")
                    if verdict == "kill":
                        out["killed"] = alert
                        break
            out["repeats"] = reps
            out["measurement"] = M.make(reps, self.wl.primary_metric, meta={"label": label})
        finally:
            self.adapter.stop(srv)
        if out["killed"]:
            return out
        if self.env["capabilities"].get("kernel_trace") == "cupti-inject":
            tlog = self.task.logs / f"{label}-server-traced.log"
            trace = self.task.artifacts / f"{label}.trace.txt"
            trace.parent.mkdir(parents=True, exist_ok=True)
            srv = self.adapter.start(self.bin_dir, self.model_path, config, self._next_port(), tlog,
                                     trace_out=trace, gpu=self.gpus[0])
            try:
                out["profile"] = self.adapter.profile(srv, self.wl, self.sampling, trace)
            finally:
                self.adapter.stop(srv)
            for k, v in (out["profile"].get("observed_config") or {}).items():
                if v is not None and out["effective"].get(k) in (None, "unresolved"):
                    out["effective"][k] = v
            out["profile"]["tracer_overhead_pct"] = round(
                (1 - out["profile"]["bench"][self.wl.primary_metric]
                 / out["measurement"]["summary"][self.wl.primary_metric]["mean"]) * 100, 1)
        led = self.theory_ledger(out["effective"])
        if "profile" in out:
            led.fill_measured(out["profile"]["nodes"])
        else:
            led.fill_measured({"request.time_per_token": 1000.0 / out["measurement"]["summary"][self.wl.primary_metric]["mean"]})
        out["ledger"] = led
        return out

    # ---------------------------------------------------------- experiments
    def run(self, exp: dict, incumbent: dict, baseline_tokens: list, on_alert) -> dict:
        wt = self.task.framework
        exp["base"] = {"config": dict(incumbent["config"]), "commit": incumbent["commit"],
                       "metric": incumbent["metric"]}
        self._transition(exp, "preparing")
        action = exp["action"]
        config = {**incumbent["config"], **(action.get("knobs") or {})}
        exp["config"] = config
        if action["type"] == "code":
            try:
                exp["execution"] = self._apply_code(exp)
            except Exception as e:  # noqa: BLE001
                exp["error"] = f"code change failed: {e}"
                # the executor may have edited and compiled: restore BOTH source and build artifacts
                # so that bin_dir keeps matching the incumbent commit (devlog IMP-9)
                workspace.reset_to(wt, incumbent["commit"])
                self.rebuild()
                self._transition(exp, "failed")
                return exp
        self._transition(exp, "running")
        try:
            res = self.measure(exp["id"], config, incumbent_mean=incumbent["metric"], on_alert=on_alert)
        except Exception as e:  # noqa: BLE001
            exp["error"] = f"measure failed: {e}"
            if action["type"] == "code":
                workspace.reset_to(wt, incumbent["commit"])
                self.rebuild()
            self._transition(exp, "failed")
            return exp
        exp["effective"] = res.get("effective")
        if res["killed"]:
            exp["killed"] = res["killed"]
            exp["measurement"] = M.make(res.get("repeats", []), self.wl.primary_metric)
            if action["type"] == "code":
                workspace.reset_to(wt, incumbent["commit"])
                self.rebuild()
            self._transition(exp, "killed")
            return exp
        exp["measurement"] = res["measurement"]
        led: Ledger = res["ledger"]
        write_yaml(self.task.ledger_dir / f"{exp['id']}.yaml", led.to_dict())
        exp["ledger"] = f"ledger/{exp['id']}.yaml"
        exp["ledger_summary"] = {"measured": led.measured, "theory": led.theory}
        exp["top_kernels"] = (res.get("profile") or {}).get("top_kernels", [])[:6]
        self._transition(exp, "measured")
        lossy = self.is_lossy(config)
        exp["lossy"] = lossy
        exp["gates"] = {"correctness": G.correctness(baseline_tokens, res["tokens"], lossy),
                        "reproducible": G.reproducible(res["measurement"])}
        exp["compare"] = M.compare(incumbent["measurement"], res["measurement"], self.wl.higher_is_better)
        self._transition(exp, "gated")
        self._transition(exp, "finished")
        return exp

    def is_lossy(self, config: dict) -> bool:
        """Derived from adapter knob metadata, never from the planner's claim."""
        for k, v in config.items():
            allowed = (self.adapter.knobs.get(k) or {}).get("lossy_unless")
            if allowed and v not in allowed:
                return True
        return False

    def rebuild(self) -> None:
        self.adapter.build(self.task.framework, self.task.logs / "build.log")

    def _apply_code(self, exp: dict) -> dict:
        wt = self.task.framework
        playbook = (ASSETS / "playbooks" / "code-change.md").read_text()
        codemap_p = ASSETS / "adapters" / self.adapter.name / "codemap.md"
        req = Request(
            id=f"{exp['id']}-exec", kind="execute", point="code_change", workdir=wt,
            instructions=playbook + "\n\n## Codemap\n" + (codemap_p.read_text() if codemap_p.exists() else ""),
            brief={"hypothesis": exp["hypothesis"], "instructions": exp["action"].get("instructions"),
                   "node": exp.get("node"), "framework": self.adapter.name, "worktree": str(wt),
                   "build_hint": "cmake --build build -j --target llama-server (sherpa rebuilds after you finish)"},
            schema={"type": "object", "required": ["status", "summary"],
                    "properties": {"status": {"enum": ["done", "gave_up"]}, "summary": {"type": "string"},
                                   "files_changed": {"type": "array", "items": {"type": "string"}}}},
            budget={"max_turns": 60, "timeout_s": 7200})
        t0 = time.time()
        out = self.backend.call(req)
        if out["status"] != "done":
            raise RuntimeError(f"executor gave up: {out['summary']}")
        commit = workspace.commit_all(wt, f"sherpa {self.task.id} {exp['id']}: {exp['hypothesis'][:60]}")
        if not commit:
            raise RuntimeError("executor reported done but worktree is clean")
        self.rebuild()
        return {"commit": commit, "summary": out["summary"], "seconds": round(time.time() - t0, 1)}
