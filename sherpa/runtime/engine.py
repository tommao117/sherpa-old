"""Engine: routes events to decision points (assets/policies/hooks.yaml), executes structured decisions,
enforces invariants. Mechanism only; all judgement is in policies.

Event chain (mechanism events are handled here; hooked events go to decision points):
  task_start --setup--> baseline_ready --plan--> slot_free --run--> experiment_{finished,killed,failed}
  --analyze--> experiment_concluded --> (slot_free | round_done) --judge--> (judge_continue --plan--> ... | task_concluded)
  task_concluded --deliver--> done
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from sherpa.kernel import facts as F
from sherpa.kernel import gates as G
from sherpa.kernel import invariants as I
from sherpa.kernel import knowledge as K
from sherpa.model import constraints, features, physics
from sherpa.model.ledger import Ledger
from sherpa.model.specs import ArchSpec, HwSpec, Workload
from sherpa.runtime import adapter as adapter_mod
from sherpa.runtime import env as env_mod
from sherpa.runtime import workspace
from sherpa.runtime.backends import Request
from sherpa.runtime.events import Bus
from sherpa.runtime.scheduler import Scheduler
from sherpa.util import ASSETS, DATA, append_jsonl, now_iso, read_json, read_yaml, write_yaml

DEFAULT_SAMPLING = {"temperature": 0.8, "top_k": 40, "top_p": 0.95}


class Engine:
    def __init__(self, task, backend, args: dict):
        self.task, self.backend, self.args = task, backend, args
        self.bus = Bus(task.p("events.jsonl"))
        self.hooks = read_yaml(ASSETS / "policies" / "hooks.yaml")["decisions"]
        self.subscribers: dict[str, list[str]] = {}
        for point, spec in self.hooks.items():
            for ev in spec["triggers"]:
                self.subscribers.setdefault(ev, []).append(point)
        self.mech = {"task_start": self._setup, "slot_free": self._run_next,
                     "experiment_concluded": self._after_conclude, "task_concluded": self._deliver}
        self.queue: list[dict] = []
        self.experiments: list[dict] = []
        self.n_decisions = 0
        self.t_start = time.time()
        self.done = False

    # =============================================================== loop
    def run(self) -> None:
        self.bus.emit("task_start")
        while not self.done:
            ev = self.bus.pop()
            if ev is None:
                raise RuntimeError("event queue drained without task_concluded (broken event chain)")
            I.check_assets_frozen(self.frozen) if hasattr(self, "frozen") else None
            handled = False
            if ev.type in self.mech:
                self.mech[ev.type](ev)
                handled = True
            for point in self.subscribers.get(ev.type, []):
                getattr(self, f"_on_{point}")(ev)
                handled = True
            if not handled:
                self._default(ev)

    def _default(self, ev) -> None:
        """Fallbacks when no decision point subscribes (keeps the chain alive if policies are removed)."""
        if ev.type.startswith("experiment_") and ev.type != "experiment_concluded":
            self._conclude(ev.data["exp_id"], analysis=None)
        elif ev.type == "round_done":
            self.bus.emit("task_concluded", reason="no judge configured")
        elif ev.type == "baseline_ready":
            self.bus.emit("task_concluded", reason="no planner configured")

    # =========================================================== decisions
    def decide(self, point: str, trigger: str, brief: dict) -> dict:
        I.check_assets_frozen(self.frozen)
        spec = self.hooks[point]
        policy = (ASSETS / "policies" / spec["policy"]).read_text()
        schema = read_json(ASSETS / "policies" / spec["output"])
        self.n_decisions += 1
        rid = f"{self.n_decisions:03d}-{point}"
        req = Request(id=rid, kind="decide", point=point, instructions=policy, brief=brief, schema=schema,
                      model=spec.get("model", "strong"))
        t0 = time.time()
        out = self.backend.call(req)
        append_jsonl(self.task.decisions_dir / "decisions.jsonl",
                     {"id": rid, "point": point, "trigger": trigger, "time": now_iso(), "backend": self.backend.name,
                      "seconds": round(time.time() - t0, 1), "brief": brief, "output": out})
        return out

    # ============================================================== setup
    def _setup(self, ev) -> None:
        a, t = self.args, self.task
        self.frozen = I.assets_fingerprint()
        self.env = env_mod.probe()
        if not self.env["gpus"]:
            raise RuntimeError("no GPU found")
        self.hw = HwSpec.match(self.env["gpus"][0]["name"])
        model_dir = self._resolve_model(a["model"])
        self.arch = ArchSpec.from_hf(model_dir, weight_dtype=a.get("dtype", "bf16"))
        self.wl = Workload.load(a.get("workload", "throughput-default"))
        self.sampling = DEFAULT_SAMPLING
        fw_name, _, ref = a["framework"].partition("@")
        self.adapter = adapter_mod.load(fw_name, self.env)
        ref = ref or self.adapter.meta["default_ref"]
        cache = workspace.repo_cache(fw_name, self.adapter.meta["repo"])
        try:
            workspace.resolve(cache, ref)
        except RuntimeError:
            workspace.fetch(cache)
        wt = workspace.create_worktree(cache, ref, t.framework, f"sherpa/{t.id}")
        self.capacity = constraints.capacity(self.arch, self.hw, self.wl)
        ctx = {"task": t.id, "created": now_iso(),
               "inputs": {"model": a["model"], "model_dir": str(model_dir), "dtype": self.arch.weight_dtype,
                          "framework": fw_name, "ref": ref, "workload": self.wl.id, "hw": a.get("hw", "local")},
               "model_arch": self.arch.to_dict(), "hw": self.hw.__dict__, "workload": self.wl.to_dict(),
               "sampling": self.sampling, "framework": {"name": fw_name, **wt},
               "env_fingerprint": env_mod.fingerprint(self.env), "assets_fingerprint": self.frozen,
               "capacity": self.capacity, "precision": constraints.precision(self.arch.weight_dtype, "f16"),
               "budget": {"max_experiments": a.get("max_experiments", 6), "max_seconds": a.get("max_seconds", 3 * 3600)}}
        t.save_context(ctx)
        self.ctx = ctx
        # build + model
        b = self.adapter.build(t.framework, t.logs / "build.log")
        self.bin_dir = Path(b["bin_dir"])
        slug = Path(model_dir).name.lower()
        self.model_path = self.adapter.prepare_model(t.framework, model_dir, DATA / "models" / slug,
                                                     self.arch.weight_dtype, t.logs / "prepare-model.log")
        self.sched = Scheduler(t, self.adapter, self.env, self.arch, self.hw, self.wl, self.sampling,
                               self.model_path, self.bin_dir, self.backend, self.bus)
        # baseline
        res = self.sched.measure("baseline", {})
        if not res["measurement"]["trustworthy"]:
            raise I.InvariantViolation(f"baseline measurement not trustworthy: {res['measurement']['problems']}")
        led: Ledger = res["ledger"]
        write_yaml(t.ledger_dir / "baseline.yaml", led.to_dict())
        self.baseline = self._as_incumbent(res, commit=wt["base_commit"], exp_id="baseline")
        self.baseline_tokens = res["tokens"]
        self.incumbent = dict(self.baseline)
        self._save_state()
        self.bus.emit("baseline_ready", metric=self.baseline["metric"])

    def _resolve_model(self, m: str) -> Path:
        p = Path(m)
        if p.exists():
            return p
        from huggingface_hub import snapshot_download  # lazy: only for hub ids
        return Path(snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]))

    def _as_incumbent(self, res: dict, commit: str, exp_id: str) -> dict:
        led: Ledger = res["ledger"] if isinstance(res["ledger"], Ledger) else Ledger.from_dict(res["ledger"])
        eff = res["effective"]
        phys = physics.bounds(self.arch, self.hw, self.wl, min(eff.get("slots") or 1, self.wl.concurrency))
        feats = features.project(self.arch, self.hw, self.wl, phys, eff, self.capacity, led.measured, led.theory)
        m = res["measurement"]
        return {"exp_id": exp_id, "config": dict(res["config"]), "commit": commit, "effective": eff,
                "measurement": m, "metric": m["summary"][self.wl.primary_metric]["mean"],
                "ledger": led.to_dict(), "features": feats}

    def _save_state(self) -> None:
        self.task.save_state({"incumbent": {k: v for k, v in self.incumbent.items() if k != "ledger"},
                              "baseline_metric": self.baseline["metric"], "queue": [e["id"] for e in self.queue],
                              "experiments": [{"id": e["id"], "state": e["state"], "outcome": e.get("outcome")}
                                              for e in self.experiments],
                              "decisions": self.n_decisions, "elapsed_s": round(time.time() - self.t_start)})

    # =========================================================== briefs
    def _knobs_catalog(self) -> list[dict]:
        return [{"name": k, "type": v["type"], "default": v.get("default"), "values": v.get("values"),
                 "node": v["node"], "kind": v["kind"], "doc": v.get("doc"), "lossy_unless": v.get("lossy_unless")}
                for k, v in self.adapter.knobs.items()]

    def _exp_summary(self, e: dict) -> dict:
        return {"id": e["id"], "hypothesis": e["hypothesis"], "action": e["action"],
                "predicted_pct": e["predicted"]["delta_pct"], "observed_pct": (e.get("compare") or {}).get("delta_pct"),
                "outcome": e.get("outcome"), "state": e["state"], "analysis": (e.get("analysis") or {}).get("conclusion")}

    def _budget_left(self) -> dict:
        b = self.ctx["budget"]
        return {"experiments_left": b["max_experiments"] - len(self.experiments),
                "seconds_left": round(b["max_seconds"] - (time.time() - self.t_start))}

    def _bound_summary(self) -> dict:
        eff = self.incumbent["effective"]
        cur_B = min(eff.get("slots") or 1, self.wl.concurrency)
        cur = physics.bounds(self.arch, self.hw, self.wl, cur_B)
        best_B = self.capacity["max_useful_batch"]
        best = physics.bounds(self.arch, self.hw, self.wl, best_B)
        return {"current_decode_batch": cur_B, "upper_bound_tok_s_at_current_batch": round(physics.throughput_upper_bound(cur), 1),
                "max_useful_batch": best_B, "upper_bound_tok_s_at_max_batch": round(physics.throughput_upper_bound(best), 1),
                "incumbent_tok_s": round(self.incumbent["metric"], 1)}

    # ===================================================== decision points
    def _on_plan(self, ev) -> None:
        left = self._budget_left()
        n_max = min(3, left["experiments_left"])
        if n_max <= 0:
            self.bus.emit("round_done", reason="budget exhausted")
            return
        inc = self.incumbent
        brief = {
            "goal": {"metric": self.wl.primary_metric, "higher_is_better": self.wl.higher_is_better,
                     "workload": self.wl.to_dict()},
            "context": {"model": self.arch.to_dict(), "hw": self.hw.__dict__, "framework": self.ctx["framework"]["name"],
                        "framework_ref": self.ctx["inputs"]["ref"], "env": self.ctx["env_fingerprint"]},
            "features": inc["features"], "capacity": self.capacity, "bounds": self._bound_summary(),
            "incumbent": {"config": inc["config"], "effective": inc["effective"], "metric": inc["metric"],
                          "summary": {k: round(v["mean"], 3) for k, v in inc["measurement"]["summary"].items()}},
            "ledger_diagnosis": inc["ledger"]["diagnosis"],
            "top_kernels": (self.experiments[-1].get("top_kernels") if self.experiments else None),
            "knobs": self._knobs_catalog(),
            "knowledge_suggestions": K.query(inc["features"], self.ctx["framework"]["name"],
                                             inc["ledger"]["diagnosis"], context=self.capacity),
            "experiments_so_far": [self._exp_summary(e) for e in self.experiments],
            "resources": {"gpus": len(self.sched.gpus), "max_experiments_this_round": n_max},
            "budget": left, "codemap": f"assets/adapters/{self.ctx['framework']['name']}/codemap.md",
        }
        out = self.decide("plan", ev.type, brief)
        for i, x in enumerate(out["experiments"][:n_max]):
            eid = f"e{len(self.experiments) + len(self.queue) + 1:02d}"
            exp = {"id": eid, "state": "pending", "hypothesis": x["hypothesis"], "node": x["node"], "kind": x["kind"],
                   "action": x["action"], "predicted": x["predicted"], "planned_by": f"{self.n_decisions:03d}-plan",
                   "knowledge_used": x.get("knowledge_used", [])}
            I.check_prediction_declared(exp)
            self.queue.append(exp)
            self.task.save_experiment(exp)
        self._save_state()
        self.bus.emit("slot_free") if self.queue else self.bus.emit("round_done", reason="planner proposed nothing")

    def _on_monitor_sync(self, exp: dict, alert: dict, repeats: list) -> str:
        self.bus.emit("watcher_alert", exp_id=exp["id"], alert=alert)
        self.bus.q.pop()  # handled synchronously below, keep only the log entry
        if "monitor" not in self.hooks:
            return "continue"
        brief = {"experiment": {k: exp[k] for k in ("id", "hypothesis", "action", "predicted")}, "alert": alert,
                 "repeats_so_far": repeats, "incumbent_metric": self.incumbent["metric"],
                 "server_log_tail": self._log_tail(self.task.logs / f"{exp['id']}-server.log", 40)}
        return self.decide("monitor", "watcher_alert", brief)["action"]

    def _on_monitor(self, ev) -> None:  # async path unused in v0 (watcher alerts are synchronous)
        pass

    def _on_analyze(self, ev) -> None:
        exp = self._exp(ev.data["exp_id"])
        brief = {"experiment": {k: exp.get(k) for k in ("id", "hypothesis", "node", "kind", "action", "predicted",
                                                         "state", "outcome", "error", "killed", "compare", "gates",
                                                         "effective", "top_kernels", "execution")},
                 "ledger_before": {"diagnosis": exp["_base_ledger"]["diagnosis"]},
                 "ledger_after": (exp.get("ledger_summary") or {}).get("measured"),
                 "theory_after": (exp.get("ledger_summary") or {}).get("theory"),
                 "measurement_summary": {k: {"mean": round(v["mean"], 3), "cv": round(v["cv"], 4)}
                                         for k, v in ((exp.get("measurement") or {}).get("summary") or {}).items()}}
        brief["experiment"]["gates"] = {k: {"pass": v["pass"]} for k, v in (exp.get("gates") or {}).items()}
        out = self.decide("analyze", ev.type, brief)
        self._conclude(exp["id"], analysis=out)

    def _on_judge(self, ev) -> None:
        brief = {"baseline_metric": self.baseline["metric"], "incumbent": {"config": self.incumbent["config"],
                 "metric": self.incumbent["metric"], "exp_id": self.incumbent["exp_id"]},
                 "improvement_pct": round((self.incumbent["metric"] / self.baseline["metric"] - 1) * 100, 2),
                 "bounds": self._bound_summary(), "ledger_diagnosis": self.incumbent["ledger"]["diagnosis"],
                 "features": self.incumbent["features"],
                 "experiments": [self._exp_summary(e) for e in self.experiments], "budget": self._budget_left()}
        out = self.decide("judge", ev.type, brief)
        if out["decision"] == "continue" and self._budget_left()["experiments_left"] > 0:
            self.bus.emit("judge_continue", direction=out.get("direction"))
        else:
            self.bus.emit("task_concluded", reason=out.get("reason"))

    # ========================================================= mechanism
    def _exp(self, eid: str) -> dict:
        return next(e for e in self.experiments if e["id"] == eid)

    def _run_next(self, ev) -> None:
        if not self.queue:
            self.bus.emit("round_done")
            return
        b = self._budget_left()
        if b["experiments_left"] <= 0 or b["seconds_left"] <= 0:
            self.queue.clear()
            self.bus.emit("round_done", reason="budget exhausted")
            return
        exp = self.queue.pop(0)
        self.experiments.append(exp)
        exp["_base_ledger"] = self.incumbent["ledger"]
        exp["_base_features"] = self.incumbent["features"]
        exp = self.sched.run(exp, self.incumbent, self.baseline_tokens,
                             on_alert=lambda alert, reps: self._on_monitor_sync(exp, alert, reps))
        # deterministic acceptance: gates + significant improvement (hard gate, OQ-8)
        if exp["state"] == "finished":
            ok = all(g["pass"] for g in exp["gates"].values()) and exp["compare"]["improved"]
            if ok:
                I.check_accept(exp)
                exp["outcome"] = "accepted"
                res = {"ledger": Ledger.from_dict({"theory": exp["ledger_summary"]["theory"],
                                                    "measured": exp["ledger_summary"]["measured"]}),
                       "effective": exp["effective"], "measurement": exp["measurement"], "config": exp["config"]}
                commit = (exp.get("execution") or {}).get("commit") or self.incumbent["commit"]
                self.incumbent = self._as_incumbent(res, commit=commit, exp_id=exp["id"])
            else:
                exp["outcome"] = "rejected"
                if exp["action"]["type"] == "code":
                    workspace.reset_to(self.task.framework, self.incumbent["commit"])
                    self.sched.rebuild()
        else:
            exp["outcome"] = {"killed": "killed", "failed": "failed"}[exp["state"]]
        self.task.save_experiment({k: v for k, v in exp.items() if not k.startswith("_")})
        self._save_state()
        self.bus.emit(f"experiment_{exp['state'] if exp['state'] != 'finished' else 'finished'}", exp_id=exp["id"])

    def _conclude(self, eid: str, analysis: dict | None) -> None:
        exp = self._exp(eid)
        exp["analysis"] = analysis
        fw = self.ctx["framework"]["name"]
        base_cfg = exp.get("base", {}).get("config") or {}
        is_cfg = exp["action"]["type"] == "config"
        sig_apply = {"knobs": exp["action"].get("knobs")} if is_cfg else \
            {"code": True, "id": exp["action"].get("id", exp["id"])}
        fact = {
            "task": self.task.id, "experiment": eid,
            "context": {"model_arch": self.arch.to_dict(), "hw": self.hw.id, "workload": self.wl.to_dict(),
                        "framework": {"name": fw, "ref": self.ctx["inputs"]["ref"],
                                      "commit": exp.get("base", {}).get("commit")},
                        "env": self.ctx["env_fingerprint"], "sampling": self.sampling},
            "features": exp["_base_features"],
            # signature is a derived value kept only as a snapshot; readers recompute it (DES-18)
            "action": {**exp["action"], "base_config": base_cfg,
                       "signature": K.signature(fw, sig_apply, base_cfg if is_cfg else None)},
            "predicted": {**exp["predicted"], "by": exp["planned_by"], "assets": self.frozen,
                          "knowledge_used": exp.get("knowledge_used", [])},
            "observed": {"delta_pct": (exp.get("compare") or {}).get("delta_pct"), "compare": exp.get("compare"),
                         "effective": exp.get("effective"), "ledger_before": exp["_base_ledger"],
                         "ledger_after": exp.get("ledger_summary"), "error": exp.get("error"),
                         "killed": exp.get("killed")},
            "gates": {k: v["pass"] for k, v in (exp.get("gates") or {}).items()},
            "outcome": exp["outcome"],
            "analysis": analysis,
        }
        exp["fact_id"] = F.write(fact)
        exp["state"] = "concluded"
        I.check_fact_written(exp)
        self.task.save_experiment({k: v for k, v in exp.items() if not k.startswith("_")})
        self._save_state()
        self.bus.emit("experiment_concluded", exp_id=eid, fact=exp["fact_id"])

    def _after_conclude(self, ev) -> None:
        self.bus.emit("slot_free") if self.queue else self.bus.emit("round_done")

    # ============================================================ deliver
    def _deliver(self, ev) -> None:
        t, inc = self.task, self.incumbent
        d = t.p("deliverable")
        d.mkdir(parents=True, exist_ok=True)
        script = self.adapter.launch_script(self.bin_dir, self.model_path, inc["config"])
        (d / "launch.sh").write_text(script)
        (d / "launch.sh").chmod(0o755)
        # deliverable gate: the script alone brings up a healthy server that answers
        import subprocess
        port = 18199
        log = t.logs / "deliverable-check.log"
        with open(log, "w") as f:
            proc = subprocess.Popen(["bash", str(d / "launch.sh"), "--port", str(port), "--host", "127.0.0.1"],
                                    stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        healthy = answered = False
        for _ in range(240):
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                    healthy = True
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)
        if healthy:
            r = requests.post(f"http://127.0.0.1:{port}/completion", json={"prompt": "Hello", "n_predict": 8}, timeout=60)
            answered = r.status_code == 200 and bool(r.json().get("content"))
        proc.send_signal(2)
        proc.wait(timeout=30)
        gate = G.deliverable(healthy, answered)
        deliv = {"branch": f"sherpa/{t.id}", "commit": inc["commit"], "base_commit": self.ctx["framework"]["base_commit"],
                 "framework": self.ctx["framework"]["name"], "ref": self.ctx["inputs"]["ref"],
                 "config": inc["config"], "model": str(self.model_path), "launch": "deliverable/launch.sh",
                 "baseline_metric": self.baseline["metric"], "final_metric": inc["metric"],
                 "improvement_pct": round((inc["metric"] / self.baseline["metric"] - 1) * 100, 2),
                 "accepted_experiment": inc["exp_id"], "gate": gate, "reason": ev.data.get("reason")}
        write_yaml(d / "deliverable.yaml", deliv)
        self._write_report(deliv)
        I.check_assets_frozen(self.frozen)
        self._save_state()
        self.done = True

    def _write_report(self, deliv: dict) -> None:
        L = [f"# Sherpa task {self.task.id}", "",
             f"- model: `{self.ctx['inputs']['model']}` ({self.arch.weight_dtype})",
             f"- hw: {self.hw.name} | framework: {self.ctx['framework']['name']}@{self.ctx['inputs']['ref']}",
             f"- workload: {self.wl.id} (P={self.wl.prompt_tokens}, O={self.wl.output_tokens}, C={self.wl.concurrency})",
             f"- **{self.wl.primary_metric}: {deliv['baseline_metric']:.1f} -> {deliv['final_metric']:.1f} "
             f"({deliv['improvement_pct']:+.1f}%)**", f"- final config: `{json.dumps(deliv['config'])}`",
             f"- deliverable gate: {'PASS' if deliv['gate']['pass'] else 'FAIL'}", "", "## Bounds", "```",
             json.dumps(self._bound_summary(), indent=2), "```", "", "## Baseline ledger", "```",
             Ledger.from_dict(self.baseline["ledger"]).render(), "```", "", "## Final ledger", "```",
             Ledger.from_dict(self.incumbent["ledger"]).render(), "```", "", "## Experiments", "",
             "| id | action | predicted | observed | outcome | conclusion |", "|---|---|---|---|---|---|"]
        for e in self.experiments:
            s = self._exp_summary(e)
            act = json.dumps(e["action"].get("knobs")) if e["action"]["type"] == "config" else "code: " + e["action"].get("instructions", "")[:60]
            L.append(f"| {s['id']} | `{act}` | {s['predicted_pct']:+} | {s['observed_pct'] if s['observed_pct'] is not None else '-'} "
                     f"| {s['outcome']} | {(s['analysis'] or '')[:160]} |")
        self.task.p("report.md").write_text("\n".join(L) + "\n")

    @staticmethod
    def _log_tail(p: Path, n: int) -> str:
        return "\n".join(p.read_text().splitlines()[-n:]) if p.exists() else ""
