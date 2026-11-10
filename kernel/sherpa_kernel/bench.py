"""Task Bench: tasks, deliverables, scoring.

A task fixes (model, machine, framework base, workload, objective, contract). A deliverable is what
an optimizer hands in: base commit (+ patch for code tasks), build flags, model dtype, server args, env.

Score of a deliverable on a task (design §3.3):

    r     = paired performance ratio candidate / reference, measured interleaved in this session
            (cancels slow machine drift; reference = expert artifact shipped with the task)
    f     = floor / reference ratio, measured at calibration (bench/baselines/<machine>/<task>.json)
    score = (r - f) / (1 - f)          0 ~ framework defaults, 1 ~ expert reference, >1 beats it
    valid = oracle passes, no measurement-contract violation, ratio trustworthy
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from . import doctor, guard, llamacpp as L, loadgen as G, machine, oracle as O, records, stats
from .util import BENCH, read_json, read_yaml, run, sha256_bytes, sha256_file, stable_hash, write_json

LOSSLESS_DTYPES = {"bf16", "f16", "f32"}
MAX_SCORE_NOISE = 0.25     # admissible task / trustworthy score: 95% score band within +-0.25
TARGET_SCORE_NOISE = 0.15  # adaptive pairing stops once the 95% score band is within +-0.15


# ---------------------------------------------------------------------------------------- tasks
@dataclass
class Task:
    id: str
    raw: dict
    path: Path

    @property
    def kind(self) -> str:
        return self.raw["kind"]

    @property
    def workload(self) -> G.Workload:
        return G.Workload.from_dict(self.raw["workload"])

    @property
    def higher_is_better(self) -> bool:
        return self.raw["objective"]["higher_is_better"]

    @property
    def metric(self) -> str:
        return self.raw["objective"]["metric"]

    @property
    def contract(self) -> dict:
        return self.raw.get("contract") or {}

    def deliverable(self, which: str) -> dict:
        return load_deliverable(BENCH / "deliverables" / self.id / f"{which}.yaml")

    def reference_data(self) -> dict:
        return read_json(BENCH / self.raw["model"]["reference"])

    def reference_sha256(self) -> str:
        return sha256_file(BENCH / self.raw["model"]["reference"])

    def baseline_path(self, machine_id: str) -> Path:
        return BENCH / "baselines" / machine_id / f"{self.id}.json"

    def hash(self) -> str:
        return stable_hash(self.raw)[:16]


def load_task(task_id: str) -> Task:
    p = BENCH / "tasks" / f"{task_id}.yaml"
    raw = read_yaml(p)
    if raw["id"] != task_id:
        raise ValueError(f"{p}: id mismatch")
    return Task(task_id, raw, p)


def list_tasks() -> list[Task]:
    return [load_task(p.stem) for p in sorted((BENCH / "tasks").glob("*.yaml"))]


# --------------------------------------------------------------------------------- deliverables
def load_deliverable(p: Path) -> dict:
    """Read a deliverable file and materialise its patch text (patch_file / patch_from)."""
    d = read_yaml(p) if p.suffix in (".yaml", ".yml") else json.loads(p.read_text())
    d = dict(d)
    if d.get("patch_file"):
        d["patch"] = (p.parent / d["patch_file"]).read_text()
    elif d.get("patch_from"):
        d["patch"] = L.diff(L.resolve(d["base"]), L.resolve(d["patch_from"]))
    d.setdefault("patch", "")
    return d


def validate(task: Task, d: dict) -> list[str]:
    errs = []
    fw = task.raw["framework"]
    if d.get("framework", "llama.cpp") != fw["name"]:
        errs.append(f"framework {d.get('framework')} != {fw['name']}")
    try:
        if L.resolve(d.get("base", fw["base"])) != L.resolve(fw["base"]):
            errs.append(f"base {d.get('base')} is not the task base {fw['base']}")
    except RuntimeError as e:
        errs.append(str(e))
    if task.kind == "config" and d.get("patch"):
        errs.append("config task: source patches are not allowed")
    if task.kind == "code":
        fixed = task.raw.get("fixed") or {}
        for k in ("args", "env", "cmake"):
            if k in fixed and (d.get(k) or ([] if k != "env" else {})) != fixed[k]:
                errs.append(f"code task: '{k}' is fixed to {fixed[k]!r}")
    allowed = task.contract.get("model_dtypes") or sorted(LOSSLESS_DTYPES)
    if d.get("model_dtype", "bf16") not in allowed:
        errs.append(f"model_dtype {d.get('model_dtype')} not allowed by the task contract {allowed}")
    try:
        L.check_args(d.get("args") or [])
    except ValueError as e:
        errs.append(str(e))
    return errs


def deliverable_id(d: dict) -> str:
    return stable_hash({k: d.get(k) for k in ("framework", "base", "cmake", "model_dtype", "args", "env")}
                       | {"patch": sha256_bytes((d.get("patch") or "").encode())})[:16]


# ------------------------------------------------------------------------------------ materialise
def hf_snapshot(model: dict) -> Path:
    org, name = model["hf"].split("/")
    snap = Path.home() / ".cache/huggingface/hub" / f"models--{org}--{name}" / "snapshots" / model["revision"]
    # a cache can hold config/tokenizer without weights (partial earlier downloads)
    if not (snap / "config.json").exists() or not any(snap.glob("*.safetensors")):
        run([L.convert_python(), "-c", "import sys; from huggingface_hub import snapshot_download as s; "
             f"print(s('{model['hf']}', revision='{model['revision']}', "
             "allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.tiktoken']))"], timeout=3600)
    return snap


@dataclass
class Prepared:
    build: L.Build
    model_path: str
    model_sha256: str
    deliverable: dict


def prepare(task: Task, d: dict, log_dir: Path) -> Prepared:
    b = L.build(d.get("base", task.raw["framework"]["base"]), d.get("patch") or "", d.get("cmake") or [], log_dir)
    conv = L.build(task.raw["model"].get("converter", task.raw["framework"]["base"]))
    m = L.prepare_model(hf_snapshot(task.raw["model"]), task.raw["model"]["slug"], d.get("model_dtype", "bf16"),
                        conv, log_dir / "model.log")
    return Prepared(b, m["path"], m["sha256"], d)


# ----------------------------------------------------------------------------------------- score
def policies() -> dict:
    return read_yaml(BENCH / "oracle" / "policy.yaml")


def policy_for(task: Task) -> dict:
    return policies()["lossy" if task.contract.get("lossy") else "lossless"]


def _segment(p: Prepared, wl: G.Workload, seed: int, label: str, log_dir: Path) -> dict:
    t0 = time.time()
    with L.start(p.build, p.model_path, p.deliverable.get("args") or [], p.deliverable.get("env") or {},
                 log_dir / f"{label}.log") as s:
        startup = time.time() - t0
        before = machine.gpu_state()
        raw = G.run_load(s.url, wl, seed)
        after = machine.gpu_state()
        quiet = s.quiet
    raw.update(label=label, startup_s=round(startup, 2), gpu_before=before, gpu_after=after,
               quiet=quiet, seconds=round(time.time() - t0, 1))
    return raw


def _ratio_so_far(segs: list[dict], task: Task) -> dict:
    vals = {"cand": [], "ref": []}
    for s in segs:
        vals[s["who"]].append(G.metrics(s, task.workload)[task.metric])
    n = min(len(vals["cand"]), len(vals["ref"]))
    return stats.paired_ratio(vals["cand"][:n], vals["ref"][:n], task.higher_is_better)


def score(task_id: str, deliverable: dict | Path, *, pairs: int | None = None, sherpa_commit: str | None = None,
          note: str | None = None, allow_unsealed: bool = False, require_doctor: bool = True,
          oracle_only: bool = False, run_oracle: bool = True, kind: str = "score",
          max_pairs: int | None = None) -> dict:
    t_start = time.time()
    task = load_task(task_id)
    diffs = guard.verify()
    if diffs and not allow_unsealed:
        raise guard.GuardViolation(f"bench differs from its manifest: {diffs[:5]}")
    doc = doctor.latest_pass()
    if require_doctor and not doc:
        raise RuntimeError("no passing doctor record for this machine/driver/CUDA: run `sherpa-judge doctor`")
    mid = machine.machine_id()
    if task.raw.get("machine") and task.raw["machine"] != mid:
        raise RuntimeError(f"task {task.id} is bound to {task.raw['machine']}, this machine is {mid}")
    d = load_deliverable(deliverable) if isinstance(deliverable, Path) else dict(deliverable)
    d.setdefault("patch", "")
    errs = validate(task, d)
    rid = records.new_id(kind)
    log_dir = records.root() / "artifacts" / rid / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    rec = {"task": task.id, "task_hash": task.hash(), "bench": guard.bench_version(), "bench_intact": not diffs,
           "machine": machine.fingerprint(), "doctor": doc["id"] if doc else None, "sherpa_commit": sherpa_commit,
           "note": note, "deliverable": {k: v for k, v in d.items() if k != "patch"},
           "deliverable_id": deliverable_id(d), "patch_sha256": sha256_bytes(d["patch"].encode()) if d["patch"] else None,
           "invalid": errs}
    if d["patch"]:
        rec["patch"] = records.save_artifact(rid, "patch", d["patch"])
    if errs:
        rec["seconds"] = round(time.time() - t_start, 1)
        records.write(kind, rec, rid)
        return {**rec, "id": rid, "derived": derive(rec, task)}

    t0 = time.time()
    cand = prepare(task, d, log_dir)
    ref_d = task.deliverable("reference")
    ref = prepare(task, ref_d, log_dir) if not oracle_only else None
    rec["build"] = {"candidate": cand.build.info["key"], "reference": ref.build.info["key"] if ref else None,
                    "model_sha256": cand.model_sha256, "seconds": round(time.time() - t0, 1)}

    # correctness, under the task's own load
    if run_oracle:
        pol = policy_for(task)
        t0 = time.time()
        with L.start(cand.build, cand.model_path, d.get("args") or [], d.get("env") or {},
                     log_dir / "oracle.log") as s:
            oraw = O.run(s.url, task.reference_data(), task.workload, task.contract, n_samples=pol["n_samples"])
        rec["oracle"] = {"raw": records.save_artifact(rid, "oracle", oraw), "seconds": round(time.time() - t0, 1),
                         "reference_sha256": task.reference_sha256()}
        bp = task.baseline_path(mid)
        oracle_ok = O.evaluate(oraw, task.reference_data(), policies(), task.contract,
                               ref_accuracy=read_json(bp).get("floor_accuracy") if bp.exists() else None)["pass"]
    else:
        oracle_ok = True

    # performance: interleaved candidate / reference pairs (A B, B A, ...) with identical prompts per pair;
    # a pair holds the machine lock so nothing runs between its two halves. Adaptive: after the minimum,
    # keep adding pairs until the uncertainty that matters is small enough: the 95% score band when the
    # task is calibrated on this machine, else the standard error of the ratio.
    if not oracle_only:
        wl = task.workload
        meas = task.raw.get("measure", {})
        n_min = pairs or meas.get("pairs", stats.MIN_PAIRS)
        # an invalid deliverable scores 0 anyway: measure the minimum only (score_if_valid for diagnosis)
        n_max = n_min if (kind == "probe" or not oracle_ok) else max(n_min, max_pairs or meas.get("max_pairs", 12))
        bpath = task.baseline_path(mid)
        f = read_json(bpath).get("floor_ratio") if bpath.exists() and kind == "score" else None
        segs = []
        t0 = time.time()
        for i in range(n_max):
            seed = 10_000 + 97 * i
            order = [("cand", cand), ("ref", ref)] if i % 2 == 0 else [("ref", ref), ("cand", cand)]
            with machine.exclusive(f"pair {i}"):
                for who, p in order:
                    segs.append({"who": who, "pair": i, **_segment(p, wl, seed, f"{who}-{i}", log_dir)})
            if i + 1 >= n_min:
                se = _ratio_so_far(segs, task)["se"]
                if (f is not None and 2 * se / (1 - f) <= TARGET_SCORE_NOISE) or (f is None and se <= stats.SE_MAX):
                    break
        rec["perf"] = {"raw": records.save_artifact(rid, "perf", segs), "pairs": len(segs) // 2,
                       "seconds": round(time.time() - t0, 1)}
    rec["seconds"] = round(time.time() - t_start, 1)
    records.write(kind, rec, rid)
    return {**rec, "id": rid, "derived": derive(rec, task)}


def derive(rec: dict, task: Task | None = None) -> dict:
    """Recompute every derived quantity of a score record from its raw artifacts."""
    task = task or load_task(rec["task"])
    out: dict = {"valid": not rec.get("invalid"), "reasons": list(rec.get("invalid") or [])}
    if rec.get("invalid"):
        out["score"] = None
        return out
    bpath = task.baseline_path(rec["machine"]["machine"])
    base = read_json(bpath) if bpath.exists() else {}
    if base and base.get("task_hash") != task.hash():
        out["reasons"].append(f"baseline {bpath.relative_to(BENCH)} was calibrated for another version of the task")
        base = {}
    if "oracle" not in rec:
        out["oracle"] = None
        if rec.get("kind") == "score":
            out["valid"] = False
            out["reasons"].append("oracle was not run")
    else:
        ev = O.evaluate(records.load_artifact(rec["oracle"]["raw"]), task.reference_data(), policies(),
                        task.contract, ref_accuracy=base.get("floor_accuracy"))
        out["oracle"] = {k: v for k, v in ev.items() if k != "trajectories"}
        if rec["oracle"].get("reference_sha256") != task.reference_sha256() or ev["unknown_items"]:
            out["stale"] = True       # measured against another version of the oracle reference
            out["reasons"].append("oracle observations were made against a different reference "
                                  f"(re-derived on the overlap; unknown items: {ev['unknown_items']})")
        out["oracle"]["sampling"] = {
            "pass": ev["sampling"]["pass"], "pooled_out_of_nucleus": ev["sampling"]["pooled_out_of_nucleus"],
            "worst_g_ratio": max((p["g"] / p["g_limit"] for p in ev["sampling"]["probes"].values() if p["g_limit"]),
                                 default=None)}
        if not ev["pass"]:
            out["valid"] = False
            failed = [k for k in ("tokens", "logprobs", "sampling", "context", "accuracy")
                      if ev.get(k) and not ev[k]["pass"] and ev[k].get("gating", True)]
            out["reasons"].append("oracle failed: " + ", ".join(failed))
    if "perf" not in rec:
        out["score"] = None
        return out
    wl = task.workload
    segs = records.load_artifact(rec["perf"]["raw"])
    vals = {"cand": [], "ref": []}
    viol = []
    for s in segs:
        m = G.metrics(s, wl)
        vals[s["who"]].append(m[task.metric])
        v = G.contract_violations(s, wl)
        if v:
            viol.append(f"{s['label']}: {v[0]}")
        if s.get("quiet") and not s["quiet"]["quiet"]:
            viol.append(f"{s['label']}: machine was busy before measuring (cpu {s['quiet']['cpu_busy']:.0%})")
    ratio = stats.paired_ratio(vals["cand"], vals["ref"], task.higher_is_better)
    out["metric"] = {"name": task.metric, "candidate": stats.summarize(vals["cand"]),
                     "reference": stats.summarize(vals["ref"])}
    out["ratio_to_reference"] = {k: ratio[k] for k in ("gmean", "cv", "se", "n", "samples")}
    if viol:
        out["valid"] = False
        out["reasons"] += viol
    if base.get("floor_ratio"):
        f = base["floor_ratio"]
        out["floor_ratio"] = f
        out["score"] = (ratio["gmean"] - f) / (1 - f)
        # Trust is judged in score units: the same ratio uncertainty matters less on a task with
        # more headroom. The calibration's own uncertainty is carried into the band.
        band = 2 * ratio["se"] / (1 - f)
        out["score_noise"] = 2 * math.hypot(ratio["se"], base.get("floor_ratio_se") or 0.0) / (1 - f)
        problems = stats.trustworthy(ratio, se_max=math.inf)
        if not problems and band > MAX_SCORE_NOISE:
            problems = [f"score band +-{band:.2f} > {MAX_SCORE_NOISE} after {ratio['n']} pairs"]
    else:
        out["score"] = None
        out["reasons"].append(f"no baseline for this machine ({bpath.relative_to(BENCH)}): run calibrate")
        problems = stats.trustworthy(ratio)
    if problems:
        out["valid"] = False
        out["reasons"] += [f"measurement not trustworthy: {p}" for p in problems]
    if not out["valid"]:
        out["score_if_valid"], out["score"] = out.get("score"), 0.0 if out.get("score") is not None else None
    return out


# ------------------------------------------------------------------------------------- calibrate
def calibrate(task_id: str, pairs: int = 5) -> dict:
    """Human action: verify floor and reference pass the oracle, measure floor/reference ratio,
    and write bench/baselines/<machine>/<task>.json (re-seal the bench afterwards)."""
    task = load_task(task_id)
    res, floor_acc = {}, None
    for which in ("floor", "reference"):
        r = score(task_id, task.deliverable(which), oracle_only=True, allow_unsealed=True, kind="calibration",
                  note=f"calibrate {which} oracle")
        rec = records.get(r["id"])
        if task.contract.get("accuracy") and which == "floor":
            floor_acc = O.accuracy_of(records.load_artifact(rec["oracle"]["raw"]).get("accuracy") or [])["accuracy"]
        # the floor defines the accuracy reference, so only non-accuracy gates apply to it here
        reasons = [x for x in r["derived"]["reasons"] if not ("no baseline" in x or x == "oracle failed: accuracy")]
        res[which] = {"record": r["id"], "oracle_pass": not reasons, "reasons": reasons}
        if reasons:
            raise RuntimeError(f"{task_id}: {which} deliverable fails the oracle: {reasons}")
    r = score(task_id, task.deliverable("floor"), pairs=pairs, max_pairs=2 * pairs, allow_unsealed=True,
              kind="calibration", note="calibrate floor vs reference", run_oracle=False)
    dv = r["derived"]
    bad = [x for x in dv["reasons"] if "no baseline" not in x]
    if bad:
        raise RuntimeError(f"{task_id}: calibration measurement invalid: {bad}")
    ratio = dv["ratio_to_reference"]
    if ratio["gmean"] >= 1 - max(2 * ratio["se"], 0.03):
        raise RuntimeError(f"{task_id}: reference is not clearly better than floor (ratio {ratio['gmean']:.3f})")
    # A task must be discriminative: its headroom (1 - f) has to be large against the measurement
    # noise, otherwise the score band covers most of [0, 1]. Observed: a prefill-heavy task with
    # f = 0.93 and pair CV 7% would have a 95% score band of +-0.7.
    noise = 2 * ratio["se"] / (1 - ratio["gmean"])
    if noise > MAX_SCORE_NOISE:
        raise RuntimeError(f"{task_id}: not discriminative: headroom {1 - ratio['gmean']:.1%} vs standard error "
                           f"{ratio['se']:.1%} gives a score band of +-{noise:.2f} (> {MAX_SCORE_NOISE})")
    base = {"task": task_id, "task_hash": task.hash(), "machine": r["machine"]["machine"], "record": r["id"],
            "floor_ratio": ratio["gmean"], "floor_ratio_cv": ratio["cv"], "floor_ratio_se": ratio["se"],
            "pairs": ratio["n"],
            "metric": task.metric, "floor_metric": dv["metric"]["candidate"]["mean"],
            "reference_metric": dv["metric"]["reference"]["mean"], "floor_accuracy": floor_acc,
            "oracle_records": res, "judge": records.judge_version(),
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json(task.baseline_path(r["machine"]["machine"]), base)
    return base
