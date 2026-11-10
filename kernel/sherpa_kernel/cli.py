"""sherpa-judge: the only way results about Sherpa are produced.

  sherpa-judge doctor                         machine health vs public scoreboard
  sherpa-judge score <task> <deliverable>     score a deliverable (yaml/json) on a task
  sherpa-judge measure <task> <deliverable>   quick non-final measurement (no oracle, 1 pair)
  sherpa-judge show <record-id>               re-derive and print a record
  sherpa-judge history [--task T]             scores so far (derived on read)
  sherpa-judge report [--sherpa-commit C]     latest score per task, mean by split (train / heldout)
  sherpa-judge bench list | verify | seal     bench content and integrity
  sherpa-judge bench reference <task>         (human) generate the HF correctness reference
  sherpa-judge bench calibrate <task>         (human) measure floor/reference on this machine
  sherpa-judge guard <path>...                exit 1 if a change touches the judge
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import bench as B
from . import doctor, guard, llamacpp as L, machine, records
from .util import BENCH, KERNEL, run


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _summary(r: dict) -> dict:
    d = r["derived"]
    out = {"id": r["id"], "task": r["task"], "valid": d["valid"], "score": d.get("score"),
           "score_if_valid": d.get("score_if_valid"), "score_noise": d.get("score_noise"),
           "ratio_to_reference": (d.get("ratio_to_reference") or {}).get("gmean"),
           "metric": {k: (d.get("metric") or {}).get(k, {}).get("mean") for k in ("candidate", "reference")},
           "reasons": d["reasons"], "seconds": r.get("seconds")}
    if "oracle" in d:
        o = d["oracle"]
        out["oracle"] = {"pass": o["pass"], "tokens": {k: o["tokens"][k] for k in ("match", "tie", "hard", "error")},
                         "lp_mae": o["logprobs"]["mae"], "sampling": o["sampling"], "context": o["context"]["pass"]}
    return out


def cmd_doctor(a) -> int:
    r = doctor.run_doctor()
    _print({k: r[k] for k in ("id", "pass", "problems", "measured", "scoreboard")})
    return 0 if r["pass"] else 1


def cmd_score(a) -> int:
    r = B.score(a.task, Path(a.deliverable), pairs=a.pairs, sherpa_commit=a.sherpa_commit, note=a.note,
                allow_unsealed=a.allow_unsealed)
    _print(_summary(r))
    return 0 if r["derived"]["valid"] else 2


def cmd_measure(a) -> int:
    """Quick, non-final measurement for an optimizer's inner loop: same contract, no oracle, 1 pair.
    Never counts as a score (record kind 'probe')."""
    r = B.score(a.task, Path(a.deliverable), pairs=a.pairs, sherpa_commit=a.sherpa_commit, note=a.note,
                allow_unsealed=True, run_oracle=False, kind="probe")
    _print(_summary(r))
    return 0


def cmd_show(a) -> int:
    r = records.get(a.record)
    if r["kind"] == "doctor":
        _print(r)
        return 0
    _print(_summary({**r, "derived": B.derive(r)}))
    return 0


def cmd_history(a) -> int:
    for r in records.read("score"):
        if a.task and r["task"] != a.task:
            continue
        try:
            d = B.derive(r)
        except Exception as e:  # noqa: BLE001
            d = {"valid": False, "score": None, "reasons": [f"cannot derive: {e}"]}
        print(json.dumps({"id": r["id"], "task": r["task"], "deliverable": r["deliverable_id"],
                          "sherpa_commit": r.get("sherpa_commit"), "valid": d["valid"], "score": d.get("score"),
                          "note": r.get("note")}, ensure_ascii=False))
    return 0


def cmd_report(a) -> int:
    """Bench-level view of one Sherpa version: latest score per task, aggregated by split.
    Held-out is the number that judges Sherpa; train is shown for diagnosis only."""
    latest: dict[str, dict] = {}
    current = guard.bench_version()
    stale = 0
    for r in records.read("score"):
        if a.sherpa_commit and r.get("sherpa_commit") != a.sherpa_commit:
            continue
        if r.get("bench") != current:     # scores are only comparable within one sealed bench version
            stale += 1
            continue
        latest[r["task"]] = r
    rows, by_split = [], {}
    for t in B.list_tasks():
        r = latest.get(t.id)
        d = B.derive(r) if r else None
        s = d.get("score") if d else None
        rows.append({"task": t.id, "split": t.raw["split"], "record": r["id"] if r else None,
                     "valid": d["valid"] if d else None, "score": s})
        by_split.setdefault(t.raw["split"], []).append(0.0 if s is None else s)
    _print({"sherpa_commit": a.sherpa_commit, "bench": current, "ignored_other_bench_versions": stale, "tasks": rows,
            "mean_score": {k: sum(v) / len(v) for k, v in by_split.items()},
            "missing": [x["task"] for x in rows if x["record"] is None]})
    return 0


def cmd_bench(a) -> int:
    if a.op == "list":
        mid = machine.machine_id()
        for t in B.list_tasks():
            bp = t.baseline_path(mid)
            print(f"{t.id:40s} {t.kind:6s} {t.raw['split']:8s} {t.metric:18s} "
                  f"{'calibrated' if bp.exists() else 'NOT calibrated'} on {mid}")
        return 0
    if a.op == "verify":
        diffs = guard.verify()
        _print({"bench_version": guard.bench_version(), "intact": not diffs, "diffs": diffs})
        return 0 if not diffs else 1
    if a.op == "seal":
        print(guard.seal())
        return 0
    if a.op == "reference":
        t = B.load_task(a.task)
        conv = L.convert_python()
        out = BENCH / t.raw["model"]["reference"]
        snap = B.hf_snapshot(t.raw["model"])
        with machine.exclusive(f"reference {t.id}"):
            run([conv, str(KERNEL / "sherpa_kernel" / "reference.py"), "--model", str(snap),
                 "--prompts", str(BENCH / "oracle" / "prompts.jsonl"), "--out", str(out)], timeout=7200)
        print(out)
        return 0
    if a.op == "calibrate":
        _print(B.calibrate(a.task, pairs=a.pairs or 5))
        return 0
    return 1


def cmd_guard(a) -> int:
    try:
        guard.check_paths(a.paths)
    except guard.GuardViolation as e:
        print(e, file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sherpa-judge", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("doctor").set_defaults(fn=cmd_doctor)
    s = sp.add_parser("score")
    s.add_argument("task")
    s.add_argument("deliverable")
    s.add_argument("--pairs", type=int)
    s.add_argument("--sherpa-commit")
    s.add_argument("--note")
    s.add_argument("--allow-unsealed", action="store_true", help="development only; recorded in the result")
    s.set_defaults(fn=cmd_score)
    s = sp.add_parser("measure", help="quick non-final measurement (no oracle), for inner loops")
    s.add_argument("task")
    s.add_argument("deliverable")
    s.add_argument("--pairs", type=int, default=1)
    s.add_argument("--sherpa-commit")
    s.add_argument("--note")
    s.set_defaults(fn=cmd_measure)
    s = sp.add_parser("show")
    s.add_argument("record")
    s.set_defaults(fn=cmd_show)
    s = sp.add_parser("history")
    s.add_argument("--task")
    s.set_defaults(fn=cmd_history)
    s = sp.add_parser("report")
    s.add_argument("--sherpa-commit")
    s.set_defaults(fn=cmd_report)
    s = sp.add_parser("bench")
    s.add_argument("op", choices=["list", "verify", "seal", "reference", "calibrate"])
    s.add_argument("task", nargs="?")
    s.add_argument("--pairs", type=int)
    s.set_defaults(fn=cmd_bench)
    s = sp.add_parser("guard")
    s.add_argument("paths", nargs="+")
    s.set_defaults(fn=cmd_guard)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
