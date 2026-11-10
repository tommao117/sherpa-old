"""sherpa CLI."""
from __future__ import annotations

import argparse
import json
import sys
import traceback

from sherpa.util import DATA, read_yaml


def cmd_run(a) -> int:
    from sherpa.runtime import backends
    from sherpa.runtime.engine import Engine
    from sherpa.runtime.task import Task
    t = Task.new()
    print(f"task {t.id}  ->  {t.dir}", flush=True)
    be = backends.make(a.backend, t.decisions_dir)
    args = {"model": a.model, "hw": a.hw, "framework": a.framework, "workload": a.workload, "dtype": a.dtype,
            "max_experiments": a.max_experiments, "max_seconds": a.max_hours * 3600}
    try:
        Engine(t, be, args).run()
    except Exception:
        (t.p("error.txt")).write_text(traceback.format_exc())
        raise
    print((t.p("report.md")).read_text())
    return 0


def cmd_env(a) -> int:
    from sherpa.runtime import env
    print(json.dumps(env.probe(), indent=2))
    return 0


def cmd_pending(a) -> int:
    roots = [DATA / "tasks" / a.task / "decisions"] if a.task else list((DATA / "tasks").glob("*/decisions")) + \
        list((DATA / "proposals").glob("*"))
    for d in roots:
        for req in sorted(d.glob("*.request.md")):
            rid = req.name[: -len(".request.md")]
            if not (d / f"{rid}.response.json").exists():
                print(req)
    return 0


def cmd_ledger(a) -> int:
    from sherpa.model.ledger import Ledger
    p = DATA / "tasks" / a.task / "ledger" / f"{a.exp}.yaml"
    d = read_yaml(p)
    print(Ledger.from_dict(d).render())
    print("\ndiagnosis:")
    for r in d.get("diagnosis", []):
        print(f"  {r['node']:28s} gap={r['gap']:.4f} ms/tok ({r['gap_share'] * 100:.1f}%)")
    return 0


def cmd_facts(a) -> int:
    from sherpa.kernel import facts
    if a.aggregate:
        print(json.dumps(facts.aggregate(), indent=2))
    else:
        from sherpa.kernel import knowledge
        for f in (facts.by_task(a.task) if a.task else facts.all_facts()):
            print(f"{f['id']}  {knowledge.fact_signature(f):40s} pred={f['predicted'].get('delta_pct')}  "
                  f"obs={f['observed'].get('delta_pct')}  {f['outcome']}")
    return 0


def cmd_metabench(a) -> int:
    from sherpa.kernel import metabench
    r = metabench.replay(None)
    print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=2))
    for row in r["rows"]:
        print(row)
    return 0


def cmd_evolve(a) -> int:
    from sherpa.kernel import evolve
    from sherpa.runtime import backends
    if a.op == "propose":
        be = backends.make(a.backend, DATA / "proposals")
        print(evolve.propose(a.task, be))
    elif a.op == "evaluate":
        r = evolve.evaluate(a.proposal)
        print(json.dumps({k: v for k, v in r.items() if k != "rows_new"}, indent=2))
    elif a.op == "apply":
        print(json.dumps(evolve.apply(a.proposal, approve=a.approve, approver=a.approver), indent=2))
    elif a.op == "history":
        for r in evolve.history():
            print(json.dumps(r))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sherpa")
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run", help="full optimization task")
    r.add_argument("--model", required=True)
    r.add_argument("--hw", default="local")
    r.add_argument("--framework", required=True, help="name@ref, e.g. llama.cpp@b12242")
    r.add_argument("--workload", default="throughput-default")
    r.add_argument("--dtype", default="bf16")
    r.add_argument("--backend", default="handoff", choices=["handoff", "codebuddy"])
    r.add_argument("--max-experiments", type=int, default=6)
    r.add_argument("--max-hours", type=float, default=3)
    r.set_defaults(fn=cmd_run)
    sp.add_parser("env", help="probe machine profile").set_defaults(fn=cmd_env)
    p = sp.add_parser("pending", help="list handoff requests waiting for an answer")
    p.add_argument("--task")
    p.set_defaults(fn=cmd_pending)
    l = sp.add_parser("ledger")
    l.add_argument("task")
    l.add_argument("exp", nargs="?", default="baseline")
    l.set_defaults(fn=cmd_ledger)
    f = sp.add_parser("facts")
    f.add_argument("--task")
    f.add_argument("--aggregate", action="store_true")
    f.set_defaults(fn=cmd_facts)
    sp.add_parser("metabench").set_defaults(fn=cmd_metabench)
    e = sp.add_parser("evolve")
    e.add_argument("op", choices=["propose", "evaluate", "apply", "history"])
    e.add_argument("--task", nargs="*", default=[])
    e.add_argument("--proposal")
    e.add_argument("--approve", action="store_true")
    e.add_argument("--approver", help="identity of the approver; must differ from the proposer")
    e.add_argument("--backend", default="handoff", choices=["handoff", "codebuddy"])
    e.set_defaults(fn=cmd_evolve)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
