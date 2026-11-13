from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .runtime.simulation import demo
from .runtime.environment import plan_environment
from .runtime.store import BusyError, Store, read_json


def output(value):
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sherpa", description="Lightweight runtime prototype; no real models")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("SHERPA_DATA", Path(__file__).resolve().parents[2] / "sherpa-data")))
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("demo", help="run only synthetic tools and a scripted decision agent")
    run.add_argument("--scenario", choices=["normal", "failure", "invalid", "slow", "timeout"], default="normal")
    run.add_argument("--delay", type=float, default=0.1)
    run.add_argument("--seconds", type=float, default=60)
    commands.add_parser("runs", help="list persisted runs")
    agents = commands.add_parser("agent", help="inspect configured providers and role-local candidate pools")
    agent_commands = agents.add_subparsers(dest="agent_command", required=True)
    listing = agent_commands.add_parser("list", help="show configuration only; no provider request")
    listing.add_argument("--config", type=Path)
    environment = commands.add_parser("env", help="inspect and plan; does not install or load models")
    env_commands = environment.add_subparsers(dest="env_command", required=True)
    plan = env_commands.add_parser("plan", help="read declarations and inspect host; save a non-executable plan")
    plan.add_argument("source", type=Path)
    plan.add_argument("--profile", help="installation entry point, e.g. jev for setup_jev.sh")
    plan.add_argument("--gpus", default="", help="comma-separated indices or UUIDs; proposed only, not reserved")
    plan.add_argument("--gpu-count", type=int, default=1)
    plan.add_argument("--intent", choices=["optimize", "serve"], default="optimize")
    plan.add_argument("--json", action="store_true")
    check = env_commands.add_parser("check-contract", help="offline contract diagnostics only; no commands, model calls or run creation")
    check.add_argument("task", type=Path, help="Task JSON with versioned execution contract")
    check.add_argument("--action", required=True, type=Path, help="probe/build/run_once action JSON")
    check.add_argument("--approval", type=Path, help="explicit task-hash-bound approval data; not an authenticated execution permit")
    completion = env_commands.add_parser("plan-completion", help="draft pinned llama.cpp commands only; no file checks, device probes or execution")
    completion.add_argument("task", type=Path)
    completion.add_argument("--locations", required=True, type=Path, help="explicit workspace/toolchain paths in JSON")
    fixture = env_commands.add_parser("test-executor", help="run fixed synthetic subprocess fixtures; never build/load user inputs")
    fixture.add_argument("--scenario", choices=["normal", "failure", "missing-artifact", "wrong-output", "malformed", "timeout", "slow", "output-limit"], default="normal")
    fixture.add_argument("--seconds", type=float, default=30)
    fixture.add_argument("--max-calls", type=int, default=3)
    for name in ("prepare-code", "explore", "resume"):
        explore = env_commands.add_parser(name, help="prepare-code pins inputs without LLM; explore investigates them; resume reuses a pinned run")
        if name != "resume":
            explore.add_argument("--framework", required=True, help="user-selected HTTPS Git URL or local code directory")
            explore.add_argument("--model", required=True, type=Path, help="local model directory (may equal framework input)")
            explore.add_argument("--revision", help="framework branch/tag/commit; remote default HEAD is resolved once")
            explore.add_argument("--device", choices=["cpu", "gpu", "unspecified"], default="unspecified")
        else:
            explore.add_argument("run_id", help="stopped incomplete exploration whose evidence should be reused")
        explore.add_argument("--allow-model-calls", action="store_true", help="send selected local evidence to configured providers")
        explore.add_argument("--allow-unpriced", action="store_true", help="allow unknown provider fees; dollar cap is not hard-guaranteed")
        explore.add_argument("--model-budget", type=float, required=name != "prepare-code", default=0,
                             help="positive reported-cost ceiling in USD; prepare-code uses no model budget")
        explore.add_argument("--framework-max-mib", type=int, default=512, help="framework workspace disk budget in MiB")
        explore.add_argument("--seconds", type=float, default=300, help="additional wall-clock budget for this segment")
        explore.add_argument("--max-calls", type=int, default=24, help="additional tool/provider calls")
        explore.add_argument("--max-rounds", type=int, default=12, help="additional daily decisions, including rejected decisions")
        explore.add_argument("--max-delegations", type=int, default=1, help="additional expert delegations")
        explore.add_argument("--candidate-timeout", type=float, default=90)
        explore.add_argument("--config", type=Path, help="current agent config; versions are recorded separately from source run")
        explore.add_argument("--json", action="store_true")
    for name in ("status", "events", "cancel", "trace"):
        command = commands.add_parser(name)
        command.add_argument("run_id")
        if name == "events":
            command.add_argument("--tail", type=int, default=20)
        if name == "cancel":
            command.add_argument("--reason", default="user requested cancellation")
    args = parser.parse_args(argv)
    try:
        if args.command == "agent":
            from .agent import Config
            output(Config.load(args.config).describe())
            return 0
        if args.command == "env" and args.env_command == "plan-completion":
            from .runtime.execution_contract import load_document
            from .platform.llama_completion import plan_completion
            report = plan_completion(load_document(args.task), load_document(args.locations))
            output(report)
            return 0 if report["status"] == "draft" else 2
        if args.command == "env" and args.env_command == "check-contract":
            from .runtime.execution_contract import load_document, validate_execution
            report = validate_execution(load_document(args.task), load_document(args.action),
                                        load_document(args.approval) if args.approval else None)
            output(report)
            return 0 if report["status"] == "valid" else 2
        store = Store(args.data_root)
        if args.command == "env" and args.env_command == "test-executor":
            from .runtime.fixture_execution import fixture_demo
            run = fixture_demo(store, args.scenario, seconds=args.seconds, max_calls=args.max_calls,
                               announce=lambda value: print(f"run_id={value}", flush=True))
            state = run.state()
            output({"run_id": run.id, "status": state["status"], "result_status": state["result_status"],
                    "reason": state["stop_reason"], "cost": state["cost"], "synthetic": True,
                    "readiness": state["readiness"], "installed": False})
            return 0 if state["status"] == "completed" else 2
        if args.command == "env" and args.env_command in {"prepare-code", "explore", "resume"}:
            from .runtime.explore import explore_environment
            run, proposal = explore_environment(store, getattr(args, "source", None), allow_model_calls=args.allow_model_calls,
                allow_unpriced=args.allow_unpriced, model_budget=args.model_budget, seconds=args.seconds,
                max_calls=args.max_calls, max_rounds=args.max_rounds, max_delegations=args.max_delegations,
                device=getattr(args, "device", "unspecified"), candidate_timeout=args.candidate_timeout, config_path=args.config,
                resume_from=args.run_id if args.env_command == "resume" else None,
                framework=getattr(args, "framework", None), model=getattr(args, "model", None),
                revision=getattr(args, "revision", None), framework_max_mib=args.framework_max_mib,
                prepare_only=args.env_command == "prepare-code",
                announce=None if args.json else lambda value: print(f"run_id={value}", flush=True))
            state = run.state()
            if args.json:
                output({"run_id": run.id, "status": state["status"], "reason": state["stop_reason"], "proposal": proposal,
                        "resume_from": state.get("resume_from"), "cost": state["cost"], "cumulative_cost": state.get("cumulative_cost"),
                        "framework": state.get("framework"), "model": state["task"]["model"], "result_status": state.get("result_status")})
            else:
                print(f"run: {state['status']}; {state['stop_reason']}")
                print(f"cost: {json.dumps(state['cost'])}")
                if state.get("resume_from"):
                    print(f"continued from: {state['resume_from']} (new segment, additional budget)")
                    print(f"cumulative cost: {json.dumps(state['cumulative_cost'])}")
                print(f"decision rounds: {state.get('decision_round', 0)}/{state['exploration_limits']['max_rounds']}")
                if state.get("framework"):
                    print(f"framework commit: {state['framework']['commit'][:12]}; workspace: {state['framework']['path']}")
                    print(f"model directory: {state['task']['model']['path']}")
                if proposal:
                    print("readiness: unverified; installed: false; proposals are not commands to execute")
                    output(proposal["plan"])
                    print(f"saved: {run.path / 'environment' / 'proposal.json'}")
                else:
                    print(f"partial evidence retained: {run.path}")
            return 0 if state["status"] == "completed" and (proposal or state.get("result_status") == "code_prepared") else 2
        if args.command == "env":
            selectors = args.gpus.split(",") if args.gpus else []
            run, plan = plan_environment(store, args.source, args.profile, selectors,
                                         args.gpu_count, args.intent,
                                         announce=None if args.json else lambda value: print(f"run_id={value}", flush=True))
            if plan is None or run.state()["status"] != "completed":
                output(run.state())
                return 2
            if args.json:
                output({"run_id": run.id, "plan": plan})
            else:
                print("mode: read-only plan; no installation, model loading or LLM calls")
                print(f"profiles: {', '.join(sorted(plan['profiles']))}; selected: {plan['selected_profile'] or 'not selected'}")
                print(f"plan: {plan['status']}; environment: {plan['readiness']}")
                print(f"GPU proposal: {plan['gpu_selection']['uuids']} (not reserved)")
                for py in plan["python"]:
                    print(f"python: {py['value']} [{py['role']}] <- {py['source']}")
                for package in plan["packages"]:
                    suffix = " CONFLICT" if package["conflict"] else ""
                    sources = ', '.join(sorted({item['source'] for item in package['evidence']}))
                    print(f"  {package['name']}: {', '.join(package['versions'])}{suffix} <- {sources}")
                    for index in package["indexes"]:
                        print(f"    index: {index}")
                for issue in plan["issues"]:
                    print(f"review: {issue}")
                print(f"saved: {run.path / 'environment' / 'plan.json'}")
            return 0
        if args.command == "demo":
            _, state = demo(store, args.scenario, args.delay, args.seconds,
                            announce=lambda run_id: print(f"run_id={run_id}", flush=True))
            output(state)
            return 0 if state["status"] == "completed" else 2
        if args.command == "runs":
            output([{k: r.get(k) for k in ("id", "kind", "status", "phase", "created_at")} for r in store.runs()])
            return 0
        run = store.open_run(args.run_id)
        if args.command == "status":
            state = run.status()
            if state.get("active_call"):
                directory = run.path / "calls" / state["active_call"]
                state["active_call_detail"] = read_json(directory / "record.json")
                if (directory / "progress.json").exists():
                    state["tool_progress"] = read_json(directory / "progress.json")
            output(state)
        elif args.command == "cancel":
            output({"run_id": run.id, "cancellation_requested": run.cancel(args.reason)})
        elif args.command == "events":
            if args.tail <= 0:
                raise ValueError("tail must be positive")
            output([{k: v for k, v in event.items() if k != "snapshot"} for event in run.events()[-args.tail:]])
        else:
            output({"run": run.state(), "calls": run.records("calls"), "experiments": run.records("experiments")})
        return 0
    except (ValueError, OSError, RuntimeError, BusyError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
