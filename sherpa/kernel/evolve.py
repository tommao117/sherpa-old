"""Outer loop: asset change proposal -> metabench -> apply (only between tasks).

Permission tiers (design §7.1):
  assets/knowledge/**                       auto-apply if metabench passes
  assets/{tools,playbooks,policies,adapters}/**  require human approval
  anything outside assets/                  rejected
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import yaml

from sherpa.kernel import facts as F
from sherpa.kernel import knowledge as K
from sherpa.kernel import metabench
from sherpa.kernel.invariants import assets_fingerprint
from sherpa.runtime.backends import Request
from sherpa.util import ASSETS, DATA, append_jsonl, now_iso, read_json, read_jsonl, write_json

AUTO = ("knowledge/",)
HUMAN = ("tools/", "playbooks/", "policies/", "adapters/")
ACTION_REQUIRED = ["id", "node", "kind", "when", "effect", "apply"]

SCHEMA = {
    "type": "object", "required": ["summary", "changes"],
    "properties": {
        "summary": {"type": "string"},
        "changes": {"type": "array", "items": {
            "type": "object", "required": ["path", "op", "rationale"],
            "properties": {"path": {"type": "string"}, "op": {"enum": ["create", "replace", "delete"]},
                           "content": {"type": "string"}, "rationale": {"type": "string"}}}}}}

INSTRUCTIONS = """# Evolve：基于事实提出 Sherpa 资产变更

你在改进 Sherpa 自己（不是推理框架）。输入是最近任务的事实（实验 + 观测 + 分析）和当前资产。
输出资产变更提案，会先经过 metabench 离线回放（新知识对历史事实的预测误差不能变差），再按权限合入。

## 优先级
1. **knowledge/actions/**：把事实归纳成**机理化**的动作条目。条件写在特征上（brief.feature_names），不写组合名。
   效果优先用 `recover_gap`（相对账本节点差距的比例）或 `change_pct`；要能解释 evidence 中的观测，同时与 counter 不矛盾。
   格式见 brief.knowledge_format。一个条目对应一个动作签名（knob 名集合）。
2. **knowledge/priors/**：效率先验（例如"该卡上 bf16 GEMV 能达到带宽的 X%"）。
3. 其他资产（policies / playbooks / tools / adapters）：只在事实明确暴露问题时提出（analysis 里的 sherpa_issues），这些需要人审。
冲突是信号：同一动作在不同事实里效果相反，说明条件缺变量，要补上条件，而不是取平均。
"""

FORMAT = """id: <kebab-case>
node: <ledger node>            # e.g. request.time_per_token / step.gpu_idle / op.attention
kind: close_gap | move_bound
when: {<feature>: "<op><value>"}   # ops: > >= < <= == != ; or "in [a, b]"
effect: {metric: output_tok_per_s, change_pct: [lo, hi]}   # or recover_gap: [lo, hi]
apply: {<framework>: {knobs: {<knob>: <value or "rule">}}}
evidence: [<fact ids>]
counter: [<fact ids>]
notes: <mechanism explanation>"""


def _pdir(pid: str) -> Path:
    return DATA / "proposals" / pid


def _compact_fact(f: dict) -> dict:
    a = f.get("analysis") or {}
    return {"id": f["id"], "action": {k: f["action"].get(k) for k in ("type", "knobs", "instructions", "signature", "base_config")},
            "features": f["features"], "predicted_pct": f["predicted"].get("delta_pct"),
            "observed_pct": f["observed"].get("delta_pct"), "outcome": f["outcome"], "gates": f["gates"],
            "diagnosis_before": [{k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}
                                 for d in (f["observed"].get("ledger_before") or {}).get("diagnosis", [])[:4]],
            "ledger_after": {k: round(v, 4) for k, v in ((f["observed"].get("ledger_after") or {}).get("measured") or {}).items()},
            "analysis": {k: a.get(k) for k in ("conclusion", "prediction_error_cause", "conditions", "sherpa_issues")}}


def propose(task_ids: list[str], backend) -> str:
    from sherpa.model.features import FEATURE_DOC
    facts = [f for t in task_ids for f in F.by_task(t)]
    if not facts:
        raise ValueError("no facts for given tasks")
    pid = f"p{time.strftime('%Y%m%d-%H%M%S')}"
    d = _pdir(pid)
    d.mkdir(parents=True)
    current = {e["_file"]: {k: v for k, v in e.items() if k != "_file"} for e in K.load()}
    brief = {"tasks": task_ids, "facts": [_compact_fact(f) for f in facts],
             "current_knowledge": current, "metabench_current": {k: v for k, v in metabench.replay(None).items() if k != "rows"},
             "feature_names": FEATURE_DOC, "knowledge_format": FORMAT,
             "asset_tree": sorted(str(p.relative_to(ASSETS)) for p in ASSETS.rglob("*") if p.is_file() and "__pycache__" not in p.parts)}
    req = Request(id=f"{pid}-evolve", kind="decide", point="evolve", instructions=INSTRUCTIONS, brief=brief, schema=SCHEMA)
    from sherpa.runtime.backends.handoff import HandoffBackend
    if isinstance(backend, HandoffBackend):
        backend.dir = d
    out = backend.call(req)
    write_json(d / "proposal.json", {"id": pid, "time": now_iso(), "tasks": task_ids,
                                     "proposer": f"{backend.name}:{req.id}",
                                     "assets_before": assets_fingerprint(), **out})
    return pid


def _tier(path: str) -> str:
    if ".." in Path(path).parts or path.startswith("/"):
        return "reject"
    if path.startswith(AUTO):
        return "auto"
    if path.startswith(HUMAN):
        return "human"
    return "reject"


def _validate_knowledge_file(path: str, content: str) -> list[str]:
    if not path.startswith("knowledge/actions/"):
        return []
    try:
        d = yaml.safe_load(content) or {}
    except yaml.YAMLError as e:
        return [f"{path}: invalid yaml: {e}"]
    errs = [f"{path}: missing {k}" for k in ACTION_REQUIRED if k not in d]
    eff = d.get("effect") or {}
    if not ({"change_pct", "recover_gap"} & set(eff)):
        errs.append(f"{path}: effect needs change_pct or recover_gap")
    return errs


def evaluate(pid: str) -> dict:
    d = _pdir(pid)
    prop = read_json(d / "proposal.json")
    tiers, errors = {}, []
    for c in prop["changes"]:
        tiers[c["path"]] = _tier(c["path"])
        if tiers[c["path"]] == "reject":
            errors.append(f"{c['path']}: outside evolvable assets")
        if c["op"] != "delete":
            errors += _validate_knowledge_file(c["path"], c.get("content", ""))
    with tempfile.TemporaryDirectory() as tmp:
        newk = Path(tmp) / "knowledge"
        shutil.copytree(ASSETS / "knowledge", newk)
        for c in prop["changes"]:
            if c["path"].startswith("knowledge/"):
                p = Path(tmp) / c["path"]
                if c["op"] == "delete":
                    p.unlink(missing_ok=True)
                else:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(c["content"])
        old = metabench.replay(None)
        new = metabench.replay(newk)
    cmp = metabench.compare(old, new)
    # per-change verdicts: a human-tier change must not block the auto-tier ones (devlog DES-20)
    blocked = bool(errors) or not cmp["pass"]
    per_change = {path: ("reject" if (tier == "reject" or blocked)
                         else "auto_apply" if tier == "auto" else "needs_approval")
                  for path, tier in tiers.items()}
    result = {"proposal": pid, "time": now_iso(), "tiers": tiers, "per_change": per_change, "errors": errors,
              "metabench": cmp, "rows_new": new["rows"],
              "pending_approval": sorted(p for p, v in per_change.items() if v == "needs_approval"),
              "verdict": "reject" if blocked else
                         ("auto_apply" if all(v == "auto_apply" for v in per_change.values()) else "partial")}
    write_json(d / "evaluation.json", result)
    return result


def apply(pid: str, approve: bool = False, approver: str | None = None) -> dict:
    """Apply what the caller is allowed to apply; report what stays pending.

    `approver` must differ from the proposer: the thing that evolves cannot approve itself
    (devlog DES-21).
    """
    d = _pdir(pid)
    ev = read_json(d / "evaluation.json")
    prop = read_json(d / "proposal.json")
    if ev["verdict"] == "reject":
        raise PermissionError(f"proposal rejected: {ev['errors'] or ev['metabench']['reasons']}")
    proposer = prop.get("proposer") or "unknown"
    if approve:
        approver = approver or "unknown"
        if approver == proposer:
            raise PermissionError(f"approver must differ from proposer (both '{proposer}'): "
                                  "the thing that evolves cannot approve itself")
    before = assets_fingerprint()
    applied, pending = [], []
    for c in prop["changes"]:
        verdict = ev["per_change"].get(c["path"])
        if verdict == "reject" or (verdict == "needs_approval" and not approve):
            pending.append({"path": c["path"], "verdict": verdict})
            continue
        p = ASSETS / c["path"]
        if c["op"] == "delete":
            p.unlink(missing_ok=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(c["content"])
        applied.append({"path": c["path"],
                        "by": f"human:{approver}" if verdict == "needs_approval" else "metabench"})
    rec = {"proposal": pid, "time": now_iso(), "assets_before": before, "assets_after": assets_fingerprint(),
           "proposer": proposer, "approver": approver if approve else None,
           "applied": applied, "pending": pending}
    append_jsonl(DATA / "proposals" / "applied.jsonl", rec)
    return rec


def history() -> list[dict]:
    return read_jsonl(DATA / "proposals" / "applied.jsonl")
