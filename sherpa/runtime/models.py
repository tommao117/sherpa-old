from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TERMINAL = {"completed", "failed", "cancelled", "timed_out", "budget_exhausted", "cleanup_failed"}


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", value):
        raise ValueError("invalid identifier")
    return value


@dataclass(frozen=True)
class Task:
    id: str
    target: dict[str, Any]
    model: dict[str, Any]
    resources: dict[str, Any] = field(default_factory=lambda: {"gpus": [], "selection": "auto"})
    objective: dict[str, Any] = field(default_factory=lambda: {"metric": "throughput", "direction": "maximize"})
    workload: dict[str, Any] = field(default_factory=lambda: {"source": "default", "resolved": False})
    contract: dict[str, Any] = field(default_factory=lambda: {"preserve_behavior": True})
    budget: dict[str, Any] = field(default_factory=lambda: {"seconds": 60, "max_calls": 20, "model_cost_usd": 0})

    def __post_init__(self):
        identifier(self.id)
        if self.target.get("kind") not in {"git", "local"}:
            raise ValueError("target.kind must be git or local")
        key = "url" if self.target["kind"] == "git" else "path"
        if not self.target.get(key):
            raise ValueError(f"target.{key} is required")
        if not isinstance(self.model, dict) or not self.model:
            raise ValueError("model source is required")
        if not isinstance(self.resources.get("gpus"), list):
            raise ValueError("resources.gpus must be a list, including for a single GPU")
        if self.objective.get("direction") not in {"maximize", "minimize"}:
            raise ValueError("objective.direction must be maximize or minimize")
        for key in ("seconds", "max_calls", "model_cost_usd"):
            value = self.budget.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"budget.{key} must be finite")
            if value < 0 or (key != "model_cost_usd" and value == 0):
                raise ValueError(f"invalid budget.{key}")
        if not isinstance(self.budget["max_calls"], int):
            raise ValueError("budget.max_calls must be an integer")

    def snapshot(self) -> dict:
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def load(cls, path: Path) -> "Task":
        return cls(**json.loads(path.read_text()))


@dataclass(frozen=True)
class Action:
    kind: str
    tool: str | None = None
    parameters: dict = field(default_factory=dict)
    reason: str = ""

    def snapshot(self) -> dict:
        if self.kind not in {"tool", "experiment", "stop"}:
            raise ValueError(f"unsupported action: {self.kind}")
        return asdict(self)


@dataclass(frozen=True)
class ToolResult:
    status: str
    output: dict
    call_id: str


@dataclass(frozen=True)
class Experiment:
    id: str
    hypothesis: str
    decision_call: str
    start: dict
    candidate: dict
    measurement_contract: dict
    status: str = "pending"
    calls: list[str] = field(default_factory=list)
    conclusion: str | None = None

    def snapshot(self) -> dict:
        identifier(self.id)
        return asdict(self)
