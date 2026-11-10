"""Task: directory layout + persistent state (state lives in Sherpa, not in agent context)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from sherpa.util import DATA, read_json, read_yaml, write_json, write_yaml


@dataclass
class Task:
    id: str

    @property
    def dir(self) -> Path:
        return DATA / "tasks" / self.id

    def p(self, *parts: str) -> Path:
        return self.dir.joinpath(*parts)

    # layout
    @property
    def framework(self) -> Path: return self.p("framework")
    @property
    def ledger_dir(self) -> Path: return self.p("ledger")
    @property
    def exp_dir(self) -> Path: return self.p("experiments")
    @property
    def decisions_dir(self) -> Path: return self.p("decisions")
    @property
    def logs(self) -> Path: return self.p("logs")
    @property
    def artifacts(self) -> Path: return DATA / "artifacts" / self.id

    @classmethod
    def new(cls, prefix: str = "t") -> "Task":
        t = cls(f"{prefix}{time.strftime('%Y%m%d-%H%M%S')}")
        t.dir.mkdir(parents=True, exist_ok=False)
        return t

    def context(self) -> dict:
        return read_yaml(self.p("context.yaml"))

    def save_context(self, ctx: dict) -> None:
        write_yaml(self.p("context.yaml"), ctx)

    def state(self) -> dict:
        p = self.p("state.json")
        return read_json(p) if p.exists() else {}

    def save_state(self, st: dict) -> None:
        write_json(self.p("state.json"), st)

    def save_experiment(self, exp: dict) -> None:
        write_yaml(self.exp_dir / f"{exp['id']}.yaml", exp)
