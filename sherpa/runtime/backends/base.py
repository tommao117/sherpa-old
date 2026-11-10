"""Agent backend interface (OQ-13).

Two kinds of calls, both stateless (a fresh session each time, OQ-4):
  decide(req)   policy-layer decision: policy text + state brief -> JSON matching req.schema
  execute(req)  execution-layer work in a workdir (edit code, debug build) -> JSON matching req.schema
The backend knows nothing about inference optimization; everything domain-specific is in the request.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema


@dataclass
class Request:
    id: str                     # unique, e.g. "003-plan"
    kind: str                   # "decide" | "execute"
    point: str                  # decision point / task type, e.g. "plan", "code_change"
    instructions: str           # policy or playbook text
    brief: dict                 # structured state given to the model
    schema: dict                # JSON schema of the expected answer
    workdir: Path | None = None
    model: str = "strong"
    budget: dict = field(default_factory=lambda: {"max_turns": 30, "timeout_s": 3600})

    def prompt(self) -> str:
        return (f"{self.instructions}\n\n## State brief\n```json\n{json.dumps(self.brief, indent=2, ensure_ascii=False)}\n```\n\n"
                f"## Output\nRespond with a single JSON object matching this schema:\n"
                f"```json\n{json.dumps(self.schema, indent=2)}\n```\n")


class Backend:
    name = "base"

    def call(self, req: Request) -> dict:
        raise NotImplementedError

    @staticmethod
    def validate(req: Request, out: dict) -> dict:
        jsonschema.validate(out, req.schema)
        return out
