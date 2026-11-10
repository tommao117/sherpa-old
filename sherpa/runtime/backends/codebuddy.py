"""codebuddy CLI backend (headless `-p` mode). Requires a logged-in CLI (see devlog ENV-4)."""
from __future__ import annotations

import json
import os
import subprocess

from sherpa.runtime.backends.base import Backend, Request

MODEL_TIERS = {"strong": os.environ.get("SHERPA_MODEL_STRONG", "primary-model"),
               "fast": os.environ.get("SHERPA_MODEL_FAST", "fast-model")}


class CodebuddyBackend(Backend):
    name = "codebuddy"

    def __init__(self, binary: str = os.environ.get("SHERPA_CODEBUDDY_BIN", "codebuddy")):
        self.bin = binary

    def call(self, req: Request) -> dict:
        cmd = [self.bin, "-p", "--output-format", "json", "--json-schema", json.dumps(req.schema),
               "--model", MODEL_TIERS.get(req.model, req.model),
               "--max-turns", str(req.budget.get("max_turns", 30))]
        if req.kind == "execute":
            cmd.append("-y")
        else:
            cmd += ["--tools", ""]  # decisions are pure reasoning, no tool use
        cmd.append(req.prompt())
        r = subprocess.run(cmd, cwd=req.workdir, capture_output=True, text=True,
                           timeout=req.budget.get("timeout_s", 3600))
        if r.returncode != 0 or "Authentication required" in r.stdout:
            raise RuntimeError(f"codebuddy failed: {r.stdout[-1000:]} {r.stderr[-1000:]}")
        envelope = json.loads(r.stdout)
        result = envelope.get("structured_output") or envelope.get("result")
        out = json.loads(result) if isinstance(result, str) else result
        return self.validate(req, out)
