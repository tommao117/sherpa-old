"""Handoff backend: write the request to disk, wait for someone (human or an external agent) to answer.

    <dir>/<id>.request.md     human/agent readable prompt
    <dir>/<id>.request.json   machine readable request
    <dir>/<id>.response.json  written by the responder (validated against schema)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jsonschema

from sherpa.runtime.backends.base import Backend, Request
from sherpa.util import write_json


class HandoffBackend(Backend):
    name = "handoff"

    def __init__(self, directory: Path, poll_s: float = 2.0):
        self.dir = Path(directory)
        self.poll_s = poll_s

    def call(self, req: Request) -> dict:
        self.dir.mkdir(parents=True, exist_ok=True)
        md = self.dir / f"{req.id}.request.md"
        resp = self.dir / f"{req.id}.response.json"
        head = f"# {req.kind}: {req.point} ({req.id})\n\n"
        if req.workdir:
            head += f"workdir: `{req.workdir}`\n\n"
        md.write_text(head + req.prompt() + f"\n\nWrite the answer to `{resp}`.\n")
        write_json(self.dir / f"{req.id}.request.json",
                   {"id": req.id, "kind": req.kind, "point": req.point, "brief": req.brief, "schema": req.schema,
                    "workdir": str(req.workdir) if req.workdir else None, "model": req.model})
        deadline = time.time() + req.budget.get("timeout_s", 3600)
        last_size, stable = -1, 0
        while time.time() < deadline:
            if resp.exists():
                # responders may write non-atomically: wait until the file stops changing (devlog IMP-6)
                size = resp.stat().st_size
                stable = stable + 1 if size == last_size and size > 0 else 0
                last_size = size
                if stable >= 2:
                    try:
                        out = json.loads(resp.read_text())
                    except json.JSONDecodeError as e:
                        (self.dir / f"{req.id}.response.error.txt").write_text(f"not valid JSON yet: {e}")
                        stable = 0
                        time.sleep(self.poll_s)
                        continue
                    try:
                        jsonschema.validate(out, req.schema)
                        return out
                    except jsonschema.ValidationError as e:
                        (self.dir / f"{req.id}.response.error.txt").write_text(str(e))
                        resp.rename(resp.with_name(f"{req.id}.response.invalid.{int(time.time())}.json"))
                        last_size, stable = -1, 0
            time.sleep(self.poll_s)
        raise TimeoutError(f"no response for {req.id} within {req.budget.get('timeout_s')}s")
