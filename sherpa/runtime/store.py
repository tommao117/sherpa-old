from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .models import TERMINAL, Task, identifier


class BusyError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def locked(path: Path, blocking=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise BusyError("another run or framework-update operation is active") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def process_identity(pid: int) -> dict | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": fields[19], "state": fields[0]}
    except (OSError, IndexError):
        return None


class Store:
    def __init__(self, root: Path, project: Path | None = None):
        self.root = root.expanduser().resolve()
        self.project = (project or Path(__file__).resolve().parents[2]).resolve()
        if self.root == self.project or self.project in self.root.parents:
            raise ValueError("runtime data must be outside the Sherpa project")

    @contextmanager
    def exclusive(self):
        with locked(self.root / ".active.lock", blocking=False):
            yield

    def create_run(self, task: Task | None, kind="optimize", sources=(), trigger="manual") -> "Run":
        if kind not in {"optimize", "reflection", "environment"}:
            raise ValueError("run kind must be optimize, reflection or environment")
        if kind in {"optimize", "environment"} and task is None:
            raise ValueError("optimization/environment run needs a task")
        if task:
            location = task.target.get("path", task.target.get("url", ""))
            if location and (task.target["kind"] == "local" or Path(location).is_dir()):
                source_path = Path(location).expanduser().resolve()
                if source_path == self.root or source_path in self.root.parents:
                    raise ValueError("data directory cannot be nested inside target input")
            if task.model.get("kind") == "local" and task.model.get("path"):
                model_path = Path(task.model["path"]).expanduser().resolve()
                if model_path == self.root or model_path in self.root.parents:
                    raise ValueError("data directory cannot be nested inside model input")
        for source in sources:
            self.open_run(source).state()
        run_id = f"{kind}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:10]}"
        path = self.root / "runs" / run_id
        path.mkdir(parents=True, exist_ok=False)
        run = Run(self, path)
        run.update("run_created", kind=kind, trigger=trigger, source_runs=list(sources),
                   task=task.snapshot() if task else None, status="created", phase="created",
                   created_at=now(), ended_at=None, version=None, active_call=None,
                   best=None, baseline=None, cost={"seconds": 0, "calls": 0, "model_cost_usd": 0})
        return run

    def open_run(self, run_id: str) -> "Run":
        path = self.root / "runs" / identifier(run_id)
        if not (path / "run.json").is_file():
            raise FileNotFoundError(f"unknown run: {run_id}")
        return Run(self, path)

    def runs(self) -> list[dict]:
        return [read_json(p) for p in sorted((self.root / "runs").glob("*/run.json"))]

    def propose(self, reflection_id: str, plan: dict) -> str:
        reflection = self.open_run(reflection_id).state()
        if reflection["kind"] != "reflection":
            raise ValueError("proposals must originate from a reflection run")
        required = ("problem", "change", "scope", "verification")
        if any(not isinstance(plan.get(k), str) or not plan[k].strip() for k in required):
            raise ValueError(f"proposal requires {required}")
        with self.exclusive():
            proposal_id = "proposal-" + uuid.uuid4().hex[:12]
            record = {"id": proposal_id, "reflection_run": reflection_id,
                      "source_runs": reflection["source_runs"], "plan": plan,
                      "state": "proposed", "created_at": now(), "review": None}
            atomic_json(self.root / "proposals" / proposal_id / "proposal.json", record)
            self.open_run(reflection_id).event("proposal_created", proposal_id=proposal_id)
        return proposal_id

    def review(self, proposal_id: str, reviewer: str, approve: bool, comment: str) -> dict:
        if not reviewer.strip() or not comment.strip():
            raise ValueError("reviewer and review rationale are required")
        path = self.root / "proposals" / identifier(proposal_id)
        with self.exclusive(), locked(path / ".lock"):
            record = read_json(path / "proposal.json")
            if record["state"] != "proposed":
                raise ValueError("proposal has already been reviewed")
            record.update(state="approved" if approve else "rejected",
                          review={"reviewer": reviewer, "approve": approve, "comment": comment, "time": now()})
            atomic_json(path / "review.json", record["review"])
            atomic_json(path / "proposal.json", record)
        return record


class Run:
    def __init__(self, store: Store, path: Path):
        self.store, self.path, self.id = store, path, path.name

    def state(self) -> dict:
        return read_json(self.path / "run.json")

    def _commit(self, entity: str, value: dict, event: str, detail: dict):
        with locked(self.path / ".state.lock"):
            # The journal is authoritative if a crash occurs before snapshot replacement.
            journal = self.path / "events.jsonl"
            if journal.exists():
                with journal.open("r+b") as handle:
                    handle.seek(0, 2)
                    end = handle.tell()
                    cursor = end
                    while cursor:
                        cursor -= 1
                        handle.seek(cursor)
                        if handle.read(1) == b"\n":
                            cursor += 1
                            break
                    if cursor < end:
                        handle.truncate(cursor)
                        handle.flush()
                        os.fsync(handle.fileno())
            item = {"time": now(), "event": event, "entity": entity, "snapshot": value, **detail}
            with journal.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if entity:
                atomic_json(self.path / entity, value)

    def update(self, event: str, **changes):
        # A run has one execution writer; cancellation uses its own request file.
        path = self.path / "run.json"
        value = read_json(path) if path.exists() else {"id": self.id, "schema_version": 1}
        if value.get("status") in TERMINAL and changes.get("status", value["status"]) != value["status"]:
            raise ValueError("terminal run cannot be restarted")
        value.update(changes, updated_at=now())
        self._commit("run.json", value, event, {})

    def event(self, event: str, **detail):
        self._commit("", None, event, detail)

    def save(self, collection: str, item_id: str, value: dict, event: str):
        if collection not in {"calls", "experiments"}:
            raise ValueError("unknown entity collection")
        self._commit(f"{collection}/{identifier(item_id)}/record.json", value, event, {"item_id": item_id})

    def records(self, collection: str) -> list[dict]:
        if collection not in {"calls", "experiments"}:
            raise ValueError("unknown entity collection")
        return [read_json(p) for p in sorted((self.path / collection).glob("*/record.json"))]

    def recover_snapshots(self):
        with locked(self.path / ".execution.lock", blocking=False), locked(self.path / ".state.lock"):
            for item in self.events():
                entity = item["entity"]
                if entity:
                    destination = (self.path / entity).resolve()
                    if self.path.resolve() not in destination.parents:
                        raise ValueError("journal contains unsafe entity path")
                    atomic_json(destination, item["snapshot"])

    def events(self) -> list[dict]:
        out = []
        path = self.path / "events.jsonl"
        if not path.exists():
            return out
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.endswith("\n"):
                    break
                out.append(json.loads(line))
        return out

    def cancel(self, reason="user requested cancellation"):
        with locked(self.path / ".cancel.lock"):
            if self.state()["status"] in TERMINAL:
                return False
            atomic_json(self.path / "cancel.json", {"time": now(), "reason": reason})
            self.event("cancel_requested", reason=reason)
        return True

    def cancellation(self) -> dict | None:
        path = self.path / "cancel.json"
        return read_json(path) if path.exists() else None

    def status(self) -> dict:
        value = self.state()
        owner = value.get("owner")
        if owner:
            actual = process_identity(owner["pid"])
            value["owner_alive"] = bool(actual and actual["state"] not in {"Z", "X"}
                                        and actual["start_ticks"] == owner["start_ticks"])
            value["interrupted"] = value["status"] == "running" and not value["owner_alive"]
        return value
