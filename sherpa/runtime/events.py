"""In-process event bus (v0). Every event is also appended to the task's event log."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from sherpa.util import append_jsonl, now_iso


@dataclass
class Event:
    type: str
    data: dict = field(default_factory=dict)


class Bus:
    def __init__(self, log: Path):
        self.q: deque[Event] = deque()
        self.log = log

    def emit(self, type_: str, **data) -> None:
        ev = Event(type_, data)
        self.q.append(ev)
        append_jsonl(self.log, {"time": now_iso(), "type": type_, "data": data})

    def pop(self) -> Event | None:
        return self.q.popleft() if self.q else None
