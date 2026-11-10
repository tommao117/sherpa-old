from __future__ import annotations

from pathlib import Path

from sherpa.runtime.backends.base import Backend, Request


def make(name: str, handoff_dir: Path) -> Backend:
    if name == "handoff":
        from sherpa.runtime.backends.handoff import HandoffBackend
        return HandoffBackend(handoff_dir)
    if name == "codebuddy":
        from sherpa.runtime.backends.codebuddy import CodebuddyBackend
        return CodebuddyBackend()
    raise KeyError(f"unknown backend {name}")


__all__ = ["Backend", "Request", "make"]
