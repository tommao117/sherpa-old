from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path

from ..runtime.store import digest

DEFAULT = Path(__file__).with_name("models.toml")


class Config:
    def __init__(self, data: dict):
        self.data = copy.deepcopy(data)
        if data.get("version") != 1:
            raise ValueError("unsupported agent configuration version")
        if set(data.get("pools", {})) != {"daily", "expert"}:
            raise ValueError("daily and expert candidate pools are required")
        for role, names in data["pools"].items():
            if not names or len(names) != len(set(names)):
                raise ValueError(f"{role}: candidates must be non-empty and unique")
            for name in names:
                entry = data.get("models", {}).get(name)
                if not entry or entry.get("backend") not in data.get("backends", {}):
                    raise ValueError(f"unconfigured candidate: {name}")
                if entry.get("enabled", True) and (not entry.get("model_id") or entry["model_id"] == "auto"):
                    raise ValueError(f"explicit provider model_id required: {name}")
        self.version = digest(self.data)

    @classmethod
    def load(cls, path: Path | None = None):
        path = path or Path(os.environ.get("SHERPA_AGENTS", DEFAULT))
        return cls(tomllib.loads(path.read_text(encoding="utf-8")))

    def candidates(self, role):
        if role not in self.data["pools"]:
            raise ValueError("unknown agent role")
        return [(name, copy.deepcopy(self.data["models"][name])) for name in self.data["pools"][role]]

    def describe(self):
        return {"config_version": self.version, "verified_online": False,
                "pools": {role: [{"name": name, **entry} for name, entry in self.candidates(role)]
                          for role in ("daily", "expert")}}
