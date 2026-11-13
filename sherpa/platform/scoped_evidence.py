"""The same evidence tools applied to two explicit, non-interchangeable input roots."""
from __future__ import annotations

import copy
import os
import stat
from pathlib import Path

from .exploration import CATALOG, EvidenceTools, validate


class ScopedEvidence:
    version = "framework-model-evidence-v2"

    def __init__(self, framework, model, scratch, check):
        self.binding = {"framework": {"source": framework["source"], "commit": framework["commit"]},
                        "model": {"kind": "local", "path": model["path"],
                                  "identity": "path plus content hashes of observed files; weights not hashed or loaded"}}
        self.roots = {}
        try:
            self.roots["framework"] = EvidenceTools(Path(framework["path"]), scratch, check, max_text_bytes=8 * 1024 * 1024)
            self.roots["model"] = EvidenceTools(Path(model["path"]), scratch, check)
        except BaseException:
            self.close()
            raise
        self.catalog = copy.deepcopy(CATALOG)
        for name, tool in self.catalog.items():
            if name != "machine":
                tool["parameters"]["scope"] = "REQUIRED: framework or model; paths are relative to the chosen root"
                tool["limits"] += "; framework source text <=8 MiB, model text <=128 KiB; no weight loading"

    def close(self):
        for root in self.roots.values():
            root.close()

    def validate(self, name, parameters):
        if not isinstance(parameters, dict):
            raise ValueError("tool parameters must be an object")
        if name == "machine":
            validate(name, parameters)
        else:
            if parameters.get("scope") not in self.roots:
                raise ValueError("file tools require explicit scope=framework or model")
            validate(name, {k: v for k, v in parameters.items() if k != "scope"})

    def execute(self, name, parameters):
        self.validate(name, parameters)
        scope = parameters.get("scope", "model")
        result = self.roots[scope].execute(name, {k: v for k, v in parameters.items() if k != "scope"})
        if name != "machine":
            result["subject"] = {"scope": scope, **self.binding[scope]}
        return result

    def validate_model_files(self, paths):
        for path in paths:
            try:
                fd = self.roots["model"].open_path(path)
                try:
                    if not stat.S_ISREG(os.fstat(fd).st_mode):
                        raise ValueError("selected model input must be an existing regular file")
                finally:
                    os.close(fd)
            except OSError as error:
                raise ValueError(f"model file {path!r} is unavailable or unsafe; choose an observed file") from error
