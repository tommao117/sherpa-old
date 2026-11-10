"""Time ledger tree. Unit for every node: ms per output token.

    request.time_per_token          1000 / throughput (wall clock, client side)
      request.outside_gpu_window    time not covered by the GPU activity window (queueing, http, startup tail)
      step.gpu_window
        step.gpu_busy
          op.gemm / op.attention / op.other / op.memcpy
        step.gpu_idle               gaps between kernels: CPU scheduling, sampling, launch, sync
"""
from __future__ import annotations

from dataclasses import dataclass, field

TREE = {
    "request.time_per_token": ["request.outside_gpu_window", "step.gpu_window"],
    "step.gpu_window": ["step.gpu_busy", "step.gpu_idle"],
    "step.gpu_busy": ["op.gemm", "op.attention", "op.other", "op.memcpy"],
}
LEAVES = ["request.outside_gpu_window", "step.gpu_idle", "op.gemm", "op.attention", "op.other", "op.memcpy"]
ALL_NODES = ["request.time_per_token", "request.outside_gpu_window", "step.gpu_window", "step.gpu_busy",
             "op.gemm", "op.attention", "op.other", "op.memcpy", "step.gpu_idle"]


@dataclass
class Ledger:
    theory: dict = field(default_factory=dict)    # node -> ms/token lower bound
    measured: dict = field(default_factory=dict)  # node -> ms/token
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_bounds(cls, b: dict, meta: dict | None = None) -> "Ledger":
        th = {n: 0.0 for n in ALL_NODES}
        th.update(b["ops"])
        th["step.gpu_idle"] = b["step.gpu_idle"]
        cls._rollup(th)
        return cls(theory=th, meta={**(meta or {}), "physics": {k: v for k, v in b.items() if k != "ops"}})

    @staticmethod
    def _rollup(d: dict) -> None:
        for parent in ["step.gpu_busy", "step.gpu_window", "request.time_per_token"]:
            kids = TREE[parent]
            if all(k in d for k in kids):
                d[parent] = sum(d[k] for k in kids)

    def fill_measured(self, m: dict) -> None:
        self.measured = dict(m)

    def gaps(self) -> dict:
        return {n: self.measured.get(n, 0.0) - self.theory.get(n, 0.0) for n in ALL_NODES if n in self.measured}

    def diagnose(self) -> list[dict]:
        """Leaves ranked by absolute gap (gap in ms/token already == gap x time share)."""
        total = self.measured.get("request.time_per_token") or 1.0
        out = []
        for n in LEAVES:
            if n not in self.measured:
                continue
            g = self.measured[n] - self.theory.get(n, 0.0)
            out.append({"node": n, "measured": self.measured[n], "theory": self.theory.get(n, 0.0),
                        "gap": g, "gap_share": g / total,
                        "efficiency": (self.theory.get(n, 0.0) / self.measured[n]) if self.measured[n] > 0 else None})
        return sorted(out, key=lambda x: -x["gap"])

    def to_dict(self) -> dict:
        return {"meta": self.meta, "theory": self.theory, "measured": self.measured, "gaps": self.gaps(),
                "diagnosis": self.diagnose() if self.measured else []}

    @classmethod
    def from_dict(cls, d: dict) -> "Ledger":
        return cls(theory=d["theory"], measured=d.get("measured", {}), meta=d.get("meta", {}))

    def render(self) -> str:
        depth = {"request.time_per_token": 0, "request.outside_gpu_window": 1, "step.gpu_window": 1,
                 "step.gpu_busy": 2, "step.gpu_idle": 2, "op.gemm": 3, "op.attention": 3, "op.other": 3,
                 "op.memcpy": 3}
        lines = [f"{'node':34s} {'theory':>9s} {'measured':>9s} {'gap':>9s}   (ms / output token)"]
        for n in ALL_NODES:
            t = self.theory.get(n, 0.0)
            m = self.measured.get(n)
            ms = f"{m:9.4f}" if m is not None else f"{'-':>9s}"
            gs = f"{m - t:9.4f}" if m is not None else f"{'-':>9s}"
            lines.append(f"{'  ' * depth[n] + n:34s} {t:9.4f} {ms} {gs}")
        return "\n".join(lines)
