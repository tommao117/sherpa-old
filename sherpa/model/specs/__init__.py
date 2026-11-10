"""Structured specs: hardware, model architecture, workload."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from sherpa.util import read_json, read_yaml

SPECS = Path(__file__).parent

DTYPE_BYTES = {"bf16": 2.0, "f16": 2.0, "f32": 4.0, "q8_0": 34 / 32, "q4_0": 18 / 32, "q4_k_m": 4.85 / 8}


@dataclass
class HwSpec:
    id: str
    name: str
    memory_gb: float
    mem_bw_gbps: float
    peak_tflops: dict

    @classmethod
    def match(cls, gpu_name: str) -> "HwSpec":
        for f in sorted((SPECS / "hw").glob("*.yaml")):
            d = read_yaml(f)
            if any(m.lower() in gpu_name.lower() for m in d["match"]):
                return cls(id=d["id"], name=gpu_name, memory_gb=d["memory_gb"],
                           mem_bw_gbps=d["mem_bw_gbps"], peak_tflops=d["peak_tflops"])
        raise KeyError(f"no hw spec matches GPU '{gpu_name}', add one under {SPECS/'hw'}")


@dataclass
class ArchSpec:
    """Dense decoder-only transformer (Qwen3/Llama family)."""
    family: str
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    tied_embeddings: bool
    weight_dtype: str = "bf16"

    @classmethod
    def from_hf(cls, model_dir: Path, weight_dtype: str = "bf16") -> "ArchSpec":
        c = read_json(Path(model_dir) / "config.json")
        heads = c["num_attention_heads"]
        return cls(family=c.get("model_type", "unknown"), hidden=c["hidden_size"],
                   layers=c["num_hidden_layers"], heads=heads,
                   kv_heads=c.get("num_key_value_heads", heads),
                   head_dim=c.get("head_dim", c["hidden_size"] // heads),
                   intermediate=c["intermediate_size"], vocab=c["vocab_size"],
                   tied_embeddings=c.get("tie_word_embeddings", False), weight_dtype=weight_dtype)

    # parameter counts
    @property
    def layer_params(self) -> int:
        q = self.hidden * self.heads * self.head_dim
        kv = 2 * self.hidden * self.kv_heads * self.head_dim
        o = self.heads * self.head_dim * self.hidden
        mlp = 3 * self.hidden * self.intermediate
        return q + kv + o + mlp

    @property
    def matmul_params(self) -> int:
        """Params touched by GEMMs per token (layers + lm_head)."""
        return self.layers * self.layer_params + self.vocab * self.hidden

    @property
    def weight_bytes_per_step(self) -> float:
        """Bytes of weights read per forward step (embedding lookup is negligible)."""
        return self.matmul_params * DTYPE_BYTES[self.weight_dtype]

    def kv_bytes_per_token(self, kv_dtype: str = "f16") -> float:
        return 2 * self.layers * self.kv_heads * self.head_dim * DTYPE_BYTES[kv_dtype]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Workload:
    id: str
    goal: str
    primary_metric: str
    higher_is_better: bool
    prompt_tokens: int
    output_tokens: int
    seed: int
    concurrency: int
    num_requests: int
    warmup_requests: int
    repeats: int

    @classmethod
    def load(cls, name_or_path: str) -> "Workload":
        p = Path(name_or_path)
        if not p.exists():
            p = SPECS / "workload" / f"{name_or_path}.yaml"
        d = read_yaml(p)
        s = d["synthetic"]
        return cls(id=d["id"], goal=d["goal"], primary_metric=d["primary_metric"],
                   higher_is_better=d["higher_is_better"], prompt_tokens=s["prompt_tokens"],
                   output_tokens=s["output_tokens"], seed=s["seed"], concurrency=d["concurrency"],
                   num_requests=d["num_requests"], warmup_requests=d["warmup_requests"],
                   repeats=d["repeats"])

    def to_dict(self) -> dict:
        return asdict(self)
