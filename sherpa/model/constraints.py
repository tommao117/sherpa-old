"""Non-time constraint dimensions: capacity (VRAM) and precision.

The time ledger cannot express "this batch size OOMs" or "this quantization loses accuracy";
these are checked here and exposed as features/limits for the planner.
"""
from __future__ import annotations

from sherpa.model.specs import DTYPE_BYTES, ArchSpec, HwSpec, Workload

LOSSLESS_WEIGHT_DTYPES = {"bf16", "f16", "f32"}


def capacity(arch: ArchSpec, hw: HwSpec, wl: Workload, kv_dtype: str = "f16",
             runtime_overhead_gb: float = 1.5, ctx_per_slot: int | None = None) -> dict:
    weights_gb = arch.matmul_params * DTYPE_BYTES[arch.weight_dtype] / 1e9
    ctx = ctx_per_slot or (wl.prompt_tokens + wl.output_tokens)
    kv_per_slot_gb = ctx * arch.kv_bytes_per_token(kv_dtype) / 1e9
    free_gb = hw.memory_gb - weights_gb - runtime_overhead_gb
    max_slots = int(free_gb / kv_per_slot_gb) if kv_per_slot_gb > 0 else 0
    return {
        "weights_gb": round(weights_gb, 3),
        "kv_per_slot_gb": round(kv_per_slot_gb, 4),
        "max_slots_by_memory": max_slots,
        "max_useful_batch": min(max_slots, wl.concurrency),
    }


def precision(weight_dtype: str, kv_dtype: str) -> dict:
    lossy = weight_dtype not in LOSSLESS_WEIGHT_DTYPES or kv_dtype not in LOSSLESS_WEIGHT_DTYPES
    return {"lossy": lossy, "weight_dtype": weight_dtype, "kv_dtype": kv_dtype}
