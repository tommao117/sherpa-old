"""Project (model, hw, workload, config, measured ledger) into the feature space.

Knowledge conditions (`when:`) are written against these names, never against combo names.
"""
from __future__ import annotations

from sherpa.model.specs import ArchSpec, HwSpec, Workload

FEATURE_DOC = {
    "ridge_point": "峰值算力 / 带宽 (FLOP/byte)",
    "decode_batch": "实际 decode batch（min(slots, 并发)）",
    "decode_arith_intensity": "decode GEMM 的算术强度 (FLOP/byte)",
    "decode_bound": "memory | compute",
    "concurrency": "负载并发",
    "slots": "服务端并行槽位数",
    "slots_over_concurrency": "slots / concurrency",
    "weight_dtype": "权重精度",
    "kv_dtype": "KV cache 精度",
    "weights_gb": "权重大小",
    "vram_headroom_gb": "显存余量",
    "gpu_idle_ratio": "实测 GPU 空闲占比（相对 GPU 窗口）",
    "outside_window_ratio": "实测 GPU 窗口之外的时间占比",
    "gemm_efficiency": "GEMM 理论下界 / 实测",
    "prompt_tokens": "输入长度",
    "output_tokens": "输出长度",
}


def project(arch: ArchSpec, hw: HwSpec, wl: Workload, physics: dict, effective_config: dict,
            capacity: dict, measured: dict | None = None, theory: dict | None = None) -> dict:
    ridge = hw.peak_tflops["bf16_tensor"] * 1e12 / (hw.mem_bw_gbps * 1e9)
    slots = effective_config.get("slots", 1)
    f = {
        "ridge_point": round(ridge, 1),
        "decode_batch": physics["decode_batch"],
        "decode_arith_intensity": round(physics["decode_arith_intensity"], 2),
        "decode_bound": physics["decode_bound"],
        "concurrency": wl.concurrency,
        "slots": slots,
        "slots_over_concurrency": round(slots / wl.concurrency, 3),
        "weight_dtype": arch.weight_dtype,
        "kv_dtype": effective_config.get("kv_dtype", "f16"),
        "weights_gb": capacity["weights_gb"],
        "vram_headroom_gb": round(hw.memory_gb - capacity["weights_gb"], 2),
        "prompt_tokens": wl.prompt_tokens,
        "output_tokens": wl.output_tokens,
    }
    if measured:
        win = measured.get("step.gpu_window") or 0
        tot = measured.get("request.time_per_token") or 0
        if win:
            f["gpu_idle_ratio"] = round(measured.get("step.gpu_idle", 0) / win, 3)
        if tot:
            f["outside_window_ratio"] = round(measured.get("request.outside_gpu_window", 0) / tot, 3)
        if theory and measured.get("op.gemm"):
            f["gemm_efficiency"] = round(theory["op.gemm"] / measured["op.gemm"], 3)
    return f
