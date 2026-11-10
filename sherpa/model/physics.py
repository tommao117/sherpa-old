"""Physics: theoretical lower bounds from (arch, hw, workload, effective config).

All outputs are in **ms per output token** so they can be added up along the ledger tree.
Only dense decoder-only models on a single GPU are covered in v0.
"""
from __future__ import annotations

from sherpa.model.specs import ArchSpec, HwSpec, Workload


def _lb_ms(flops: float, bytes_: float, hw: HwSpec, peak_key: str) -> tuple[float, str]:
    t_compute = flops / (hw.peak_tflops[peak_key] * 1e12)
    t_mem = bytes_ / (hw.mem_bw_gbps * 1e9)
    return max(t_compute, t_mem) * 1e3, ("compute" if t_compute > t_mem else "memory")


def bounds(arch: ArchSpec, hw: HwSpec, wl: Workload, decode_batch: int,
           kv_dtype: str = "f16", peak_key: str = "bf16_tensor") -> dict:
    """Lower bound per output token, split by ledger op node."""
    B = max(1, decode_batch)
    P, O = wl.prompt_tokens, wl.output_tokens
    ctx_avg = P + O / 2
    kv_tok = arch.kv_bytes_per_token(kv_dtype)

    # ---- decode step at batch B ----
    gemm_ms, gemm_bound = _lb_ms(2 * arch.matmul_params * B, arch.weight_bytes_per_step, hw, peak_key)
    attn_flops = B * 4 * arch.layers * arch.heads * arch.head_dim * ctx_avg
    attn_ms, attn_bound = _lb_ms(attn_flops, B * ctx_avg * kv_tok, hw, peak_key)
    decode_gemm_per_tok = gemm_ms / B
    decode_attn_per_tok = attn_ms / B

    # ---- prefill (amortized over output tokens) ----
    pre_gemm_flops = 2 * arch.layers * arch.layer_params * P + 2 * arch.vocab * arch.hidden
    pre_gemm_ms, _ = _lb_ms(pre_gemm_flops, arch.weight_bytes_per_step, hw, peak_key)
    pre_attn_flops = 2 * arch.layers * arch.heads * arch.head_dim * P * P  # causal: QK + PV, half matrix
    pre_attn_ms, _ = _lb_ms(pre_attn_flops, P * kv_tok, hw, peak_key)
    # prefill of B requests can be batched together, weights read once per step
    pre_gemm_per_tok = pre_gemm_ms / O
    pre_attn_per_tok = pre_attn_ms / O

    return {
        "decode_batch": B,
        "decode_bound": gemm_bound,
        "decode_arith_intensity": 2 * arch.matmul_params * B / arch.weight_bytes_per_step,
        "ops": {
            "op.gemm": decode_gemm_per_tok + pre_gemm_per_tok,
            "op.attention": decode_attn_per_tok + pre_attn_per_tok,
            "op.other": 0.0,
            "op.memcpy": 0.0,
        },
        "step.gpu_idle": 0.0,
        "detail": {
            "decode_step_ms": gemm_ms + attn_ms,
            "decode_gemm_ms_per_step": gemm_ms,
            "decode_attn_ms_per_step": attn_ms,
            "prefill_ms_per_request": pre_gemm_ms + pre_attn_ms,
        },
    }


def throughput_upper_bound(b: dict) -> float:
    per_tok = sum(b["ops"].values()) + b["step.gpu_idle"]
    return 1000.0 / per_tok if per_tok > 0 else float("inf")
