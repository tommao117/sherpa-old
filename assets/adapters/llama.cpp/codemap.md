# llama.cpp 源码地图（b12242）

> 做代码级优化时给执行层 agent 的导航。仓库自带 `AGENTS.md`（贡献规范），改代码前先读。

| 账本节点 | 位置 | 说明 |
|---|---|---|
| 调度 / batch 组成 | `tools/server/server-context.cpp` | slot 管理、`update_slots` 主循环、continuous batching、prefill/decode 混排 |
| step（一次 forward） | `src/llama-context.cpp` (`decode`) | ubatch 切分、graph 构建与复用、logits 回拷 |
| 计算图构建 | `src/llama-graph.cpp`, `src/models/` | 各架构的图；Qwen3 在 `src/models/` 下 |
| KV cache | `src/llama-kv-cache.cpp` | unified KV、槽位分配 |
| 采样（CPU） | `common/sampling.cpp`, `src/llama-sampler.cpp` | 每个 token 在 CPU 上对整个 vocab 做 softmax/top-k，是 `step.gpu_idle` 的嫌疑来源 |
| 采样（GPU，实验） | `-bs` / backend sampling, `ggml/src/ggml-cuda/top-k.cu`, `argsort.cu` | |
| CUDA 后端入口 | `ggml/src/ggml-cuda/ggml-cuda.cu` | CUDA graph capture/复用（`GGML_CUDA_DISABLE_GRAPHS`）、算子分派、kernel 融合 |
| GEMV（decode, bf16/f16） | `ggml/src/ggml-cuda/mmvf.cu` | `mul_mat_vec_f`，batch 很小时走这里 |
| GEMM（小 batch, 浮点） | `ggml/src/ggml-cuda/mmf.cu` | `mul_mat_f` |
| GEMM（大 batch） | cuBLAS（`ggml-cuda.cu` 中 `ggml_cuda_op_mul_mat_cublas`） | |
| Attention | `ggml/src/ggml-cuda/fattn*.cu` | flash attention 各变体（vec/tile/mma） |
| 量化 GEMM | `ggml/src/ggml-cuda/mmq.cu`, `mmvq.cu` | |
