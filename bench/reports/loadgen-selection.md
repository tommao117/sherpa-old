# 压测客户端选型（OQ-19）

**结论**：用评判者自有的流式客户端（`kernel/sherpa_kernel/loadgen.py`，约 200 行 aiohttp），并用**服务端自报的逐请求计时**做独立校验；不采用 `vllm bench serve` / GuideLLM 作为测量实现。

## 实测对比（llama.cpp b12242，Qwen3-0.6B bf16，RTX 5070 Ti，P=256 O=128 N=128）

| 项 | `vllm bench serve`（vLLM 0.22 dev） | 评判者客户端 |
|---|---|---|
| 请求成功率（`-np 32`, C=32） | **86–91%**（114/110/117 of 128，ServerDisconnected） | 100%（`force_close`，每请求一个连接） |
| 实际输入 token（应为 32768） | **28160–29952**（random 数据集先解码成文本，服务端重新分词） | 32768（直接发送 token id） |
| 默认配置下吞吐 | 659 / 667 / 665 tok/s（含失败请求） | 673 / 701 / 710 tok/s |
| 逐请求原始数据 | 只有汇总 | 每请求每个 chunk 的时间戳 + usage + timings，可重算 |
| 依赖 | 整个 vLLM + torch | aiohttp |

## 独立校验：客户端时钟 vs 服务端时钟

服务端在每个请求的最后一个 SSE 事件里报告 `predicted_ms` / `predicted_per_token_ms`。两者独立计时：

| 并发 | 吞吐 tok/s | TPOT 客户端 / 服务端 (ms) | decode 跨度 客户端/服务端 |
|---|---|---|---|
| 1 | 374 | 2.442 / 2.443 | 0.9994 |
| 32 | 1246 | 23.161 / 23.197 | 0.9985 |
| 64 | 1288 | 46.505 / 46.515 | 0.9998 |

偏差 < 0.15%：客户端在 C ≤ 64 时不是瓶颈，也不引入偏差。TTFT 高于服务端 `prompt_ms` 是预期的（包含排队和首个 decode step）。

## 契约中由客户端强制的项

| 陷阱 | 处理 |
|---|---|
| prompt cache / 前缀复用 | 每个请求、每次会话的 prompt 都是随机且唯一的 token 序列；另发 `cache_prompt: false`；用 `timings.prompt_n` 检测跳过的 prefill |
| 输出长度 | `ignore_eos` + `max_tokens`；`usage.completion_tokens` 必须等于要求值 |
| 服务端虚报 | 流出的 chunk 数 < usage 的 50% 视为虚报（chunk 可合并多个 token：服务端会扣住不完整的 UTF-8 字节） |
| keep-alive 断连 | 每请求独立连接 |

## 何时需要重新评估

- 并发 > 64 或单机吞吐远高于 ~1300 tok/s 时，重复上面的服务端时钟校验；偏差 > 1% 时换 Rust 客户端（design E6）
- 接入非 llama.cpp 框架时，`timings` 字段不可用，需要另找独立时钟（例如服务端 `/metrics`）
