# 正确性 oracle 标定（OQ-21）

**结论**：参考取自 HF transformers（fp32），与任何推理框架无关；所有检查都在**题目自身的并发负载下**进行。阈值由正确构建的实测分布 + 故意注入的故障共同确定（`bench/oracle/policy.yaml`）。原始数据：`data/oracle-calibration.json`。

## 设计

```
参考（一次性，HF fp32）           候选（每次评分，负载下）
prompt ids ──► greedy 轨迹 + top-20 logprob     native /completion：tokens（精确 id）+ n_probs
6 个高熵探针 ──► top-64 分布                     500 次 1-token 采样 / 探针（T=0.7, top_k=40, top_p=0.9）
                                                 一个 4096-token 请求（上下文契约）
```

| 检查 | 判据 | 抓什么 |
|---|---|---|
| tokens | 首个分歧点上参考自身的概率差 δ < τ 记为 tie，否则 hard；hard 率必须为 0 | 计算被改坏 |
| logprobs | 公共前缀上被选 token 的 \|Δlogprob\| 均值、top-k KL | API 返回错误概率 |
| sampling | 按 llama.cpp 采样链（top_k → top_p → min_p → temperature）从参考 logits 推出分布；G 检验 vs 蒙特卡洛零分布；nucleus 外采样质量（跨探针汇总） | 采样器语义变化（greedy 永远走不到） |
| context | 题目契约长度（4096）的请求必须被服务 | 静默缩短每请求上下文 |

## 标定结果（Qwen3-0.6B bf16，b12242，背景负载 C=32，每配置 1–2 次）

| 配置 | hard | tie δ 最大 | lp MAE | KL | G/限值 最大 | nucleus 外（汇总） | context | 结论 |
|---|---|---|---|---|---|---|---|---|
| 默认 | 0 | 0.036 | 0.0026 | 0.00014 | 0.61 | ≤0.2% | ✓ | 通过 |
| `-np 32 -c 131072` | 0 | 0.041 | 0.0028 | 0.00014 | 0.45 | ≤0.2% | ✓ | 通过 |
| `-np 32 -c 131072 -bs` | 0 | 0.041 | **0.43** | **27** | 0.60 | ≤0.1% | ✓ | **logprobs 失败** |
| 故障：`--samplers top_k;temperature`（丢 top_p） | 0 | 0.041 | 0.0026 | 0.00013 | 0.62 | **2.7%** | ✓ | **sampling 失败** |
| 故障：`--samplers top_k;top_p`（丢 temperature） | 0 | 0.031 | 0.0027 | 0.00013 | **1.45–3.5** | 0 | ✓ | **sampling 失败** |
| `-np 32 -c 32768` | 0（+3 error） | 0.024 | 0.0027 | 0.00012 | 0.65 | ≤0.1% | **✗** | **context 失败** |

阈值（≥5 倍安全边际）：τ = 0.2，hard 率 = 0，lp MAE ≤ 0.02，KL ≤ 0.005，G ≤ 1.2 × 零分布 q99.9，nucleus 外 ≤ 1%（汇总）/ 4%（单探针）。

## 标定中发现的问题

| # | 发现 | 意义 |
|---|---|---|
| 1 | **`-bs`（backend sampling）在并发负载下返回错误的 logprobs**：单请求时与 CPU 采样完全一致，并发时几乎每个位置都错（误差 0.1–4 nats），但 token 完全正确 | v0 接受了 `-bs`。只看 batch=1 的 greedy token 的门禁对此完全失明（v0 DES-14 的同类问题）。原始数据：`data/bs-logprob-errors.json` |
| 2 | **`-np 32 -c 32768` 把每请求上下文静默缩到 1024**；`-np 32 -c 131072` 在 16GB 上放不下，`--fit`（默认开）**静默把层挪到 CPU 并关闭 FA** | v0 的 `-np 32` 同样把每请求上下文从 40960 切到 1280。题目必须写明上下文契约；参考配置一律加 `-fit off` |
| 3 | OpenAI `logprobs.content` 和原生 `completion_probabilities` 都会**合并不完整 UTF-8 的 byte-fallback token**，不是可靠的 token 序列（prompt t25：`歴` 被拆成两个 token） | oracle 用原生 `/completion` 的 `tokens` 字段取 id，再按 id 对齐概率 |
| 4 | 并发下的 greedy 分歧全部发生在参考自身几乎打平的位置（δ ≤ 0.041 nats） | 无损动作的判据不能是"逐 token 完全一致"，而是"分歧只发生在打平处" |
| 5 | **模型自带的采样默认值会改变请求语义**：转换器把 `generation_config.json` 写进 GGUF，llama-server 以此为默认值。Qwen2.5-1.5B-Instruct 因此默认 `repeat_penalty 1.1`、`top_p 0.8`，"greedy"请求在会重复前文的位置换了 token（框架默认构建 44 条中 34 条 hard 分歧，采样探针 nucleus 外 11%）；显式 `repeat_penalty 1.0` 后与参考逐 token 一致 | 评判者的每个请求（oracle、负载、精度集）都显式给出**全部**采样参数（`loadgen.NEUTRAL_SAMPLING`），任务的采样参数叠加其上 |
| 6 | KL 在原始数据经 JSON 存档后重算时变成 6.7（键从 int 变成 str，候选 top-k 全部查不到） | 只在"读时重算"路径出现，正说明派生量必须从存档重算并有单测覆盖（`test_comparison_survives_json_roundtrip`） |

## 局限

- 采样检验的功效：能抓住丢掉 temperature / top_p 这类语义变化；对很小的分布扭曲（TV < ~0.05）不敏感
- 参考 logits 来自 fp32，候选是 bf16：G 检验留了 1.2 倍余量
- 有损契约改用 GSM8K-100 精度（相对地板构建的下降 ≤ `max_drop`）+ 宽松的采样检查；token 和 logprob 指标照常报告但不作为门禁
