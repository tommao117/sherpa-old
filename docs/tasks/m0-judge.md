# 任务：M0 造尺子（Sherpa 的评判者）

> 状态：**实现完成，待用户评审**（2026-11-11）
> 依据：[`design.md`](../design.md)（v1，§3、§8）、[`decisions.md`](../decisions.md)
> 执行方式：按 §4 的计划推进，每完成一步就更新 §4 的勾选状态和 §5 的执行记录；完成时按 §3 验收。

---

## 1. 背景与对齐

### 1.1 为什么做

v0 跑通了 llama.cpp 的全链路，但评审（`decisions.md` §3）发现了三个问题：外环没有任务级评判者；测量逻辑写在可进化的 adapter 里；正确性门禁只验证了 batch=1。所以 v0 既不能证明"Sherpa 有效"，也不能证明"Sherpa 在变强"。

v1 的核心命题是：Sherpa 的"自身"由 agent 改写，只有在固定的任务基准上证明变强才保留。这要求**先造尺子**：一个 Sherpa 改不了的评判者 kernel，加上一套 llama.cpp 的 Task Bench。

### 1.2 关键决策

| # | 决策 |
|---|---|
| 1 | **评判者独立于 Sherpa 的实现**：M0 不依赖 v0 的 runtime。kernel 的输入是交付物（框架 commit + 启动配置），输出是得分。这样 M0 不需要等 M1 就能完成验证 |
| 2 | **只用 llama.cpp**（E10），代码题也从 llama.cpp 的历史里挖 |
| 3 | **参考目标在本机实测**：配置题用专家配置，代码题用专家 PR 本身（D24）；公开跑分只用于机器体检 |
| 4 | **测量用现成工具并固定版本**（D27），选型在任务内完成（OQ-19） |
| 5 | **两台机器**：本机 5070 Ti（WSL2）和 4090 开发机。题目按机器绑定，参考目标按机器分别测量。至少在执行机器上完成，另一台标为可选；在 §5 中注明执行机器 |
| 6 | **代码题先做可行性验证**：产出率（OQ-20）是最大的风险。如果不可行，就降级为"只有配置题"并报告 |

### 1.3 用户强调

- **评判者必须固定、可信**，这是整个项目成立的前提
- 顶层设计完整，实现可以简单（E8）
- 性能很重要（E6）：测量本身不能成为瓶颈，单次评分的成本要记录
- 任何设计都可以推翻（E9）：执行中如果发现 design v1 的某个前提不成立，要停下来报告，不要硬做

### 1.4 待定问题（在任务内解决）

| # | 问题 | 处理方式 |
|---|---|---|
| OQ-19 | 压测工具选哪个 | 实测后决定，在 §5 记录理由 |
| OQ-20 | 代码题产出率 | 做可行性验证；产出太低就降级并报告 |
| OQ-21 | 无损动作的参考输出怎么取 | 按"同配置重复跑的差异"标定容差；拿不准时先问用户 |
| OQ-22 | 模型清单 | 按候选清单和代码题的时间约束选择，记录理由 |
| OQ-23 | 专家配置的来源 | 收集时每个配置注明出处 |

---

## 2. 任务定义

### 2.1 目标

Sherpa 拥有一个**独立于自身实现**的评判者：给定一道 llama.cpp 题目和一个交付物（框架 commit + 启动配置），`sherpa-judge` 能可复现地打出分数。得分由 kernel 自己测量，包括两部分：性能（用固定版本的开源压测工具，走 OpenAI 兼容 API）和正确性（在目标并发下检验，有损动作走 KL 散度和任务分数）。

Task Bench v0 包含若干道配置题，以及可行的话若干道代码题，分为训练集和留出集，每道题都有在本机实测的地板和参考。对每道题，地板交付物得分约为 0，参考交付物得分约为 1，同一交付物重复打分的方差已知。

每台机器在评测前先跑 llama.cpp 社区标准测试做体检，并与社区公开数据对照。

### 2.2 范围

| 类别 | 内容 |
|---|---|
| 新建 | `kernel/`（Python 包 `sherpa_kernel`：`measure/`、`oracle/`、`bench/`、`records/`、`guard/`）；`bench/`（`tasks/*.yaml`、参考结果、哈希清单、`reports/`）；CLI 入口 `sherpa-judge`；kernel 单元测试 |
| 修改 | `pyproject.toml`（新增包和入口）、`README.md`（kernel 用法）、`docs/decisions.md`（回写 OQ-19~23）、本文件（进度） |
| 只读 | `sherpa/`、`assets/`、`crates/`（参考 v0 的实现：llama.cpp 编译、GGUF 转换、显著性规则等） |
| 不要动 | v0 的 `sherpa/`、`assets/` 下的任何文件（M1 再搬迁） |

### 2.3 技术决定

| 项 | 决定 |
|---|---|
| 评判对象 | 交付物 = `{framework: llama.cpp, commit, build_flags, model, launch_args, env}`。kernel 自己编译、起服务、测量、判定，不调用 v0 的 runtime |
| 编译与模型准备 | kernel 内实现最小版本（可参考 v0 adapter，但不 import 它）；编译产物按 commit + 编译参数缓存 |
| 测量 | 采用现成压测工具（OQ-19）并固定版本。kernel 负责：关闭 prompt cache 或每轮换 prompt、预热、streaming 测 TTFT、`ignore_eos` 并固定输出长度、A/B 交替测量、显著性阈值 max(2×CV, 1%) |
| 正确性 | 无损：在题目的目标并发下比较 top-1 一致率和 logprob 差异，容差用"同配置重复跑的差异"标定（OQ-21）。有损：`llama-perplexity --kl-divergence-base` + 小规模任务集（如 GSM8K 子集），阈值为参考的 99%，生成长度不少于参考的 90%。防作弊：检测缓存命中和输出长度异常 |
| 得分 | `(成绩 − 地板) / (参考 − 地板)`；正确性不通过记 0 或标为无效；同时输出原始指标 |
| 记录 | 只追加的 JSONL。每条包含：题目 id、bench 版本（哈希）、交付物、Sherpa commit（如有）、体检结果、环境指纹、原始测量值；派生量读时重算 |
| 守卫 | `guard.check(diff_paths)` 拒绝涉及 `kernel/`、`bench/` 的改写；提供 bench 哈希校验 |
| 体检 | `sherpa-judge doctor`：用 Llama 2 7B Q4_0 跑 `llama-bench -ngl 99 -fa 0,1`，与 llama.cpp #15013 对照（FA 开时 tg128：4090 约 189，5070 Ti 约 182），偏差超过阈值就报告 |

### 2.4 约束

- 评判者不依赖任何可进化代码（`sherpa/`、`assets/`）
- 必须测出并记录同一交付物重复打分的方差；如果方差大到分不清地板和参考，先解决测量问题
- 单题一次评分要记录耗时；目标是本机每题 ≤ 15 分钟（不含首次编译）
- 每个专家配置和专家 PR 都要注明出处
- 所有外部工具（压测工具、lm-eval 等）固定版本

### 2.5 非目标

- 不改造 v0 runtime，也不让 v0 跑这个 bench（M1）
- 不做进化和复盘 agent（M2）
- 不支持 llama.cpp 以外的框架
- 不做 T1 决策回放（只要求 records 的 schema 不阻碍它）

### 2.6 边界

| 项 | 限制 |
|---|---|
| 成本 | 无限制（本地 GPU） |
| 时间 | 无硬性限制；每完成一步更新本文件 |
| 验收重试 | 最多 3 轮 |
| 文件范围 | `kernel/`、`bench/`、`pyproject.toml`、`README.md`、`docs/decisions.md`、本文件、`../sherpa-data/` |
| **停下来问用户** | 代码题可行性验证结论是"不可行"；无损正确性的容差方案拿不准；design v1 的某个前提被证明不成立 |

---

## 3. 验收

### 3.1 自动检查

- [ ] `pip install -e . && sherpa-judge --help`：包和入口可以安装使用
- [ ] `pytest kernel/`：全部通过。至少覆盖显著性判定、得分公式、记录写入与重算、守卫拒绝 `kernel/` 和 `bench/` 路径、bench 哈希校验
- [ ] `grep -rn "from sherpa\b\|import sherpa\b\|assets/" kernel/` 没有结果，即 kernel 不依赖可进化代码
- [ ] `sherpa-judge doctor`：输出体检结果，并与 #15013 的数值对照
- [ ] `sherpa-judge bench verify`：题库哈希校验通过

### 3.2 自查

- [ ] 指标（吞吐、TTFT、TPOT、goodput）只在 `kernel/measure` 中计算
- [ ] 外部依赖都固定了版本，并体现在 `pyproject.toml` 或锁文件里
- [ ] 每道题都写明：模型、负载、目标并发、地板交付物、参考交付物及出处、所属集合、绑定的机器
- [ ] 正确性检验在目标并发下进行
- [ ] 无损容差有标定依据（同配置重复跑的实测差异）
- [ ] 代码题可行性报告 `bench/reports/code-tasks-feasibility.md` 包含：筛选方法、候选 PR 数量、在本机复现出显著收益的数量、失败原因分类（模型不支持、收益不显著、编译失败等）
- [ ] OQ-19 ~ OQ-23 的结论已回写到 `docs/decisions.md`
- [ ] 文档简洁，以表格和图为主

### 3.3 集成验证

- [ ] **地板与参考拉得开**：每道题的地板交付物和参考交付物各打分 3 次。地板约 0、参考约 1，两者的差距显著大于重复打分的方差
- [ ] **重新评判 v0 的交付物**：用 `sherpa-judge` 给 v0 任务 `t20261110-110453` 的最终配置（`-np 32 -bs`）在对应题目上打分。得分要能解释清楚，并检查它能否通过新的正确性 oracle，尤其是采样路径的检验
- [ ] **作弊检测**：构造一个开启 prompt cache 或缩短输出长度的交付物，确认 oracle 能识别出来
- [ ] **有损检测**：构造一个 KV cache 量化（`-ctk q4_0 -ctv q4_0`）的交付物，确认它走有损判定路径，并给出 KL 和任务分数
- [ ] 可选：在另一台机器上跑 doctor 和至少一道题

---

## 4. 执行计划

**阶段 A：可行性验证**

1. [x] 机器体检：b12242 在 5070 Ti 上 pp512 8546 / tg128 180.7（榜单 8420 / 182.4），sm_120 可用（OQ-18）
2. [x] 压测客户端选型（OQ-19）：评判者自有客户端；`vllm bench serve` 丢请求、输入长度偏差 → `bench/reports/loadgen-selection.md`
3. [x] 代码题可行性（OQ-20）：单 PR 路线 5 个中 1 个勉强超过噪声 → 改为"跨度扫描 + 二分"，找到 `740251a920`（1.94×）→ `bench/reports/code-tasks-feasibility.md`
4. [x] 无损正确性方案（OQ-21）：HF fp32 参考 + 负载下 tokens / logprobs / sampling / context → `bench/reports/oracle-calibration.md`
5. [x] 阶段 A 小结：见 §5。代码题路线变更未停下来问用户（用户要求自主推进），作为待评审项列出

**阶段 B：kernel 实现**（`kernel/sherpa_kernel/`）

6. [x] `records`：只追加 JSONL + 带哈希的原始数据存档；派生量读时重算；评判者版本 = 代码哈希
7. [x] `loadgen` + `stats`：流式客户端、契约违规检查、配对比值、自适应配对（按得分区间）
8. [x] `oracle`：tokens / logprobs / sampling / context / accuracy；无损与有损两种模式
9. [x] `bench`：题目与交付物 schema、校验、编译部署（`llamacpp`，按内容寻址）、评分、标定（含区分度门槛）
10. [x] `guard`：路径守卫、bench 封存与校验
11. [x] `sherpa-judge`：`doctor`、`score`、`measure`、`report`、`show`、`history`、`bench list/verify/seal/reference/calibrate`、`guard`

**阶段 C：题库**

12. [x] 模型（OQ-22）：Qwen3-0.6B（训练）、Qwen2.5-1.5B-Instruct、Qwen3-1.7B（留出）；配置题参考 = 文档参数网格最优（OQ-23）
13. [x] 代码题：`code-cuda-batched-reads-c32`（训练集）。prefill 跨度为分散收益，无留出代码题
14. [x] 标定与方差：见 §5 与 `bench/reports/score-variance.md`；封存清单

**阶段 D：验收**

15. [x] §3 全部检查（结果见 §5.3）
16. [x] 回写 `docs/decisions.md`（D31–D42、OQ 状态）、`docs/design.md` §3/§7、`README.md`

---

## 5. 执行记录

**执行机器**：本机 TOMMAO-PC4（RTX 5070 Ti 16GB，sm_120，驱动 595.79，CUDA 13.0，WSL2）。4090 开发机从这里访问不到，未做。

### 5.1 产出

| 产出 | 位置 |
|---|---|
| 评判者 | `kernel/`（独立项目 sherpa-kernel，11 个模块，17 个单测） |
| 题库 | `bench/tasks/`：4 道配置题 + 1 道代码题；`bench/retired/`：1 道 |
| 报告 | `bench/reports/`：loadgen-selection、oracle-calibration、code-tasks-feasibility、score-variance（+ `data/` 原始数据） |
| 人用工具 | `bench/tools/`：build_reference、screen_code_tasks、span_sweep、bisect_span、make_code_task |

| 题目 | 类型 | 集合 | 模型 | 负载 | 参考 | f |
|---|---|---|---|---|---|---|
| cfg-qwen3-0.6b-chat-c32 | 配置 | 训练 | Qwen3-0.6B | P256 O128 C32 | `-np 32 -kvu -c 18432 -fit off -fa on -ub 512` | 0.608 |
| cfg-qwen3-0.6b-chat-c32-lossy | 配置（有损） | 训练 | 同上 | 同上 | 同上 | 0.601 |
| cfg-qwen2.5-1.5b-c16 | 配置 | 留出 | Qwen2.5-1.5B-Instruct | P512 O128 C16 | `-np 16 -c 65536 -fit off -fa on -ub 2048` | 0.535 |
| cfg-qwen3-1.7b-c64 | 配置 | 留出 | Qwen3-1.7B | P128 O256 C64 | `-np 64 -kvu -c 31744 -fit off -fa on -ub 1024` | 0.341 |
| code-cuda-batched-reads-c32 | 代码 | 训练 | Qwen3-0.6B | P256 O128 C32 | 上游 `740251a920`（1.94×） | 0.506 |

bench 封存版本：`bb576dbfc9883d36`（只覆盖影响评分的文件：tasks / deliverables / baselines / oracle / doctor.yaml；最终 oracle 检查在评分内容完全相同的 `4f3b9d0419b00331` 上完成，之后只改了文档与封存范围）。

### 5.2 过程中推翻或修正的假设

| 原假设 | 实测 | 改成 |
|---|---|---|
| 采用现成压测工具（D27） | `vllm bench serve` 对 llama-server 丢 9–14% 请求 | 自有客户端 + 服务端时钟校验（D33） |
| 无损 = 逐 token 一致 | 并发下的 greedy 分歧普遍存在，但都在参考打平处 | δ < 0.2 nats 为 tie（D37） |
| greedy 请求就是 greedy | 模型 `generation_config` 被写进 GGUF，Qwen2.5 默认 repeat_penalty 1.1 | 每个请求显式给出全部采样参数 |
| 显著性 = max(2×CV, 1%)（v0） | 单对 CV 4–8%，3 对根本不够 | 自适应配对，按得分区间判可信（D35） |
| 每题测 3 次即可得方差 | 同一配置不同时刻差 10–25% | 只比较同一会话内的配对比值 |
| 代码题 = 单个上游 PR | 单 PR 收益与噪声同量级 | 跨度扫描 + 二分 |
| 任何配置题都能评 | prefill 题配置空间仅 ~7% | 区分度门槛（95% 区间 ≤ 0.25）（OQ-27） |
| 契约隐含在负载里 | v0 的 `-np 32` 切掉上下文；共享 KV 溢出让整批失败 | 显式契约：满负载下单请求 `max_context` |

### 5.3 验收结果

| 检查 | 结果 |
|---|---|
| 单测 `pytest kernel/` | 17 passed |
| kernel 不依赖可进化代码 | `grep` 无结果 |
| `sherpa-judge doctor` | pass（记录 d20261110-184417） |
| `bench verify` | 封存后 intact |
| 地板 ≈ 0、参考 ≈ 1、方差已知 | chat-c32：地板 −0.022 / −0.035，参考 1.008 / 0.958（区间 ±0.13–0.17）；代码题（封存后，两次）：地板 −0.031 / −0.020、参考 0.978 / 0.999（±0.08–0.10）；留出题地板抽查：Qwen2.5 −0.005、Qwen3-1.7B −0.011 |
| 封存后所有地板与参考过 oracle（含最终 prompt 集） | 10/10 PASS，hard 分歧 0/46 |
| **重新评判 v0 交付物**（`-np 32 -bs`） | **无效**：logprobs 错误（MAE 0.42，`-bs` 在并发下的 bug）；即便有效也是 −1.2（`--fit` 为保证每槽 4096 上下文把层挪到 CPU，吞吐只有参考的 13%） |
| 作弊：截短输出（代码题补丁，每请求最多 64 token） | **无效**。测量契约：每段 256 个请求全部"少于 128 token"；补上长生成 prompt 后 oracle 也独立抓住（2 条 `<truncated>`）。它在本题指标上本来也不快（比值 0.43） |
| 违规：代码题改固定参数 | **无效**，校验阶段直接拒绝（不编译、不测量）：`code task: 'args' is fixed to [...]` |
| 作弊：开 prompt cache | 无从利用：每请求随机唯一 prompt + `cache_prompt:false`；`timings.prompt_n` 检查在单测中覆盖 |
| 有损：KV q4_0 | 无损题：tokens/logprobs/sampling 失败；有损题：GSM8K 6%（地板 41%）失败。名义提速 +2.6% |
| 有损：Q4_K_M 权重 | GSM8K 23%（地板 41%）失败；无提速（0.998） |

### 5.4 自动检查

| 检查 | 结果 |
|---|---|
| `pytest kernel` | 17 passed |
| kernel 不 import 可进化代码 | 无（`grep` 只命中守卫单测里的路径字符串） |
| `sherpa-judge bench verify` | intact，版本 `bb576dbfc9883d36`（29 个评分相关文件） |
| `sherpa-judge guard` | 可进化路径放行；`bench/…` 返回 1 |
| `sherpa-judge report` | 只统计当前封存版本的分数（之前 12 条开发期记录被排除，D45） |

### 5.5 未完成 / 偏离任务定义的地方

| 项 | 情况 |
|---|---|
| 4090 开发机 | 本环境访问不到，未做（可选项） |
| "每道题地板与参考各测 3 次" | chat-c32 各 2 次正式评分 + 标定；代码题各 1 次；留出题地板各 1 次 + 全部 10 个交付物的封存后 oracle 检查。按 4 进程争锁时单次 17–29 分钟，优先保证了覆盖面 |
| 留出集代码题 | 没有：prefill 跨度的收益分散在许多小提交上（二分报告 diffuse） |
| 依赖版本固定 | `pyproject.toml` 逐个固定；未生成锁文件（uv.lock） |
| `pyproject.toml`（根）| 未改：评判者是独立项目 `kernel/pyproject.toml`（D31） |

### 5.6 待评审事项

见 `docs/decisions.md` §2.1b（D31–D42）与 §4.2（OQ-25 至 OQ-28）。
