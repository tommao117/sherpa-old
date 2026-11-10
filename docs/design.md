# Sherpa 顶层设计（v1）

> 状态：v0 已跑通一次，经独立 review 后重写为 v1。下一步是 **M0：造尺子**（[`tasks/m0-judge.md`](./tasks/m0-judge.md)）。
> 配套文档：[`decisions.md`](./decisions.md)（决策、v0 评审结论、待定问题）、[`devlog.md`](./devlog.md) 与 [`retro-v0.md`](./retro-v0.md)（v0 的原始记录，仅作历史参考）。
>
> **v1 与 v0 的根本区别**：v0 按"代码归人改 / 数据可进化"划分权限，外环用"知识条目对事实的预测误差"做评判。v1 改为按 **评判者 / 被评判者** 划分，外环用 **Task Bench（任务级基准）** 评判，进化的单位是 **对 Sherpa 自身仓库的 commit**。

---

## 0. 定位

Sherpa（夏尔巴人，领路人）是一个**自迭代的推理优化框架**。它**不做推理**，而是去优化已有的推理框架。

```
输入：模型 × 卡 × 推理框架@版本 × 负载        例：Qwen3-0.6B × RTX 4090 × llama.cpp@b12242
  │
  ▼  Sherpa
部署 → bench → profile → 诊断 → 优化（配置 / 代码）→ 验证 → …… → 交付
  │
  ▼
输出：① 交付物（框架分支 + commit + 启动命令）
      ② Sherpa 自身变强（它自己的代码、工具、知识、流程都会被改写）
```

| 特性 | 说明 |
|---|---|
| 框架无关 | 长期目标覆盖 vLLM / SGLang / llama.cpp / 自研框架；**当前阶段只用 llama.cpp 打磨 Sherpa** |
| 动作无边界 | 调参、开关特性、改源码、写 kernel 都可以，门禁就是唯一边界 |
| 自迭代 | base 版本没有专业知识，越用越强；可进化的范围是 Sherpa 的整个"自身" |
| 形态 | 现阶段是 CLI；以后加运营页面，再往后做成平台 |

---

## 1. 第一性原理

### 1.1 推理优化是带约束的搜索

```
给定  Context = (模型, 卡, 框架+版本, 负载, 目标)
寻找  Action  = 配置 / 特性 / 代码改动
使得  目标最优（吞吐、TTFT、TPOT、成本……）
约束  精度不掉、稳定、可复现、可交付
```

### 1.2 专家为什么越做越快

```
解决时间 ≈ 实验次数 × 单次实验成本
             │            └─ 靠【工具】降低
             └─ 靠【判断】降低：方向选得准，知道上限在哪、何时该停
```

| 专家积累的东西 | 泛化性 | 占不占上下文 |
|---|---|---|
| 机理认知（例：小 batch decode 受访存限制） | 极强 | 少 |
| 诊断套路 | 强 | 少 |
| 工具 | 强，且确定 | **不占** |
| 个案记忆 | 弱 | 按需取用 |

### 1.3 核心原则

**能进化的东西，必须由不随它一起进化的东西来评判。** 否则系统会学会降低标准（Goodhart 效应）。

### 1.4 核心命题（v1 新增）

> **Sherpa 的"自身"（代码 + 资产）由 agent 不断改写；一次改写只有在一套固定的任务基准上证明"变强了"，才会被保留。**

这句话有三个前提，缺一个都不成立：

| 前提 | v0 现状 | v1 的做法 |
|---|---|---|
| 评判者是固定的 | 测量逻辑写在可进化的 adapter 里 | 测量、正确性判定、评分都上收到 kernel |
| 被改写的是整个"自身" | 只有知识条目能自动进化 | 除评判者之外全部可进化，包括代码 |
| "变强了"可以测量 | 只测知识条目的预测误差，而且是在训练集上测 | Task Bench：在留出任务上比较不同 Sherpa 版本 |

---

## 2. 总体架构

```
            ┌──────── 评判者 kernel（人维护，Sherpa 改不了）────────┐
            │ 测量契约 · 正确性 oracle · Task Bench · 记录与溯源 · 守卫 │
            └──────────▲─────────────────────────┬──────────────┘
         结果只能由这里产生 │                         │ 只合入不退化的改写
   ┌──────────────────────┴─────────────────────────▼───────────────┐
   │ 被评判者：Sherpa 的"自身"（全部可进化，包括代码）                      │
   │ runtime · adapter · 工具 · 认知模型（账本/物理/特征）· 知识 · policy │
   └──────▲──────────────────────────────────────────────┬────────┘
          │ 改写提案 = 对 sherpa 仓库的 commit                  │ 执行
     ┌────┴──────┐                                    ┌────▼──────┐
     │ 复盘 agent  │ ◄──────── 任务全过程的记录 ────────── │  优化任务    │
     └───────────┘                                    └───────────┘
                         大脑：LLM（外部提供，经 backend 抽象接入）
```

### 2.1 一条关键约束：结果只能由 kernel 产生

runtime 可以随便进化，但它不能"自己宣布"结果：

- **测量**：只能调用 kernel 的测量接口，指标由 kernel 计算
- **接受改动**：只能由 kernel 判定（门禁 + 显著性），并写入 kernel 的记录
- **任务得分**：由 kernel 根据它自己的记录和交付物独立计算

所以 runtime 无论怎么改写，都伪造不了结果，**这就是 runtime 可以放进可进化一侧的前提**。

---

## 3. 评判者（kernel）

### 3.1 测量契约

| 项 | 规定 |
|---|---|
| 接口 | 统一走 OpenAI 兼容 API（`/v1/completions`、`/v1/chat/completions`），与框架无关 |
| 负载生成 | 采用一个现成的开源压测工具并**固定版本**（候选：`vllm bench serve`、GuideLLM、genai-bench；见 OQ-19）。工具版本本身就是评判者的一部分 |
| 指标 | 吞吐（tok/s）、TTFT、TPOT、e2e、goodput（满足 SLO 的请求占比），都由 kernel 计算 |
| 测量陷阱 | 契约里统一处理：关闭 prompt cache 或每轮换 prompt、预热、streaming 测 TTFT、固定 `ignore_eos` 和输出长度 |
| 比较方式 | 候选和 incumbent **交替测量**（A/B/A/B），不和过期结果比；显著性阈值 = max(2×CV, 1%)（沿用 v0 已验证的规则） |
| 机器体检 | 每台机器先跑 llama.cpp 社区标准测试（Llama 2 7B Q4_0，`llama-bench -ngl 99 -fa 0,1`），和社区数值对照。对不上说明环境有问题，不进入优化 |

adapter 只负责"怎么起服务、怎么观察内部"，**不负责算指标**。

### 3.2 正确性 oracle

| 动作类型 | 判据 | 参考来源 |
|---|---|---|
| 无损 | **在目标并发下**（不只是 batch=1）对比候选输出与参考输出，判据是 top-1 一致率 / logprob 差异，容差要能覆盖 batch 组成带来的浮点差异（具体方法见 OQ-21） | vLLM batch invariance 的思路 |
| 涉及采样路径 | 固定种子下的 token 分布检验 | v0 DES-14 |
| 有损：分布层面 | KL 散度 + top-1 一致率（`llama-perplexity --kl-divergence-base`） | llama.cpp |
| 有损：任务层面 | GSM8K 等任务分数，判定方式是"期望值 ± 容差" | vLLM CI、lm-evaluation-harness |
| 阈值规则 | 精度不低于参考的 99%；**生成长度不少于参考的 90%**（防止靠少生成 token 刷速度） | MLPerf |
| 防作弊 | 检测缓存命中、跳过计算、改负载之类的捷径（v0 IMP-3 就是同一类问题） | GSO HackDetector |

门禁与动作作用的节点绑定：动作触及哪里，就启用对应的检查（v0 DES-14 的结论）。

### 3.3 Task Bench

**一道题** = (模型, 卡, 框架@版本, 负载, 预算) + **参考目标** + 地板。

| 题型 | 题目 | 参考目标（都在本机实测） |
|---|---|---|
| **配置题** | (模型, 卡, llama.cpp@版本, 负载) | 已知专家配置里最好的那个：官方文档推荐、社区配置等 |
| **代码题** | llama.cpp 历史上某个性能 PR 的**父 commit** | 专家 PR 本身（合入后的版本），前提是它在我们的卡和模型上确实有收益 |
| **陌生题** | 训练集里从未出现过的组合 | 同配置题 |

```
得分 = (Sherpa 成绩 − 地板) / (参考 − 地板)      必须先通过正确性 oracle
地板 = 框架默认配置（代码题：父 commit + 默认配置）
天花板 = 物理上界（roofline），只用来看还剩多少空间
Sherpa 超过参考时，参考上调为 Sherpa 的成绩
```

| 规则 | 说明 |
|---|---|
| 训练集 / 留出集 | **按任务划分**。复盘 agent 只能看训练集任务的记录；留出集只用于打分 |
| 版本得分 | 在留出集上汇总：得分、达到参考 90% 所需的实验次数、墙钟时间、失败率 |
| 方差 | 先测出同一版本重复打分的方差，改进幅度必须超过方差才算数 |
| 硬件绑定 | 参考目标按机器测量，题目绑定到具体的卡 |
| 防过拟合 | 定期加入新题、轮换题目；保留"陌生题"这一类 |

**分层评测**（否则进化跑不动）：

| 层 | 内容 | 成本 |
|---|---|---|
| T0 | 单元测试 | 秒级 |
| T1 | 决策回放：把历史任务中某个决策点的状态交给新版本，看它是否选中事后证明最好的动作 | 分钟级 |
| T2 | 完整跑 Task Bench | 小时级 |

### 3.4 记录与溯源

- 每个任务记录：**Sherpa commit**、bench 版本、机器体检结果、环境指纹
- kernel 的记录只追加。原始上下文必须保存，**派生量一律读时重算**（v0 DES-18 的准则）
- 记录有两个用途：①给 Task Bench 打分；②作为 T1 的**可回放决策状态**
- 记录**不进 LLM 上下文**，复盘 agent 通过工具查询

### 3.5 守卫

- 进化提案在路径层面**禁止修改** `kernel/` 和 `bench/`
- bench 题库和参考结果做哈希校验
- 对评判者的修改只能走人工 PR

---

## 4. 被评判者：Sherpa 的"自身"

### 4.1 范围

kernel 之外的全部内容：runtime、adapter、env 探测、tracer、卡规格的获取、特征、物理、账本、知识与其解释器、policy、brief 组装、决策点、playbook、工具。

v0 里那些"只能人改"的教训（rpath、CUPTI 路径、prompt cache 陷阱、给特征补变量、让 notes 进入 brief……），在 v1 里都可以由 Sherpa 自己学进来。

### 4.2 认知模型：账本、物理、特征

从 v0 保留，但**定位改为"仪器之一"**，不再是唯一的认知模型。

```
request.time_per_token            客户端视角（单位：ms / output token）
  request.outside_gpu_window
  step.gpu_window
    step.gpu_busy → op.gemm / op.attention / op.other / op.memcpy
    step.gpu_idle                 需要能继续下钻（v0 议题 2）
```

| 部分 | 说明 |
|---|---|
| 理论 vs 实测 | 每个节点有理论下界和实测值，差距按"差距 × 时间占比"排序（v0 已证明有效） |
| 两类动作 | close_gap（缩小差距）/ move_bound（移动下界） |
| 特征 | 知识的适用条件写在特征上，不写组合名；**特征集合本身可以进化**（发现冲突时可以补变量） |
| 可证伪性 | 物理下界一旦被实测击穿就证明它错了，所以它可以安全地进化 |
| 效率先验 | 可以从公开数据直接推出。例：4090 上 llama.cpp 的 tg128 达到带宽峰值的约 68% |
| 局限 | 时间维度之外还要考虑容量和精度；单一标量指标表达不了 SLO 权衡，以后要支持多目标 |

### 4.3 知识

| 准则 | 说明 |
|---|---|
| 按机理组织 | 不按组合组织 |
| 成熟后沉降 | 个案 → 机理/规则 → 工具/代码 |
| 冲突是信号 | 说明漏了变量，要重构知识，而不是追加 |
| 写了必须有人读 | v0 中 priors、notes、side_effects 写进去后都没人读。知识必须进入决策上下文，或被工具执行，否则就是废料 |

知识的具体形态（when/effect 的格式等）属于被评判者，**由 Task Bench 的结果来决定哪种形态更好**，不再预先规定。

### 4.4 手脚：adapter 与环境

adapter 契约（来自 v0 的实证）：

| 阶段 | 职责 |
|---|---|
| build | 编译框架（含工具链探测，例如 pip 版 CUDA 的 rpath） |
| prepare_model | 模型格式转换（例如 HF → GGUF），产物缓存 |
| deploy | 起服务、健康检查、停服务；提供 OpenAI 兼容端点 |
| effective_config | 读出**实际生效**的配置（它和输入配置可能不同） |
| observe | profile、trace，把 kernel 映射到账本节点 |
| deliver | 生成启动脚本 |

**测量不归 adapter**，统一由 kernel 负责（§3.1）。

环境类知识（新卡型、CUDA 布局、profile 能力）是出现频率最高的一类经验，必须能进化。卡规格优先**用 microbench 实测**，白皮书数字只作参考。

### 4.5 Loop：机制与策略分离

```
策略层   Planner · Monitor · Analyst · Judge ……（LLM 决策点）
机制层   资源池 · 实验生命周期 · 事件总线 · 动作词表
执行层   agent backend（codebuddy headless / claude code / ……，无状态）
```

| 要点 | 说明 |
|---|---|
| 决策点 = 资产 | 一个决策点由 触发条件 + 上下文组装器 + policy + 输出 schema 构成，全部可进化（v0 中 brief 写死在引擎里，这一点要改） |
| 动作词表 | 开实验、叫停、开 probe（产出信息而非性能）、下结论、交付 |
| 分级上报 | 规则 → 快模型 → 强模型 |
| backend 无状态 | 每次调用开新会话；关掉 CLI 自带的持久记忆（例如 codebuddy 的 auto-memory） |
| 单次实验成本 | 每个实验按阶段记录耗时（编译、起服务、测量、trace、LLM），降低它本身就是进化目标 |

---

## 5. 进化（外环）

```
任务结束（只用训练集任务）
  → 复盘 agent 读全过程：事件、决策、日志、失败、预测误差
  → 产出若干 commit 提案（类型不限：修 adapter / 写工具 / 改 policy / 补特征 / 加知识）
  → T0 → T1 →（定期）T2
  → 不退化则合入，否则丢弃；合入后的版本号就是 git commit
```

| 规则 | 说明 |
|---|---|
| 进化时机 | **任务期间资产冻结**，进化只在任务之间进行（已确认） |
| 人审范围 | 只有对评判者的修改需要人审；被评判者的改写由 bench 裁决 |
| 提案者 ≠ 审批者 | 由 bench 自动裁决，天然满足 |
| v0 devlog 的启示 | v0 真正发生学习的地方是开发 agent 手写的 devlog，复盘 agent 就是把这件事变成正式机制 |

---

## 6. 两层循环（v1）

```
┌─ 外环：Sherpa 进化 ─────────── 评判者：Task Bench（留出集）──────┐
│                                                              │
│   ┌─ 内环：一次优化任务 ───── 评判者：测量契约 + oracle ─────┐    │
│   │  资产冻结；结果只能由 kernel 产生                       │    │
│   └───────────────────────────┬─────────────────────────┘    │
│                               │ 记录                          │
│                               ▼                              │
│        复盘 agent → commit 提案 → T0/T1/T2 → 合入 / 丢弃        │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. 仓库布局

```
sherpa/                        仓库
├── kernel/                    评判者（人改）           Python 包：sherpa_kernel
│   ├── measure/               测量契约（封装固定版本的压测工具）
│   ├── oracle/                正确性 oracle
│   ├── bench/                 Task Bench 的执行与评分
│   ├── records/               记录与溯源
│   └── guard/                 路径守卫、哈希校验
├── bench/                     题库数据（人改）：tasks/*.yaml、参考结果、哈希
├── sherpa/                    被评判者（Sherpa 可改） Python 包：sherpa
│   ├── runtime/  model/  adapters/  tools/
│   ├── knowledge/  policies/  playbooks/
│   └── cli/
├── crates/                    原生代码（按归属分属 kernel 或 sherpa）
└── docs/

../sherpa-data/                运行时数据（不进仓库）：repos / models / tasks / records / artifacts
```

v0 已有代码在 M1 搬迁到这个布局；M0 只新建 `kernel/` 和 `bench/`。

---

## 8. 里程碑

| 里程碑 | 内容 | 退出标准 |
|---|---|---|
| **M0 造尺子** | 测量契约 + oracle + 机器体检 + Task Bench（llama.cpp 的配置题和代码题，分训练集/留出集）+ 记录与溯源 + 守卫 | 给定任意交付物都能打分；地板约 0、参考约 1；**同一交付物重复打分的方差已知** |
| M1 诚实的第 0 版 | v0 搬到新布局；测量改走 kernel；修复可移植性问题；真实 backend 无人值守 | 在两台机器上无人值守跑完 bench，得到第 0 版的分数 |
| M2 自进化 | 复盘 agent + commit 提案 + 分层门禁 | **在留出集上，第 N 版显著优于第 0 版**（这是整个项目的验收标准） |
| M3 扩展 | vLLM、自研框架、云上卡型；bench 持续扩充、轮换 | 新组合首次上手的用时持续下降 |

当前阶段**只用 llama.cpp**，原因：v0 已经接好、编译快、有现成的社区跑分可以对照，而且代码题可以从它的历史里挖。

---

## 9. v0 资产的处置

| v0 组件 | 处置 |
|---|---|
| `kernel/measure.py`（均值、CV、compare） | 并入 `kernel/measure`；指标计算从 adapter 上收到这里 |
| `kernel/gates.py` | 被 `kernel/oracle` 取代（补上目标并发、采样路径、有损判定） |
| `kernel/metabench.py`（知识回放） | 被 Task Bench 取代；回放的思路保留为 T1 |
| `kernel/evolve.py` | 重写为"复盘 agent + commit 提案"（M2） |
| `kernel/knowledge.py`、`invariants.py`（部分） | 知识解释器移到被评判者一侧；与接受判定相关的不变量留在 kernel |
| `kernel/facts.py` | 演化为 `kernel/records`（加入 Sherpa commit，支持决策回放） |
| `model/`、`runtime/`、`assets/` | 移到 `sherpa/` 包下，整体可进化（M1） |
| `crates/cupti-trace` | 保留，归 sherpa 一侧；修复它对 CUDA 目录布局的假设 |
| 已经被验证有效的设计 | 理论 vs 实测账本、确定性的接受规则、条件写在特征上、派生量重算、资产冻结、handoff 通道、逐项裁决 |

---

## 附录 A：开源 bench 调研摘要（2026-11-10）

| 类别 | 代表 | 对 Sherpa 的作用 |
|---|---|---|
| 服务压测工具 | `vllm bench serve`、GuideLLM、genai-bench、genai-perf、inference-perf | 测量契约的实现候选（OQ-19） |
| 性能基准榜 | InferenceX（原 InferenceMAX，数据中心卡 + 大 MoE）、MLPerf Inference（数据中心卡，最小 LLM 为 Llama 3.1-8B）、llama.cpp CUDA 跑分帖 #15013 | 前两者到 M3 才有直接对照价值，现在借鉴方法：吞吐–交互性 Pareto 曲线、SLO 约束；InferenceX 公开的启动配置可作为专家配置来源。#15013 覆盖了我们的两张卡 |
| 功能/精度基准 | lm-evaluation-harness、vLLM CI 的 GSM8K 测试、MLPerf 精度规则、`llama-perplexity` 的 KL 模式、vLLM batch invariance | 正确性 oracle（§3.2） |
| 优化型 agent 的 bench | GSO（102 题，10 个仓库，含 llama-cpp）、SWE-fficiency（498 题，榜首约为专家加速比的 0.225×）、KernelBench（fast_p） | **Task Bench 的范式**：base commit + 性能测试 + 专家 commit；按专家加速比归一化；防 reward hacking |

llama.cpp #15013 中两张卡的数据（Llama 2 7B Q4_0，FA 开）：

| 卡 | pp512 (t/s) | tg128 (t/s) | 提交 commit |
|---|---|---|---|
| RTX 4090 | 14770.63 | 188.96 | 2241453 |
| RTX 5070 Ti | 8419.56 | 182.43 | 933414c |

链接：[InferenceX](https://github.com/SemiAnalysisAI/InferenceX) · [llama.cpp #15013](https://github.com/ggml-org/llama.cpp/discussions/15013) · [GSO](https://github.com/gso-bench/gso) · [SWE-fficiency](https://swefficiency.com/) · [GuideLLM](https://github.com/vllm-project/guidellm)

注：GSO 中 llama-cpp 任务的数量、InferenceX 配置文件的具体目录，调研时没有核实到，用到时需要再确认。
