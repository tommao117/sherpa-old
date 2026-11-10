# v0 复盘：议题与待决策

> **[历史文档]** 本文的议题已在 2026-11-10 的独立评审中处理，处理结果见 [`decisions.md`](./decisions.md) §3，现行设计见 [`design.md`](./design.md)（v1）。下文保留原样，作为 v0 的原始记录。

> 本文档供**复盘讨论**使用，是一次完整走查（llama.cpp × Qwen3-0.6B × RTX 4090，内环 + 外环全链路跑通）之后的产物。
>
> **接手的 agent 请按以下顺序读**：
> 1. [`design.md`](./design.md) —— 顶层设计（写于实现之前，部分已被实证修正，见本文 §4）
> 2. [`decisions.md`](./decisions.md) —— 已确认的决策与原始待定问题（OQ-1 ~ OQ-18）
> 3. [`devlog.md`](./devlog.md) —— 走查中记录的 **35 条**问题原始记录（ENV/DES/IMP 编号，本文大量引用）
> 4. 本文档 —— 把 35 条问题收敛成 **8 个待决策议题**，按优先级排序
>
> **复盘的目标**：逐个确认 §3 的议题，把结论回写进 `design.md` / `decisions.md`，然后决定 v1 的范围。

---

## 1. 这次实际跑了什么

```
输入：Qwen3-0.6B (bf16) × RTX 4090 (sm_89, 24GB, WSL2) × llama.cpp@b12242
负载：throughput-default（P=256, O=128, 并发 32, 128 请求, 重复 3 次）
目标：output_tok_per_s
```

### 内环（任务 `t20261110-110453`，已交付）

| 实验 | 动作 | 类型 | 预测 | 实测 | 结果 |
|---|---|---|---|---|---|
| baseline | 默认（`-np` auto → slots=4） | – | – | 877 tok/s | – |
| e01 | `-np 32` | move_bound | +80% | **+75.4%** | accepted |
| e02 | `+ -bs`（backend sampling） | close_gap | +40% | **+23.5%** | accepted |
| e03 | 代码级：消除 step 间 GPU 空闲 | close_gap | +20% | – | failed（人为中止） |

- **877 → 1900 tok/s（+116.6%）**
- 门禁：correctness（greedy 逐 token，6 个 prompt 全等）、reproducible（CV 3.3~4.6%）、deliverable（`launch.sh` 独立起服务并应答）全部通过
- 产物：3 条事实、7 次 LLM 决策、4 份账本（baseline/e01/e02 + 理论）、交付物（分支 + commit + 启动脚本）

### 账本（ms / output token）

```
节点                        baseline(理论/实测)      final(理论/实测)
request.time_per_token       0.343 / 1.238           0.084 / 0.537
  request.outside_gpu_window 0     / 0.084           0     / 0.030
  step.gpu_window            0.343 / 1.154           0.084 / 0.508
    step.gpu_busy            0.343 / 0.729           0.084 / 0.284
      op.gemm                0.306 / 0.458           0.048 / 0.083
      op.attention           0.037 / 0.096           0.037 / 0.076
      op.other               0     / 0.107           0     / 0.091   ← e02 搬进来的 GPU 采样
      op.memcpy              0     / 0.068           0     / 0.034
    step.gpu_idle            0     / 0.425           0     / 0.224   ← 始终是第一差距
```

### 外环（提案 `p20261110-121449`，已合入）

- 产出 4 条知识（2 条动作 + 2 条效率先验）→ metabench 通过 → **自动合入**
- 产出 2 条资产变更（`policies/plan.md`、`adapters/llama.cpp/knobs.yaml`）→ **人审合入**
- metabench：覆盖率 0 → 100%，MAE 0.36%，符号正确率 100%
- **进化闭环实证**：用 baseline 的特征重新查知识库

  | | 进化前 | 进化后 |
  |---|---|---|
  | baseline 特征 | `knowledge_suggestions: []` | `raise-slots-to-concurrency  +75.0%  {parallel: 32}` |
  | e01 后特征 | `[]` | `offload-sampling-to-gpu  +23.9%  {backend_sampling: true}` |

  同一批输入，第二次不需要 LLM 推理即可给出方向；且条件正确失效（`slots_over_concurrency < 0.5` 在 e01 后不再匹配）。

### 实现规模

| | |
|---|---|
| 代码 | 2766 行（Python + 1 个 CUPTI tracer C 文件） |
| 资产 | 19 个文件（含进化产出的 4 条知识） |
| 环境 | 无系统 CUDA（用 pip `nvidia/cu13`）、无 nsys/ncu（自写 CUPTI tracer）、无 perf/gdb、codebuddy CLI 未登录（用 handoff backend） |

---

## 2. 设计中被证实有效的部分

这些不需要讨论，但复盘时值得明确保留，因为后面的改动不应破坏它们。

| 设计 | 实证 |
|---|---|
| **理论账本 vs 实测账本** | 一眼定位瓶颈。baseline 的 `op.gemm` 实测 0.458 / 理论 0.306，而"当前 batch 上界 2915 vs 最大 batch 上界 11849"直接指出 move_bound 空间，e01 就是这么来的 |
| **条件写在特征上** | `slots_over_concurrency < 0.5` 这条在 e01 后自动失效，换成另一条动作。换模型/换卡也能迁移，无需改写知识 |
| **动作分 close_gap / move_bound** | 两者的预测方法完全不同（分摊 vs 消除），分类直接决定了怎么推算 |
| **确定性锚点 + LLM 决策** | 接受/拒绝由规则定（门禁 + 显著性），LLM 只负责出假设和解释。e02 预测偏乐观 1.7 倍，但不影响判定正确 |
| **预测误差作为学习信号** | e02 的 +40% vs +23.5% 直接暴露了"成本搬运"这个机理盲点（DES-16），这是本次最有价值的知识产出 |
| **事实不进上下文** | LLM 只看到聚合结果和当前账本，没有历史台账膨胀问题（对应 OQ-1 的担心） |
| **任务期间资产冻结** | `assets_fingerprint()` 在每次决策和交付时校验，全程无违例 |
| **handoff backend** | 原为 codebuddy 未登录的权宜之计，实际成了调试 policy 和人工接管的正式通道（ENV-4） |

---

## 3. 待决策议题（按优先级）

### 议题 1 ★★★ 外环的锚点失效：离线回放检测不到知识过拟合

**事实**（devlog DES-19）：首轮 evolve 的 metabench 给出 MAE **0.36%**、符号正确率 100%、`pass + improved`。但明细是：

| 事实 | 预测 | 实测 | 误差 |
|---|---|---|---|
| f001 | 75.0（区间 `[60,90]` 的中点） | 75.39 | 0.39 |
| f002 | 23.85（`recover_gap [0.3,0.4]` 算出） | 23.52 | 0.33 |

这两条知识**是看过这两条事实之后才写的**，区间中点自然贴着观测值。用同一批事实评测由它们归纳出的知识，等于**在训练集上测试**。MAE 0.36% 不代表知识准，只代表它背下了答案。

**为什么严重**：design §1.3 的核心原则是"能进化的东西，必须由不随它一起进化的东西来评判"。内环满足（物理下界、真实测量、门禁都不由 Sherpa 生成），**外环不满足**——事实本应独立，但当知识由这批事实归纳而来时，独立性就没了。这是 Goodhart 效应的入口。

**候选方案**：

| # | 方案 | 代价 |
|---|---|---|
| 1 | **留出法**：事实分归纳集/保留集，提案只看归纳集，metabench 在保留集评 | 冷启动阶段事实太少，分不动 |
| 2 | **前瞻验证**：合入时只记"待验证"，由**后续任务**的新事实回头评分，不合格回滚 | 反馈有延迟 |
| 3 | **惩罚自由度**：静态 `change_pct` 几乎必然能拟合；`recover_gap` 从账本确定性算出，自由度低得多。按 effect 形式给不同权重，甚至禁止用静态区间解释自己的来源事实 | 限制知识表达 |
| 4 | **要求可证伪**：知识必须写明"什么条件下失效"，由后续实验去撞；`counter` 为空的条目可信度上限压低 | 需要主动设计反例实验 |

**当前倾向**：**2 + 3 组合**——合入门槛看形式自由度（结构化、低自由度的形式更容易自动合入），真实可信度靠前瞻验证累积。
**关联**：OQ-1（事实库）、OQ-12（metabench 设计）。**这两个 OQ 应该和本议题合并讨论。**

---

### 议题 2 ★★★ 账本只有一个维度，`step.gpu_idle` 无法下钻

**事实**（devlog DES-12）：`step.gpu_idle` 的理论下界恒为 0，所以诊断永远把它排第一（baseline 34%、final 42%），但账本树到这里就断了，答不出"CPU 在干什么"。e03 失败的根本原因就是这个：Planner 只能说"去消除 idle"，无法定位成因。

**已有的手工证据**（正是账本给不出来的东西）：
- 框架自带 `DEBUG_TIMINGS` 显示 server 主循环开销极小（pre_decode 0.013ms、post_decode 0.095ms、CPU 采样 0.0016ms/step），而 `llama_decode` 本身 10.8~11.3ms/step → 空闲在 `llama_decode` 内部
- trace 分析：e02 有 138 个 ≥5ms 大停顿（占窗口 18%），而该轮正好 **128 个请求**，数量吻合
- 源码：每个请求开始/结束都调 `llama_set_sampler` → 置 `sched_need_reserve` → 整个 scheduler 重建 + CUDA graph 失效（`llama-context.cpp:1739/1758`）
- **这条路径只在开了 backend sampling 后才触发**，意味着 e02 一边消除 CPU 采样、一边引入了每请求一次的 graph 重建

**需要决策**：
1. `step.gpu_idle` 怎么继续分解？候选：框架侧 NVTX 打点（需改框架，但可沉淀进 adapter）、CPU profiler（本机缺 perf/gdb）、CUPTI 的 driver API callback（已有 tracer 可扩展）
2. "空闲"要不要给非零理论下界？（kernel launch 本身有固有成本）
3. `model/physics` 当前只覆盖 GPU 时间，CPU 侧成本（采样 ∝ vocab × token）没有物理模型 → 关联 OQ-15

---

### 议题 3 ★★ 缺"诊断型动作"，不变量过严

**事实**（devlog DES-13）：不变量要求"实验开始前必须声明预测收益"（design §5.4），于是每个实验都必须改东西并预测性能。但 e03 真正有价值的工作是"插桩 + 读 trace + 读源码"——**产出是信息，不是性能**，在当前模型里无法表达。

**提案**：动作分两类

| 类型 | 预测什么 | 走性能门禁 | 改 incumbent | 产出事实 |
|---|---|---|---|---|
| `optimize` | 性能变化 % | 是 | 可能 | 是 |
| `probe` | **信息**："我预期能区分假设 A 和 B" | 否 | 否 | 是（结论是信息） |

不变量"必须声明预测收益"只对 `optimize` 成立；`probe` 的对应不变量是"必须声明要区分哪些假设"，事后由 Analyst 判断是否真的区分开了。

**附带**：`policies/plan.md` 已在本次 evolve 中加入第 10 条（瓶颈成因不明时优先做定位实验），但 runtime 还不支持 `probe` 动作类型，属于半成品。

---

### 议题 4 ★★ 门禁有假阴性，且与动作无关

**事实**（devlog DES-14）：e02 改的是采样实现，而正确性门禁用 greedy（`top_k=1`）逐 token 对比——**这条路径根本不经过 `top_p`/`temperature` 的采样逻辑**。门禁通过了，但什么都没验证。我后来是靠读源码确认各采样器都有 backend 实现才放心，**Sherpa 自己没有这个能力**。

**为什么比假阳性危险**：错误的优化会被接受 → 写进事实 → 被 evolve 沉淀成知识 → 污染从门禁漏进知识库。这是污染的主要入口，比"知识追加导致膨胀"（E3 最初的担心）严重得多。

**提案**：门禁与动作作用的节点绑定（`gates_required_by_node`）

| 动作触及 | 需要的门禁 |
|---|---|
| 采样 | 固定种子下的 token 分布检验 / 对采样参数的敏感性 |
| KV cache | 长上下文一致性 |
| 调度 / batch | 并发下的输出一致性 |
| 量化等有损 | eval 集 + 容差（原 OQ-9） |

**关联**：OQ-8（硬性门禁形式）、OQ-9（有损动作）。v0 的接受规则（门禁全过 + `delta > max(2×CV, 1%)`）可作为 OQ-8 的一个已验证答案。

---

### 议题 5 ★★ 知识的 effect 表达力不足

**事实**（devlog DES-16）：e02 把采样从 CPU 搬到 GPU，`step.gpu_idle` −0.126 但 `op.other` **+0.066**。预测 +40% 实测 +23.5%，误差几乎全来自没算新增成本。

**根因**：优化动作对成本有三种命运，而 `effect` 只能表达前两种：
- **消除**：成本真的消失
- **分摊**：每 step 固定成本摊到更多 token（按 batch 缩放）
- **搬运**：成本从一个节点移到另一个节点 ← **表达不了**

**提案**：`effect` 从标量改成多节点向量

```yaml
effect:
  metric: output_tok_per_s
  nodes:
    step.gpu_idle: { recover_gap: [0.3, 0.4] }
    op.other:      { add_ms_per_token: [0.06, 0.08] }
```

由 `kernel/knowledge.py` 在当前账本上确定性求值。本次提案里 `offload-sampling-to-gpu` 用 `side_effects` 字段勉强记录了，但**解释器不读它**，属于"写了不生效"的半成品。

**顺带**（DES-22，已修）：知识里的符号值（`parallel: concurrency`）原本被原样拼成 `-np concurrency`，三道门都没拦住。启示是"条件写在特征上"只解决了 `when` 一侧，`apply` 一侧同样需要能引用特征；且**声明了不校验的元数据等于没有**。

---

### 议题 6 ★★ 事实 schema 的设计准则

**事实**（devlog DES-18）：`f002` 的动作签名原本是 `knobs:backend_sampling,parallel`，把没变的 knob 也算进去了（因为配置是在 incumbent 上叠加的）。按这个签名归纳知识，条目会声称"要同时设 parallel"——知识直接被污染。

**已修**：签名只算相对 `base_config` 的 diff，且所有读取方一律重算，不信任事实里存的值。效果是历史事实在新语义下自动修正，**零数据迁移**：

```
f002 stored: llama.cpp:knobs:backend_sampling,parallel
     recomputed: llama.cpp:knobs:backend_sampling
```

**要确认的准则**：**事实只存原始上下文，派生量一律可重算**（存了就会在语义演进时全废）。这和 OQ-1 讨论里"存账本快照以便重新投影特征"是同一条原则。建议把它写成 `kernel/facts` 的显式契约。

**仍未定**（原 OQ-1 剩余部分）：hot/cold/raw 三层存储、淘汰策略、hot 集上限。本次只有 3 条事实，这些机制一条都没被压力测试过。

---

### 议题 7 ★ 外环的权限与流程机制

三条已修但需要确认设计：

| # | 问题 | 已修成 |
|---|---|---|
| DES-20 | `evolve apply` 全有或全无：2 个人审档把 4 个自动档一起挡住 | 逐项裁决（`verdict: partial`），自动档先落地、人审档挂起 |
| DES-21 | 提案者可以自己批准自己，人审门禁形同虚设 | 记录 proposer/approver 身份并强制不同；实测自批被拒 |
| IMP-9 | 代码实验失败只回滚源码、不重新编译，残留二进制污染后续测量 | 失败路径也 rebuild |

**要确认**：
1. 权限分级（design §7.1 / P1）是否就按本次实现固化？`knowledge/**` 自动、其余人审
2. IMP-9 的更彻底做法：`bin_dir` 带 commit 指纹，测量前校验与 HEAD 一致（作为 `kernel/measure` 的前置检查）。回滚的单位是"源码 + 派生产物"，不只是源码
3. 关联 OQ-6（自修改合入权）、OQ-7（资产 overlay）

---

### 议题 8 ★ 测量与负载的可信度

| # | 问题 | 现状 |
|---|---|---|
| IMP-2 | CUPTI tracer 开销约 **7%**（890 → 827 tok/s）。bench 与 profile 分两次跑，账本实测值来自 traced run，与 bench 吞吐差约 7% | 每次记录 `tracer_overhead_pct`。待定：降低开销（抽样采集）还是按比例校准 |
| IMP-3 | prompt cache 会跳过 prefill，必须显式 `cache_prompt: false` + 每轮换种子。这种"测量陷阱"现在靠 adapter 作者记得 | 应上升为 `kernel/measure` 的契约检查 |
| IMP-4 | 不用 streaming 时 TTFT 只能用 `e2e − predicted_ms` 近似（含排队）。高并发下 Python 客户端会让测量失真 | 对应 E6，需要 Rust 压测客户端 |
| IMP-7 | 我曾怀疑压测客户端制造"请求波次"污染 `gpu_idle`，实测否掉：把窗口切 10 段，空闲率 58/57/60/57/58/58/55/65/56/56 **均匀分布** | **这正是"事实检验结论"的作用**——没这步会按错结论改测量代码。但负载模型问题仍在：闭环、等长、同步化 ≠ 真实流量 → OQ-10 |

---

## 4. design.md 需要修订的地方

| design 位置 | 问题 | 依据 |
|---|---|---|
| §3.1 账本树 | 根节点应是"客户端视角时间"，需新增 `request.outside_gpu_window`；节点单位统一为 **ms/output token** | DES-3, DES-4 |
| §3.1 账本树 | `step.gpu_idle` 需可继续分解；时间维度之外需显式的容量/精度约束（已落为 `model/constraints.py`） | DES-12, §3.5 |
| §4.4 知识示例 | `effect` 要支持多节点向量 + `recover_gap`；`apply` 要支持引用特征的符号值 | DES-7, DES-16, DES-22 |
| §5.3 hooks | 决策点之间改为**事件链**（`experiment_concluded → round_done → judge → judge_continue → plan`），而非同一事件上的执行顺序；并定义决策点的输入契约（上一个决策的输出如何进入下一个） | DES-10, DES-15 |
| §5.4 不变量 | "必须声明预测收益"只对 `optimize` 动作成立 | DES-13 |
| §5.5 门禁 | 门禁与动作作用节点绑定，不是固定三条 | DES-14 |
| §6 adapter 契约 | 补齐为 **build / prepare_model / deploy / bench / profile / deliver** 六段，另加 `effective_config()`（有效配置 ≠ 输入配置）与 knob 元数据校验 | DES-1, DES-2, DES-22 |
| §7 目录 | 拆成两张图：概念分层（权限边界）+ 物理布局（可导入代码 vs 可自改数据）；`sherpa-data/` 补 `repos/ models/ tools/ proposals/` | DES-17, DES-1 |
| §7.1 权限 | 改为**逐项裁决** + 提案者/审批者隔离 | DES-20, DES-21 |
| §10 v0 范围 | 实际完成度与此表的差异需对账（metabench 做了离线回放而非"只留接口"；evolve 做了完整合入而非"只产出提案"） | – |

**decisions.md 需更新**：OQ-1 部分解决（§3 议题 6）、OQ-2/OQ-3 已定（llama.cpp + 本机 4090）、OQ-8 有了 v0 答案、OQ-13 因 ENV-4 改用 handoff、OQ-17/18 已实测（无 nsys，自写 CUPTI tracer；sm_89 非 sm_120）。环境现状（§5）需按实测订正（4090 而非 5070 Ti）。

---

## 5. 复盘建议顺序

1. **议题 1**（外环锚点失效）——动摇设计根基，且决定 OQ-1 / OQ-12 怎么定
2. **议题 2**（账本下钻）——卡住所有代码级优化，是 e03 失败的直接原因
3. **议题 4**（门禁假阴性）——污染进入知识库的主要入口
4. **议题 3 / 5**（probe 动作、effect 表达力）——提高迭代效率与预测准确度
5. **议题 6 / 7 / 8**（事实准则、外环权限、测量可信度）——多数已有实现，确认即可
6. 最后：按结论修订 `design.md` / `decisions.md`，定 v1 范围

## 6. 可复现的现场

| 内容 | 位置 |
|---|---|
| 任务全过程 | `sherpa-data/tasks/t20261110-110453/`（context / events.jsonl / decisions / experiments / ledger / logs / report.md / deliverable） |
| 事实 | `sherpa-data/facts/t20261110-110453.jsonl` |
| 外环提案与评测 | `sherpa-data/proposals/p20261110-121449/`（proposal.json / evaluation.json） |
| 进化产出的知识 | `assets/knowledge/actions/*.yaml`、`assets/knowledge/priors/*.yaml` |
| kernel trace（原始） | `sherpa-data/artifacts/t20261110-110453/*.trace.txt`（81MB / 20MB / 54MB） |
| 机器画像 | `sherpa-data/env.yaml` |

常用命令：

```bash
python3 -m sherpa.cli.main env                    # 探测环境
python3 -m sherpa.cli.main ledger t20261110-110453 baseline
python3 -m sherpa.cli.main facts [--aggregate]
python3 -m sherpa.cli.main metabench              # 离线回放（注意议题 1：当前结果不可信）
python3 -m sherpa.cli.main evolve history
python3 -m sherpa.cli.main pending                # 列出等待回答的 handoff 请求
```

> 注意：`run` 需要 LLM 决策。v0 的 `--backend handoff` 会把请求写成文件等人/外部 agent 回答（`*.request.md` → 写 `*.response.json`，**务必先写临时文件再 `mv`**，见 IMP-6）。`--backend codebuddy` 未经实测（ENV-4）。
