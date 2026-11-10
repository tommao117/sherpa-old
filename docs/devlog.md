# v0 开发日志：问题与设计反思

> 目的：第一次端到端走通（llama.cpp × Qwen3-0.6B × RTX 4090）时，记录**遇到的问题**和**觉得不合理的设计**，供复盘。
> 记录格式：`[编号] 现象 → 临时处理 → 对设计的启示`。编号前缀：ENV 环境，DES 设计，IMP 实现。

## 环境

**[ENV-1] 机器和文档记录的不一致。** decisions.md §5 写的是 5070 Ti（sm_120），这台实际是 4090（sm_89），没有 codebuddy CLI，也没有 cargo/rustc（只有 `~/.workbuddy/binaries/rust`）。
→ 以实测为准。
→ **启示**：环境事实不能靠人写进文档，应该由 `runtime/env` 在每次任务开始时**探测**，结果写入任务的 `context.yaml`（同时也是事实的环境指纹）。

**[ENV-2] 没有系统 CUDA toolkit。** 只有 vllm venv 里 pip 装的 `nvidia/cu13`（nvcc 13.0、cublas、cupti 都有）。
→ 编译时用 `-DCUDAToolkit_ROOT=<pip cu13>`；可执行文件链接时找不到传递依赖 `libcudart`，需要加 `-Wl,-rpath-link,<cu13>/lib -Wl,-rpath,<cu13>/lib`。
→ **启示**：①"工具链在哪"是机器级配置，需要一个机器画像（`sherpa-data/env.yaml`），由探测生成、人可以覆盖。② 这种踩坑经验（"pip 版 CUDA 需要 rpath-link"）正是**手脚类知识**，应该沉淀进 adapter 的 build 逻辑，而不是 knowledge/actions。现有知识体系只覆盖"优化动作"，没有覆盖"怎么让东西跑起来"的**操作知识**。

**[ENV-3] 没有 nsys/ncu。**
→ 自己写一个 CUPTI 注入库（`CUDA_INJECTION64_PATH`）做 kernel 级 trace。
→ **启示**：profile 能力本身依赖环境。adapter 的 profile 应该声明"需要哪些能力"，由 env 告诉它"有哪些能力"，不够时降级（比如只做 bench 级账本）。

**[ENV-4] codebuddy CLI 需要交互式 `/login`，headless 无法鉴权。**
→ 实现了 `handoff` backend：决策请求写成文件，由外部（人或 IDE 里的 agent）写回响应。这次走查中，所有 LLM 决策点都由 IDE agent 通过 handoff 完成。
→ **启示**：backend 接口天然要支持"异步 / 人在环中"。handoff 不只是权宜之计，它也是**人工接管**和**调试 policy** 的正式通道。

## 设计

**[DES-1] 走查漏了三个阶段：build、模型准备、仓库缓存。**
- llama.cpp 必须编译（约 2.5 分钟，32 核）。adapter 契约原来只有 deploy/bench/profile，没有 `build`。
- llama.cpp 只认 GGUF，需要 HF → GGUF 转换，而转换脚本在**框架仓库里**（`convert_hf_to_gguf.py`），所以模型准备依赖框架 worktree。产物缓存在 `sherpa-data/models/<slug>/<dtype>.gguf`。但缓存 key 只有 dtype，没有带框架版本；转换器升级后可能产出不同的文件。
- 每个任务都要 clone 框架太慢，于是加了 bare 仓库缓存 `sherpa-data/repos/<fw>.git`，worktree 从它创建。
→ **启示**：adapter 契约应该是 `build / prepare_model / deploy / bench / profile / deliver` 六段。目录图里 `sherpa-data/` 下要补 `repos/`、`models/`、`tools/`（编译出来的原生工具）、`proposals/`。

**[DES-2] 有效配置 ≠ 输入配置。** 默认 `-np -1`（auto）实际是 4 个 slot；`-fa auto` 实际是否启用，默认日志级别下看不到，只能从 kernel trace 里有没有 `flash_attn_*` 推断。
→ adapter 增加 `effective_config()`（读 `/props` + 日志），并用 trace 观测补全。
→ **启示**：特征投影、物理下界、事实记录都必须基于**有效配置**，否则"开了 X"的事实可能是假的（X 其实本来就开着）。这是事实可信度的前提，应该写进 adapter 契约。

**[DES-3] 账本的归一化单位。** 吞吐负载里 prefill 和 decode 混在一起，按 phase 分节点没法从 trace 里干净地拆开（同一个 step 里可能两者都有）。v0 统一用 **ms / output token**，理论值 = decode 下界 + 分摊到每个输出 token 的 prefill 下界。
→ 可加总，诊断排序简单，但丢掉了 phase 维度。要分 phase，tracer 需要知道每个 step 的 batch 组成（需要框架打点，比如 NVTX）。

**[DES-4] 账本新增节点 `request.outside_gpu_window`。** 客户端看到的 wall time 比 GPU 活动窗口长（排队尾部、HTTP、首尾空转），原来的树里没有地方放这部分时间。
→ **启示**：账本根节点应该是"客户端视角的时间"，下面先拆"GPU 窗口内 / 外"，再往下拆。

**[DES-5] "诊断"不值得单独成模块。** 原定加 `model/diagnose`，实现下来就是 ledger 上 20 行的排序函数，放在 `model/ledger.py` 里。等诊断变复杂了（多维约束、组合多个信号）再拆出来。

**[DES-6] 知识解释器放在 kernel。** design 没说"`when` / `effect` 怎么求值"这段代码放哪。它决定了"预测对不对"的语义，如果放在可进化资产里，进化就能改评判标准（Goodhart）。所以放在 `kernel/knowledge.py`，知识**内容**在 `assets/knowledge/`。

**[DES-7] effect 用静态区间太弱，增加 `recover_gap`。** "开 X 提升 10%~20%"在不同组合上必然不准。改成"消除节点 N 实测差距的 a~b 比例"，预测值由当前账本**确定性**地算出来，跨组合自动缩放。这才是"条件写在特征上"在 effect 一侧的对应物。

**[DES-8] 有损判定不能信 planner。** 动作是否有损（决定正确性门禁怎么判）由 adapter `knobs.yaml` 的 `lossy_unless` 元数据确定性地算出，不能让 LLM 声明，否则 LLM 可以绕过门禁。
→ **启示**：凡是会影响门禁判定的属性，都必须来自锚点或确定性代码。

**[DES-9] 接受 / 拒绝改成确定性规则。** design 里"是否接受改动"没说谁来定。v0 规则是：所有门禁通过，并且 `measure.compare` 判定为显著提升（> max(2×CV, 1%)）。Analyst 只负责解释，不能改结果。这就是 OQ-8 "硬性门禁"的一个具体形式。

**[DES-10] hooks 触发顺序有歧义。** design §5.3 里 plan 和 judge 都挂在 `experiment_concluded` 上，谁先执行没有定义。改成**事件链**：`experiment_concluded →（队列空）round_done → judge → judge_continue → plan`。决策点之间通过事件串联，而不是靠同一个事件上的执行顺序。

**[DES-11] 一次 plan 出多个实验，但串行执行时基线会漂移。** 预测是相对 plan 时的 incumbent 做的，如果第一个实验被接受，第二个实验的基线就变了，预测误差的计算就不准了。v0 允许一次最多 3 个（减少 LLM 轮次），事实里记录了实际的 `base_config`。
→ 并行实验时这个问题会更明显，需要定义"预测相对什么"。

**[DES-12] 账本只能说"GPU 空着"，说不出"CPU 在干什么"。** `step.gpu_idle` 的理论下界恒为 0，所以诊断永远把它排第一（baseline 34%、e02 42%），但账本树到这里就断了，没法再往下分解。e03 卡住就是因为这个：Planner 只能说"去消除 idle"，具体是谁占着 CPU，账本答不出来。本机还没有 perf / gdb，连外部 profiler 都没有。
→ **启示**：账本现在只有"时间去哪了"一个维度，缺"谁在占用"。`step.gpu_idle` 需要能继续分解（CPU 采样 / 调度 / graph 重建 / 请求间隙）。这要么靠框架打点（NVTX），要么靠 CPU profiler，属于 adapter 要声明的 profile 能力的一部分。

**[DES-13] 缺"诊断型动作"。** 现在每个实验都必须改配置或改代码，并声明预测收益（不变量 P3）。但真实专家排查时的第一步常常是"先加个打点看看"——产出是**信息**，不是性能。e03 我实际上就是先开 `DEBUG_TIMINGS` 插桩、再看 trace，这一步在 Sherpa 的模型里无法表达。
→ **启示**：动作需要分两类：`optimize`（有性能预测，走门禁和接受判定）和 `probe`（有**信息**预测："我预期能区分假设 A 和 B"，不走性能门禁，产出事实但不改 incumbent）。不变量"必须声明预测收益"只对前者成立。

**[DES-14] 正确性门禁有假阴性，这比假阳性危险。** e02 改的是采样实现（backend sampling），而门禁用的是 greedy（top_k=1）逐 token 对比——这条路径根本不经过 top_p/temperature 的采样逻辑。门禁通过了，但实际上没有验证采样语义是否等价。我后来读源码确认 top_k/top_p/temp/dist 都有 backend 实现才放心，**但这是我在门禁之外做的，Sherpa 自己没有这个能力**。
→ **启示**：门禁不能是固定的三条，必须和动作作用的节点挂钩（`gates_required_by_node`）。动作触及采样 → 需要分布检验；触及 KV → 需要长上下文检验。否则错误的优化会被接受，还会被 evolve 沉淀成知识——污染从门禁漏进知识库。

**[DES-15] 决策点之间的信息传递没有定义。** Judge 输出了 `direction`（"下一轮往哪个方向查"），事件里也带上了，但 `_on_plan` 组装 brief 时没有读它——Planner 根本没看到。这是实现漏洞，但暴露的是设计空白：design §5 只定义了"事件触发决策点"，没定义"上一个决策的输出如何进入下一个决策的输入"。
→ **启示**：决策点需要显式的输入契约（brief 由哪些部分构成），其中一部分来自事件载荷。否则决策点之间是哑的，LLM 的判断传不下去。

**[DES-16] 账本节点之间会"搬运成本"，effect 表达不了。** e02 把采样从 CPU 搬到 GPU：`step.gpu_idle` -0.126，但 `op.other` +0.066（GPU 采样 kernel）。预测 +40% 实测 +23.5%，误差几乎全来自没算新增成本。
→ **启示**：`effect` 不能只写"目标节点 -x%"，要能写成多节点的向量：`{step.gpu_idle: recover_gap(0.3~0.4), op.other: +0.06ms/tok}`。否则预测系统性偏乐观，而预测误差又是外环进化的主要信号，会连带把进化方向带偏。

**[DES-17] 实现时偏离了 design §7 的目录树（已获用户同意保留）。** design 里 `kernel/ model/ assets/ runtime/ crates/ cli/` 是同一层级的六个顶层目录。实际落成：

```
sherpa/                     ← 仓库根
├── sherpa/                 ← Python 包：kernel/ model/ runtime/ cli/ util.py
├── assets/                 ← 资产（adapters / knowledge / policies / playbooks / tools）
├── crates/                 ← 原生代码（cupti-trace）
├── docs/  pyproject.toml  .gitignore
```

直接原因是 Python 打包：`kernel`、`model`、`runtime`、`events` 这些名字作为**顶层模块**太通用，容易和第三方包撞名，而且不成包就没法 `pip install` / 提供 `sherpa` 命令。

但更本质的原因是：design §7 那棵树把**两类生命周期完全不同的东西**画在了同一层级。

| | 代码（kernel/model/runtime/cli/crates） | 资产（assets/） |
|---|---|---|
| 是什么 | 程序，被 import | 数据，被读取和**写入** |
| 谁改 | 人，走 PR | Sherpa 自己（evolve） |
| 运行时可写 | 不应该（装到 site-packages 后是只读的） | 必须 |
| 能否多来源叠加 | 不能 | 需要（OQ-7 的 base + 团队 overlay） |
| 加载方式 | `import` | `importlib` 动态加载 / 读文件 |

`assets/adapters/<fw>/adapter.py` 最能说明问题：它是 Python 代码，但**不能是包的一部分**——它要能被 Sherpa 自己改写和提案；如果它跟 `runtime/` 一起被 import，"资产进化"就等于"改代码"，权限边界就没了。所以它必须走动态加载（`runtime/adapter.py` 的 `spec_from_file_location`）。

→ **启示**：design §7 说"目录即权限边界"是对的，但那棵树是**概念分层**，物理布局还要再切一刀：**可导入的代码** vs **可被自己改写的数据**。建议 design §7 拆成两张图（概念分层 + 物理布局），并明确资产根可以是多个 overlay 路径。

**[DES-18] 事实里不能存派生量。** `f002` 的动作签名原本是 `knobs:backend_sampling,parallel`——因为它的配置是在 incumbent（已含 `parallel: 32`）之上叠加的，签名把没变的 knob 也算进去了。按这个签名归纳知识，条目会声称"要同时设 parallel"，知识直接被污染。
→ 已修：签名只算**相对 base_config 的 diff**，且所有读取方（metabench / aggregate）**一律重算，不信任事实里存的值**。效果是历史事实在新语义下自动修正，无需数据迁移（`f002` 重算后为 `knobs:backend_sampling`）。
→ **启示**：这和 OQ-1 讨论里"存账本快照以便重新投影特征"是同一条原则——**事实只存原始上下文，派生量必须可重算**。凡是存了派生量，语义一演进历史事实就全废。事实的 schema 设计准则应该是："能重算的绝不存（或只作为快照存、读时重算）"。

**[DES-19] 离线回放无法检测知识过拟合（本次最严重的发现）。** 首轮 evolve 的 metabench 结果：覆盖率 0 → 100%，MAE **0.36%**，符号正确率 100%，`verdict: pass + improved`。但明细是：

| 事实 | 预测 | 实测 | 误差 |
|---|---|---|---|
| f001 | 75.0（区间 `[60,90]` 的中点） | 75.39 | 0.39 |
| f002 | 23.85（`recover_gap [0.3,0.4]` 算出） | 23.52 | 0.33 |

这两条知识是**看过这两条事实之后才写的**，区间中点自然贴着观测值。用同一批事实评测由它们归纳出的知识，等于**在训练集上测试**。MAE 0.36% 不代表知识准，只代表它背下了答案。

于是 §2 两层循环的"锚点"在外环这一侧是**失效的**：事实本应是不随知识进化的独立评判者，但当知识由这批事实归纳而来时，它就不再独立。内环的锚点（物理下界、真实测量、门禁）没有这个问题，因为它们不由 Sherpa 生成。

→ **候选对策**（需要讨论，直接关系 OQ-1 / OQ-12）：
1. **留出法**：事实按任务/时间分成归纳集与保留集，提案只能看归纳集，metabench 在保留集上评。代价：冷启动阶段事实太少，分不动。
2. **前瞻验证**：新知识的价值由它对**未来**事实的预测误差决定。即提案合入时只记"待验证"，等下个任务产生新事实后再回头评分，不合格就回滚。这更像真实的科学方法，也天然解决独立性，代价是反馈有延迟。
3. **惩罚自由度**：`change_pct` 这类静态区间几乎必然能拟合，而 `recover_gap` 这类从账本确定性算出的形式自由度低得多。metabench 应该对不同 effect 形式给不同权重，甚至禁止用静态区间去"解释"自己的来源事实。
4. **机理可证伪性**：要求知识写明"在什么条件下会失效"，并由后续实验去撞。counter 为空的条目可信度上限要压低。
→ 我倾向 **2 + 3 的组合**：合入门槛看形式自由度（结构化、低自由度的形式更容易自动合入），实际可信度靠前瞻验证累积。

**[DES-20] evolve apply 是全有或全无的。** 本次提案含 4 个自动档（knowledge）+ 2 个人审档（policies/plan.md、adapters/knobs.yaml），`verdict` 直接变成 `needs_approval`，**连自动档也一起被挡住**。合理的行为是：自动档立即合入，人审档单独挂起等待。
→ **启示**：提案的粒度应该是**单个变更**，而不是整个提案包。`evaluate` 要给出逐项裁决，`apply` 要能分批执行。

**[DES-21] 外环缺"提案者 ≠ 审批者"的隔离。** 这次走查里，提案是我写的，`--approve` 也是我点的，人审档的门禁形同虚设。design §7.1 假定了"人"在审，但没有机制保证**审的人不是提的人**。
→ **启示**：`evolve` 需要记录提案者与审批者身份（backend 名 / 会话 id / 人），且不允许相同。这是防 Goodhart 的一部分：能进化的东西不能自己批准自己。

**[DES-22] 知识里的"符号值"没有求值语义。** 归纳出的条目写的是 `apply: {llama.cpp: {knobs: {parallel: concurrency}}}`——意思是"把 slots 设成等于并发数"。这是正确的泛化写法（写死 `32` 就只对这一个负载有效），但 `knowledge.py` 只把它当字符串透传，adapter 原样拼成 `-np concurrency`，**而且三道门都没拦住**：knowledge 没校验值域、adapter 的 `command()` 不检查类型、knobs.yaml 声明了 `type: int` 却没人用。
→ 已修：knob 值支持表达式（`concurrency` / `max_useful_batch` 等来自特征和容量的符号），由 `kernel/knowledge.py` 在查询时按当前上下文求值；`adapter.command()` 增加按 `knobs.yaml` 的 `type`/`values` 做类型与值域校验，非法值直接抛错。
→ **启示**：①"条件写在特征上"只解决了 `when` 一侧，`apply` 一侧同样需要能引用特征，否则动作无法泛化。② knobs.yaml 已经声明了类型，却没有任何代码使用——**声明了不校验的元数据等于没有**。这类"把错误挡在测量之前"的检查属于 adapter 契约的一部分。

## 实现

**[IMP-1] YAML 1.1 把 `on:` 解析成布尔值 `True`。** hooks.yaml 的键 `on` 读出来是 `True`，于是报 KeyError。改名为 `triggers`。

**[IMP-2] tracer 开销约 7%**（890 → 827 tok/s）。所以 bench（不开 trace）和 profile（开 trace）分开跑，账本实测值来自 traced run，它的总时间和 bench 的吞吐对不上，误差约 7%。每次记录 `tracer_overhead_pct`。
→ 以后要么降低开销（只采样部分 step），要么把账本按比例校准到 untraced 的吞吐。

**[IMP-3] 测量中的 prompt cache。** 每轮重复用相同的 prompt，llama-server 默认会复用 prompt cache，导致 prefill 被跳过。需要显式设置 `cache_prompt: false`，并让每轮使用不同的 prompt（用种子偏移）。这种"测量陷阱"应该进 `kernel/measure` 的契约检查，而不是靠 adapter 作者记得。

**[IMP-4] 不用 streaming 时 TTFT 只能近似。** 用 `e2e − predicted_ms` 近似，其中包含排队时间。Python 客户端在高并发 streaming 下会让测量失真（E6），这就是 Rust 压测客户端的用处。

**[IMP-5] `--json-schema` 的 headless 输出格式没有实测**（ENV-4 卡住了），`CodebuddyBackend` 里取 `structured_output` / `result` 字段的写法是推测的。

**[IMP-6] handoff 有读写竞态。** 响应方非原子地写 JSON 时，backend 可能读到半个文件，判定为非法并改名归档，内容就丢了（003-analyze 实际发生了一次）。
→ 已修：等文件大小稳定两个轮次再解析；JSON 不完整只重试，只有**schema 不匹配**才归档。响应方也应该写临时文件再 `mv`。

**[IMP-7] 一个被实测否掉的猜测（留作记录）。** 我怀疑压测客户端制造了"请求波次"（128 个请求、并发 32、长度全相同 → 批次同步化），波次间隙的 GPU 空闲被算进了 `step.gpu_idle`，污染账本。
→ 实测：把 trace 窗口切成 10 段分别算空闲率，e01 是 58/57/60/57/58/58/55/65/56/56，e02 是 49/32/60/36/42/53/30/61/43/35——**空闲均匀分布在整个窗口，不集中在头尾**。猜测不成立，idle 是持续性的，e03 的方向是对的。
→ **启示**：这正是"事实检验结论"的作用。如果没有这个检查，我会按错误的结论去改测量代码。不过负载模型的问题依然存在（闭环、等长、同步化不是真实流量），应该作为 OQ-10 的一部分。

**[IMP-8] e03 的线索（尚未证实）。** e02 的 trace 里有 138 个 ≥5ms 的大停顿，合计 1512ms / 窗口 8317ms = 18%；而这一轮正好是 **128 个请求**，数量高度吻合。读源码：`server-context.cpp` 在每个请求开始（2081 行）和 slot 释放（457 行）都调用 `llama_set_sampler`，而它会置 `sched_need_reserve = true`（`llama-context.cpp:1739/1758`），触发整个 scheduler 重建。只有开了 backend sampling 才会走到这条路径——`set_sampler(nullptr)` 在没有设过 sampler 时会提前返回（1707 行）。
→ 如果成立，这解释了 e02 为什么只拿到 +23.5%：它在消除 CPU 采样的同时**引入了每请求一次的 graph 重建**。优化方向是 sampler 链结构不变时跳过 reserve。
→ **启示**：这种"优化引入新开销"的模式（DES-16）不是偶然，而是常态。

**[IMP-9] 代码实验失败时只回滚源码、没有重新编译。** 执行层可能已经改过并编译过，`scheduler` 的失败路径只做了 `git reset`，`build/` 里仍是改动后的二进制——`bin_dir` 与 HEAD 不一致，会污染后续实验的测量和最终交付物。e03 放弃时实际发生了（残留带 `DEBUG_TIMINGS` 插桩的 `libllama-server-impl.so`）。
→ 已修：失败路径也调用 `rebuild()`。
→ **启示**：回滚的单位不是"源码"，而是"**源码 + 由它派生的产物**"。凡是有编译产物的框架都有这个问题；更彻底的做法是让 `bin_dir` 带 commit 指纹，测量前校验指纹与 HEAD 一致（可作为 `kernel/measure` 的前置检查）。
