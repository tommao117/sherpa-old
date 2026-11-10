# Sherpa

自迭代推理优化框架。它不做推理，而是在给定的 模型 × 卡型 × 推理框架 上完成部署、bench、profile、优化、交付，并在每次使用后让自己变得更强。

当前阶段：**v0 全链路已跑通，等待复盘。**

首次走查结果（Qwen3-0.6B × RTX 4090 × llama.cpp@b12242，吞吐负载）：

| | |
|---|---|
| 内环 | 877 → **1900 tok/s（+116.6%）**，2 个动作被接受，门禁全过，已交付 |
| 外环 | 产出 4 条知识（自动合入）+ 2 条资产变更（人审合入） |
| 进化闭环 | 同一 baseline 特征，进化前无建议，进化后可直接给出 `-np 32`（+75%） |

| 文档 | 内容 |
|---|---|
| [`docs/design.md`](docs/design.md) | 顶层设计：第一性原理、两层循环、机理模型、知识体系、agent loop、目录结构、v0 范围 |
| [`docs/decisions.md`](docs/decisions.md) | 用户强调的要点、已确认的决策、默认方案、待定问题、环境现状 |
| [`docs/devlog.md`](docs/devlog.md) | **走查中记录的 35 条问题**（环境 4 / 设计 22 / 实现 9），含现象、临时处理、对设计的启示 |
| [`docs/retro-v0.md`](docs/retro-v0.md) | **复盘入口**：把 35 条问题收敛成 8 个待决策议题，并列出 design 需修订处 |

## 接手须知

**要复盘设计**：读 `retro-v0.md`，它指明了阅读顺序和讨论顺序。

**要继续开发**：

```bash
python3 -m sherpa.cli.main env       # 探测机器（GPU / CUDA / profile 能力），写入 sherpa-data/env.yaml
python3 -m sherpa.cli.main pending   # 列出等待回答的决策请求
```

完整命令与现场数据位置见 `retro-v0.md` §6。

## 布局

```
sherpa/        Python 包（只能人改）
  kernel/        锚点：facts / measure / gates / invariants / knowledge 解释器 / metabench / evolve
  model/         机理模型：specs / physics / ledger / features / constraints
  runtime/       机制层：engine / scheduler / events / watchers / backends / workspace / env / task
  cli/
assets/        可进化资产（Sherpa 自己会改，走 evolve 提案）
  adapters/<fw>/   框架适配器（含 adapter.py，动态加载而非 import）
  knowledge/       actions/ 动作目录、priors/ 效率先验
  policies/        决策点 policy + hooks.yaml + 输出 schema
  playbooks/       执行层：某类子任务怎么做
crates/        原生代码（cupti-trace：CUPTI 注入式 kernel tracer，替代缺失的 nsys）
docs/

../sherpa-data/   运行时数据（不进代码仓）：env.yaml / repos / models / tools / tasks / facts / artifacts / proposals
```

为什么 Python 包多套一层、为什么 `adapter.py` 不走 import：见 `devlog.md` DES-17。
