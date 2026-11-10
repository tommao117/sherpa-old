# Sherpa

自迭代推理优化框架。它不做推理，而是在给定的 模型 × 卡型 × 推理框架 上完成部署、bench、profile、优化和交付，并在每次使用后让自己变得更强。

**核心命题**：Sherpa 的"自身"（代码 + 资产）由 agent 不断改写；一次改写只有在固定的任务基准（Task Bench）上证明变强了，才会被保留。

## 当前阶段

```
v0 跑通 ──► 独立评审 ──► 设计 v1 ──► M0 造尺子 ◄── 现在 ──► M1 诚实的第 0 版 ──► M2 自进化 ──► M3 扩展
```

- v0：在 llama.cpp × Qwen3-0.6B × RTX 4090 上跑通全链路（877 → 1900 tok/s）。但评审结论是：它证明了管线能走通，还没有证明"Sherpa 有效"或"Sherpa 在变强"
- 当前任务：[`docs/tasks/m0-judge.md`](docs/tasks/m0-judge.md)，为 llama.cpp 建立评判者 kernel 和 Task Bench
- 现阶段只用 llama.cpp 打磨 Sherpa

## 文档

| 文档 | 内容 | 状态 |
|---|---|---|
| [`docs/design.md`](docs/design.md) | 顶层设计 v1：核心命题、评判者 / 被评判者、Task Bench、正确性 oracle、进化、里程碑、开源 bench 调研 | **现行** |
| [`docs/decisions.md`](docs/decisions.md) | 用户强调的要点、决策（含状态）、v0 评审结论、待定问题、环境现状、下一步 | **现行** |
| [`docs/devlog.md`](docs/devlog.md) | v0 走查中记录的 35 条问题 | 历史 |
| [`docs/retro-v0.md`](docs/retro-v0.md) | v0 复盘的 8 个议题（处理结果见 `decisions.md` §3） | 历史 |

| [`docs/tasks/m0-judge.md`](docs/tasks/m0-judge.md) | 当前任务 M0：背景、任务定义、验收、执行计划、执行记录 | **进行中** |

**接手的 agent**：先读 `design.md` 和 `decisions.md`，再按 `docs/tasks/m0-judge.md` 执行。

## 布局

v1 的目标布局见 `design.md` §7。M0 新建 `kernel/` 和 `bench/`；下面的 v0 代码会在 M1 搬迁。

```
sherpa/        v0 Python 包：kernel / model / runtime / cli
assets/        v0 资产：adapters / knowledge / policies / playbooks
crates/        原生代码（cupti-trace：CUPTI 注入式 kernel tracer）
docs/          设计、决策、任务文档（docs/tasks/）

../sherpa-data/   运行时数据（不进仓库）
```

v0 的常用命令（M1 之前仍可使用）：

```bash
python3 -m sherpa.cli.main env       # 探测机器，写入 sherpa-data/env.yaml
python3 -m sherpa.cli.main pending   # 列出等待回答的 handoff 决策请求
```
