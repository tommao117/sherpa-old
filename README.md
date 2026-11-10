# Sherpa

自迭代推理优化框架。它不做推理，而是在给定的 模型 × 卡型 × 推理框架 上完成部署、bench、profile、优化和交付，并在每次使用后让自己变得更强。

**核心命题**：Sherpa 的"自身"（代码 + 资产）由 agent 不断改写；一次改写只有在固定的任务基准（Task Bench）上证明变强了，才会被保留。

## 当前阶段

```
v0 跑通 ──► 独立评审 ──► 设计 v1 ──► M0 造尺子 ◄── 现在（实现完成，待评审）──► M1 诚实的第 0 版 ──► M2 自进化 ──► M3 扩展
```

- **M0**：评判者（`kernel/`）和 llama.cpp 的 Task Bench（`bench/`）已实现，并在本机（RTX 5070 Ti，WSL2）上标定。任务文档与执行记录：[`docs/tasks/m0-judge.md`](docs/tasks/m0-judge.md)
- 现阶段只用 llama.cpp 打磨 Sherpa

## 文档

| 文档 | 内容 | 状态 |
|---|---|---|
| [`docs/design.md`](docs/design.md) | 顶层设计 v1；§3（评判者）已按 M0 实测修订 | **现行** |
| [`docs/decisions.md`](docs/decisions.md) | 决策（D31–D42 为 M0 实施决定，**待评审**）、v0 评审结论、待定问题、环境现状 | **现行** |
| [`docs/tasks/m0-judge.md`](docs/tasks/m0-judge.md) | M0 任务：定义、验收、执行记录 | **待评审** |
| [`kernel/README.md`](kernel/README.md) | 评判者：模块与命令 | 现行 |
| [`bench/README.md`](bench/README.md) | 题库：结构、交付物格式、得分、人工流程 | 现行 |
| [`bench/reports/`](bench/reports/) | 压测客户端选型、oracle 标定、代码题可行性、得分方差 | 现行 |
| [`docs/devlog.md`](docs/devlog.md)、[`docs/retro-v0.md`](docs/retro-v0.md) | v0 原始记录 | 历史 |

## 布局

```
kernel/        评判者（人改）：独立项目 sherpa-kernel，命令 sherpa-judge
bench/         题库数据（人改，封存）
sherpa/        v0 Python 包（M1 搬迁、改为调用 kernel）
assets/        v0 资产
crates/        原生代码（cupti-trace）
docs/

../sherpa-data/   运行时数据（不进仓库）：repos/、judge/（构建、模型、记录）
```

## 快速开始（评判者）

```bash
cd kernel && uv venv -p 3.12 .venv && uv pip install -p .venv -e ".[test]"
.venv/bin/pytest
.venv/bin/sherpa-judge doctor                         # 机器体检
.venv/bin/sherpa-judge bench list                     # 题目与标定状态
.venv/bin/sherpa-judge score cfg-qwen3-0.6b-chat-c32 my-deliverable.yaml --sherpa-commit <sha>
```
