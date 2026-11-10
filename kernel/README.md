# sherpa-kernel（评判者）

Sherpa 中**唯一产生结果**的部分：测量契约、正确性 oracle、Task Bench 评分、记录与溯源、守卫。只能人改；Sherpa 的进化提案不得触碰 `kernel/` 和 `bench/`。

独立的 Python 项目，依赖逐个固定版本，有自己的 venv：Sherpa 其余部分怎么变，评判者的行为都不变。

```bash
cd kernel && uv venv -p 3.12 .venv && uv pip install -p .venv -e ".[test]"
.venv/bin/pytest                      # 纯逻辑单测（无 GPU）
.venv/bin/sherpa-judge --help
```

## 模块

| 模块 | 职责 |
|---|---|
| `llamacpp.py` | 评判者自有的 llama.cpp 驱动：按内容寻址编译（commit + patch + 参数 + 工具链）、固定转换器的 GGUF 转换、起停服务。隔离 conda 等外来工具链 |
| `loadgen.py` | 流式压测客户端（OpenAI 兼容），保存逐请求原始时间戳；指标与契约违规全部由原始数据推出 |
| `oracle.py` | 负载下的正确性：token 轨迹（tie/hard）、logprob、采样分布（llama.cpp 语义）、上下文契约、GSM8K 精度（有损契约） |
| `reference.py` | 用 HF transformers 生成参考轨迹（一次性，需要 torch；评分时不需要） |
| `stats.py` | 配对比值、显著性 max(2×CV, 1%)、可信度 |
| `bench.py` | 题目 / 交付物 / 校验 / 评分 / 标定 |
| `records.py` | 只追加记录 + 带哈希的原始数据存档；派生量读时重算；记录评判者版本（代码哈希） |
| `guard.py` | 路径守卫；bench 封存与校验 |
| `machine.py` | 机器身份、工具链、**全机互斥锁**（所有编译与测量）、**静默检查**（CPU 忙时不测量） |
| `doctor.py` | 机器体检（llama.cpp 社区 CUDA 跑分） |

## 命令

| 命令 | 用途 | 谁用 |
|---|---|---|
| `sherpa-judge doctor` | 体检；评分前必须有一条本机通过的记录 | 人 |
| `sherpa-judge score <task> <deliverable.yaml> [--sherpa-commit C]` | 正式评分（oracle + 3 对交替测量） | Sherpa / 人 |
| `sherpa-judge measure <task> <deliverable.yaml>` | 快速非正式测量（无 oracle，1 对），供内环探索 | Sherpa |
| `sherpa-judge report [--sherpa-commit C]` | 各题最新得分，按 train / heldout 汇总 | M1/M2 |
| `sherpa-judge show <id>` / `history` | 重新派生并查看记录 | 任何人 |
| `sherpa-judge bench list / verify / seal / reference / calibrate` | 题库维护 | 人 |
| `sherpa-judge guard <path>...` | 改动触及评判者时退出码 1 | 进化流程 |

运行时数据在 `../sherpa-data/judge/`（`src/` 构建、`models/` GGUF、`records/` 记录与存档、`machine.lock`）。
