# Task Bench（评判者数据，只能人改）

本目录与 `kernel/` 一起构成 Sherpa 的评判者。Sherpa 的任何进化提案都不允许修改这里（`sherpa-judge guard` 会拒绝）。改动后必须由人重新封存：`sherpa-judge bench seal`。

```
bench/
├── tasks/<id>.yaml                 题目：模型 × 机器 × 框架基线 × 负载 × 目标 × 契约
├── deliverables/<id>/
│   ├── floor.yaml                  地板：框架默认配置（代码题：PR 的父提交）
│   └── reference.yaml              参考：专家产物（配置题：文档参数网格的最优解；代码题：上游 PR）
├── baselines/<machine>/<id>.json   本机标定：floor/reference 比值、floor 精度（calibrate 产生）
├── oracle/
│   ├── prompts.jsonl               正确性 prompt：40 短文本 + 4 长上下文（≤3.6k token）+ 6 采样探针
│   ├── references/<model>.json     HF transformers（fp32）参考轨迹，与任何推理框架无关
│   ├── policy.yaml                 oracle 阈值（有标定依据）
│   └── gsm8k-100.jsonl, gsm8k-shots.json   有损契约的任务精度集
├── doctor.yaml                     机器体检：llama.cpp 社区 CUDA 跑分
├── retired/                        退役题目及原因（不参与评分）
├── tools/                          人用工具：参考网格、PR 筛选、跨度扫描、二分、生成代码题
├── reports/                        选型、标定、可行性、方差报告（及原始数据 data/）
└── MANIFEST.sha256                 封存清单；bench 版本 = 其哈希
```

## 交付物格式（Sherpa 提交给评判者的东西）

```yaml
framework: llama.cpp
base: b12242                 # 必须等于题目的 framework.base
patch_file: my.diff          # 仅代码题；相对本文件路径，git diff 格式（配置题禁止）
cmake: []                    # 额外 CMake 参数（代码题由题目固定）
model_dtype: bf16            # 必须在 contract.model_dtypes 内
args: ["-np", "32", "-fa", "on"]   # llama-server 参数；-m/--host/--port 等由评判者保留
env: {}
```

评判者自己编译（按内容寻址缓存）、转换模型、起服务、测量、判定；不使用 Sherpa 构建的任何产物。

## 得分

```
r     = 候选 / 参考 的配对性能比（同一会话内交替测量 A B B A ...，同一对使用相同 prompt）
f     = 地板 / 参考 的比值（标定时测得，存于 baselines/）
score = (r − f) / (1 − f)        地板 ≈ 0，参考 ≈ 1，> 1 表示超过专家
有效  = oracle 通过 ∧ 测量契约无违规 ∧ 得分 95% 区间 ≤ ±0.25；无效记 0（score_if_valid 保留供诊断）
```

至少 3 对，自适应加对直到 95% 区间 ≤ ±0.15（最多 12 对）。分数只在同一封存版本内可比。

## 人工流程

| 场景 | 命令 |
|---|---|
| 新机器 / 驱动或 CUDA 变化 | `sherpa-judge doctor` |
| 新模型 | `sherpa-judge bench reference <task>` |
| 新配置题的参考 | `python bench/tools/build_reference.py <task>` |
| 新题 / 新机器的标定 | `sherpa-judge bench calibrate <task>` |
| 任何改动之后 | `sherpa-judge bench seal` |
| 找代码题 | `span_sweep.py`（版本跨度找大差距）→ `bisect_span.py`（定位到单个提交）→ `screen_code_tasks.py`（复测）→ `make_code_task.py` |
