# 验收用交付物（M0）

不参与评分（不在封存范围内），用于复现 `docs/tasks/m0-judge.md` §5.3 的验收结果。

| 文件 | 题目 | 预期 | M0 实测 |
|---|---|---|---|
| `v0-final.yaml` | cfg-qwen3-0.6b-chat-c32 | 无效（logprobs） | 无效：lp MAE 0.42；score_if_valid −1.2（`--fit` 把层挪到 CPU） |
| `kvq4.yaml` | chat-c32 / chat-c32-lossy | 无效 | 无损题：tokens/logprobs/sampling 失败；有损题：GSM8K 6%（地板 41%） |
| `wq4.yaml` | cfg-qwen3-0.6b-chat-c32-lossy | 无效（精度） | GSM8K 23%；无提速（0.998） |
| `cheat-cap-output.yaml` + `.diff` | code-cuda-batched-reads-c32 | 无效 | 测量契约：输出不足；oracle：2 条长生成 `<truncated>` |
| `illegal-config-on-code-task.yaml` | code-cuda-batched-reads-c32 | 校验拒绝 | `code task: 'args' is fixed to [...]` |

```bash
cd kernel && .venv/bin/sherpa-judge score <task> ../bench/reports/fixtures/<file>.yaml
```
