# Agent 开发计划

更新：2026-11-13｜设计：[Agent](../design/agent.md)

## 状态

统一 Backend 协议、角色候选池、调用追踪和 CodeBuddy CLI 适配器已实现。DeepSeek 有真实调查记录，GPT6 Luna 有故障切换记录；其他候选未逐一在线验证。专家咨询有离线覆盖，尚未真实验证专家任务执行。

配置位于 `sherpa/agent/models.toml`，优先级为命令行 `--config`、`SHERPA_AGENTS`、默认配置。

| 角色 | 候选顺序 |
|---|---|
| daily | `deepseek-v4.1-flash` → `GPT6 Luna`（`gpt-6-luna`）→ `hy4`（`hy4std`） |
| expert | `Claude-Opus-5.5` → `GPT-6-Astra` → `Claude-Opus-5` → `GPT-6.1-Sol` |

## 后续任务

- 在授权预算内验证限定专家咨询：范围清晰、失败可返回日常决策、费用与结果可追踪。
- 新提供方通过 Backend 接入，复用相同请求与验证契约。
- 以任务完成时间、总 token、错误与人工介入评估调用效率，保留正反样例。

## 实现约定

- CodeBuddy CLI 使用无工具、空 MCP、独立临时目录和不保存会话配置，检查 user turn 回显。回显校验不证明远端完整上下文可见。
- 单路调用要求至少 4 GiB 主机可用内存、swap 不过半；进程组 RSS 限额 768 MiB，Node 堆 384 MiB。RSS 是轮询保护。
- `max_output_mib` 默认 16，只限制本次 stdout/stderr 合计；内部日志和缓存不受该阈值限制。原全局 4 MiB 文件限制曾触发 `SIGXFSZ`，已移除。
- `process.json` 保存真实退出码、信号、内存采样、输出阈值与清理结果。费用来自提供方报告，不提供未知账单的硬封顶。

回归入口：`tests/test_agent.py`。环境报告的字段修正与证据展示在[环境准备计划](environment.md)跟踪。
