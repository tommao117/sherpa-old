# Sherpa

**小模型主导、实验验证、双环迭代的推理优化框架。**

Sherpa 面向用户指定的推理代码和模型：小模型选择行动，工具执行，大模型处理限定任务；通过行为验证和性能测量改进目标框架，再从运行记录中提炼通用能力。

目前可运行代码准备、只读环境调查和模拟优化流程，已提供固定 llama.cpp 版本的受控 GPU 冒烟试测入口。真实行为/性能优化闭环尚未验证，阶段状态见[开发总览](docs/plan/overview.md)。

## 主要能力

- **固定输入**：HTTPS Git 或本地框架目录与模型目录分开管理，独立工作区固定 commit。
- **模型驱动调查**：按需读文档、定位源码、观察机器，形成有依据的运行方案。
- **可观察、可续跑**：记录决策、工具结果、版本和成本；支持状态查询、取消、证据恢复及草案修正。
- **可替换 Agent**：模型提供方经 Backend 接入，角色内串行故障切换。

## 快速开始

环境：Linux/WSL2、Python 3.11+、Git。在项目根目录直接运行，无需安装目标模型依赖。

### 运行模拟任务

```bash
python3 -B -m sherpa.cli demo --scenario normal
```

模拟结果标记为 `synthetic`。输出的 Run ID 用于查询和取消：

```bash
python3 -B -m sherpa.cli status <run-id>
python3 -B -m sherpa.cli events <run-id> --tail 20
python3 -B -m sherpa.cli trace <run-id>
python3 -B -m sherpa.cli cancel <run-id> --reason "停止测试"
```

数据默认位于仓库外的 `../sherpa-data/`，可用 `SHERPA_DATA` 或子命令前的 `--data-root` 指定。

### 准备框架代码

以下示例要求模型已位于本地目录；可将框架 URL 换成本地源码目录。

```bash
python3 -B -m sherpa.cli env prepare-code \
  --framework https://github.com/ggml-org/llama.cpp \
  --model /path/to/model --device gpu
```

此步骤只准备代码并固定版本，不调用决策模型、不编译或加载目标模型。可用 `--revision` 指定框架版本。

### 发起只读调查

需先准备可用的 CodeBuddy CLI 和 provider 访问权限。`agent list` 查看配置，不发送请求；配置文件为 `sherpa/agent/models.toml`，候选顺序见[Agent 计划](docs/plan/agent.md)。

```bash
python3 -B -m sherpa.cli env resume <prepare-run-id> \
  --allow-model-calls --allow-unpriced --model-budget 1 \
  --seconds 300 --max-calls 24 --max-rounds 12
```

该命令会将选中的文本证据发送给远端决策模型。预算为本段新增额度；美元阈值基于已报告费用，不能保证未知账单硬封顶。方案位于 `runs/<run-id>/environment/proposal.json`，标记为 `proposed/unverified`，不会执行其中的安装步骤。

也可用 `env explore --framework ... --model ...` 搭配相同授权和预算参数，从新任务开始；`env plan /path/to/source` 仅扫描本地声明与机器，不调用模型。各命令的参数通过 `--help` 查看。

### 离线检查执行契约

```bash
python3 -B -m sherpa.cli env check-contract task.json --action action.json --approval approval.json
```

输出字段诊断和待执行声明，不创建 Run、不调用模型或执行程序。JSON 格式见[执行契约](docs/design/environment.md)；检查通过仍需执行前认证授权与复核环境。

### 验证执行器

```bash
python3 -B -m sherpa.cli env test-executor --scenario normal
```

运行固定的伪探测、伪构建和伪推理子进程，验证契约、产物、响应与清理。可用 `--scenario timeout` 或 `wrong-output` 检查失败路径；Run 可用 `status/events/trace/cancel` 观察。结果始终为 `synthetic`，不编译用户框架、不加载模型或使用 GPU。

固定 llama.cpp 版本还支持离线命令规划：`env plan-completion task.json --locations locations.json`。只输出 `draft`，不探测工具/设备或运行程序；输入格式与约束见[环境准备设计](docs/design/environment.md)。

### 受控真实冒烟（试测入口）

```bash
python3 -B -m sherpa.cli env completion-template > input.json
```

填写模板中的干净本地 llama.cpp 检查点、模型目录、GPU UUID、CUDA 架构、工具路径和预算。模板中的 Q8、请求与资源数值仅是待确认声明；此命令不检查机器。

```bash
python3 -B -m sherpa.cli env stage-completion input.json --allow-input-preparation
```

准备命令复制本地代码并读取/哈希所选权重，不下载、构建或探测 GPU。成功后返回 Run ID 和票据哈希。先审查该 Run 的 `completion-ticket.json`、`completion-plan.json`，显卡空闲且确认范围后，才能执行：

```bash
python3 -B -m sherpa.cli env run-completion RUN_ID --approve-ticket-hash TICKET_HASH
```

这一步会检查 GPU、构建并完成一次请求。缺依赖、忙卡、观测不明或预算耗尽会停止，保留日志，不重试。采用固定测试流程，不调用决策模型；成功仅表示该请求的 GPU 冒烟通过，JEV 行为和性能仍需单独验证。限定版本与试测进度见[环境准备计划](docs/plan/environment.md)。

## 文档与测试

- [整体设计与架构图](docs/design/overview.md)
- [开发路线与模块计划](docs/plan/overview.md)
- [文档写作规范](AGENTS.md)

离线回归：

```bash
python3 -B -m unittest discover -s tests -q
```
