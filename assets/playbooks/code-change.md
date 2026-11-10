# Playbook：修改推理框架源码

你在一个推理框架的 git worktree 里工作（`workdir`）。目标是实现 `brief.instructions` 描述的优化。

1. 先读仓库自带的 `AGENTS.md` / `CONTRIBUTING.md`（如果有），以及下面的 codemap，定位相关代码。
2. 改动要**最小且可解释**：只改为实现假设所必需的部分，不要顺手重构。
3. 改完后自己编译一次确认通过（`brief.build_hint`）。编译失败要修好；修不好就回滚并返回 `gave_up`。
4. **不要 commit**，不要改 git 配置。Sherpa 会负责提交、重新编译、测量和门禁。
5. 返回 `summary`：改了什么、为什么预期有效、有什么风险。
