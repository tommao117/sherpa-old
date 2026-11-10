# Monitor

Watcher 在实验进行中发现了异常（例如中间结果明显低于 incumbent）。决定：继续跑完、叫停。

- 看 `server_log_tail`：是否走了慢路径、是否有报错 / 警告、是否在重新编译 / 预热？
- 如果有明确证据表明这次实验不可能超过 incumbent，叫停（`kill`）以节省 GPU 时间。
- 如果异常可能是暂时的（预热、首轮抖动），继续（`continue`）。
