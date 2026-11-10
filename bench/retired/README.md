# 退役题目

| 题目 | 原因 | 证据 |
|---|---|---|
| cfg-qwen3-0.6b-prefill-c8 | 不具区分度：配置层面的空间太小。参考（`-np 8 -c 32768 -ub 1024`）只比默认配置快 ~7%（f ≈ 0.93），而配对噪声 CV ≈ 7%，10 对后标准误仍 2.4%，得分 95% 区间约 ±0.7 | 标定记录（kind=calibration，task=cfg-qwen3-0.6b-prefill-c8）。prefill 的空间在代码层面：b9000 → b12242 的 pp2048 +28.5%（`reports/data/span-sweep.jsonl`），作为代码题候选 |
