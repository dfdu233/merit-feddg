# 最少需要补充的数据

为了最快完成机制实验，不需要重新上传模型，也暂时不需要重跑完整 benchmark。请优先补下面三类数据。

## P0：必须先确认

- 生成当前主表 MERIT 行的**完整 commit SHA**；
- 实际 decoder / protocol 名称；
- resolved config 或启动命令；
- receiver checkpoint / revision；
- 数据 manifest / split；
- scorer 版本与 generation/prompt contract。

## P1：若服务器已有，直接导出即可

同一 development/source manifest 上以下 arm 的逐例结果或 run 目录：

`generalist`, `joint_all`, `isolated_mean`, `isolated_geomedian`, `bard`。

每例最少字段：

- dataset
- receiver
- sample_id
- image_id，或去标识化 patient cluster
- method
- prediction
- upstream scorer 给出的逐例 score
- delivered source-group IDs / count
- run/config identity

如果 run 目录已有 BARD trace，再附：

- candidate token
- anchor token
- selected token
- commit reason

没有这些 trace 时，**不要为了导出先重跑**。

## P2：只有缺少必要 arm 才需要 GPU 信息

- 可复用 expert cache 路径；
- development/source manifest；
- 当前 GPU 与显存；
- 允许的 GPU 小时或墙钟预算。

**优先给 P0 + P1。** 这样可以先在 CPU 上完成配对分析并决定是否值得启动 GPU。
