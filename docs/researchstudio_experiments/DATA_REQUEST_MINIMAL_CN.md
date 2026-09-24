# 最少需要补充的数据

为了最快完成机制实验，不需要重新上传模型，也暂时不需要重跑完整 benchmark。请优先补下面三类数据。

## P0：必须先确认

- 生成当前主表 MERIT 行的**完整 commit SHA**；
- 如果运行时工作区有未提交修改，附差异文件清单及补丁 SHA；不能只报 commit；
- 实际 decoder / protocol 名称；
- resolved config 或启动命令；
- receiver checkpoint / revision；
- 数据 manifest / split；
- scorer 版本与 generation/prompt contract。
- 对应的逐例输出目录、完整 TEST 样本数、输出/缺失/空答/未完成数量，以及原始与补答协议是否分列。

## P1：若服务器已有，直接导出即可

同一 development/source manifest 上以下 arm 的逐例结果或 run 目录：

`generalist`, `joint_all`, `isolated_mean`, `isolated_geomedian`, `bard`。

优先选择**同一 receiver、同一份 manifest、同一原生专家缓存**下已经存在的 run；各 arm 必须覆盖同一批 `sample_id`，不要只导出错误样本或各自不同的子集。已有 TEST 结果可以做描述性配对分析，但不能再用其标签调 quorum、阈值或专家路由。

每例最少字段：

- dataset
- receiver
- sample_id
- image_id，或去标识化 patient cluster
- method
- prediction
- upstream scorer 给出的逐例 score
- `finished`、空答、解析状态与实际输出 token 数
- delivered source-group IDs / count，精确到每例；`joint_all` 与 isolated arms 的证据 ID/内容 hash 是否相同
- image/question/prompt hash、native evidence-cache identity、scorer hash
- run/config identity

对 CE 另给解析后的选项或 Yes/No；对短 OE 给原始答案和每例 token-recall 分数。报告生成任务单独给 BLEU-4、ROUGE-L、METEOR，不与 VQA 合成同一个 score。

如果 run 目录已有 BARD trace，再附：

- candidate token
- anchor token
- selected token
- commit reason
- 若已有，再给每个 source branch 的同前缀 token 分数或其可复现缓存，以便 Stage 2 在**相同分支**上做 consensus；没有保存时明确标注“需重推理”，不可宣称零成本复用。

没有这些 trace 时，**不要为了导出先重跑**。

## P2：只有缺少必要 arm 才需要 GPU 信息

- 可复用 expert cache 路径；
- development/source manifest；
- 当前 GPU 与显存；
- 允许的 GPU 小时或墙钟预算。

还需要提供输出根目录剩余磁盘空间、checkpoint/processor 精确版本、每例 branch-forward 次数和 wall time。新实验先做 8–32 例五臂真实 canary，通过输出与配对身份检查后，才扩展到预先声明的完整集合。若只有曾看过结果的 TEST split，应把消融标为描述性分析，并另外寻找未用于选参的开发集。

**优先给 P0 + P1。** 这样可以先在 CPU 上完成配对分析并决定是否值得启动 GPU。
