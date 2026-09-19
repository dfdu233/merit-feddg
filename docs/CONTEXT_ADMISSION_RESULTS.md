# Context admission：完成结果（2026-09-19）

## 结论

已完成旧 24 例终态审计和新 1793 例完整 TRAIN 五臂对照。简单的同模型相关性/用途判断没有起到过滤作用：两种策略各 734 次真实调用，全部选择准入。新生成候选覆盖率为 **0/1793**；这不是新方法成功，也不能据此宣称医学改善。停止本轮扩展，不进入全量 test 或部署，不按这些分数修改规则。

实验代码提交 `53c1274`，独立分支 `experiments/context-admission-train-v1`，基于 `3349259`。后续报告提交只修正指标说明和补充脱敏导出，不改变已运行 Gate 或回答生成。

## 新实验完整 TRAIN

1793 题、313 张图像。指标为冻结当前 ANCHOR CLOSED 解析得分与 OPEN answer-token-recall 的逐题平均；CLOSED 包含解释性答案解析，不是仅首 token 的严格 EM，更不是临床准确率。所有臂用同一清单、参考和评分文件重评。

| 臂 | 任务分数 % | 对 compact 变化 pp | 改善/伤害题数 | 新增判断调用 |
|---|---:|---:|---:|---:|
| generalist | 44.0297 | -4.0967 | 48 / 126 | 0 |
| compact | 48.1263 | 0 | 0 / 0 | 0 |
| relevance | 48.1263 | 0 | 0 / 0 | 734 |
| scope | 48.1263 | 0 | 0 / 0 | 734 |
| scope_complement | 44.0297 | -4.0967 | 48 / 126 | 复用 scope 判断 |

compact 相对 generalist 的图像聚类配对 bootstrap 95% CI 为 **+2.5637 至 +5.6678 pp**。两种 Gate 相对 compact 的差异和 CI 都为 0，是输出相同导致的退化比较，不是非劣效统计证明。原有 126 个收益全部保留、48 个伤害全部保留；不能归功于新 Gate。complement 因没有被拒绝的包而全部复用 generalist，不能解释为有效反事实位置/内容对照。

实际送达覆盖：646/1793 题有包（36.03%），1147 题无包；558 题一个包、88 题两个包，共 734 包。未发现标签 B/C/D，也没有预算不足的 Gate 调用。5 例 canary 的 10 次 generalist/compact 文本与 token parity 全通过。最终候选没有部分保留分支，因此该分支只有 CPU 覆盖，不能说真实候选生成机制已获得验证。

历史按已送达专家组合分组（相关性描述，不是专家的因果贡献）：

| 专家组合 | 题数 | compact 改善 generalist | compact 伤害 generalist |
|---|---:|---:|---:|
| CheXagent 文本 | 529 | 114 | 33 |
| CheXagent + CXR 解剖 | 80 | 8 | 10 |
| Biomed 解剖分类 | 29 | 4 | 4 |
| Biomed 解剖分类 + CheXagent | 8 | 0 | 1 |
| 无送达包 | 1147 | 0 | 0 |

解剖组合的伤害值得后续独立机制检查，但组合、病例难度和问题分布相互混杂，不能据此删除某专家或制定疾病黑名单。当前实验只说明这个同模型类别判断未区分这些收益与伤害。

## 成本与旧资源限制

- relevance 判断累计 312.23 秒，scope 331.72 秒，合计 643.96 秒；平均每包约 0.43/0.45 秒。
- 每题摊销新增约 0.174/0.185 秒；每有证据题约 0.483/0.514 秒。模型加载合计 28.38 秒，canary parity 11.33 秒。
- 新专家调用 0，新回答生成调用 0；两者是复用策略的结果，不是省掉了医疗专家需求。
- 历史 compact 记录时间 1098.05 秒，generalist 929.28 秒。历史工具审计 1270 次调用，其中 734 个 ok、536 个 empty_or_unusable_evidence，工具 trace 时间 73.00 秒。这些是继承记录，可能包含缓存，不盲目与外层时间相加。
- 两卡与已有其他任务并发，以上为调用累计耗时，不是独占双卡吞吐基准。没有终止其他作业，没有升级共享环境。

## 旧 24 例 revision 终态审计

完整的是终态记录，而非所有臂都生成了可用答案。保持旧输出；预算会挤掉证据的一臂明确 unavailable，不运行替代输入。该项仅在 failure-inclusive 分数中计 0，原始 text 仍为 null；另外提供 available-only 配对结果。

| 臂 | failure-inclusive 分数 % | 空输出 | 工程不可用 | 对 compact 改善/伤害 |
|---|---:|---:|---:|---:|
| generalist | 33.52 | 0 | 0 | 3 / 2 |
| compact | 35.93 | 0 | 0 | 0 / 0 |
| revision_no_evidence | 25.19 | 3 | 0 | 2 / 3 |
| revision_evidence | 21.02 | 5 | 1 | 3 / 5 |

本小样本不支持继续扩大原 revision。额外完成 40 次回答调用约 49.97 秒，加载 12.43 秒，另保留历史调用成本。旧 scorer 的实际 hash 与当前文件不同，虽然两者都标 v13；使用独立 `configs/context-admission-current-scorer.json` 对所有臂统一重评，未改旧 frozen 或评分器源码。不要将本次分数与历史不同 hash 的分数混用。

## 身份、路径与复现

- 模型：原 LLaVA-Med v1.5 Mistral 7B，原 CLIP 336，原 64-token、2048 输入预算。不是其他会话的 1024-token 正式 test 设置。
- 原完整 TRAIN 基准：`/home/dbw/merit-feddg/runs/vqarad-official-train-v3/transport-fixed/0379506d53573b3a4ba8eb1c1e3648604b22fdf781949f780b6b1091d3f8dbd6`。
- manifest：`/home/dbw/merit-feddg/runs/vqarad-official-protocol-v3/data/train/manifest.jsonl`。references 只用于离线评价。原官方 TRAIN 与 test 存在共享图像，患者级隔离不可验证。
- 新运行根：`/home/dbw/merit-feddg-context-admission/runs/context-train-v2`，identity `e3d338c11b73acb5b8876a9ca03c898620095a5e67449a85555567f4736ade3b`，complete.json 已核验 1793 行精确 ID 集。
- 旧终态运行：同 worktree 下 `runs/revision-terminal-v1`。日志、原始输出留服务器，不上传病例。
- 环境：`/home/dbw/merit-feddg/.venv/bin/python`；无训练、下载或升级。宿主 GPU0 与容器 CUDA0（宿主 GPU1）分别运行一个分片，均后台 tmux，正常结束。
- CPU：`PYTHONPATH=. .../python -m pytest -q -o addopts=''`，**1032 passed in 16.41s**；新文件 Ruff 通过，CLI 帮助和预检通过。完整日志在运行根 `cpu-summary.log`、`canary.log`、`shard-0.log`、`shard-1.log`、`evaluate.log`。

执行时公共环境：

```bash
cd /home/dbw/merit-feddg-context-admission
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
export HF_HOME=/home/dbw/ANCHOR/hf_cache HF_HUB_CACHE=/home/dbw/ANCHOR/hf_cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# 使用现有 .venv/bin/python；以下参数分别在对应设备环境执行
python scripts/run_context_admission_train.py --output runs/context-train-v2 --check-only
python scripts/run_context_admission_train.py --output runs/context-train-v2 --gpu-uuid GPU-3846413a-4238-d307-b1f3-10c2dfbe002c --canary
python scripts/run_context_admission_train.py --output runs/context-train-v2 --gpu-uuid GPU-3846413a-4238-d307-b1f3-10c2dfbe002c --shard-count 2 --shard-index 0
# 宿主机 GPU0，CUDA_VISIBLE_DEVICES=0
python scripts/run_context_admission_train.py --output runs/context-train-v2 --gpu-uuid GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023 --shard-count 2 --shard-index 1
python scripts/run_context_admission_train.py --output runs/context-train-v2 --merge-only
python scripts/evaluate_context_admission.py --run runs/context-train-v2
python scripts/evaluate_context_admission.py --run runs/revision-terminal-v1 --revision-terminal --scorer-pin configs/context-admission-current-scorer.json
```

已存在评分文件会拒绝覆盖。原始评分描述曾称 strict CLOSED；最终脱敏报告明确更正为当前解释性 CLOSED 解析，数值未改变。评分代码真实哈希见 JSON。

文献与研究边界见 [调研报告](CONTEXT_ADMISSION_RESEARCH.md)。公开聚合结果见 [完整 TRAIN](../reports/context-admission-train-v1/full-train-1793-final.json) 和 [24 例终态](../reports/context-admission-train-v1/revision-terminal-24-final.json)。没有上传患者问题、回答、图像、原生证据或权重。

## 下一步判断，不在本轮自动执行

“模型觉得适配”没有形成可用决策。下一步应先用独立的、带合法正对照的用途任务区分选项偏好、指令执行失败与真实能力不足，再讨论机制；不能直接换标签顺序后挑最高任务分数。最终关注的应是证据对答案的条件效用，而不只是主题相关性。缺少逐条适用性真值、包内过滤和组合效用检查，是当前结论的边界。文献已有大量过滤/审核先例，本轮没有足够证据支持 ICLR 创新性或医学收益。
