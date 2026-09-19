# Huatuo：有文献与源码依据的连续小实验

## 固定研究边界

目标是修复 Huatuo 上 MERIT 相对 Baseline 的退步，而不是把另一骨干的结果当作证据。
不训练，不拟合温度，不加数值准入阈值，不增加疾病/yes-no 特例，不用 test 分数选策略。
原专家池、路由、原图、原生证据、正式1024-token回答设置保持不变。
所有新方法是论文机制的医学适配，不宣称完整复现或算法原创。

原有4个 TRAIN-only 图像用于开发。预先固定同一个旧24例清单的剩余20例用于下一次确认；
这不是新 benchmark 划分，也不是完整训练集。确认集结果一旦被查看，后续修改不能再称
在这20例上的结果为独立验证。没有查询测试答案或据其挑选病例。

## 文献到代码的明确映射

| 论文 | 正式来源与官方源码 | 本轮采用 | 未采用及限制 |
|---|---|---|---|
| Self-Refine，NeurIPS2023 | [论文](https://papers.nips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html)；[官方代码](https://github.com/madaan/self-refine/tree/9a206d41e5d2d0c241bb441f41eeadb945afaa55) | init→actionable feedback→refine；固定一次迭代；对照反馈能否看到专家原生证据 | 检查了 commongen/run.py、feedback.py、task_iterate.py 和 responsegen/run.py。不复制任务词表、示例、数值评分或词形启发式；没有声称其非医学结果保证医学纠错。 |
| RARR，ACL2023 | [论文](https://aclanthology.org/2023.acl-long.910/)；[官方代码](https://github.com/anthonywchen/RARR/tree/51a1a10fe5bada837a368f98cb55288ac5168c9e) | 从原答案开始；agreement gate 判断冲突；仅冲突时做最小编辑；另设无条件编辑对照 | 检查了 utils/agreement_gate.py、utils/editor.py、prompts/rarr_prompts.py。没有 Bing 搜索/问题生成；现有原生专家证据替代检索，因此不是完整 RARR。官方特别警告生成证据可含幻觉，本轮不将其当作真值。 |

RARR 的官方实现用文本中“disagrees”判定是否编辑，并在解析失败时可能保留原答案。
本轮改为明确记录 AGREES/DISAGREES/IRRELEVANT/UNKNOWN；不能解析时工程失败，不
隐式变成保留。UNKNOWN 是不确定关系，不是医学错误。首次模型输出将解释接在
“Decision: AGREES.”后，旧严格行解析拒绝。新版本仅支持这个通用序列化形式，仍拒绝
多个决策和歧义表达；6个正负解析检查通过。未依据医学答案调整解析。

最小编辑适配要求保留受支持内容，不把原生分数/测量值升级成无依据的事实。这是对
证据语义的要求，不是对某疾病、分数区间或答案类型加规则。原文输出全部保留。

## 实际执行

独立 worktree `/home/dbw/merit-feddg-huatuo-refine`，分支
`experiments/huatuo-self-refine-v1`，基于 `2bf719c`。
环境/权重全复用，没有安装或升级依赖。两张授权 GPU 的 UUID 显式检查；可用显存不足
时不启动，不杀其他任务。实验在 tmux 后台执行。

- `runs/self-refine-dev4-v1`：4例、两个新臂、16次反馈/修订调用，加原答案复现检查。
- `runs/rarr-dev4-v1`：保留工程失败记录，因决策行格式不匹配停止，不能作为完整医学结果。
- `runs/rarr-dev4-v2`：同4例，两个新臂，全部完成；16次实际调用。
- `runs/native-confirm20-v1`：剩余20例真实 Huatuo 路由、专家调用、原 MERIT 和 Baseline，
  已完成；不重新运行已经完成的4例，也不混入 LLaVA 预测。
- `runs/rarr-confirm20-v1`：冻结 v2 算法与提示，复用上述20例控制，双卡调度分片。

运行入口：`scripts/run_huatuo_self_refine.py`、`scripts/run_huatuo_rarr.py`。
`scripts/run_huatuo_admission_probe.py` 只增加旧清单的调度范围和 native-only 模式，
不用新 Gate 重新评判已失败的策略。离线评分沿用原 ANCHOR，代码哈希单列，CLOSED
解释性解析 + OPEN token recall 的混合任务分数，不称临床准确率。

## 已完成开发4例结果

| 方法 | 混合任务分数 | 对 compact 改善/伤害 | 对 Baseline 改善/伤害 |
|---|---:|---:|---:|
| Baseline | 80.56% | 1/0 | — |
| compact MERIT | 55.56% | — | 0/1 |
| Self-Refine，不再提供专家证据 | 5.56% | 0/2 | 0/3 |
| Self-Refine，保留专家证据 | 55.56% | 0/0 | 0/1 |
| Baseline-anchored minimal editor | 80.56% | 1/0 | 0/0 |
| RARR-style agreement/editor | 80.56% | 1/0 | 0/0 |

无条件 minimal editor 真正生成4个候选，但仍未超过 Baseline。agreement gate 的12次
判断全部 AGREES，零编辑调用、全部复用 Baseline，不能称为有效 Gate 或新方法收益。
Self-Refine 的结果不支持继续放量它。仅将 anchored editor 及其 agreement 对照保持
冻结并送入20例确认，而不是直接做完整测试集或宣称显著提升。

## 评价纪律

完整结果必须核对全部病例 ID，报告新调用、空输出、回退、改善与伤害；不以单个好例子
结束分析。20例即使正向也不能支撑 SOTA；下一步扩大 TRAIN 的独立确认后才考虑正式
测试。若20例为负，明确否定这项适配，不在这20例上反复选提示后当成 holdout。
测试集只用于最终冻结比较，不用于探索。

## 完成的20例确认

| 方法 | 混合任务分数 | 对 compact 改善/伤害 | 对 Baseline 改善/伤害 |
|---|---:|---:|---:|
| Baseline | 36.33% | 2/1 | — |
| compact MERIT | 33.33% | — | 1/2 |
| Baseline-anchored minimal editor | 46.33% | 3/0 | 2/0 |
| RARR-style agreement/editor | 36.33% | 2/1 | 0/0 |

最小编辑臂20/20真实生成候选，相对 Baseline +10pp、compact +13pp，是初步正信号，
不是显著性或 SOTA 结论。agreement 总计52 AGREES、5 UNKNOWN，无 DISAGREES，
再次零编辑、完全复用 Baseline；不将它带入主方法。
77次新调用合计151.29s，其中20次无条件编辑、57次判断；加载与继承开销另外记录。
原始结果留 runs，公开脱敏聚合见 reports/rarr-confirm20-v1.json。

后续保持同一个 minimal editor 提示与1024-token设置，扩至剩余87个 VQA-RAD
TRAIN-only 图像（每图一题、按固定哈希选择，排除已用24图及 test 图），并在 SLAKE
的 TRAIN-only 图像上做独立小样本。新增同预算、不提供专家证据的 anchored editor
控制，以区分额外重答收益与专家收益。不会将20例反复复用为独立确认集。
