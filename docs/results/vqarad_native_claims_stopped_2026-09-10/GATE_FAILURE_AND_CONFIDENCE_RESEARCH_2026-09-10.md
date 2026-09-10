# 医学多专家 VQA 中 Gate 失效与证据置信度：停止快照、机制诊断与改进路线

## 摘要

本报告冻结 2026-09-10 按要求停止的 VQA-RAD `native_claims` 实验，并回答三个问题：当前 Gate 为什么不能区分有用、无关和误导证据；怎样把“置信度”改成证据带来医学收益的概率或保守下界；如何只用 source 数据完成可发表且不污染 target 的验证。六臂共同完成 362/451 例，`hybrid_gate` 的严格统一分为 0.4845，比 `hybrid_all` 高 0.0076，却比逐字复现的 Generalist 低 0.0261；平均耗时从 0.676 s 增至 3.140 s。当前实现没有概率意义上的 Gate confidence：专家 `confidence` 均为 `null`，XRV 数值明确是未校准独立 sigmoid，BiomedParse mask 也未校准；所谓 Gate 信号是局部移除敏感性。它衡量模型是否依赖某个预测区域，而不衡量证据是否正确或是否改善最终答案。更严重的是，152 个纯语义条目因无法做空间检验而以 `semantic_only_unverified` 被直接接受。结论是：当前主要失败属于目标错位、未校准与自动接受未知证据，而不是已训练域泛化 Gate 的过拟合。建议把决策目标定义为 source-only、图像分组校准的 `P(ΔU>0)` 或其置信下界，分离适用性、专家可靠性、冲突、因果利用和分布偏移五个维度，并以选择性风险—覆盖曲线、harmful-accept rate、净收益和耗时为共同约束。本文仅分析，不修改算法和阈值。

## 1. 范围与研究问题

冻结问题如下：

1. 当前 Gate 为什么把“生成器依赖证据”误当成“证据有医学收益”？
2. 哪些顶会方法能让置信度对应期望增益/风险，而不是相似度、显著性或词面变化？
3. 在不使用 target 答案、不按结果放宽阈值的前提下，怎样进行最小而充分的 source-only 验证？

实验使用完整 451 例 official-test 清单，没有重新划分 target；停止时六臂共同完成 362 例。该 362 例由运行完成顺序产生，不是预注册子集，因此所有分数只用于诊断，不能写成最终论文结果。原始紧凑结果见 [`per-case-results.jsonl`](per-case-results.jsonl)，聚合及全部严格 good/bad case 见 [`evaluation-summary.json`](evaluation-summary.json)。

## 2. 研究方法与证据边界

本报告组合三类证据：代码路径和逐例 trace 用来确定“系统实际上做了什么”；冻结评测器和专门 Greedy 文件用来排除 Baseline 漂移；论文及作者/官方代码用于比较机制，而不是把别的数据集结论直接移植到医学 VQA。检索优先使用 ACL Anthology、PMLR、NeurIPS、OpenReview、CVF Open Access 和论文作者 GitHub。纳入标准是同行评审顶会/可靠主档及可核验官方实现；排除只有二手解读、未核验代码和低可信聚合页的结果。

反证检查包括：把严格 CE 解析失败与医学判断翻错分开；检查 Gate 相对 Generalist 和相对 `hybrid_all` 两种差值；检查证据是否真正进入最终生成；检查专家原始分数是否有校准语义；检查所谓 Gate 是否真在“选专家”。

## 3. 停止时结果

### 3.1 六臂共同样本

| 实验臂 | Strict unified | CE accuracy | CE parse | OE recall | 内容诊断 | 平均耗时 |
|---|---:|---:|---:|---:|---:|---:|
| Generalist | 0.5106 | 0.6262 | 0.9903 | 0.3579 | 0.5134 | 0.676 s |
| semantic_all | 0.5238 | 0.6262 | 0.9272 | 0.3885 | 0.5486 | 1.448 s |
| entry_all | 0.4797 | 0.5485 | 0.8689 | 0.3888 | 0.5156 | 1.488 s |
| entry_filtered | 0.4797 | 0.5485 | 0.8689 | 0.3888 | 0.5156 | 1.478 s |
| hybrid_all | 0.4769 | 0.5485 | 0.8689 | 0.3824 | 0.5129 | 1.577 s |
| hybrid_gate | 0.4845 | 0.5680 | 0.8641 | 0.3744 | 0.5232 | 3.140 s |

`semantic_all` 相对 Generalist 为 +0.0132（29 improve / 20 harm），但原生条目化使 `entry_all` 相对 `semantic_all` 下降 0.0441（11 / 27）。属性过滤与未过滤逐字完全相同：2,784 个检查通过、仅 1 个拒绝，因此它当前不是有效消融。空间通道 `hybrid_all-entry_filtered` 为 -0.0028。Gate 相对 `hybrid_all` 回收 +0.0076（9 / 7），但相对 Generalist 仍为 -0.0261（21 / 27）。

内容诊断不能替代主指标，但能识别格式混淆。例如 `vqarad-official-test-0002` 的 Generalist 以 `No` 开头并得 1，Gate 输出“The chest x-ray does not show...”医学语义仍是否定，却因未以 `No` 开头得 0。因此 27 个 strict harm 不能全部称为医学错误。另一方面，下文的 0053、0168、0312 确实发生了医学极性翻错。

### 3.2 Baseline 与运行成本

停止时已有 364 个 Generalist 输出，和 ANCHOR 专门评测的 LLaVA-Med Greedy 文件逐字一致 364/364。故本次共同样本上的相对下降不是 Prompt、模型权重或 Baseline 文件漂移。

`hybrid_gate` 平均耗时是 Generalist 的 4.65 倍，也是 `hybrid_all` 的 1.99 倍；每例多约 2.46 s（相对 Generalist）或 1.56 s（相对无门控混合臂）。在收益仍为负时，不支持放量。

### 3.3 Gate 实际接受了什么

362 例中，Gate 审计 408 个条目：246 接受、162 拒绝。接受项中 93 个来自正局部移除增益，152 个是 `semantic_only_unverified`，另 1 个是空间预算耗尽后的语义回退。也就是说，大多数接受项没有空间验证。按接受原因组合分组后，没有任何一组相对 Generalist 的平均差值为正：

| 例级接受原因集合 | n | 相对 Generalist | improve / harm |
|---|---:|---:|---:|
| 无接受证据 | 210 | 0.0000 | 0 / 0 |
| 局部增益 + 未验证语义 | 73 | -0.0388 | 10 / 12 |
| 仅未验证语义 | 60 | -0.0767 | 8 / 10 |
| 仅局部增益 | 18 | -0.1111 | 3 / 5 |

这是诊断关联而非因果估计，但直接否定“正局部移除增益可当作高医学置信度”的解释。

## 4. 直观 good/bad cases

### 4.1 Good：Gate 最终纠正 Generalist

`vqarad-official-test-0179` 问“肺血管是否增大”，参考为 `no`。Generalist 与 `hybrid_all` 均答 `Yes`；Gate 答 `No`。调用顺序和信号为：

- XRV findings：前四项为 Effusion 0.840、Mass 0.813、Lung Opacity 0.506、Enlarged Cardiomediastinum 0.413；这些是未校准 sigmoid，不直接回答肺血管问题。其空间条目局部移除增益 -0.00694，被拒绝。
- CheXagent：原生输出 `No`，`confidence=null`；因为没有可用空间张量，以 `semantic_only_unverified` 自动接受。
- BiomedParse：输出 left/right lung mask；一个条目的局部移除增益仅 +0.000281，也被接受。

这个 case 的直接纠错信息来自 CheXagent 的 `No`，但 Gate 并未证明它正确；恰好正确的未验证语义与一个极小正空间增益共同进入生成。因此这是结果 good case，不是 Gate 置信度校准成功的证据。

`vqarad-official-test-0346` 问右侧膈肌下是什么，参考 `free air`。Generalist 答 `mass`，CheXagent 原生输出 `Free air`，最终 Gate 答对；XRV 的最高项是 Lung Opacity 0.173，局部增益仅 +0.0000817。真正有判别力的仍是未验证语义输出，而不是 Gate 的空间分数。

### 4.2 Bad：无关/错误证据干扰正确 Generalist

`vqarad-official-test-0053` 问纵隔是否增宽，参考 `yes`。Generalist 和 `hybrid_all` 均答 `Yes`，但 Gate 输出“The chest X-ray does not show...”。XRV 条目被负增益拒绝；CheXagent 原生输出错误的 `No`，却因 `semantic_only_unverified` 被自动接受；BiomedParse lung masks 被拒绝。这里 Gate 明确让错误专家覆盖了正确通用模型。

`vqarad-official-test-0168` 问是否有肺水肿，参考 `no`。Generalist 和 CheXagent 都是否定，但 Gate 最终答 `Yes`。XRV 前四项 Effusion 0.840、Mass 0.813、Lung Opacity 0.506、Enlarged Cardiomediastinum 0.413；Gate 对其中空间激活给出仅 +0.000619 的局部增益并接受。该专家甚至没有把 `Edema` 列为前四项，正增益只说明模型对被扰动区域敏感，无法证明肺水肿成立。

`vqarad-official-test-0312` 问肝脏是否典型，参考 `no`。Generalist 指出大囊性病变并答 `No`；BiomedParse 仅返回 liver、aorta、duodenum、esophagus、gallbladder、left adrenal、left kidney、pancreas 等解剖结构 mask，不提供“正常/异常”诊断。两个条目局部移除增益为 +0.01728 和 +0.00351，均被接受，最终却翻成错误的 `Yes, typical liver`。这是“定位存在”被生成器误读成“器官正常”的典型属性错配。

以上图片路径、全部原生输出、每个 Gate 的 paired log-probability、扰动几何和 token IDs 均可按 ID 在结果 JSONL 中定位。

## 5. 为什么当前 Gate 效果差

### 5.1 优化目标错位：faithfulness/dependence 不等于 correctness/utility

实现比较“模糊预测支持区”与“模糊平移控制区”时，证据候选相对基线候选的教师强制序列分数差。它可回答“回答是否更依赖这一区域”，不能回答“专家说法是否为真”或“加入后是否提高主指标”。ERASER 明确把 rationale 的忠实影响与任务正确性作为不同维度；当前 Gate 只有前者的近似，没有后者。[ERASER, ACL 2020](https://aclanthology.org/2020.acl-main.408/)

最新的 Draft-and-Refine 同样用 question-conditioned relevance map、遮挡和外部视觉专家优化 utilization；论文报告总体收益，但其度量本身仍是利用程度，不自动成为医学正确率。[DnR, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Jeong_Draft_and_Refine_with_Visual_Experts_CVPR_2026_paper.html) 其[官方代码](https://github.com/EavnJeong/Draft-and-Refine-with-Visual-Experts)值得复用其问题条件化 relevance map 和 expert rendering，而不应照搬“利用率即可信度”的解释。

### 5.2 未知被当成通过，且没有真正的 confidence

所有代表性专家 `confidence` 都是 `null`。XRV payload 自述 `uncalibrated_independent_sigmoid` 和 `positive_threshold=not_calibrated_for_query_domain`；BiomedParse 自述 `mask_probabilities_calibrated=false`。空间条目只用 `gain > 1e-6` 的数值容差，语义条目则无法检验时直接接受。因此系统不存在“高置信有用、低置信误导”的概率变量。

Guo 等人在 ICML 2017 将校准定义为预测概率应对应真实正确频率，并指出现代网络常失准；温度缩放虽是有用基线，也必须在独立校准集上拟合。[Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a) Ovadia 等进一步表明常规后处理校准在分布偏移下会退化，模型边际化方法更稳健。[Ovadia et al., NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html) 因此不能把 source 上的原生 sigmoid 或 test-time 小正增益直接称为 target 医学置信度。

### 5.3 这不是一个学习型专家路由器

运行模式为 `all_evidence`：在兼容描述符中固定取第一个、调用、再继续下一个。它没有根据预计医学收益比较专家。Gate 是 post-acquisition entry admission，而非 expert selection。RouteLLM 的核心不同：它用偏好/胜负数据学习强弱模型的相对收益，路由分数约定为强模型 win rate，并在部署前校准成本—质量阈值。[RouteLLM, ICLR 2025](https://openreview.net/pdf?id=8sSqNntaMr)，[官方实现](https://github.com/lm-sys/RouteLLM)。当前系统缺少这种“调用谁会赢过 Generalist”的监督目标。

### 5.4 原生条目化改变了提示分布

整包 `semantic_all` 为 +0.0132，而拆成条目立即变为 -0.0309；属性过滤又 0 改变。拆分 XRV 排名表会把每一个弱 finding 都变成独立、显著的文本事实，同时移除父包 confidence 并重复包装字段。这既破坏相对排序/共同不确定性，也使 LLaVA-Med 更容易把“候选类别存在”误读成诊断。CE parse 从 0.9272 降至 0.8689也说明提示分布改变了回答契约。

### 5.5 专家能力与问题属性不匹配

BiomedParse 的器官 mask 只说明模型定位了某个结构，不能回答病变、正常性、严重程度和测量；XRV disease scores 也不能回答列表外诊断、准确位置或否定性结论。当前 `attribute_check` 只拒绝空占位和少数“空间问题+裸 yes/no”，2,785 次仅拒绝 1 次。它是接口检查，不是 relevance、entailment 或 correctness 检查。

医学场景还存在标注者分歧；AISTATS 2021 的 diagnostic uncertainty calibration 区分类别概率与专家间诊断不确定性，说明单一硬标签置信度本身也可能不充分。[Mimori et al., AISTATS 2021](https://proceedings.mlr.press/v130/mimori21a.html)

## 6. 相关方法分类与可迁移部分

### 6.1 证据如何进入视觉语言模型

Visual Evidence Prompting 将小视觉模型的专业输出符号化为 prompt，并在通用 LVLM 上降低幻觉；它支持“专家输出应显式、可审计地进入上下文”，但不解决医学专家输出错误时如何拒绝。[VEP, ACL 2025](https://aclanthology.org/2025.acl-long.205/) MedVP 用框、圈、scribble 等视觉提示并对模型做视觉提示感知微调，证明区域提示形式会改变医学 VQA 表现；它也说明当前 training-free 注入与经过适配训练的 prompt 并不等价。[MedVP, NAACL 2025](https://aclanthology.org/2025.naacl-long.587/) DnR 提供问题条件化区域和遮挡利用率，适合改进当前固定 anatomy mask 的相关性；ERASER 则提醒必须同时报告 sufficiency/comprehensiveness 与任务结果。

可迁移结论是：保留结构化 packet 和排序，不把每项自动声明成事实；采用 question-conditioned region；分别审计“送达、使用、正确、收益”。仅有送达或使用不能算成功。

### 6.2 选择性预测、校准与路由

SelectiveNet 直接优化 risk–coverage，而不是只阈值化一个预训练网络的最大置信度。[SelectiveNet, ICML 2019](https://proceedings.mlr.press/v97/geifman19a.html) UAI 2024 的系统评测也显示后处理 confidence estimator 的形式会显著影响选择性分类，必须用 AURC/风险—覆盖而非单点准确率评价。[Cattelan & Silva, UAI 2024](https://proceedings.mlr.press/v244/cattelan24a.html) RouteLLM 用相对胜负目标路由，正对应本项目“新答案是否胜过 Generalist”。Conformal Risk Control 能在交换性等假设下用独立校准数据约束单调风险，但并不自动解决 domain shift；若假设不成立，保证不可直接宣称。[Angelopoulos et al., ICLR 2024](https://openreview.net/pdf?id=33XGfHLtZg)

可迁移结论是：Gate 应输出可验证的选择分数，阈值必须在 source calibration 上冻结；主图应是风险—覆盖/收益—覆盖曲线，而非只挑一个 test threshold。

### 6.3 医学工具代理与分布偏移

MedRAX 展示了把多种胸片工具接入 agent 并用专门 ChestAgentBench 评测的可行性，[论文](https://proceedings.mlr.press/v267/fallahpour25a.html)与[官方代码](https://github.com/bowang-lab/MedRAX)适合参考工具 schema、任务分类和逐工具评测；它不等价于证明每次工具调用安全。Ovadia 的 shift 结果要求本项目把 source calibration 和 target reliability drift 单独报告；Mimori 的医学标注不确定性要求在 source 上保留多标注/软标签（如可用），不要把争议病例硬压成虚假精确概率。MedVP 则表明医学视觉提示需要适配训练，解释了当前未经适配的原生条目容易造成 prompt shift。

## 7. 推荐的新 Gate 定义（仅方案，不实施）

### 7.1 决策目标

对问题、图像、Generalist 草稿、专家及其证据定义：

`U = primary_mixed_score - λ_harm * harmful_flip - λ_time * extra_seconds`

Gate 学习的不是 `P(expert label)` 或显著性，而是
`p_gain = P(U(y_with_evidence)-U(y_generalist) > 0 | source features)`。
在 source calibration 上得到保守下界 `LCB(p_gain)` 或直接得到 `LCB(E[ΔU])`；仅当下界大于预先冻结的成本/风险界限时接受，否则回退 Generalist。医学部署应允许第三态 `abstain`，而不是把 unknown 当 accept。

### 7.2 五个分离维度

1. **适用性**：模态、问题类型、所求属性是否落在专家 capability/scope 内。
2. **专家可靠性**：按专家×模态×问题属性在 source 上校准的正确率、Brier/ECE 和样本量下界；XRV 原始 sigmoid 不直接复用。
3. **证据一致性/冲突**：多专家、原图 Generalist 与证据之间的支持/矛盾/未知；解剖存在不能蕴含正常或病变。
4. **因果利用**：question-conditioned removal/sufficiency，只作为“模型用了它”的信号；不能单独通过。
5. **偏移与不确定性**：source-to-current embedding distance、专家间分歧、变换稳定性；偏移越大应扩大置信区间或拒绝。

最终分数不要简单相乘未校准数值。可以用 source-only logistic/GBDT/meta-router 学习 `ΔU`，随后按专家/任务做温度或 isotonic 校准；若坚持 training-free，只能采用保守的 capability hard rules、独立专家一致和 unknown→reject，并明确它不是校准概率，通常会牺牲覆盖率。

### 7.3 两级决策

先做 evidence admission，再做 answer arbitration：第一层决定证据是否具备适用性与最低可靠性；第二层比较同一输出契约下的 `y0=Generalist` 和 `y1=evidence`，只有 source-calibrated 预期收益为正才提交 `y1`。这样可以避免一个被接受但无助的条目直接重写正确答案，也能把 closed answer 的 Yes/No 格式约束与医学内容分开。

## 8. 不污染 target 的验证计划

1. 从现有 source 数据建立按图像/患者分组的 train/calibration/test；查重图像 hash、问题模板和近重复，禁止 VQA-RAD official-test 进入拟合。
2. 冻结 Generalist、专家、候选生成和统一输出契约，离线产生 `y0/y1` 与完整 trace；reference 只用于计算 source `ΔU` 标签。
3. 对五维特征做最小模型与强规则两条线：保守 training-free rule；source-fitted utility router。二者不可混成一个“training-free”贡献。
4. 在 source calibration 冻结阈值，报告 ECE、Brier、AURC、risk@coverage、coverage@fixed-risk、harmful-accept rate、净统一分、CE parse、OE recall 和额外时延；置信区间按图像 cluster bootstrap。
5. 必做消融：native score only、applicability only、causal utilization only、reliability only、conflict only、完整 Gate；另报 unknown→accept 与 unknown→reject，证明当前自动接受语义未知的代价。
6. 固定后只运行一次完整 target；不根据 target good/bad cases 改阈值。当前 362 例仅用于定位机制错误，不能成为下一版本的参数选择集。
7. 放量门槛应预先写明：Generalist 非劣（统一分置信区间下界）、harmful-accept 明显下降、Gate 的 risk–coverage 优于原生 score/随机选择、非零证据覆盖且平均耗时在预算内。零调用、纯词面改善和只提高利用率均不算成功。

## 9. 开放问题与论文创新定位

最有价值的创新不是再加一个相似度分数，而是“医学证据的条件增益校准”：将专家的任务适用性、原生不确定性、跨专家冲突、因果利用与域偏移拆开，并对相对 Generalist 的收益给出可检验选择性风险。理论上可研究 source 校准在有界 shift 下的风险上界；方法上可做结构化 evidence packet 与两级 accept/commit；实证上可构建 expert-specific harm taxonomy。主要风险是 source 与 target 条件差异过大，使校准失效；这必须通过 shift-aware 区间扩大和 target 一次性审计诚实呈现，而不能靠 test 调阈值修补。

## 10. 结论

当前实验不支持放量，也不支持“Gate 高置信即医学有用”。它确实比完全无门控的混合臂少损失，但仍弱于逐字对齐的 Generalist，并付出约 4.65 倍耗时。错误主要来自：局部依赖代替正确性、语义未知自动接受、专家原生分数未校准、能力属性错配、条目化提示漂移，以及不存在真正的收益型专家路由。下一步应先在严格 source-only、图像分组数据上把 Gate 目标改为相对 Generalist 的预期净收益/风险，再冻结后进行完整 target 评测。

## 参考文献与代码

1. Li et al. Visual Evidence Prompting. ACL 2025. https://aclanthology.org/2025.acl-long.205/
2. Zhu et al. MedVP. NAACL 2025. https://aclanthology.org/2025.naacl-long.587/
3. Jeong et al. Draft and Refine. CVPR 2026. https://openaccess.thecvf.com/content/CVPR2026/html/Jeong_Draft_and_Refine_with_Visual_Experts_CVPR_2026_paper.html
4. DnR official code. https://github.com/EavnJeong/Draft-and-Refine-with-Visual-Experts
5. DeYoung et al. ERASER. ACL 2020. https://aclanthology.org/2020.acl-main.408/
6. Guo et al. Calibration. ICML 2017. https://proceedings.mlr.press/v70/guo17a
7. Geifman & El-Yaniv. SelectiveNet. ICML 2019. https://proceedings.mlr.press/v97/geifman19a.html
8. Ovadia et al. Uncertainty under shift. NeurIPS 2019. https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html
9. Cattelan & Silva. Post-hoc confidence estimators. UAI 2024. https://proceedings.mlr.press/v244/cattelan24a.html
10. Ong et al. RouteLLM. ICLR 2025. https://openreview.net/pdf?id=8sSqNntaMr
11. RouteLLM official code. https://github.com/lm-sys/RouteLLM
12. Angelopoulos et al. Conformal Risk Control. ICLR 2024. https://openreview.net/pdf?id=33XGfHLtZg
13. Mimori et al. Diagnostic Uncertainty Calibration. AISTATS 2021. https://proceedings.mlr.press/v130/mimori21a.html
14. Fallahpour et al. MedRAX. ICML 2025. https://proceedings.mlr.press/v267/fallahpour25a.html
15. MedRAX official code. https://github.com/bowang-lab/MedRAX

