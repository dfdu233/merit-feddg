# 多模型协作中的上下文污染：相关性、增益与裁判能力不是同一问题

## 摘要

本报告针对 MERIT 在 Huatuo 上的负结果，比较上下文鲁棒性、自我纠错和多模态裁判三类文献，并在冻结的 TRAIN 开发图像上验证一个最小工程改动。核心问题不是如何让模型更相信专家，而是如何识别采纳专家答案相对原答案的净收益。已有研究分别揭示无关上下文干扰、自我修订破坏正确答案、以及裁判偏爱信息量而忽略图像等风险；这些发现并不构成医学场景的统一因果证明。本轮先修复裁判输出协议，保留原候选而不重写，结果消除了格式失败，但仍未超过 Baseline。离线候选上限进一步区分出 VQA-RAD 的候选质量瓶颈与 SLAKE 的选择能力瓶颈。后续不应靠提示堆叠或测试集调参把两者混在一起。

## 1. 研究问题

- RQ1：与问题相关的专家证据，为何仍可能降低通用模型质量？
- RQ2：为何增加自我纠错或同模型 Gate，不能自然保证保住 Baseline？
- RQ3：在不训练、不设数值准入阈值的约束下，怎样用最小改动区分并缓解失败？

读者是准备分析算法机制、设计下一轮实验的 MERIT 研究者。本文不把工程成功写成临床改善，也不把组合已有模块写成已成立的新颖性。

## 2. 检索与核验方法

2026-09-19 进行三条独立检索：上下文/解码理论、自我纠错反证、裁判偏差与测量；每条至少两轮，由经典机制扩展到2025–2026更新，再核对正式论文与作者源码。另检索受约束生成，判断输出接口修复的边界。关键词覆盖 irrelevant context、context-aware decoding、self-correction verification、VLM judge bias、pairwise ordering、constrained decoding；通过相关论文引用链补充相反证据。

纳入下列14篇已核验正式论文。未确认 venue 或缺乏方法证据的预印本不承担结论；未找到官方代码的论文明确不声称复用了其实现。本文是问题导向综述，不声称穷尽整个领域。计算机科学基准结果和本地医学开发实验分开讨论，测试集没有用于选规则。

## 3. 按失败位置组织文献

三类轴是流水线中的失败位置，不是给论文贴互斥标签；跨位置的方法会显式说明。

| 位置 | 要区分的量 | 主要文献 | 本地可观测诊断 |
|---|---|---|---|
| 证据进入模型 | 相关性、可靠性、对当前答案的增量价值 | [1–4] | 加证据前后改善/伤害、真实证据与控制输入 |
| 生成或修改答案 | 提出正确候选、保留原有正确内容 | [5–9] | 不重写控制、同预算盲编辑、候选上限 |
| 选择并执行 | 判断质量、位置偏差、输出合法性 | [10–14] | 顺序互换、有效判决率、选择遗憾、视觉干预 |

这个分类避免把所有负结果都解释成一个 Gate 问题：候选不存在、候选有但选错、以及根本没有合法判决，是不同事件。

## 4. 为什么“更多相关信息”不保证更好

[GSM-IC 的干扰实验 [1]](https://proceedings.mlr.press/v202/shi23a.html)通过不改变正确答案的无关句观察退步，而 [RetRobust [2]](https://arxiv.org/abs/2310.01558)研究检索噪声下的鲁棒性；两者支持设置上下文干预对照，却不能直接证明当前医学案例中的因果链。RetRobust 的主方法包含微调，不符合本轮 training-free 条件；其过滤思路还可能丢弃真正有用信息。

相关性判断近似回答“证据谈的是不是这个问题”，真正的选择任务则是“采纳这条证据会不会比现有答案更好”。两者没有一般等价关系：同一器官的分类结果可能相关，但其原生分数不能直接变成当前患者的确诊事实；anatomy mask 也不能被提升成病变证据。这些是本地待核查的语义失配机制，而不是已由相关性标签证明的原因。

理想决策者得到额外观测后可以忽略它，因此其最优期望效用不会变差；这个论证依赖能够实现最优策略，不适用于固定提示、有限容量模型的实际推理。若专家只是同一图像的确定性变换，信息论上没有新增患者观测，但仍可能通过计算上的专长使通用模型更容易读取信息。不能因此把专家重复表达当成独立证据，也不能据此断言专家必然无价值。

### 4.1 对比解码并不自动识别真假

[CAD [3]](https://aclanthology.org/2024.naacl-short.69/)增强对上下文的遵循，而 [CD [4]](https://aclanthology.org/2023.acl-long.687/)对比强弱语言模型并使用 plausibility 约束；两者的对比分支具有不同角色，不能将“医学专科专家”直接等同于 CD 的 expert。原 CD 的过滤含数值参数，本轮不照搬。

令正确与错误两候选的原始 log-odds 为 m0，上下文引起的变化为 Δc。以下是我们的二候选代数推导，不是医学正确性定理：

```
m_context = m0 + Δc
m_CAD     = m0 + (1 + α) Δc
```

当 m0>0 且 Δc<-m0，上下文可以翻转原本正确的排序；正 α 会放大同一方向。因而“受上下文影响大”不是“上下文正确”的证据。这个推导解释了为何不能仅把原有软引导权重加大，却没有证明实际每个 bad case 的 Δc 都满足上述条件。

## 5. 为什么自我修订不是独立验证

[Self-Refine [5]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html)在其任务中展示迭代反馈收益，而 [Huang 等 [6]](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html)报告无外部反馈的推理自纠错可破坏正确答案；任务、目标和反馈条件不同，不能只选择一边。我们原有无专家编辑也退步，与这种风险一致，但不足以证明所有自纠错无效。

[RARR [7]](https://aclanthology.org/2023.acl-long.910/)引入外部研究证据并最小编辑，[ProCo [8]](https://aclanthology.org/2024.emnlp-main.714/)则用可核对的关键条件重建替代抽象自评；后者是“外部反馈并非逻辑必要条件”的反例。二者都不保证没有可靠图像裁判时的医学纠错。RARR 已有 agreement gate，因此“加一个 Gate 再编辑”本身不是新机制。

[Stechly 等 [9]](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f3c5e56274140e0420baa3916c529210-Abstract-Conference.html)在推理/规划任务中强调验证器质量，并比较简单重新提示与复杂反馈；结合 [5–8]，应分开衡量搜索新答案和识别正确答案的能力。形式任务的 sound verifier 不能直接类比当前医学 VLM。文本格式错误也不能被直接统计为医学选择错误。

## 6. 为什么同意、一致和合法输出仍不等于正确

[MT-Bench [10]](https://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)分析位置、篇幅和自我偏好；[LLaVA-Critic [11]](https://openaccess.thecvf.com/content/CVPR2025/html/Xiong_LLaVA-Critic_Learning_to_Evaluate_Multimodal_Models_CVPR_2025_paper.html)使用专门训练过的多模态裁判。因此“同一个模型再问一次”不能当作已获得同等评价能力，交换顺序也只处理部分位置问题。

更新的 [BIRCH [12]](https://aclanthology.org/2026.acl-long.703/)指出裁判会偏爱信息丰富却与图像冲突的答案，而 [MM-JudgeBias [13]](https://aclanthology.org/2026.acl-long.1162/)通过问题、图像和答案扰动揭示模态忽略。这两项结果意味着受限标签不能消除输入层的内容偏好。BIRCH 的图像纠正锚点值得后续研究，但其锚点由模型生成，不应在医学应用中称真值。

[PICARD [14]](https://aclanthology.org/2021.emnlp-main.779/)限制不合法解码 token；与 [10–13] 的判断质量问题不同，它针对可执行语法。我们只借鉴“在生成时保证合法控制输出”，没有复制其 SQL parser、beam 搜索或训练设置，也没有将语法合法性解释成医学正确性。

## 7. 本轮最小实现与真实结果

### 7.1 只改决策接口，保留两候选和原评测

独立分支 `experiments/huatuo-judge-channel-v1`，基于 `5d827db`。
`scripts/run_huatuo_pairwise.py` 增加 `--decision-channel finite_choice`，默认仍为旧 free_text。
同一图像/问题，两份匿名原答案交换顺序；用已有 `finite_choice_constraint` 对 A/B/C 做实际 token 级约束。
两顺序均选 compact 才采用它；平局或不一致保留 generalist。没有新增数值阈值、训练或疾病特例。
这一离散保守选择规则是显式工程选择，不是无害保证。C 也不是校准置信度。

旧 free_text 输出保留；新的选择器不改任何患者答案文本或 token IDs，只复用完整原生候选。
8-token 上限用于控制标签，不改变旧1024-token回答预算。省去长解释会同时改变推理形式，所以不能把本实验解释为纯粹的正则解析器消融。

官方源码检查：FastChat `play_a_match_pair/run_judge_pair`（提交 `587d5cfa1609a43d192cedb8441cac3c17db105d`）的交换/映射；PICARD `seq2seq/utils/picard_model_wrapper.py` 的 token admissibility mask；LLaVA-Critic 官方 model card 的图像/问题/两答案接口。实际运行复用本仓库控制解码，不安装 PICARD/FastChat，不声称运行了 LLaVA-Critic 权重。

先各8例真实 canary，全部完成且非全保留，再原样完成已固定87/64图像开发清单。
这些 TRAIN 图像此前已看过结果，明确属于开发诊断，不是新 holdout；未启动测试集评测。
参考答案仅进入离线冻结 ANCHOR scorer。旧 Baseline、路由、专家和 compact 没有重跑。

### 7.2 完整开发诊断

| 方法 | VQA-RAD87 | SLAKE64 |
|---|---:|---:|
| Baseline | 60.3448% | 52.8646% |
| 原 compact MERIT | 53.4483% | 52.9167% |
| 前轮无专家编辑 | 48.4674% | 48.6979% |
| 前轮有专家编辑 | 56.5134% | 49.4792% |
| 本轮双顺序有限标签选择 | 59.1954% | 50.8333% |
| 两候选离线 oracle，仅诊断 | 62.6437% | 62.2917% |

混合任务分数沿用 CLOSED parser + OPEN token recall，不称临床准确率。选择器相对 Baseline：
VQA-RAD 改善0、伤害1，95%图像配对 bootstrap 差值区间[-3.4483,0]pp；SLAKE 改善3、伤害4，区间[-9.375,+5.000]pp。没有可靠净增益，不能宣传不劣保证。

VQA-RAD47例两答案文本不同，94次判决，选择 compact9例，双顺序不一致15例；其余40例文本相同无需调用。SLAKE34例不同，68次判决，选择 compact14例，顺序不一致12例，另3例涉及 C；其余30例相同。全部162次调用返回合法控制标签，零格式失败，但医学选择仍可错。

新增调用累计时间31.4585s +19.3961s；与每题最长512-token解释相比机制上更短，但不据共享服务器计时宣称固定倍数加速。加载、历史生成、路由和专家预取是额外成本；旧阶段成本不完整，不能算零。新生成患者答案数为0是本选择机制的设计，不是候选生成覆盖率100%；两份候选的可用覆盖为151/151，实际选择变化为23例。

### 7.3 净收益与候选上限

令离线评分为 s，b 为 Baseline、e 为已有专家候选、g∈{0,1}为选专家与否：

```
Δ = E[g · (s(e)-s(b))]
  = 选中改善的分数质量 − 选中伤害的分数质量
oracle = E[max(s(b), s(e))]
```

这是对固定候选的恒等分解，不是训练目标或推理规则。若是二元正确率，可写作 (1-a)r-ah；本轮 OPEN 有部分分，实际用上面的加权形式。

VQA-RAD 候选改善质量2.0、伤害质量8.0，选择器捕获0改善并保留1.0伤害；即使完美选择，当前候选池相对 Baseline 也只剩2.2989pp提升空间。SLAKE 改善质量6.0333、伤害质量6.0，选择器捕获2.2并引入3.5伤害；已有候选的离线上限仍比 Baseline 高9.4271pp。前者不仅是 Gate 问题，后者更明确存在选择质量瓶颈。oracle 使用真值，只供离线诊断，绝不进入推理。

## 8. 综合解释与下一步

相关性 Gate 忽略原答案，不能直接估计增量效用；正向 CAD 可能增加对错误上下文的服从；重写同时改变事实、语言和表面形式；有限标签只消除了接口不确定性。上述是不同机制，不应一起归结为“模型不够大”。本轮修复使我们能真正观察选择错误，而不是从格式失败推断语义失败。

下一轮优先在 SLAKE 的未用 TRAIN-only 图像验证一项有视觉可证伪性的机制，而非重选已用图像或以当前开发分数调提示。可核对条件的 ProCo 思路与图像锚点 BIRCH 提供不同路径：前者要求存在可确定核对的条件，后者要求锚点本身可靠。先验证这些前提，并做同候选的无图/错图诊断，才值得追加独立裁判权重。任何控制输入扰动仅作标记清楚的离线实验，不能伪装为真实患者证据。

## 9. 局限与反证

既有证据并没有证明所有同模型 Gate 都无效；也没有证明独立裁判一定更好。本文保留 Self-Refine/ProCo 的正面结果与 Huang/Stechly 的负面条件。医学真值、中文词面评分和病人级独立性均仍有限制：图像像素去重不等于患者隔离，token recall 不衡量整段事实安全。当前两数据集也不能代表 PathVQA 或报告生成。

不把换图不敏感单独当作“没看图”的证明，因为问题有时可由语言或两候选确定；需要图像信息确实决定胜负的受控对。双顺序一致仍可共同选错。未看到神经概率，就不能将 A/B/C 选择当作已测量 uncertainty。不得从这一开发结果宣称 SOTA、创新性或统计保证。

## 10. 结论

RQ1：相关性不保证可靠性和增量价值；在错误上下文下增强服从甚至可能反向作用。
RQ2：提出、编辑、验证分别受限，同模型重复调用没有独立正确性保证；本地格式失败只能说明接口未执行成功。
RQ3：已落地有限控制输出和候选复用，实际消除了格式失败，但未取得净性能提升。当前应优先检验可证伪的视觉依据和候选选择，而不是继续扩增阈值、反思层数或重写轮次。

## References

[1] Shi et al., "Large Language Models Can Be Easily Distracted by Irrelevant Context," ICML, 2023.

[2] Yoran et al., "Making Retrieval-Augmented Language Models Robust to Irrelevant Context," ICLR, 2024.

[3] Shi et al., "Trusting Your Evidence: Hallucinate Less with Context-aware Decoding," NAACL, 2024.

[4] Li et al., "Contrastive Decoding: Open-ended Text Generation as Optimization," ACL, 2023.

[5] Madaan et al., "Self-Refine: Iterative Refinement with Self-Feedback," NeurIPS, 2023.

[6] Huang et al., "Large Language Models Cannot Self-Correct Reasoning Yet," ICLR, 2024.

[7] Gao et al., "RARR: Researching and Revising What Language Models Say, Using Language Models," ACL, 2023.

[8] Wu et al., "Large Language Models Can Self-Correct with Key Condition Verification," EMNLP, 2024.

[9] Stechly et al., "On the Self-Verification Limitations of Large Language Models on Reasoning and Planning Tasks," ICLR, 2025.

[10] Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," NeurIPS Datasets and Benchmarks, 2023.

[11] Xiong et al., "LLaVA-Critic: Learning to Evaluate Multimodal Models," CVPR, 2025.

[12] Zou et al., "When Vision-Language Models Judge Without Seeing: Exposing Informativeness Bias," ACL, 2026.

[13] Lee et al., "MM-JudgeBias: A Benchmark for Evaluating Compositional Biases in MLLM-as-a-Judge," ACL, 2026.

[14] Scholak et al., "PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models," EMNLP, 2021.
