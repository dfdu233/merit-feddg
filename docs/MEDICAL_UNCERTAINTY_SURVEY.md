# 医学 VLM 不确定性：证据的确定性、采纳价值与验证能力不是同一变量

## 摘要

已有医学 VLM 实验中，删除错误证据可能比复杂解码更有效，而低熵的错误回答又可能被验证模型接受。因此，本调研不把 uncertainty 当成一个统一的可靠率，而区分源模型输出、接收模型分布、外部答案验证与经过校准的决策保证。检索覆盖医学 VQA、通用上下文解码、语义熵和医学报告验证的论文及官方实现。文献支持把源端属性熵作为可检验的解码折扣，却不支持将它解释为医学正确概率；也支持研究跨模型 verifier，但不支持把更换模型视为错误独立。服务器实验因此保留固定权重对照，单独研究原生属性熵，并用常数/打乱消融及视觉/空白图验证器对照检验机制，不训练模型或使用测试集选规则。

## 1. 研究问题与范围

- RQ1：医学 VLM 通常如何估计 uncertainty，各自测量的对象是什么？
- RQ2：源端专家 uncertainty 是否可以作为双分支 alpha，并提供接收端熵以外的信息？
- RQ3：额外视觉模型能否有效选择候选，其错误相关性、视觉依赖与成本如何验证？

读者是正在开发 MERIT 的研究者。用户附件是研究参考，不是已运行的实现；其 NumPy 代码已审阅并复制到独立模块，原固定实验不动。探索只用已有完整 TRAIN manifest，最终 test 规则必须另行冻结。

## 2. 检索方法

检索日期 2026-09-15。三个独立检索视角分别覆盖医学 uncertainty、解码公式与代码、verifier 及反证，随后主线程补充校准/选择性预测和本地模型来源。检索词包含 medical VQA uncertainty、semantic entropy、adaptive contrastive decoding、source reliability、medical visual verification、conformal prediction。优先 ACL Anthology、CVF、MICCAI、Nature、作者 arXiv 与官方 GitHub；排除无法核实的工作和只有第三方概述的细节。2026 年预印本明确标注，不等同于已发表的会议结论。核心工作先核实存在，再核查摘要/正文或代码对具体机制的支持。

## 3. 按测量对象分类，而不是把不同数字混成 confidence

| 对象 | 常见量/方法 | 它能说明什么 | 不能直接说明什么 |
|---|---|---|---|
| 原生预测分布 | 类别熵、相关二元属性熵、token predictive entropy | 模型输出是否集中 | 集中输出是否正确、适合当前问题 |
| 跨预测稳定性 | ensemble/MC sampling、语义熵、视觉扰动一致性 | 预测随随机性或输入变化的稳定程度 | 稳定的系统错误是否存在 |
| 接收模型对证据的反应 | 两分支熵、JSD、top-2 margin | 证据如何改变语言分布 | 外部专家自身的可靠性 |
| 答案的外部核验 | 图像判别、参考报告蕴含、跨模型选择 | 某个候选是否被另一信息源支持 | 所有 verifier 错误是否与 actor 独立 |
| 校准后的决策 | selective prediction、conformal prediction | 给定数据与假设下的风险/覆盖行为 | 无条件的单病例医学正确概率 |

分类可相互组合，但归因需保持独立：例如 VASE 同时涉及语义采样与视觉扰动，不能把其统计量叫原生分类熵。

## 4. 从“犹豫”到“可靠”之间缺少什么

[Semantic Entropy](https://www.nature.com/articles/s41586-024-07421-0) 对多次回答进行语义分组，比字符串变化更贴近意义分歧；[VASE](https://papers.miccai.org/miccai-2025/1005-Paper0083.html) 进一步比较视觉条件，解决纯语言先验可能遮蔽图像影响的问题。前者并不保证发现反复出现的稳定错误；后者也面临弱扰动影响不足、强扰动破坏临床信息的张力。因此，采样次数、语义比较器和变换规则都必须固定，不能翻转有侧别的问题后仍假定语义保持。

[Expert-CFG](https://openaccess.thecvf.com/content/ICCV2025/html/Liang_Uncertainty-Driven_Expert_Control_Enhancing_the_Reliability_of_Medical_Vision-Language_Models_ICCV_2025_paper.html) 使用预测熵触发人工专家介入，与 VASE 的自动幻觉检测任务不同；其[附录](https://arxiv.org/html/2507.09209v1)包含低熵错误，提醒我们不能用“熵低”替代适用性。它也不是“小模型 uncertainty 自动决定 alpha”的直接验证。

VASE 的 arXiv 编号 2503.20504 在 2026-02-04 的 v2 已扩展并更名为 [UniVRSE](https://arxiv.org/html/2503.20504v2)。报告应将 MICCAI 2025 最终版和此预印本分开：输入对的预测不确定性与某一具体回答的事实真实性不是同一个对象。

本地可复用的 [TorchXRayVision](https://github.com/mlmed/torchxrayvision) 给出独立病征输出；相比人为构造 BiomedCLIP 目录 softmax，其原生二元标签空间更适合明确研究相关属性熵。然而训练域外校准仍未知。使用 `op_threshs=None`、原生 sigmoid 和官方中心裁剪/resize，不拟合温度，也不对疾病之间做 softmax。未看见的图像边缘不属于模型已验证区域。

## 5. alpha 自适应已有直接先例，新增价值应来自信息源

相较于 [CAD](https://aclanthology.org/2024.naacl-short.69/) 对上下文分支做外推，[ACD](https://aclanthology.org/2024.findings-emnlp.136/) 在有/无上下文分支之间做熵驱动插值；[AdaCAD](https://aclanthology.org/2025.naacl-long.581/) 则用 JSD 改变外推强度。[CoCoA](https://aclanthology.org/2025.emnlp-main.348/) 又结合上下文置信与冲突信号。因此“alpha 不是常数”本身没有足够新意。

设两分支共享已生成前缀，只有选定证据不同：

\[
z=(1-s_t)z_0+s_tz_E,\qquad
\rho_t=H_0/(H_0+H_E).
\]

| 对照 | 权重 | 信息来源 |
|---|---|---|
| 无新增证据 | 0 | 原 incumbent |
| 普通证据生成 | 1 | 同样的新增证据 |
| 固定 Blend / CAD | 0.5 / 1.5 | 预先固定 |
| ACD | rho | 接收模型熵 |
| Source-only | 1-U | 当前原生属性分布 |
| Source-ACD | (1-U) rho | 两类信息的启发式组合 |

后两者不是 ACD 官方方法，也不是 Bayes 正确率。对二元属性，U 为该属性 sigmoid 的二元熵除以 ln(2)。source-ACD 只会把 ACD 权重往零收缩；如果原模型自信地犯错，正确专家也可能被过度削弱。故 source-only、同均值常数与同属性打乱 U 缺一不可，不能只和固定 0.5 比较就归因于 uncertainty 的识别力。

实现审查发现几个边界。ACD [固定版本代码](https://github.com/younanna/ACD/blob/362bb3b91b66ba7de60c4252c3bd36c6de505b3d/generation/gen_wrapper.py) 在 processors/warpers 后算熵，并把同一个新 token 追加到两分支。本实验使用无额外截断的完整有限词表分数、float64 稳定熵、零分母取 0.5 的数值约定。复用独立 KV 而不共用条件缓存。AdaCAD [官方代码](https://github.com/HanNight/AdaCAD/blob/main/group_decode_fileio_adacad.py) 的自然对数 JSD 与 [CoCoA 公开脚本](https://github.com/infusion-zero-edit/CoCoA/blob/main/group_decode_fileio_CoCoA.py) 的附加罚项也要求标清版本；本实验不声称复现这两项。

## 6. verifier：参考指标、无参考视觉判断与自验证不能混称

[GREEN](https://aclanthology.org/2024.findings-emnlp.21/) 和 [RadFact](https://github.com/microsoft/RadFact) 都需要参考报告来评价报告内容，[RadGraph](https://github.com/Stanford-AIMI/radgraph) 的图谱相似度也依赖参考图谱；它们不能在不提供真值的推理阶段自动成为视觉事实裁判。相较之下，[Phrase-grounded Fact-checking](https://papers.miccai.org/miccai-2025/0693-Paper3526.html) 研究图像—finding—位置验证，但要训练新模型，会议页没有可直接复用的代码/权重入口，不符合当前零新增训练实验条件。

跨模型复用可以测试，却不提供错误独立保证。[Verification Mirage](https://arxiv.org/abs/2605.10850)（2026-05 预印本）专门指出医学 VQA 自验证和跨模型验证的错误耦合；与 [MT-Bench 的 judge 偏差研究](https://arxiv.org/abs/2306.05685) 一起，它支持同时核验候选交换顺序与图像依赖，而不是只数 judge 接受率。

本地有完整 [OpenMed/Qwen2.5-3B-MedVL](https://huggingface.co/OpenMed/Qwen2.5-3B-MedVL/blob/main/README.md) 权重，因此将其与 LLaVA-Med actor 分离：judge 只接收原图、问题和匿名候选，不接收原始专家分数/alpha/参考答案。每对 A/B 与 B/A 各调用一次；两个顺序都选新候选才替换，否则保留 incumbent。用同尺寸空白图对照检验视觉依赖，同模型 judge 另作诊断。该模型经 SynthVision 医学 VQA 微调，训练重叠未独立排除；不同模型绝不意味着未见过这些数据。

## 7. 为什么这轮不做 calibration 或 conformal guarantee

[MICCAI 2025 的 SCP 研究](https://papers.miccai.org/miccai-2025/0955-Paper4783.html)依赖带标签 calibration 集和交换性假设；[Overconfidence and Calibration in Medical VQA](https://arxiv.org/abs/2604.02543) 与 [Calibrated Triage, Not Autonomy](https://arxiv.org/abs/2606.15910)（均为 2026 预印本）进一步把医学 confidence 的测量与实际决策风险分开。因此，未经校准的 U→alpha 不能借用 conformal coverage 的保证。本轮不拟合温度、阈值或校准器；不学习 gate，也不把所有病例的标签用于冻结常数。

## 8. 预注册式最小实验与反证

本地完整 TRAIN 基准为 1793 行 `transport-fixed/0379506d...`，保留 `compact_rows`、`compact_all`、`semantic_all`。在同一完整 manifest 上，按纯问题正向语法、原 modality 与官方标签取 12 题：Effusion/Cardiomegaly/Pneumothorax 各最先出现的 4 张不同图像。不会创建 test 子集或按答案挑样本。相同图像可跨属性出现，离线 bootstrap 按图像聚类，不能当独立患者。

每例保留原始有效专家证据，再增加一份真实 XRV 证据。无范围聚焦的完整标签表与只保留所问标签的普通文字分别记录，避免把适用性收益归给熵。随后固定 0.5/1.5、ACD、source-only、source-ACD、同属性平均 1-U 和循环打乱 U 使用同一对分支。完整词表分数不从旧轨迹复用。

新增源端推理一次/例；其成本单列，所有候选复用它。主 verifier 只评估预先指定的 source-ACD 候选与原 incumbent，不从多个候选里事后择优。空白图仅作 verifier 机制对照，不当作患者图像增强或改变 actor 输入。所有 judge 调用、平局、未知、顺序不一致、格式失败均计入分母。

反证条件：source-only 不胜同均值/打乱控制，不能说 U 识别了有用证据；source-ACD 没有新增正向收益，不继续堆信号；视觉 judge 不胜空白图，不能说它真正使用视觉信息；全保留或无可修复候选，不能宣布 gate 成功。旧 ANCHOR v11 保留作对齐指标，另列严格 leading yes/no 诊断，避免“not”免责声明带来的评分假象。

## 9. 开放问题与结论

RQ1：医学 uncertainty 并无单一通用标量；分布集中、预测稳定、视觉支持和校准风险有不同前提。RQ2：源端 U 可以作为可审计的折扣，但价值只能通过相同平均强度与打乱消融证明，不能靠公式新颖性或更低输出熵。RQ3：额外视觉模型可以做候选选择实验，但要证实它具备增益识别能力、视觉依赖和合理成本；现有研究不支持把它当临床安全层。

本轮未实现生成专家多采样/NLI 语义熵，也未把 mask 稳定性或检索排名包装成正确率。后续只有在原生分类属性实验出现可复验机制证据时，才考虑固定预算的生成式采样、独立 holdout 和跨专家迁移；本轮 12 题不能支撑 ICLR 级性能或临床结论。实测结果和命令另见 `UNCERTAINTY_TRAIN_RESULTS.md`。

## 参考来源

核心论文与官方代码已在各论点旁直接链接。其中 CAD/ACD/AdaCAD/CoCoA 为已发表 NLP 解码工作；Semantic Entropy 为 Nature 2024；VASE、SCP 与 phrase-grounded fact-checking 为 MICCAI 2025；Expert-CFG 为 ICCV 2025；Verification Mirage、UniVRSE v2 和两项 2026 calibration/triage 工作按预印本处理。没有把附件中的合成测试算作真实医学实验。
