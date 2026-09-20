# 当前 C 算法的创新性核查（2026-09-20）

本文评价实际执行的 `project`，不是尚未通过的整条 A→B/C→D/E，也不把工程可运行等同于科学有效。完整 TEST 正在运行，最终效果见后续结果报告。

## 实际方法

同一冻结模型、同一图像和已生成前缀，分别计算无专家上下文、原始专家上下文、行列布局切换、证据包逆序的四个分布。把专家与基础分布的 log 概率差作基础分布加权中心化；两个布局变化构成最多二维的候选干扰空间。在基础分布的 Fisher 加权内积下，删除专家增量落在该空间内的分量，用剩余增量解码。空证据时回退到原始模型。

对应代码：`merit_feddg/algorithm_chain/decoding.py::project_residual`、`generate` 和 `native.py::build_controls`。变换保持证据包内容，但“这些布局差异只代表无关干扰”是待验证假设。加权正交成立不意味着医疗回答正确，也不保证对未见布局或后续生成路径不变。

## 最接近的已有工作

以下根据论文原文或作者摘要核查，未借用不同数据集上的分数作横向比较。

| 工作 | 已有贡献与当前重叠 | 当前具体区别 |
|---|---|---|
| Shi 等，2024，*Trusting Your Evidence: Hallucinate Less with Context-aware Decoding*（NAACL） | 根据有/无上下文分布差修正解码；免训练上下文增量不是新概念 | CAD 放大上下文差异；当前试图删除增量中对布局敏感的分量。[原文](https://aclanthology.org/2024.naacl-short.69/) |
| Leng 等，2024，*Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding*（CVPR） | 对输入干预后比较输出分布，在推理时抑制幻觉 | VCD 扰动视觉输入；当前保持图像不变，改变专家证据的呈现方式。[原文](https://openaccess.thecvf.com/content/CVPR2024/papers/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.pdf) |
| Gan、Song、Yu、Li，2026，*Mitigating Vision-Text Order Bias in Vision-Language Model*（CVPR Findings） | DOCD 已用双顺序分布校正视觉语言顺序偏差 | 该方法调整视觉/文本位置并对比 logits；当前变换专家证据布局，并作加权子空间删除。[论文页](https://openaccess.thecvf.com/content/CVPR2026F/html/Gan_Mitigating_Vision-Text_Order_Bias_in_Vision-Language_Model_CVPRF_2026_paper.html) |
| Dasgupta、Tanvir、Zhong，2026，*Invariant Features in Language Models: Geometric Characterization and Model Attribution*（预印本） | 语义保持变化与干扰子空间的分离已有明确研究 | 该工作研究隐表示和模型归属；当前作用于每一步完整词表分布，并用于医疗问答生成。[原文](https://arxiv.org/html/2605.06458v1) |
| Xue、Xiong、Ma、Zhang，2026，*Distill What the Student Can See: Fisher-Projected On-Policy Distillation for Vision-Language Models*（预印本） | 最接近数学操作：中心化 log 概率差、分布加权 Fisher 内积、扰动估计子空间及投影 | FP-OPD 保留视觉可达子空间内的教师修正，用于训练蒸馏；当前删除证据布局空间内的上下文修正，用于冻结模型解码。[方法原文](https://arxiv.org/html/2608.01263v2) |

核查使用“context-aware/visual contrastive decoding”“order bias decoding”“Fisher projection log probability”“invariant nuisance subspace”等关键词。未检索到完全相同的实现不构成首创证明。不能声称首创 Fisher 投影、分布差解码、干扰子空间或顺序鲁棒化。

## 创新边界与关键风险

可能成立的增量贡献是：**利用同一专家证据的布局变化，在逐 token 分布空间删除不稳定的上下文增量，并实测修复错误与保留收益之间的权衡。** 这属于已有几何与解码思想上的方法扩展；目前不足以称为范式创新。

最关键的机制风险是“稳定但错误”和“敏感但正确”无法通过两种布局变换自动区分。投影也可能只是削弱全部专家作用，让模型回到较强的 Baseline。必须同时超过 Baseline、原生 MERIT 和相同四流计算的布局平均对照；只超过原生 MERIT 不足以证明这个投影有价值。

冻结 TEST 已包括上述四臂。尚未包括等强度的简单残差收缩、欧氏代替 Fisher 的投影、随机子空间、未参与构造的布局，以及其他模型。这些是论文机制归因的缺口，不会在看到 TEST 后临时加入调参或筛选。新版本必须回到开发集设计，再用新的独立验证。

现有开发集 pooled279 的算法−Baseline 为 +0.1135 个百分点，图像聚类 95% 区间 [-3.1662, 3.3692]；原收益保留13/24。既有证据没有建立稳定净收益，原冻结筛选结论仍是失败。完整 TEST 将用于固定版本的可行性判断，不能倒改开发决策。

## 评价原则

最终分别回答：单卡是否实际跑通、对现有基础模型是否有净收益、对匹配计算对照是否有独立收益、代价是否合理、现有证据能支撑多大的创新主张。自动 CLOSED/OPEN 指标不是临床正确率；OPEN token recall 对冗长答案可能宽容，必须结合输出长度、错误/收益保留和分层表现解释。

用户可用单张宿主机 GPU1；7B 冻结推理已有实测可行。数据已齐备，工程风险主要是对照一致性与耗时。每周研究时间、目标会议和写作期限未知，不能据此虚构个人能力或毕业可行性。检索到多个近期近邻，支持“竞争密集、贡献需收窄”的判断，但不能准确预测研究生命周期。
