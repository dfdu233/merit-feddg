# v0.9 研究设计：生成过程中的连续能力价值策略

本文解释方法、已有工作的关系和实验边界，不是性能报告，也不构成 ICLR 接收承诺。文献按正式发表年份标注；个别论文的 arXiv 年份早于会议年份。调研核对日期：2026-09-06。

## 1. 保持不变的研究目标

让一个冻结的通用医学 VLM，在生成过程中按需使用不同医学专用模型的原生能力；同时考虑这些专用模型可能只在狭窄数据来源上有效。目标不是把所有模型改造成同一套诊断候选分数，也不是先写完完整答案，再让另一个模型检查。

本轮把待研究问题收敛为：

> 给定图像、问题、已经生成的精确前缀和已有工具证据，现在再调用某个工具，预期能为后续回答增加多少质量？这种增量在不同源域和工具调用历史之间能否迁移？

“像 agent”描述交互形式；学术上需要研究的是有预算的顺序信息获取、条件边际效用、工具证据组合和源域偏移，而不是仅给一个工具调用循环起新名字。

v0.9 的最小实现采用一个轻量连续回归策略。主模型、视觉特征编码器和专家先保持冻结；不要求训练新的医学基础模型，不用二值“主模型会不会答错”作为策略训练目标。

## 2. 为什么调整 v0.7/v0.8 的路线

此前实验暴露了不同层次的问题，不能全部归因于一个 DG 门槛：

- 把所有证据都加入上下文，会引入无关描述、相似病例答案和额外文本长度；“调用了更多专家”不等于“获得更多有效信息”。
- 自由 JSON 控制器可能把专家、能力、scope 混配。长控制器输出本身也有明显成本。
- 一个专家在所有源样本上的平均增益为负，并不排除它在某些问题、某些前缀或某些已有证据下有正增益；整张资格卡按均值拒绝会丢掉这种互补性。
- 初始状态的单工具增益，不等于生成中途的增益；单独调用 A、B 的收益，也不等于先 A 后 B 的收益。
- Token-F1 的提高可能来自词面重合，而不是医学事实改善。

因此本轮不是把原有信任阈值调低，而是改变预测对象：从“这个工具整体合格吗”改为“这个工具在当前生成状态下还有多少边际价值”。必要的类型、图像模态、预算和数据支持检查仍保留，但不再把若干未经校准的路由置信度、OOD、可靠性概率简单相乘解释为安全概率。

## 3. 十二项最相关的已发表工作及官方实现

以下是方法设计的主要参照，不是声称逐个完整复现了这些系统。代码链接来自作者或项目官方仓库；旧项目的环境不应直接覆盖现有医学模型环境，应按许可证与依赖兼容性单独复现。

| 工作与发表位置 | 原论文／官方代码 | 借鉴点，以及不能直接继承的结论 |
| --- | --- | --- |
| **Counterfactual Risk Minimization: Learning from Logged Bandit Feedback**，ICML 2015 | [论文](https://proceedings.mlr.press/v37/swaminathan15.html) · [作者 POEM 代码](https://www.cs.cornell.edu/~adith/POEM/) | 用历史干预及收益学习策略。只有被记录的动作得到反馈时，需要处理动作支持和 logging propensity；不能把未调用工具记为零收益。本项目工具较少时直接运行同状态完整分支，可避免不必要的 IPS 方差。 |
| **Safe Policy Improvement with Baseline Bootstrapping**，ICML 2019 | [论文](https://proceedings.mlr.press/v97/laroche19a.html) · [官方代码](https://github.com/RomainLaroche/SPIBB) | 数据不足的状态—动作回归基准行为。原文有限 MDP、固定离线数据及不确定性条件下的保证，不能直接转移到任意 VLM 前缀和未知医院；“不差于基准”也不等于基准本身临床安全。 |
| **Consistent Estimators for Learning to Defer to an Expert**，ICML 2020 | [论文](https://proceedings.mlr.press/v119/mozannar20b.html) · [官方代码](https://github.com/clinicalml/learn-to-defer) | 应优化主模型与专家的系统互补性，而不只是主模型熵。原问题主要是分类决策转交给专家；这里专家提供分割、检索等观察，最终仍由医学 VLM 回答，不能直接沿用其一致性结论。 |
| **Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization**，ICLR 2020 | [论文](https://arxiv.org/abs/1911.08731) · [官方代码](https://github.com/kohpangwei/group_DRO) | 提醒不能只优化平均源域。最差已知组训练与正则化有价值，但不是任意新域的保证。当前轻量策略是 domain-balanced ridge 加留域残差，不应写成已复现 Group DRO 的 minimax 优化。 |
| **In Search of Lost Domain Generalization**，ICLR 2021 | [论文](https://arxiv.org/abs/2007.01434) · [DomainBed 官方代码](https://github.com/facebookresearch/DomainBed) | 强 ERM 基线、模型选择和公平实验协议非常重要。调参必须区分源域验证与目标域 oracle；随机哈希分组不应冒充真实医院域。 |
| **FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space**，CVPR 2021 | [论文](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_FedDG_Federated_Domain_Generalization_on_Medical_Image_Segmentation_via_Episodic_CVPR_2021_paper.html) · [官方代码](https://github.com/liuquande/FedDG-ELCFS) | 医学域变化和多源学习的直接参照。原方法含实际分散客户端、频域分布交换和局部 episodic 训练；本项目目前集中式冻结工具策略不能仅因仓库名含 FedDG 就宣称实现了该联邦算法或隐私保证。 |
| **ReAct: Synergizing Reasoning and Acting in Language Models**，ICLR 2023 | [论文](https://arxiv.org/abs/2210.03629) · [官方代码](https://github.com/ysymyth/ReAct) | 语言生成、动作、观察交替出现并非新概念。本项目需要比较通用提示式工具调用与低开销价值策略；仅增加一个 action/observation 循环不足以构成创新。 |
| **Visual Programming: Compositional Visual Reasoning without Training**，CVPR 2023 | [论文](https://arxiv.org/abs/2211.11559) · [VisProg 官方代码](https://github.com/allenai/visprog) | 不同视觉模块通过原生中间结果组合，已覆盖检测、分割、分类等能力。区别应来自医学生成状态中的条件效用与跨域实验，而非“能插很多模型”本身。 |
| **ViperGPT: Visual Inference via Python Execution for Reasoning**，ICCV 2023 | [论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Suris_ViperGPT_Visual_Inference_via_Python_Execution_for_Reasoning_ICCV_2023_paper.pdf) · [官方代码](https://github.com/cvlab-columbia/viper) | 用视觉 API 与中间结果完成组合推理，是原生能力协作的重要基线。这里不执行主模型任意生成的 Python，而只允许已登记、参数合法的工具动作。 |
| **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs**，ICLR 2024 | [论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/28e50ee5b72e90b50e7196fde8ea260e-Paper-Conference.pdf) · [ToolBench 官方代码](https://github.com/OpenBMB/ToolBench) | 工具描述、工具选择、执行和评估需要分开设计；非法动作率是实际系统指标。它不是医学影像域鲁棒性证明，也不直接覆盖同一前缀下工具对自由生成的边际作用。 |
| **Conformal Risk Control**，ICLR 2024 | [论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf) · [官方代码](https://github.com/aangelopoulos/conformal-risk) | 风险控制必须写清可交换性、有界单调损失等条件；偏移扩展也有额外假设。本项目留域残差分位数不是 conformal risk control，不能写成“90%/95% 安全保证”。 |
| **Distributionally Robust Policy Evaluation under General Covariate Shift in Contextual Bandits**，TMLR 2024（期刊） | [论文](https://arxiv.org/html/2401.11353v2) · [作者代码](https://github.com/guoyihonggyh/Distributionally-Robust-Policy-Evaluation-under-General-Covariate-Shift-in-Contextual-Bandits) | 用鲁棒连续收益模型改善策略评估，直接支持“预测工具增量”的建模方向。但该工作明确假设条件收益分布在 covariate shift 下不变，并使用源/目标密度比或估计；不能把它包装为本项目 source-only 未知域保证。 |

已有工作已经覆盖了策略收益估计、悲观改进、模型协作、工具组合与分布鲁棒性。可以探索的论文贡献是这些因素在**同前缀、原生医学能力、历史条件、真实域偏移**这一具体问题中的新机制和实验证据；是否足够创新必须由与近邻方法的差异和结果决定。

### 3.1 另外直接影响实现的工具学习、机器人和视觉工作

| 工作 | 原文 / 作者实现 | 本轮具体取舍 |
| --- | --- | --- |
| SayCan，2022 机器人方法 | [项目与论文](https://say-can.github.io/) · [作者代码](https://github.com/google-research/google-research/tree/master/saycan) | 语言上“适合做”和当前状态下“实际能做/有价值”分开。我们不照搬技能成功概率乘积，而测工具使最终回答发生的连续质量变化。 |
| Toolformer，NeurIPS 2023 | [正式论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html) | 以实际工具结果对后续预测的作用筛选调用。未核实作者正式训练仓库，不把社区实现当官方代码。本轮采用真实完整续写的配对收益，不假装复现其训练。 |
| ToolkenGPT，NeurIPS 2023 | [正式论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8fd1a81c882cd45f64958da6284f4a3f-Abstract-Conference.html) · [作者代码](https://github.com/Ber666/ToolkenGPT) | 已有冻结主模型、轻量工具参数方案；轻量工具选择本身不新。已查看工具 head/冻结参数逻辑，但未复制 GPL 实现；本轮 head 输入包括图像、前缀和原生观察，监督为连续边际收益。 |
| Visual Sketchpad，NeurIPS 2024 | [论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/fb82011040977c7712409fbdb5456647-Paper-Conference.pdf) · [作者代码](https://github.com/Yushi-Hu/VisualSketchpad) | mask/框可作为视觉上下文，不能全丢成文字。其对照也提醒只给覆盖后的图像可能有害，因此本轮固定保留原图，另加一张预测 overlay；不是照搬其完整 agent。 |
| AvaTaR，NeurIPS 2024 | [论文](https://arxiv.org/html/2406.11200v3) · [作者代码](https://github.com/zou-group/avatar) | 执行反馈与正负案例可优化工具策略。本轮用固定真实源分支监督，不依赖目标反馈优化 prompt；未复现其迭代 agent 优化。 |

本轮实际查看了 ViperGPT 的 `image_patch.py`、VisProg 的解释器/共享状态、ToolkenGPT 的模型与训练入口、ReAct 的工具环境、AvaTaR 的 agent/tool 实现，以及 Visual Sketchpad 的专家架构说明。这些是设计核查，不是声称完整运行了上述系统。没有把别的项目的任意 Python 执行器或整套旧依赖安装进医学实验环境。

### 3.2 医学近邻不能漏掉

- [MMedAgent，Findings of EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.510/) 已研究跨任务、跨模态的医疗工具选择；“医疗 agent + 多工具”不是新主张。
- [VILA-M3，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Nath_VILA-M3_Enhancing_Vision-Language_Models_with_Medical_Expert_Knowledge_CVPR_2025_paper.html) 已把医学专家信息加入 VLM 训练/使用流程；[作者框架](https://github.com/Project-MONAI/VLM-Radiology-Agent-Framework/tree/main/m3) 也使用 TorchXRayVision 等专家。最终论文必须与其专家融合/调用思路做直接对照。
- [MedRAX，ICML 2025](https://proceedings.mlr.press/v267/fallahpour25a.html) 已有胸片多工具推理工作。我们的差异必须落在**同前缀工具边际价值、原生观察作用和真实域迁移**的实证，而不是“可插拔”字样。

### 3.3 官方后端检查发现的配对实验混杂

检查 [LLaVA-Med 官方 `mm_utils.py`](https://github.com/microsoft/LLaVA-Med/blob/main/llava/mm_utils.py) 时发现，当前 `expand2square` 对非正方图可能随机改变一个像素的填充偏移。反复预填充时，NONE 和 tool 分支可能看见不同预处理图像。v0.9 profile 因此在原图和预测视图上采用固定居中填充，再调用原来的视觉 processor；不改变全局 RNG，也不改远端官方源码。配置、后端源码和像素指纹均进入缓存身份。旧入口默认行为保持不变。该问题是否影响此前具体远端提交，须以其已记录官方源码指纹和配置核对，不能直接归咎此前全部负增益。

### 3.4 2025–2026 最新近邻：进一步收紧创新主张

| 正式工作 | 原文 / 作者代码 | 与本轮的重合和边界 |
| --- | --- | --- |
| SPORT，NeurIPS 2025 | [论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/55d16334943f8728073e17139e5baa3d-Abstract-Conference.html) · [作者代码](https://github.com/SPORT-Agents/SPORT-Agents) | 在相同历史采样执行多个工具动作，再构造逐步偏好微调多模态控制器。**相同历史的多动作探索/credit assignment 已有先例**。本轮使用精确生成 token 前缀、原生记忆和连续终局 CALL/NONE 差，冻结主模型，只拟合源域小 head；这些差别仍需要真实跨域收益来成立。 |
| GiGPO，NeurIPS 2025 | [论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/420c9f777c0b4f78d515e53cf74d58b2-Abstract-Conference.html) · [作者代码](https://github.com/langfengQ/verl-agent) | 把重复 anchor state 的动作分组，用未来回报形成逐步优势并结合 episode 优势训练 agent。不能声称首次提出 same-state credit。当前 ridge 不是 GiGPO，也没有长轨迹强化学习的完整价值传播。 |
| Tool-REX（早期名 Tool-DE），ICLR 2026 Poster | [ICLR 官方确认](https://iclr.cc/virtual/2026/poster/10008173) · [方法全文](https://arxiv.org/html/2510.22670v1) · [作者代码](https://github.com/EIT-NLP/Tool-REX) | 正式题目 *Tools are under-documented: Simple Document Expansion Boosts Tool Retrieval*。Tool-Embed/Tool-Rank 训练工具相关性检索与重排。它不等于工具执行后自由回答的增量价值，但“用小模型选择工具”已充分研究；本轮不能仅以冻结小 head 构成创新。OpenReview PDF 遇验证页，方法按作者全文/源码核查，录用状态按会议官网核查。 |
| ToolRL，NeurIPS 2025 | [论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/97c5b2707228e7e3fb67e4ecc2e0e607-Abstract-Conference.html) | 工具奖励的粒度、尺度和时间结构直接影响学习。这里的 F1 回归仍只是最低限度研究基线；后续应以独立事实质量替代，而不是把答案词面奖励换名为医疗可靠性。 |

据此，本轮代码应定位为**可复现的研究基线与新机制试验台**，而非已证明的全新 ICLR 方法。值得继续验证的空白是：异构医学原生证据进入生成后，工具的条件增量在真实来源偏移下怎样失效，怎样在不重训医学主模型的条件下保留有用协作。必须直接对照视觉工具 agent、step-wise 工具学习和无跨域保守项版本，才能支撑超出工程组合的贡献。

## 4. 最小在线流程：仍由通用医学 VLM 生成答案

状态记为：

\[
s_t=(x,q,p_t,M_t,H_t,b_t),
\]

其中，\(x\) 是真实医学图像，\(q\) 是问题，\(p_t\) 是已经输出的精确 token 前缀，\(M_t\) 是已采用的原生证据记忆，\(H_t\) 是有序工具历史，\(b_t\) 是剩余预算。

执行链只需要以下循环：

1. 根据图像类型、任务、能力契约和参数合法性获得候选工具。数据集被称为 pathology，并不意味着其中每张图都是组织切片。
2. 从当前状态和每个候选的 descriptor 提取预调用特征，预测连续效用；NONE 的新增效用定义为零。
3. 只有数据支持足够、预算允许且净效用为正时，执行最高分工具；否则继续主模型生成。
4. 工具返回原生观察，经 EvidenceBridge 转成受限、可追踪的证据进入记忆。主模型保留原前缀，继续生成。
5. 在后续生成边界重算剩余工具的边际价值，直到回答结束或预算耗尽。

这不是把专家的答案候选分数直接加到主模型 logits 上；轻量策略分数只决定是否获得一条观察。最终文本仍由通用医学 VLM 产生。

也不要把“精确前缀保留的 block-wise 生成”描述成“每个 token 都调用专家”或“无需重新 prefill”。增加证据通常需要重新建立上下文，实际 token 复用、重新 prefill 和缓存成本应由 trace 与计时验证。

### 4.1 原生证据接口的含义

现有 `CapabilityRequest`、`CapabilityResult`、`EvidenceItem` 保持候选答案无关的契约。能力目录可包括：

| 能力 | 允许的原生输出 | 不能擅自升级成的结论 |
| --- | --- | --- |
| Classification | 有明确标签语义的预测及模型原生分数 | 任意问题的最终诊断；分数不自动是校准概率 |
| Segmentation | 掩膜、结构名称、面积和几何关系 | 掩膜正确或结构异常已经获得临床认证 |
| Detection | 区域、目标类别、检测分数 | 框内所有病理细节均真实存在 |
| Retrieval | 相似来源病例、来源 ID、相似度及适用性信息 | 检索病例的参考答案就是当前病例答案 |
| Generation | 专科模型的描述及其来源 | 独立且无幻觉的医学事实 |

接口能表达某能力，不代表已有可执行、任务匹配并通过真实数据评估的适配器。应单独报告“接口覆盖、实际模型覆盖、实际调用覆盖、产生收益的能力覆盖”。不应只凭注册表枚举五类能力就宣布五类协作均已验证。

医学主模型可以替换为实验环境已有的 LLaVA-Med；OpenMed/Qwen 等作为成对对照。不同主模型会改变工具收益，因此策略与模型、prompt、tokenizer、桥接版本、工具权重和数据指纹绑定，不应默认跨主模型复用资格与价值头。

## 5. Source-only 同前缀真实分支：连续监督从哪里来

对源域中的真实图像和真实问题，在固定状态 \(s\) 处分叉：

- NONE 分支：保留当前前缀与证据，不调用新工具，按固定后续生成协议完成回答 \(y^0\)。
- 工具 \(a\) 分支：从完全相同的前缀、图像和既有证据出发，仅新增该工具的原生观察，按同一后续协议完成回答 \(y^a\)。

定义连续质量增量和非负增量成本：

\[
\Delta(s,a)=Q(y^a,r)-Q(y^0,r),\qquad
c(s,a)=\max(0,T^a-T^0).
\]

其中参考答案 \(r\) 仅用于源域离线评分。当前接口接受 \(\Delta\in[-1,1]\) 和非负成本；若采用另一种无界质量指标，必须先明确重标度，不能静默截断医学损失。

使用同一前缀是为了隔离“现在多获得这条观察”的作用。图像、问题、采样种子、模型、模板、最大新生成 token、停止条件和后续策略应相同并留指纹。若两分支还改变了未来控制器、后续工具集合或答案格式，那么差值就是混合干预，不能仍解释为该单工具的边际贡献。

需要至少区分三类真实状态：

- `initial`：第一答案块之前。
- `continuation`：已经产生一段真实回答、尚无工具历史的前缀。
- `after_tool`：已有实际工具观察，测量下一工具的条件增量。

不得为了凑状态数给一个病例复制若干相同前缀，也不得根据目标正确答案选择“最容易挽救”的采集状态。初期病例回答很短时，不能因为接口支持多个 block 就声称完成了生成中途协作。

### 5.1 最小记录与标签隔离

策略训练记录包括：

| 字段 | 作用 |
| --- | --- |
| `role='source'`, `sample_id`, `group_id`, `domain` | 标明仅源域拟合；`group_id` 是独立 RGB 图像/患者组，`domain` 另行定义 |
| `state_id`, `state_kind`, `history_actions` | 前缀状态和精确有序历史；用于条件支持检查 |
| `action_key`, `features` | 专家/模态/任务/能力/scope 及预调用状态—动作特征 |
| `gain`, `cost`, `executed`, `adopted` | 连续增量、实际成本、是否执行、是否得到新增可用证据 |
| 配对输出、前缀 token、证据、计时和指纹 | 由采集层保存，用于追溯差值和缓存一致性，不直接作为预调用特征 |

预调用特征可以使用图像、问题、已有前缀、旧证据和工具 descriptor；**不能读取当前尚未执行工具将要返回的 evidence**。该证据只能进入执行后的下一状态。目标 reference、最终正确答案、评估分数都不能进入在线特征或动作选择。

全部执行过且完成配对评分的记录进入收益拟合，包含无新增证据或负增益的结果；不能只保留 adopted 样本拟合，否则会产生幸存者偏差。运行失败与合法空观察分开记录，失败不伪装成零收益。重复的同病例、同状态、同动作记录去重；相互矛盾的重复记录拒绝。

若工具数很少，源域直接运行合法动作分支比用不完整行为日志做复杂离线估计更简单。若以后仅采样部分动作，必须记录实际采样概率和覆盖范围；确定性旧日志中从未调用的动作不能得到无偏反事实结论，参见 CRM。

## 6. 实际连续策略及其保守项

### 6.1 共享 ridge 回归，而非工具整体常数

设 \(\phi(s,a)\) 是固定维度的源域可复现特征。特征必须包含状态与工具描述的交互；否则共享线性头可能退化为固定工具排序，无法回答“同一工具在什么上下文有效”。

策略用同一个设计矩阵拟合两个连续输出：质量增量 \(\hat\mu(s,a)\) 和增量成本 \(\hat c(s,a)\)。归一化均值和尺度只由拟合源域获得。若训练共有 \(G\) 个独立组、\(D\) 个域，域 \(d\) 有 \(G_d\) 个组，组 \(g\) 有 \(m_g\) 条记录，则实现采用：

\[
w_i=\frac{G}{D\,G_{d(i)}\,m_{g(i)}},\quad
\hat W=\arg\min_W \sum_i w_i\lVert z_i-\widetilde\phi_i^\top W\rVert_2^2
+\alpha\lVert W_{\text{non-intercept}}\rVert_F^2,
\]

其中 \(z_i=(\Delta_i,c_i)\)，\(\widetilde\phi_i\) 为带截距的归一化特征。这样每个域总权重相同，同域每个独立图像组总权重相同；给某图增加更多问题或前缀不会按记录条数放大其权重。预测增量限制在 \([-1,1]\)，成本限制为非负。

这不是 Group DRO 训练，也不是二值错误风险估计。模型在同一 action 上可以对不同状态给出正、负连续值；条件数据足够时，不会因该工具整体平均收益为负而整卡禁止。

### 6.2 历史条件支持：证明范围不能由初始样本数代替

条件签名定义为：

\[
k=(\text{action key},\text{state kind},\text{ordered history}).
\]

动作 key 含专家、模态、任务、能力、scope。有序历史不会排序：先 A 后 B 与先 B 后 A 是不同状态。只观察过初始调用 A，不支持“生成中途调用 A”；只观察过 A 后调用 B，不支持 C 后调用 B。

每个条件按独立图像组计算 support，要求达到配置的源域数和每域非空 adopted 组数。默认参数不是医学定理；降低门槛只能降低工程数据要求，不能自动建立统计可信性。空执行仍影响收益拟合与残差，不能因不计入非空支持就从训练中消失。

未观察条件或低支持条件回退 NONE，并明确区分 `unknown_action`、`unseen_state_history`、`insufficient_independent_support` 等原因。这表示“不能依据现有数据作出该干预”，不是“工具对该病人一定无用”。

### 6.3 留一源域的过估残差

对每个源域 \(d\)，仅用其他源域重新拟合归一化和回归头。对留出域记录计算净效用过估：

\[
e_i^{(-d)}=
\left(\hat\mu_{-d}(s_i,a_i)-\lambda\hat c_{-d}(s_i,a_i)\right)
-\left(\Delta_i-\lambda c_i\right).
\]

同条件、同独立图像组的多个状态先取最大过估，再计算域内经验高分位数，并取最差源域：

\[
P_k=\max_d\left[0,\operatorname{Quantile}_q
\left(\left\{\max_{i\in(g,d,k)} e_i^{(-d)}\right\}_g\right)\right].
\]

实现使用经验 `higher` 分位数；它没有有限样本 conformal 修正。最终部署点预测来自全部源域拟合的共享头，不是简单对 LODO 头求平均。LODO 头用于构造与审计过估残差。

最终分数为：

\[
V_{\mathrm{mean}}(s,a)=\hat\mu(s,a)-\lambda\hat c(s,a),\qquad
V_{\mathrm{robust}}(s,a)=V_{\mathrm{mean}}(s,a)-P_k.
\]

在支持和预算允许的动作中选择最大正分数；若没有则选择零价值 NONE。`robust=False` 仅去除 \(P_k\)，不绕过能力、预算或历史支持限制。当前不另叠加 ensemble 标准差、OOD 乘数或所谓多重安全概率。

\(\lambda\) 默认可为零，此时算法没有优化延迟—质量折中，即使报告了延迟也不能宣称已经学会成本最优路由。非零值必须依据源域验证和目标预算预先确定。实际计时应区分冷启动、模型加载、工具推理、证据重新 prefill、缓存命中以及固定视觉特征开销。

## 7. 必须明确的 DG 与联邦边界

### 7.1 留域残差不是正式目标域风险保证

当前方案只能称为“源域留出偏移下的经验保守策略”。不能从中推出任意未知医院上的风险上界，原因包括：

- 未知域可能包含源域完全不存在的病种、染色方式、设备、成像质量或问题类型。
- 相同特征下专家的实际效用也可能改变，不只发生 covariate shift。
- 留出域残差由不同的 head 产生，部署又在所有源域上重新拟合；不是标准独立校准条件。
- 在多个动作、状态和组合中自适应选择，会改变最终被访问状态的分布。
- 支持计数与正增益选择不构成整个策略的临床安全认证。

LODO 分位数的 0.9 只表示一个经验分位水平，不能解释为“90% 概率不会损害回答”。如要正式保证，需另外明确数据生成、目标偏移集合、有限策略类或风险控制条件并给出相应证明；不能把已有 DRO/CRC 术语贴到当前启发式上。

### 7.2 共享 source retrieval bank：这里只做 policy-level LODO

若源域配对输出预先使用一份共享 source 检索库生成，然后回归头才进行 leave-one-domain-out，那么留出的域可能仍存在于检索库中，或参与过其他训练域分支的 evidence。即使检索排除了当前图片、当前 group 或当前域，也不能因此断言整个外层 fold 完全隔离：其他域的训练分支仍可能检索到被留出域。

因此必须区分：

- **Policy-level LODO**：留出域不进入价值头拟合与归一化，但源域分支和检索库可能共享。这是当前设计可以明确描述的范围。
- **Full-pipeline LODO**：每个外层 fold 重新建立不含留出域的 source 检索库，并重新采集工具分支、桥接输出和统计；留出域仅作验证查询，不作为检索 evidence。源域内另做必要的参数选择。

第一种可用于工程迭代，不能在论文中不加说明地当作第二种。两者都必须排除最终 target 的图像、问题参考答案和标签进入训练检索库。严格 target 隔离和严格 source 外层 fold 隔离是不同检查，前者通过不代表后者已经通过。

### 7.3 真实数据不等于真实域

PathVQA/VQA-RAD 的真实图像与真实自由文本答案适合验证执行链，但哈希 proxy group 不是医院。不同数据集可作为明确的数据来源偏移，但不能直接替代跨医院域，也要注意模态和任务差异带来的混杂。

跨医院 DG 需要可审计的医院、中心、设备、染色协议或明确采集来源元数据；按患者或独立图像组拆分，并报告每域任务和类别覆盖。未给出真实域元数据时，应标记 `real_domain_metadata=false`，把结果称为代理分组机制实验。

集中式收集各源域图像、参考答案和工具分支再训练策略，不是 federated learning。只有真正按客户端实施数据边界、聚合协议和相应评估后，才可以提出联邦训练或隐私方面的结论。

## 8. 多工具协同仍有哪些难点

history-conditioned value 比无历史的单工具均值更符合设计初衷，但当前策略仍然是贪心的条件边际选择。它不自动求解多步最优计划。

例如 A 单独没有改善，B 单独也没有改善，只有 A 提供定位后 B 的区域分析才有用。如果所有单步价值都不为正，贪心策略不会先调用 A，因而到不了 A→B 的有效状态。观察到 \(\Delta(B\mid A)>0\)，也不代表在线策略已经学会为了未来收益主动执行 A。

最低限度的组合实验应预声明少量有语义的顺序，例如“区域定位/分割→区域描述”或“检索→专科解释”，在足够独立源病例上收集真实条件分支，并比较完整轨迹。建议报告：

- 每种能力的实际执行与采用数。
- 至少两种不同能力被同一病例使用的比例。
- 第二工具在已有第一证据条件下的增量，而非两次单工具增量相加。
- A→B 与 B→A、去除历史特征、去除第一证据的对照。
- 仅用于离线诊断的少量两步 oracle 上界，与真实策略分开。

如果只有检索被调用，或者所有调用仍发生在第一答案 token 前，就不能声称已验证多能力生成中协作。此时可以得到有价值的负面机制结论，但应如实缩小论文主张。

## 9. 评估：先证明效用，再讨论幻觉和 DG

### 9.1 公平对照

保留 v0.8 的原能力实验作为历史对照，不让入口变化覆盖旧结果。v0.9 至少比较：

1. 通用医学 VLM，使用相同问题、回答格式、停止条件和 token 预算。
2. 每个任务匹配的单工具固定调用，以及最好的源域选择固定工具基线。
3. 合法且任务匹配的静态 evidence 基线，证据长度受控；不要只把明显无关的全证据方案当强基线。
4. v0.8 适配后的动态控制器，记录其决策 token 与非法动作成本。
5. 连续均值价值策略。
6. 连续价值加留域过估惩罚。
7. 相同调用位置、同 schema、近似同长度的 shuffled evidence 回放，隔离正确证据内容的作用。
8. 去历史特征、只允许初始调用、固定预算等机制消融。

错误模态或 wrong-capability 检查主要验证类型约束，不等价于有效专家相对随机观察的医学贡献。工具调用率不同的两方法也不能只比较原始准确率，应给出质量—成本曲线并标明操作预算。

### 9.2 F1 不是医学幻觉率

若 \(Q\) 目前使用 Token-F1，则训练目标与结果只能直接支持“连续词面收益”。EM/F1、答案长度和同义词处理仍可报告，但：

- `kidney` 变为 `renal` 可能提高词面重合，而没有得到正确完整诊断。
- 更长的解释可能增加错误医学断言，也可能仅因多余词降低 F1。
- `1-accuracy` 不能命名为开放式医学 hallucination rate。

讨论减轻幻觉必须额外评价 claim-level supported/unsupported/contradicted facts、医学概念与关系、遗漏，以及答案—解释一致性。尽量采用与生成/工具模型独立的评分流程，保留一组医生或适格标注者盲评；自动裁判需报告提示、模型版本和偏差审计。不能按目标标签挑选成功病例后当作总体收益证据。

### 9.3 统计与域结论

以独立患者或 RGB 图像组做配对统计，不能把同图多个问题、前缀或工具分支当独立样本放大显著性。报告成对质量差、置信区间、逐域表现、负增益案例、worst-domain 指标及其真实域定义。

只有一个 target proxy group 时，worst-domain 指标等于总体值，不构成最差医院表现。小 canary 用于验证链路和审计 trace，不应通过不断挑选样本或调整目标阈值来满足“至少救回一例”后宣布有效。

所有 ridge、残差分位数、成本权重、预算和 prompt 选择只使用源域训练/验证。若用同一留出源域反复选择许多设置，最终还需独立源域验证或外层嵌套评估，不能把选择后的乐观结果当未调参估计。最终 target 标签只用于冻结系统的评价。

## 10. 代码落点、接口和本轮交付边界

已独立定义的策略接口在 `merit_feddg/capability_value.py`：

```python
artifact = fit_value_policy(
    source_records,
    provenance,
    ridge=1.0,
    min_cases_per_domain=8,
    min_domains=2,
    residual_quantile=0.9,
    cost_weight=0.0,
)
scores = score_value_policy(artifact, candidates, robust=True)
```

`candidates` 只含预调用 action、features、state kind 与 history；评分不拟合任何目标统计。返回每个动作的 `supported`、`mean_gain`、`predicted_cost`、`mean_net_gain`、`penalty`、`score` 和 `reason`。NONE 不需要拟合，其基准新增价值是零。调用方必须同时检查 `supported=True` 和 `score>0`，不能只比较 unsupported 动作返回的零分。

本轮集成的职责划分为：

| 模块/入口 | 设计职责 |
| --- | --- |
| `capability_runtime.py` | 精确前缀恢复、工具执行、证据桥接和预算控制 |
| `capability_features.py` | 冻结的图像/文本与状态—descriptor 交互特征；不接触标签 |
| `capability_value_study.py` | 源域配对分支采集、价值拟合、冻结目标评估和可追溯报告 |
| `capability_value.py` | 共享连续回归、历史支持、LODO 残差与只读评分 |
| `llava_run --study value` | 新价值策略实验入口；`capabilities` 路径保留旧实验 |

该表说明集成职责，不代表单凭本文已完成这些模块的远端运行、模型下载或真实实验。具体可用选项以合并代码的帮助信息、配置和当前测试为准；不在本文固定测试数量或承诺性能提升。

缓存至少绑定源数据清单、图像像素指纹、主模型及工具权重、tokenizer、prompt、解码预算、特征提取、桥接代码、检索库以及后续生成协议。仅匹配文件名不足以复用旧分支。源标签可进入离线 gain 缓存，但不得成为在线 feature；目标 reference 应与推理 manifest 分离。

## 11. 面向 ICLR 的候选主张，而非先验承诺

更可信的主张形式是：

> 研究冻结医学 VLM 在生成中使用异构原生工具观察的问题，通过同前缀干预学习历史条件边际价值，并分析源域偏移如何影响工具选择与组合；在真实数据、真实工具和明确域定义下检验质量、事实性与成本。

它至少要回答三个可证伪问题：

1. 学到的是状态相关的专家互补性，还是固定工具偏好、答案长度或词面匹配？
2. 留域保守项是否在相同预算下改善未知真实来源上的效用，而不是把所有工具拒绝后退回 generalist？
3. 第二种能力是否提供了条件新增信息，且真实进入主模型输出，而不是仅存在于离线缓存分数中？

如果答案仍不明确，应保留负面结果并缩小结论。通用 registry、医学模型列表、ridge、悲观分位数和工具预算都是可借鉴组件，本身不足以保证创新性。论文价值需来自清楚的问题定义、与近邻方法的实质差异、机制验证、真实跨域证据和可复现实验，而不是方法名字或模块数量。
