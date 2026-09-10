# 医学异构专家协作：结果诊断、相关工作与下一轮方案

## 1. 结论与范围

当前结果没有证明专家协作优于通用模型。最清楚的失败有三项：**接口没有完整传递专家语义；视觉对照 gate 把对照图变差误当作原图证据变好；CT/MRI 的有效专家覆盖远少于胸片。** 加权几乎没有改变结果，暂时不能把重要性权重作为得到实验支持的贡献。

分析依据为 GitHub 提交 `ac5ec9cdeb98a544194b70bfe359706ed2b84b49` 发布的 451 题完整结果。`SHA256SUMS` 五个文件全部校验通过。逐题记录包含原始分数、标签、坐标变换和 gate 的候选评分，但大体积 mask 内容被省略，因此不能据此检查病灶边界质量或完整重放模型。本报告区分日志事实、机制推断和待验证方案。

调研覆盖多智能体、大小模型协作、医学 agent、免训练可靠性及域泛化。新增核查了 12 个官方仓库的关键函数；另复核前期固定版本的 ProxyCLIP、PAI、VCD、DnR、BiomedParse 等代码。版本与源码入口见 [代码核查清单](research/verified-code-2026-09-10.json)；这些外部方法没有在本环境重新运行。优先引用顶会顶刊；未从可访问正式来源核实会议信息的项目不冒充顶会论文。

本方法的 training-free 指：只调用已有且冻结的预训练权重，不训练 bridge、router、adapter、soft prompt、校准器或重要性权重，也不做测试时梯度更新。预训练模型历史上训练过不构成冲突。当前全部题目仍用同一完整 manifest 和自由生成协议，不按 CE/OE 设置任何输出约束。

## 2. 这轮结果究竟说明什么

### 2.1 总体指标均未超过基线

| 实验组 | strict mixed | 内容诊断分数 | 通用 Token-F1 | 平均引擎时间/题 |
|---|---:|---:|---:|---:|
| 通用模型 | 29.45% | 43.64% | 7.52% | 0.53 秒 |
| 空间证据等权 | 28.85% | 41.93% | 7.36% | 0.84 秒 |
| 空间证据相关性加权 | 28.85% | 41.93% | 7.36% | 0.67 秒 |
| 空间证据＋旧 gate | 29.23% | 43.20% | 7.51% | 1.66 秒 |

来源：[原始汇总](results/vqarad_spatial_full_2026-09-10/evaluation-summary.json)。时间受到顺序运行、共享专家缓存及热启动影响，不能作为冷启动端到端加速比较。gate 组的额外生成和验证成本仍然明确存在。

等权和加权只在 3 题上文字不同，最终分数完全相同。加权组相对基线的内容诊断计数为 1 题提高、10 题降低；gate 组为 1 题提高、3 题降低。这里的“提高”首先是自动评分事件，不是经过人工确认的医学纠错。

`strict mixed` 混合了闭式题首词解析与开放题 token recall，不是纯临床正确率。全组 exact match 为零也不能单独说明全错：完整句子与短参考答案很难字面完全一致。应保留固定评测协议，同时公开逐题内容审查，不能为了分数改变模型输出语法。

### 2.2 一个确定的 gate 漏洞

旧 gate 对候选答案计算原图平均 log probability 与均值颜色对照图平均 log probability 的差，随后比较加入专家前后的增量：

\[
G = [\ell_{image}(y_{expert})-\ell_{control}(y_{expert})]
  - [\ell_{image}(y_{base})-\ell_{control}(y_{base})].
\]

`G > 0` 不意味着候选在原图上更受支持。例如第 0003 题，正确的 right 被改成 left：候选原图分数降低约 0.00379，对照图分数降低约 0.01597，于是差值仍增加 0.01217，旧 gate 就放行了。这是评分定义的漏洞，不是阈值稍微调一下就能解释的问题。

403 次可用证据决策中，340 次候选 token 完全相同，51 次差值不为正，12 次放行。**12 次放行中 9 次原图分数下降。** 完整明细和可复现脚本见 [gate-diagnosis.json](results/vqarad_spatial_full_2026-09-10/gate-diagnosis.json)、[分析脚本](../scripts/analyze_spatial_bundle.py)。

| 题号 | 参考答案与变化 | 原图增量 | 视觉对照增量 | 诊断 |
|---|---|---:|---:|---|
| 0003 | right → left；参考 right | −0.00379 | +0.01217 | 正确改错 |
| 0084 | 无气胸 → 有气胸；参考 no | −0.16490 | +0.02149 | 正确改错 |
| 0260 | 无膈下游离气体 → 有；参考 no | −0.01227 | +0.01002 | 正确改错 |
| 0033 | 基线与候选都未匹配参考 no；候选生成更多解释 | +0.28414 | +0.07471 | 自动内容评分“提高”不等于纠错 |
| 0422 | normal fat → normal fat，少一个冠词；参考 no | +0.08525 | +0.04654 | 原有错误保留，只改措辞 |
| 0428 | 左侧更清晰，增加 “in the image” | +0.01968 | +0.02586 | 正确答案的措辞变化 |

第 0033 题的“提高”尤其不能计为真实贡献：新答案仍声称安全，且添加了“X-ray 不涉及 ionizing radiation”的说法。这里按参考答案和生成内容即可发现自动指标与实际回答不一致；不据此修改运行时规则去针对该题。

如果只给已记录的 12 次决策补上“原图分数必须提升”，会挡掉其中 9 次，但留下的 3 次仍没有清楚的新纠错。**这只是回顾性诊断，不是新方案的实验结果。** 真正重跑时，前面拒绝的证据会改变后续状态，不能直接用旧日志推算新准确率。

### 2.3 统一接口丢失了什么

当前 `SpatialEvidenceBridge` 根据 mask/CAM 对通用模型自己的视觉 patch 特征做区域聚合和回写。类别字符串用于审计与词面相关性，原始分类分数没有作为语义输入传给语言模型。它更接近空间特征重分配，而不是完整的异构证据理解。

第 0084 题，胸片分类器的 Pneumothorax 原始分数约为 0.00416。这个未校准低分不能被解释成“排除气胸”，但至少说明专家输出包含了具体类别和分数信息。旧空间通道没有把这层含义交给主模型。不能仅凭有一张正激活图，就把它理解成病灶存在的证据。

这也解释了为什么不能只增强空间注入强度：增强一个不完整、甚至任务不相关的空间信号，可能更强地干扰本来正确的答案。加权公式目前主要是标签与问题词面重叠，并不是无训练可靠性权重，也没有处理不同专家分数的不同尺度。

### 2.4 专家覆盖不均衡

下表按运行时预测的模态统计，是路由结果，不是真实模态标注准确率；仅用于事后诊断，没有重新划分数据。

| 预测模态 | 题数 | 加权组实际采纳过证据 | 相对基线文字改变 |
|---|---:|---:|---:|
| 胸片 | 174 | 174 | 45 |
| CT | 172 | 51 | 2 |
| MRI | 105 | 4 | 0 |

有效证据几乎集中在胸片。BiomedParse 每题都有调用记录，并不表示每题执行了模型并产生有效 mask；不支持的解剖或序列可在模型推理前被拒绝。“625 次工具尝试”不能写成 625 次有效专家推理。

新增专家应先补实际能力缺口，而非增加注册名称。当前二维数据优先处理 CT/MRI 的输入适配、序列信息与实际可输出的类别/空间证据；不能为声称覆盖全面而接入需要三维体数据的工具，或凭空构造缺失序列、标尺、ROI。接口支持类别、分割、检测或生成，不等于这轮实验实际验证了每一类。

## 3. 相关论文与代码：哪些与创新真正重叠

### 3.1 多智能体、大小模型与医学协作

| 工作与来源 | 官方代码的具体机制 | 与本项目的关系及限制 |
|---|---|---|
| [MDAgents，NeurIPS 2024 Oral](https://arxiv.org/abs/2404.15155) | [utils.py](https://github.com/mitmedialab/MDAgents/blob/3adbd760ca809b4e7b0c1085d68314b6e7d91e1b/utils.py)：`Group.interact` 收集角色医生的文字调查，由组长汇总 | 主要是 LLM 角色与协作结构；不能把角色异构等同于分类器、分割器的原生输出异构。可借鉴按需求协作，不能直接复制其选择题格式 |
| [MedRAX，ICML 2025；会议信息见作者仓库](https://github.com/bowang-lab/MedRAX/tree/dae30e2f136ef0b2a40885a4c335386e9ffad052) | [agent.py](https://github.com/bowang-lab/MedRAX/blob/dae30e2f136ef0b2a40885a4c335386e9ffad052/medrax/agent/agent.py)：执行工具并将 `str(result)` 放入 `ToolMessage` 交回模型 | 已经是免额外训练的医学工具协作，并非所有医学 agent 都只连接一种小模型。主要集中胸片；统一工具包装本身不是我们的新贡献 |
| [HuggingGPT](https://arxiv.org/abs/2303.17580) | [awesome_chat.py](https://github.com/microsoft/JARVIS/blob/7624cf388b47334ff8a0868e7d862dde18cfda86/hugginggpt/server/awesome_chat.py)：`parse_task`、`choose_model`、`model_inference`、`response_results` | 已有语言模型协调不同任务模型，再汇总结构化结果的路径。不同于主模型内部的无训练隐藏向量对齐 |
| [Multiagent Debate](https://arxiv.org/abs/2305.14325) | [gsm/gen_gsm.py](https://github.com/composable-models/llm_multiagent_debate/blob/9846749350eb917ae5bfaaff4c645fc705b8d3af/gsm/gen_gsm.py)：把其他 agent 答案拼入后续消息 | 讨论带来多个候选，但相同错误可能被相互强化。多数赞同不是独立证据可靠性；数学数据专用输出模板不照搬 |
| [Contrastive Decoding，ACL 2023](https://aclanthology.org/2023.acl-long.687/) | [run_generation.py](https://github.com/XiangLi1999/ContrastiveDecoding/blob/170e9142e92159c1237d731e240f5eb14aabf428/text-generation/run_generation.py)：加载 student LM，联合 teacher/student 生成 | 解决同类语言生成分布协作；分类分数、mask 与 token 分布没有共同词表，不能直接套用减 logits |
| [RouteLLM](https://arxiv.org/abs/2406.18665) | [train_matrix_factorization.py](https://github.com/lm-sys/RouteLLM/blob/0b64fdafe049e596a3f5657c219329f24af24198/routellm/routers/matrix_factorization/train_matrix_factorization.py)：偏好数据、loss、backward、optimizer.step | 其 MF router 明确训练；不能因为部署时冻结就算本项目要求的 training-free。可比较成本意识，不能移植训练模块 |

因此，论文不能笼统宣称“首次用大模型协调异构小模型”。更有针对性的目标是：**不同原生任务输出在同一接口中保留必要语义和空间对应，并用证据级拒绝机制抑制负迁移。** 是否构成贡献仍取决于相对上述系统的严格实验。

### 3.2 免训练判断不等于判断正确

| 方法与来源 | 已核实实现 | 能测什么；不能测什么 |
|---|---|---|
| [Semantic Entropy，Nature 2024](https://www.nature.com/articles/s41586-024-07421-0) | [semantic_entropy.py](https://github.com/jlko/semantic_uncertainty/blob/a8d9aa8cecd5f3bec09b19ae38ab13552e0846f4/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py)：双向蕴含分组，再计算簇分布熵 | 区分措辞变化与意义变化；不能识别所有“稳定地答错”。`strict_entailment` 与默认分组条件不同，复现时必须明确 |
| [SelfCheckGPT，EMNLP 2023](https://arxiv.org/abs/2303.08896) | [modeling_selfcheck.py](https://github.com/potsawee/selfcheckgpt/blob/19b492a2a380931bf1ed0ca94a9565c9aa7b03e1/selfcheckgpt/modeling_selfcheck.py)：NLI 变体比较陈述与多次采样文本的矛盾分数 | 可做自由回答一致性诊断；采样均重复同一错误时仍会自洽。现成 NLI 权重可冻结使用，但不是无历史训练的模型 |
| [VCD，CVPR 2024](https://github.com/DAMO-NLP-SG/VCD) | [vcd_sample.py](https://github.com/DAMO-NLP-SG/VCD/blob/d6568ff81b8fd306a49e630df44f2db5c2300191/vcd_utils/vcd_sample.py)：原图与噪声图 logits 对比，并约束候选合理性 | 图像依赖性有用，但视觉差值不能当作临床正确率；我们旧 gate 也不是原论文复现 |
| [DnR，CVPR 2026，作者项目](https://github.com/EavnJeong/Draft-and-Refine-with-Visual-Experts) | [process_uq.py](https://github.com/EavnJeong/Draft-and-Refine-with-Visual-Experts/blob/d72196b1b498e982ecf6eb94e91724707c15ebf2/process_uq.py)：对专家区域干预后的回答，用冻结 CLIP 文本表示计算 fidelity/faithfulness 等 | 与“专家区域＋无训练可靠性＋回答修正”最接近，必须认真比较。源码含数据集选择项映射，不能把这部分搬成我们的题型专用策略 |
| [ProxyCLIP，ECCV 2024](https://github.com/mc-lan/ProxyCLIP) | [transformer.py](https://github.com/mc-lan/ProxyCLIP/blob/a0e38d1989c990651fda0d8fbf8aad41bde955b5/open_clip/transformer.py)：外部特征生成空间关联，作用于已有视觉表示 | 支持无训练空间关系迁移；没有证明任意输出分数或任意隐藏空间可直接对齐 |
| [PAI，ECCV 2024](https://github.com/LALBJ/PAI) | [attention.py](https://github.com/LALBJ/PAI/blob/9bbd8bd57a0b0923f996197e4bd3e02cc10b8d58/attention.py)：修改已有注意力中图像位置的权重 | 可研究图像利用不足，不能把空间平滑等同于已实现该方法 |

不能把这些指标直接归一化后求一个“可靠性加权平均”。未校准分类 sigmoid、CLIP 相似度、mask 稳定性和语言平均 log probability 含义不同。一个指标很高不应抵消另一个指标暴露的明确不适用或输入错误。

更合理的是按职责分层：先检查输入/输出契约及是否送达，再判断任务相关性，随后看扰动稳定性或语义冗余，最后判断加入证据是否改变了图像支持。每项都保留 unknown 状态和计算成本，不输出伪装成概率的综合可靠性分数。

### 3.3 域泛化与测试时适配必须分开

| 工作与来源 | 代码依据 | 对本项目的决定 |
|---|---|---|
| [DomainBed / In Search of Lost Domain Generalization，ICLR 2021](https://arxiv.org/abs/2007.01434) | [model_selection.py](https://github.com/facebookresearch/DomainBed/blob/b93c22a1cfc3b2428398272c1a116c8de1f4139e/domainbed/model_selection.py) 显式区分 IID、leave-one-out 与 oracle 选择 | 借鉴评估原则，不采用训练算法。依据反复查看的测试结果挑 gate，不能再把同一测试集当成独立泛化证明 |
| [Tent，ICLR 2021](https://github.com/DequanWang/tent) | [tent.py](https://github.com/DequanWang/tent/blob/e9e926a668d85244c66a6d5c006efbd2b82e83e8/tent.py) 的 `forward_and_adapt` 计算梯度并更新参数 | 不采用；冻结其他层不改变它是测试时训练的事实 |
| [TPT，NeurIPS 2022](https://github.com/azshue/TPT) | [tpt_classification.py](https://github.com/azshue/TPT/blob/63ecbace79694205d7884e63fdc3137a200f0b0e/tpt_classification.py) 优化 prompt、反向传播 | 不采用优化部分。可以借鉴低置信样本与增广一致性的研究问题 |
| [TDA，CVPR 2024](https://arxiv.org/abs/2403.18293) | [tda_runner.py](https://github.com/kdiAAA/TDA/blob/e697fb0c8078cdeff93daa56bcf8860702542069/tda_runner.py) 在 no_grad 下动态维护正负伪标签缓存 | 无梯度但有跨测试样本状态，且依赖共有类别空间。README 提供数据集相关配置，不能视为所有域无需适配的证明；当前不引入 |
| [Conformal Prediction Under Covariate Shift](https://arxiv.org/abs/1904.06019) | 论文要求相应统计条件，涉及校准与分布比；本轮未核查其官方实现 | 不凭空宣称拒绝规则有分布无关风险保证。禁止校准数据的约束下不能直接借用覆盖率承诺 |

固定冻结模型在新域上测试、在线更新伪标签缓存、更新神经网络参数，是三种不同实验设置。“training-free”不足以把它们混成同一类。当前只有一套经过多轮分析的 451 题结果，尚不足以证明医学域泛化。

### 3.4 医学专家的能力边界

[BiomedParse](https://github.com/microsoft/BiomedParse/blob/db5c10782dab2377db4f68bbc03f71c54572e51b/inference_utils/inference.py) 返回由指定对象提示条件化的分割输出。掩码非空不等于对象已确认存在，更不等于该对象与当前问题相关。该领域基础模型可扩展输入覆盖，但不能消除医学任务边界。

[CONCH](https://github.com/mahmoodlab/CONCH) 的现成图文空间适合其病理领域任务；不能把它的向量直接点乘 LLaVA 的 hidden state 就声称得到语义对齐。[MedSAM](https://github.com/bowang-lab/MedSAM/blob/d71e8a1a99ad751840a22a7fa3ecfb4166fb1488/MedSAM_Inference.py) 等需要明确提示的分割工具，缺少 ROI 时不能用目标标注或假设 ROI 补足。本轮不通过仓促堆叠未验证权重来扩充专家。

## 4. 本次代码方案与取舍

### 4.1 双通道接口，明确其信息损失

新增 `semantic` 通道：保留所有原生类别与分数、score semantics、scope、坐标变换、未知状态和来源，不再按 top-k 丢掉低排名类别，也不生成一个可能改写否定关系的专家总结。原始输出留在证据记录里。

语义字段通过已有 tokenizer 和冻结 token embeddings 进入主模型；可选 `semantic_spatial` 同时把原始空间证据交给既有零参数空间算子。无空间图的分类输出也能通过语义通道到达，不再仅因缺少 CAM 而排除该类别专家。

大体积 mask 不以压缩字符串冒充可读知识：语义记录中用可审计引用、尺寸及确定性几何统计代替，空间通道消费原始几何。几何统计保留原生网格坐标系下的加权质心、均值与最大值，连同原始裁剪变换一起提供；不把质心冒充边界，也不把 mask 值解释成对象存在概率。网格下采样、区域聚合、原生 adapter 的提示上限仍会损失信息，因此不能声称“所有原生张量已无损传入”。超过上下文预算时整体省略记录并报告原因，不截断半个陈述；这种原子打包也可能过于保守，应观察漏送率。

“向量化”必须说清是哪一种：现成 tokenizer 的语义 token、现成视觉编码器的 patch、外部模型的隐藏向量，不是同一个概念。相同维度不代表同一坐标系。**此版不是任意隐藏向量的统一对齐算法。** 如果论文坚持纯非文本接口，语义通道应当是诊断与对照组，不能换个名称就冒充原目标已实现。

### 4.2 多维 gate

新 gate 按以下次序执行，不拟合任何阈值或权重：

1. 保留原有能力、输入和输出验证；确认新语义记录可以送达，且不挤掉已有语义/空间记录。
2. 冻结通用模型判断原生观测是否能帮助回答当前问题。它判断实际能力，不仅判断是否同属胸片或 CT；未知时拒绝。
3. 生成同预算的基线与加入证据的自由候选。完全相同则停止。
4. 参照 Semantic Entropy 的代码思想，用双向蕴含检查纯措辞重复；确认等价时不改变答案。矛盾不是直接拒绝条件，因为真正纠错也会与旧答案矛盾。未知继续交给视觉检查，不能标成可靠性通过。
5. 用不含专家信息的验证上下文，分别记录原图增量与视觉对照增量；保守候选方案要求两项都为正。阈值 `1e-6` 仅作数值平局容差。
6. 单独记录专家扰动稳定性、校准状态、额外推理与候选 token 成本。当前 XRV 可测轻微 gamma 扰动；未支持的专家明确 unknown。稳定性这一版是审计维度，尚未用未经验证的医学阈值做硬拒绝。

相关性与蕴含的有限选项只用于内部控制判断，医学回答始终自由生成，并未按 CE/OE 改动输出约束。双候选的蕴含检查也不是完整 Semantic Entropy：没有多次随机采样、语义簇分布或熵估计，不能这样命名。

保守原图增量检查有明显代价：验证者还是同一个通用模型，可能偏爱自己的错误；序列平均分受长度和措辞影响；均值颜色对照图仍是分布外输入。因此它是有可解释动机的实验条件，而不是已证实的最优 gate，更不是可靠性证书。

### 4.3 必须有的对照

新增协议 `semantic_spatial`，五组都跑同一个完整 manifest：

| 组名 | 作用 |
|---|---|
| generalist | 统一自由生成基线 |
| semantic_all | 仅完整标量语义，检验旧接口是否丢信息 |
| hybrid_all | 相同语义＋空间算子，检验空间干预的增量 |
| hybrid_contrast | 同一混合接口＋旧视觉对照 gate |
| hybrid_gate | 同一混合接口＋相关性、语义去冗余、原图/对照双检查，另记录稳定性 |

旧 `spatial` 协议保留，便于比较纯空间方法。新协议允许此前因无空间图而排除的输出，故其专家集合可能不同；不能把新旧整体差异全部归因于 gate。新协议内部五组的候选专家来自同一配置，差异与实际呈现集合在日志中公开。

## 5. 投稿前应该怎样收敛

第一优先是跑清楚“信息有没有用”。如果 `semantic_all` 仍弱于基线，就不要继续增强空间权重；先看专家本身是否提供任务所需信息，以及主模型是否读懂未校准分数。若语义有效、加空间反而下降，空间算子就是主要问题，不能为保留论文叙事而强行启用。

第二优先是证明选择能力。除总分，报告正确改错、错误改对、原有错误保留、措辞变化、拒绝率、有效证据覆盖和推理成本。只靠拒绝几乎所有证据恢复基线，没有证明协作价值。完整报告每项 gate 的拒绝原因，不能把 unknown 合并为通过。

第三优先才是增加复杂可靠性指标。完整语义熵、多视图回答一致性、专家之间的蕴含/矛盾、DnR 式局部视觉干预，均值得作为后续可消融模块。但现在不把四五个昂贵指标一次性加上，既无法归因，也可能掩盖主模型自信但错误的问题。稳定性应比较同一专家、同一属性、几何可比的扰动，不能把多个相关专家的多数意见当独立验证。

域泛化结论需要锁定规则后，在未参与这些选择的自然外部来源上检验。不重新划分当前 451 题；该集合继续完整运行，并披露已经被用于开发诊断。若投稿前没有独立域结果，就收窄为“冻结模型下的异构证据协作”，不把泛化保证写进贡献。

两条贡献的审慎表述可为：一是保留任务语义、分数语义、空间对应与未知状态的异构证据接口；二是针对证据加入造成的负迁移，提供可拒绝、可审计的无训练选择机制。**目前是研究假设与候选实现，尚不是已成立的效果结论。** 统一 JSON/token 表示、视觉差值、蕴含检查都已有先例；新意需要来自具体信息保持机制、异构设置、可靠性分析与扎实的组合消融。

## 6. 复现与当前验证边界

回顾性分析：

```bash
python scripts/analyze_spatial_bundle.py \
  --bundle docs/results/vqarad_spatial_full_2026-09-10 \
  --output /tmp/spatial-gate-diagnosis.json
```

新 GPU 实验入口（`MANIFEST` 使用已有完整 manifest，`ARTIFACTS` 使用现有权重目录）：

```bash
python -m merit_feddg.matched_evaluation \
  --manifest "$MANIFEST" \
  --artifacts "$ARTIFACTS" \
  --config configs/matched_semantic_spatial.yaml \
  --protocol semantic_spatial \
  --output runs/matched-semantic-spatial
```

本地验证覆盖原生标量不丢失、数组引用不冒充像素、完整记录预算、实际语义与空间联合传入、相关性先于候选生成、拒绝不改变已提交前缀、原图变差的差值假阳性，以及五组协议一致性。真实医学模型没有在本环境重跑；新分数、速度、显存和跨域性能必须等待服务器结果。
