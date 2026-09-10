# 完整证据传输与冻结模型答案核验

本轮实现针对停止快照的两个可复核问题：条目化改变了实际送达的专家/类别，造成格式与内容的混杂；局部移除分数无法验证医学声明。新增 `verified_packets` 协议，不修改已经发布的旧实验定义。**没有新的真实模型收益结果，也没有经过校准的正确率或医学风险保证。**

## 通用性检查与适用边界

- 传输只处理原生字典、列表、标量和明确编码的数组；没有疾病词表、问题模板列表、CE/OE 分支或目标答案。分类、分割、检测、检索、生成均有离线保真测试。
- 每个专家包保持完整，类别次序、分数、低分项、否定标志、自定义属性、单位、置信度与坐标元数据保留。共同字段提取一次；原始数组继续留在原始审计里，语义通道保留尺寸、编码及包内数组序号。引用不是模型可自动读取的数组。
- 布局键与原生字段冲突时回退原生布局；不同数值类型（如 True、1、1.0）、缺失和 null 不合并。原始输出不被修改。
- 答案核验接口接收原图、完整问题、原答案与完整候选答案。不把裸 No/Left 从问题中剥离，不使用参考答案。修改片段仅用于审计，**不是通用医学声明抽取器**。
- 已接入 BiomedCLIP 与 CONCH 的冻结图文评分后端；默认配置仅启用已准备的 BiomedCLIP。模型适用模态必须显式声明；未知模态弃权。配置中的模态范围是实验声明，不是经临床验证的覆盖范围。不能把支持多个接口等同于所有模态都有效。
- 不新增训练、温度拟合、源域收益拟合、目标标签选择或学习式权重。旧仓库的训练代码不在新协议调用路径上。

## 与前人工作的具体关系

| 工作 | 官方实现依据 | 本次采用和未采用部分 |
| --- | --- | --- |
| CheXzero，Nature Biomedical Engineering 2022 | [zero_shot.py: run_softmax_eval](https://github.com/rajpurkarlab/CheXzero/blob/main/zero_shot.py)，对成对文本做图像比较 | 借鉴成对图文评分；本次使用完整候选答案和已有 BiomedCLIP/CONCH，比较原图余弦分数，没有复制训练流程，也没有把 softmax 当成校准概率。 |
| VEP，ACL 2025 | [论文](https://aclanthology.org/2025.acl-long.205/)，专家视觉观察作为上下文 | 保留专家上下文路线。本轮主要新增完整包压缩与受控布局消融，不主张“专家转文本”是首创。 |
| ERASER，ACL 2020 | [metrics.py: score_classifications](https://github.com/jayded/eraserbenchmark/blob/master/rationale_benchmark/metrics.py)，分类分数与 rationale 移除/保留分数分开 | 把局部依赖性与任务正确性分开；新协议不使用局部移除分数作为最终接受条件。 |
| DnR，CVPR 2026 | [process/uq.py](https://github.com/EavnJeong/Draft-and-Refine-with-Visual-Experts/blob/d72196b1b498e982ecf6eb94e91724707c15ebf2/process/uq.py)，遮挡后回答变化、CLIP 文本相似度及权重 | 旧局部 Gate 保留为已有对照；本轮新协议比较外部模型对原图下两个完整答案的偏好，不是另一个视觉利用率公式。 |
| SAR，ACL 2024 | [get_tokenwise_importance.py](https://github.com/jinhaoduan/SAR/blob/main/src/get_tokenwise_importance.py)、[token_sar](https://github.com/jinhaoduan/SAR/blob/main/src/compute_uncertainty.py)，删除 token 的语义变化和重要性加权不确定性 | 本轮不冒称实现 SAR；审计中的文字 diff 不是 SAR 的语义重要性。该方法可作为后续独立不确定性基线。 |
| Semantic Entropy，Nature 2024 | [get_semantic_ids / cluster_assignment_entropy](https://github.com/jlko/semantic_uncertainty/blob/master/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py) | 本轮没有多次采样语义聚类；两种固定核验表述不是 Semantic Entropy，也不是统计独立样本。 |
| RouteLLM，ICLR 2025 | [train_matrix_factorization.py](https://github.com/lm-sys/RouteLLM/blob/main/routellm/routers/matrix_factorization/train_matrix_factorization.py)，BCE、backward、optimizer.step | 借鉴相对收益的研究问题，不使用其训练式路由器。本轮不会声称学到了 P(采用专家会胜过 Generalist)。 |

新组合是待验证研究假设。引用提供可核查的机制来源，不能替代本项目的消融证据。

## 方法

### 完整包和两种布局

`compact_evidence.compact_records` 仅将明确编码的稠密数组值移到原始审计引用，提取各条目共同的嵌套字段，保留所有变化字段。`compact_prompt` 提供行式和列式编码。两种布局使用相同提示说明。

`evidence_transport.pack_records` 增加 companion renderer：一个包只有同时满足两种布局的实际 tokenizer 预算才可加入。选择顺序固定，后续专家的预算准入也采用共享剩余预算。因此 `compact_rows` 与 `compact_all` 的内容保持一致。若完整包仍无法装入，整包有审计地拒绝；不会只留下最高分项。这个约束用于消除实验混杂，不是宣称一种新的最优压缩算法。

### 答案提交前核验

先得到 `y0 = Generalist` 和 `y1 = compact_all`。`compact_verified` 直接复用这两个输出，不重新生成逐条候选。

`answer_arbitration.render_pairs` 为同一问题和两个完整答案产生两种固定文字表述。外部冻结图文模型批量编码四条文本，并对同一原图计算余弦相似度。每种表述的分数为：

```
m_j = sim(I, render_j(q, y1)) - sim(I, render_j(q, y0))
```

- 两个 `m_j > 1e-6`：接受候选。
- 两个 `m_j < -1e-6`：拒绝候选，返回原答案。
- 平局、表述间分歧、没有适用核验器、输入无法完整编码或评分无效：弃权，返回原答案。
- 候选与原答案完全相同：不调用核验模型。
- 核验模型 ID 与生成器或任何已送达证据的模型 ID 相同：弃权，禁止显式同模型自证。ID 检查不意味着不同模型训练数据或错误一定独立；不要给同一检查点换名称绕过该约束。

`1e-6` 是固定数值平局容差，未用结果调整。模型设为 eval，参数 requires_grad=False，评分位于 inference_mode。输入过长时拒绝隐式截断：BiomedCLIP 检查原始 tokenizer 长度，CONCH 禁止 truncate。

两种表述一致只是较保守的排序启发式，不是正确性证据充分条件。特别是图文模型的否定、比较、数量和细粒度空间理解仍可能不足。本轮没有把语义未知直接判为高置信，也没有声称彻底解决了稳定但错误的专家输出。

### 审计与成本

输出保留完整候选及修改片段、四个原生分数、两个 margin、决策原因、评分调用数和耗时。弃权/拒绝返回原答案的精确 token IDs；候选证据转为 candidate_evidence，最终 evidence 为空。原始 tool trace 是候选构造审计，不能在最终提交被拒绝后继续当成证据已影响最终输出。

总耗时计入 baseline、candidate、外部核验，以及首次加载核验器的时间；另报 verifier_initialization_seconds。缓存复用时间仍不等于独立冷启动部署时间，应分别报告。没有每条证据重新调用生成器，也没有 NLI 自评和均值图验证。

## 五臂实验

| 实验臂 | 输入表示 | 最终答案核验 |
| --- | --- | --- |
| generalist | 无专家证据 | 无 |
| semantic_all | 旧整包语义表示 | 无 |
| compact_rows | 完整包、共享字段、行式 | 无 |
| compact_all | 同样内容、列式 | 无 |
| compact_verified | 复用 compact_all 的候选 | 外部冻结图文比较 |

关键对照：

- `compact_rows` 对 `compact_all`：相同内容，只改变布局。
- `compact_all` 对 `semantic_all`：内容送达与表示共同变化，必须同时报告送达率，不能声称是纯布局效果。
- `compact_verified` 对 `compact_all`：同一个候选答案，只改变提交决策。
- 所有方法都与 Generalist 比较；仅超过旧 hybrid 不构成成功。

运行同一完整清单，不分片、不读取 answer_type、不用 CE/OE 专用提示。因旧 native_claims 快照使用 ANCHOR 提示，本轮必须重新生成统一提示下的全部对照，不能直接拼接旧分数。

```bash
python -m merit_feddg.matched_evaluation \
  --protocol verified_packets \
  --config configs/matched_verified_packets.yaml \
  --manifest /absolute/path/to/full-inference-manifest.jsonl \
  --artifacts /absolute/path/to/prepared-artifacts \
  --output runs/matched-verified-packets
```

沿用已有本地模型；启动前检查核验器快照和 revision。无需下载新模型或训练。若本地 BiomedCLIP 尚未准备，程序明确报错，不会假装跑出了验证结果。

离线评分沿用 `scripts/evaluate_matched_vector.py`，支持新臂名称及答案核验审计。评分清单可包含离线 answer_type 与 references；这些不进入推理。重点报告净得分、harmful-accept rate、修改覆盖率、拒绝/弃权、原生 margin 排序区分能力、病例级成本。置信区间应按图像分组，不能把同图不同问题当成完全独立病例。

## 论文的可检验贡献与尚缺证据

Motivation：同一个高分分类条目对不同问题价值不同；不完整送达会把“证据压缩”变成“类别选择”；区域利用率和医学修改正确性不同。

Related work：区分专家提示、视觉依赖诊断、冻结不确定性估计、成对图文判断和需要训练的路由/校准。以上表格给出具体官方代码。

Method：完整原生包、共享字段及共同预算约束，配合候选复用的外部答案提交核验。可扩展不同图文模型，不假定任意专家 latent 已对齐；标量、几何引用与语义仍有明确区别。

Experiments：上述五臂，加上固定配置跨 generalist、专家缺失/误导证据、不同完整外部数据集的验证。当前 VQA-RAD 已参与开发诊断，不再称完全未见目标域。没有真实跨域结果前，不使用“已实现域泛化”的结论。

创新性判据：需要证明完整信息约束有实际作用，外部比较的排序确实优于局部利用率，并且在相同候选、相同内容和成本口径下减少损害。单纯增加一个 CLIP 相似度模块或组合已有方法不足以保证 ICLR 发表。本次代码使这些主张可验证，**不能预先宣称已经达到录用标准**。

尚缺：真实模型端到端实验、核验器否定与属性能力评测、外部完整数据集验证、token 预算与 verifier 选择的预注册消融。当前默认不启用空间注入或多次采样语义熵，这些必须独立验证后再合并。

本地验证：686 passed、17 skipped；相关实现文件通过 Ruff，补丁通过 `git diff --check`。这些是离线逻辑与回归测试，不代表真实医学收益。
