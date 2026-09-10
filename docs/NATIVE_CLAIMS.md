# Training-free 原生条目协作：实现与实验协议

本次改动针对三项已观察到的问题：XRV 多标签包整体超预算、明显无效的自由文本进入上下文、旧 Gate 把视觉差分或同模型 NLI 当作纠错依据。新增 `native_claims` 协议用于检验改进，尚无新一轮真实模型收益结果。

## 研究依据及采用范围

| 前人工作 | 已核查的官方实现 | 本项目采用范围与区别 |
| --- | --- | --- |
| VEP，ACL 2025 | [论文](https://aclanthology.org/2025.acl-long.205/)；原生视觉专家观察构成上下文 | 保留专家的原生结构，再按原生条目传输。没有将已有专家提示包装成本项目首创。 |
| ProxyCLIP，ECCV 2024 | [custom_attn](https://github.com/mc-lan/ProxyCLIP/blob/a0e38d1989c990651fda0d8fbf8aad41bde955b5/open_clip/transformer.py)，用外部关系作用于本模型 value | 沿用仓库的零参数空间算子，用已有 VLM patch 特征与原生区域交互；不是复现 ProxyCLIP，也不是任意专家 latent 向量直接对齐。 |
| Draft and Refine with Visual Experts，CVPR 2026 | [process/uq.py](https://github.com/EavnJeong/Draft-and-Refine-with-Visual-Experts/blob/d72196b1b498e982ecf6eb94e91724707c15ebf2/process/uq.py)，相关/非相关视觉扰动与专家选择 | 借鉴局部视觉利用度检验。这里比较同一对答案在两张局部扰动图上的序列分数；不使用该工作的训练式快速选择器，不声称复现其分数。 |
| ERASER，ACL 2020 | [score_classifications](https://github.com/jayded/eraserbenchmark/blob/36467f1662812cbd4fbdd66879946cd7338e08ec/rationale_benchmark/metrics.py)，删除/保留依据后的预测变化 | 将“支持区域移除是否影响候选答案”作为 faithfulness 诊断；不把依赖性等同医学正确性。 |
| CRITIC，ICLR 2024 | [critic.py](https://github.com/microsoft/ProphetNet/blob/5cf70eb41cdaa1d8faa3e1265d95ee5792d49a53/CRITIC/src/qa/critic.py)，实际外部工具反馈后修订 | 本轮移除新协议中的自报相关性与双向 NLI 调用，但图像评分仍由冻结通用模型完成，尚未实现独立医学验证器。 |
| MedRAX，ICML 2025 | [论文](https://proceedings.mlr.press/v267/fallahpour25a.html)、[代码](https://github.com/MedRAX/MedRAX) | 异构医学工具协作已有先例；本项目应验证的差异是条目级传输、空间干预审计和有限验证预算，不能只主张“多个医学专家”。 |
| DomainBed，ICLR 2021；WILDS，ICML 2021 | [DomainBed](https://github.com/facebookresearch/DomainBed)、[WILDS](https://github.com/p-lambda/wilds) | 借鉴固定选择协议与跨域评估。当前没有新 DG 算法或跨医院有效性的证据；不做 source/target 阈值拟合。 |

以上是机制启发与组合实现，不是直接复制算法。本轮没有引入训练、源域可靠性拟合或目标域调参，也不使用 RECOMP 的训练式压缩器、GroupDRO 优化器或拟合式风险校准。

## 方法及代码

1. `claim_evidence.native_entries`：将 `findings/catalog/structures/detections/boxes/objects` 的唯一原生列表按条目拆开，记录父 ID、字段、索引。每条保留原始分数、分数语义、坐标变换和空间数组。父包与原始 summary 留在 `native_evidence` 审计中；避免把整包 summary 复制成每个条目的声明。自由文本完整保留，不声称已经完成通用医学声明抽取。
2. `attribute_check`：在 VLM 调用前排除空文本、`cxr_description` 等已知异常占位符，以及明确空间属性问题中的裸 `Yes/No`。存在性问题的 `No` 仍是有效的、但可能错误的观察。该检查只读取问题文本，不读取 CE/OE 元数据，不限制最终答案词表。未知属性、未知语言不会被猜测性拒绝。
3. 原有实际 tokenizer 预算检查按新条目执行。预算不足会逐条记录遗漏原因，原始审计仍保留全部结果；不再要求整个 XRV 列表同时装入。大条目仍可能装不下，未实现最优预算分配。语义 prompt 仅把可见条目交给空间通道，防止隐藏数组绕过文本预算。
4. `claim_gate.assess_claim_support`：对有有效区域的单条证据，生成“已接受证据状态”的答案 `y0` 和“加入此条”的答案 `y1`。两个候选都保留相同原图、问题与已提交 token 前缀。候选是探测，不提交为正式回答。
5. 还原方形补边坐标后，对预测支持区域做局部高斯模糊。另将同一像素权重图平移半幅图像，在三个固定方向中选与原区域重叠最少者作为对照。两个权重图在原图像素坐标上具有相同质量分布；不替换整幅图为均值色。平移后的区域可能仍包含病灶，因此它不是已验证的临床反事实。

记 `L(y,I)` 为固定问题/前缀、无专家文本时的每 token 平均对数似然，定义：

```
margin(I) = L(y1, I) - L(y0, I)
delta = margin(control-removal image) - margin(support-removal image)
```

只有 `delta > 1e-6` 才接受这一可测的空间条目。`1e-6` 是固定数值平局容差，未拟合。没有原图似然必须提高的硬条件，没有 NLI 前置否决。此 delta 是依赖性启发式，不是证据正确率、因果收益概率或风险上界。

6. 每例最多两个空间检查。按既有专家顺序和原生条目顺序分配，不宣称学习到最优路由。预算耗尽后可保留条目的语义，但写入 `spatial_transport_disabled`，空间算子明确忽略其数组。有效区域缺失的语义证据以 `semantic_only_unverified` 保留；没有不同平移对照的区域也只能退回语义。接受、拒绝、未知必须分别报告。
7. 已接受条目不能被后续候选从上下文或空间包中挤出。最终仍由冻结通用模型回答。空间权重继续使用既有 `equal` 模式：在专家和重复几何之间分配算子权重，不将不同模型的 sigmoid/相似度混合为正确率，也不以原始分类分数作为跨专家可靠性权重。

## 六组完整匹配实验

| 实验臂 | 新条目打包 | 低成本过滤 | 空间通道 | 局部 Gate |
| --- | --- | --- | --- | --- |
| generalist | — | — | — | — |
| semantic_all | 否，旧整包 | 否 | 否 | 否 |
| entry_all | 是 | 否 | 否 | 否 |
| entry_filtered | 是 | 是 | 否 | 否 |
| hybrid_all | 是 | 是 | 是 | 否 |
| hybrid_gate | 是 | 是 | 是 | 是，每例最多两次 |

各臂使用同一完整清单、相同原图、统一 unrestricted 64-token 简答提示和冻结权重；不按 CE/OE 切换输出契约。新协议在读取清单时移除 answer_type，并拒绝 `anchor-ce-v1` 配置和分片执行。上游已有的旧协议与分片实现保留供已有实验复现，不作为新方法入口。不要直接使用 `configs/matched_semantic_spatial.yaml` 启动本轮实验。

```bash
python -m merit_feddg.matched_evaluation \
  --protocol native_claims \
  --config configs/matched_native_claims.yaml \
  --manifest /absolute/path/to/existing-full-manifest.jsonl \
  --artifacts /absolute/path/to/existing-model-artifacts \
  --output runs/matched-native-claims
```

沿用现有依赖与本地权重。运行 identity 包含实现和配置；旧缓存不会被当作新实验结果。无需训练步骤。离线评估继续使用现有命令：

```bash
python scripts/evaluate_matched_vector.py \
  --run runs/matched-native-claims/IDENTITY \
  --manifest /absolute/path/to/existing-evaluation-manifest.jsonl \
  --references /absolute/path/to/references.json \
  --anchor-root /absolute/path/to/ANCHOR \
  --output runs/matched-native-claims/IDENTITY/evaluation-summary.json
```

带答案类型的评估清单只供离线评分；顺序、ID、图像、问题应与生成清单一致。新增输出包含逐条 gate 原因、未验证的语义接受数、局部移除分数、答案变化覆盖率，以及变化子集上的得分缺口和损害比例。得分缺口包含 OE 的分数损失，不是临床错误率。修正了多条证据把“出现专家的病例数”重复计数的问题。

## 验证、研究目标与尚未解决的问题

本地完整回归：665 passed、17 skipped；改动实现文件通过 Ruff，补丁通过 `git diff --check`。新增离线测试覆盖原始输出不可变、整包失败时条目仍可传输、属性错配、区域绑定、矩形图像补边还原、同质量平移对照、前缀不变、无 NLI、预算耗尽无空间旁路、统一六臂与 answer_type 隔离。真实模型耗时及收益必须跑完后报告，不能由单元测试推断。

实验先验证三个问题：

- `entry_all - semantic_all`：解决整包丢弃能否增加有效证据覆盖并改善答案？
- `entry_filtered - entry_all`：廉价接口检查减少多少明显无效观察和后续损害？
- `hybrid_gate - hybrid_all`：局部依赖检查能否改善空间干预的收益/损害关系，代价是多少？同时对比 `entry_filtered`，不能只超过较弱的 hybrid。

仍未解决：稳定但错误的 CheXagent 文本、完整医学属性绑定、通用器官 mask 与病变的区分、通用模型评分的相关错误、区域模糊的 OOD 影响、固定顺序造成的验证预算偏置。这里没有把这些问题伪装成已解决。尤其语义回退是未验证观察，不应在论文里计为 Gate 验证通过。

域泛化只能在保持相同配置、完整外部数据集与无目标标签选择的前提下验证。单一 VQA-RAD 的改进或人工模糊不能证明跨医院泛化。可先做跨 generalist/专家缺失的固定配置复用；是否进一步引入跨院数据取决于数据可用性。本轮不拆数据、不训练、不为了故事增加未经验证的 DG 模块。

论文贡献的候选表述是：**在冻结模型协作中，使原生证据单位、实际传输预算和局部干预验证保持一致，并显式区分可测空间支持与未验证语义回退。** 它是否具有足够研究价值，取决于上述消融和失败边界；目前不能声称已达到 ICLR 录用标准。
