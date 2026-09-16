# MedRAX 多查询证据复用：文献排重与真实任务结构审计

2026-09-16；审计身份 `medrax-reuse-structure-20260916-v1`。
独立分支 `research/medrax-reuse-audit-20260916`，基于 `21003dc`。

## 1. Evidence update

上一轮可逆图表示试验已触发停止条件。本轮没有延长该试验，转而核查预留的“同病例多查询证据复用”。完成两项工作：一手文献近邻排重；官方 ChestAgentBench 元数据与 MedRAX 工具实现审计。没有推理调用、没有训练、没有读取 gold answer/explanation 内容用于分析或推断需求。

数据来源为[官方 ChestAgentBench](https://huggingface.co/datasets/wanglab/chest-agent-bench)，revision `921e60440927f9893228d843c0206b755744252a`。仅下载约 4.9 MB 的 `metadata.jsonl`，未下载约 823 MB 的图像压缩包。文件 SHA256：`45e429e3a3b8e4dbfa064cbdaa3f2960bee90cad9e0f25501e38da67597bc2a8`。

| 元数据统计 | 实测值 |
|---|---:|
| 问题数 / 唯一问题 ID | 2500 / 2500 |
| 当前发布元数据中唯一病例 ID | 609 |
| 每病例 2 / 3 / 4 / 5 个问题的病例数 | 1 / 147 / 248 / 213 |
| 只有一组有序图像引用的病例 | 431 |
| 有多组有序图像引用的病例 | 178 |
| 图像引用总出现次数 | 4629 |
| 唯一图像引用；病例内去重后总数 | 1346；1346 |
| 不同有序图像引用组合 | 865 |
| 病例＋图像组合＋完全相同问题的重复数 | 0 |

609 是此次固定版本 metadata 的计数；不将其与论文中构建语料的病例总数混为一谈。图像身份仅按引用字符串统计，未验证像素相同。原始问题只用于精确哈希，不做语义相似度判断。

若**假设**每个问题都对每张引用图像调用一次固定、图像条件的工具、同一病例内无限缓存且无失效，则 4629 次调用可归约到 1346 个图像键，重复引用为 3283 次（70.92%）。这是明确假设下的结构性计算，**不是实测调用节省、命中率、延迟或模型效果**。它无需图方法。

相反，若直接把完整 benchmark 问题作为 VQA prompt，同病例同图像组合的精确 prompt 键没有重复。这个统计也不能推断实际 agent 的子问题是否重复，因为没有真实工具轨迹。

## 2. Affinity-map location：复用对象必须分开

| 复用对象 | 最近邻与状态 | 共同目标及对本项目的约束 |
|---|---|---|
| 执行结果与状态 | [TVCache，2026 preprint](https://arxiv.org/abs/2602.10986)；[ToolCaching，2026 preprint](https://arxiv.org/abs/2601.15335)；[KGCache，2026-08 preprint](https://arxiv.org/abs/2608.07954) | 减少重复执行。分别处理带历史状态的工具值、缓存管理、重复图邻域访问；“图管理结果缓存”已有明确近邻 |
| 计划与中间上下文 | [Agentic Plan Caching，NeurIPS 2025 Main](https://papers.nips.cc/paper_files/paper/2025/hash/9549f7d06700f0966d5f938f1d11022a-Abstract-Conference.html)；[Semantic Caching of Contextual Summaries，2025 preprint](https://arxiv.org/abs/2505.11271)；[Query-OPT，EMNLP 2024 Industry](https://aclanthology.org/2024.emnlp-industry.86/) | 复用计划、摘要或共享上下文以减少推理成本；不能仅用“复用中间过程而非最终答案”区分 |
| 多轮答案与底层状态 | [SmartCache，NeurIPS 2025 Main](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fb74b63d225f846e6032bf3e3ab0f4ec-Abstract-Conference.html)；[ReCache，2026-08 preprint](https://arxiv.org/abs/2608.19662)；[GroundedCache，2026 preprint](https://arxiv.org/abs/2605.27494) | 分别考虑上下文匹配、工具 schema KV 复用、证据支持与版本有效性。这些机制不能互相当作同一种缓存 |

APC 从已完成任务提取计划模板并适配新任务，而 TVCache 要求工具历史满足匹配条件；前者允许实例变化，后者保护执行状态。KGCache 的实体邻域缓存保持 LLM 调用序列，另设语义上下文缓存尝试减少检索规划；不能把所有节省归于推理能力提高。

SmartCache 用 Semantic Forest 匹配会话语义与上下文，ReCache 研究资源 schema 的可组合 KV 块。它们与工具观察事实复用不同，但已覆盖“树结构缓存”“组合复用”的泛化描述。这里不复述论文的加速数字，因为没有复现且衡量对象不同。

额外反证：[When KV Cache Reuse Fails in Multi-Agent Systems，ACL 2026 Main](https://aclanthology.org/2026.acl-long.327/) 发现候选间交互对 judge 决策重要。它说明不能将生成阶段的 KV 复用收益直接外推给仲裁阶段；本项目也不能把精确工具结果缓存和近似 KV 复用混在一个干预中。

## 3. Bit Flip status

尚未成立。多查询确实存在，但“存在可复用信息”不是创新；从三类文献只能得出不同层次有不同复用条件，不能说大家都只做精确缓存，也不能说现有方法完全忽略上下文或状态。

“同一病例新问题组合”目前只是可能的评估划分，还没有展示新的机制。仅把现有 APC/语义缓存/图记忆迁移到医学，不满足当前追求的研究强度。

## 4. Current Vector

**排除精确工具缓存、全图像特征复用及已有计划/上下文缓存后，MedRAX 是否仍存在需要学习关系结构才能解决的证据复用问题？**

本轮回答到的程度：真实多查询任务存在，普通缓存的混淆非常强，但没有轨迹证据证明剩余问题存在。这里是未回答，不是已经实证否定全部剩余价值。

## 5. Core / Periphery：源码给出的边界

源码来自 [MedRAX 官方仓库](https://github.com/bowang-lab/MedRAX)，commit `dae30e2f136ef0b2a40885a4c335386e9ffad052`。

| 工具 | 实际条件变量 | 应先采用的简单基线 |
|---|---|---|
| `classification.py` | 图像；固定模型和预处理 | 图像内容与模型版本键缓存一次完整病理向量 |
| `report_generation.py` | 图像；固定模型和解码 | 缓存图像报告；随机解码时需预先声明复用语义 |
| `segmentation.py` | 图像与所选 organs | 代码先预测全部 mask，再按 organs 做可视化/指标；复用全部 mask 属于显然的工程基线 |
| `grounding.py` | 图像、phrase、解码参数 | 仅图像键不足，保留 phrase 及参数 |
| `llava_med.py` | question、图像；代码为非采样生成 | 保留问题条件，不能以“同一张图”替代相同调用 |
| `xray_vqa.py` | 图像列表、prompt、解码参数 | 同时保留图像顺序、prompt 与配置 |

接口审计只支持“哪些变量影响输出”，不证明实现已具备生产级缓存或模型完全确定性。分割工具的全 mask 复用也不等于可以无条件复用旧可视化文件。

Core 完成元数据与接口审计。新 gate、专家融合、模型训练和图模块均未添加。没有修改原 MedRAX 或历史实验分支。

## 6. Minimum experiment：进入模型执行前必须满足的条件

目前不编造固定命中率的合成实验，因为那只会重演普通缓存的已知收益。下一个有判别力的步骤应使用同病例真实 agent 轨迹：

- H1：精确缓存以后仍有不同调用可由已得证据正确组合替代，且超越表格检索和 APC 类复用。
- H0：收益全部来自重复图像调用、已有缓存/摘要复用或更大的上下文预算。
- 必要基线：独立问答；按版本和参数键的精确缓存；完整图像特征缓存；平面事实表；已有计划/上下文复用。所有条件共享 actor、工具、解码和预算。
- 首先测量真实的剩余重复调用及必要性；若没有剩余可操作现象，不训练方法。
- 命中和未命中分别报告正确率、重新调用率、延迟及总成本。不能用引用次数推算端到端加速。
- 病例隔离 development 和 evaluation；当前元数据结构审计不能被当作模型测试结果。若做未见查询组合，需明确新组合如何与已有题目、图像、病例隔离。
- 不把他题 gold 答案或 explanation 当作复用证据。必须来自实际观察或模型生成，并保留错误传播风险。

这些是下一阶段的识别条件，不是已运行实验，也不是已冻结的正式样本量协议。

## 7. Novelty collision check

1. 图邻域跨查询复用：KGCache 已直接覆盖。
2. 新任务复用结构化计划：APC 已覆盖，且是 NeurIPS 主会工作。
3. 上下文条件的树状复用：SmartCache 已覆盖。
4. 中间证据摘要复用：contextual-summary caching 已覆盖。
5. 诊断证据图与补证：[CDEG，2026-08 preprint](https://arxiv.org/abs/2608.22899v2) 已覆盖此前的粗粒度方案。
6. 跨查询一致性也不能当作自动退路：[Quantifying Cross-Query Contradictions，2026 preprint](https://arxiv.org/abs/2604.14525) 已提出 case-level 可满足性与修复。该项只完成论文身份和摘要核验，不据其结果推断已彻底解决。

结论：暂不将“图缓存/多查询复用”选为论文方法。潜在剩余问题必须通过真实轨迹与强基线证明，不能靠换一个应用场景划出创新。

## 8. Velocity criterion

本轮新增知识是：609 个真实病例都带多问题；大量重复图像引用只支持普通工具缓存机会；官方工具的条件变量解释了哪些复用可直接实现，哪些需要新的证据；近邻已覆盖大部分泛化包装。

尚未获得：真实工具调用分布、语义可替代调用比例、正确性收益、图对平面结构的额外价值。GPU 空闲不能补上这些证据。

## 9. Re-vector rules

- 若真实轨迹剩余调用很少或已有表格/缓存基线解决，退出此候选，不添加图。
- 若有剩余现象，先明确其机制是信息获取、证据组合还是记忆检索；仅对一个机制建立新假设，再做精确近邻排重。
- 若数据只显示速度收益，不转述成推理能力提升；若多问题共享会引入错误传播，则同时测正确性与成本。

## 可复现产物

脚本：[audit_medrax_reuse.py](../scripts/audit_medrax_reuse.py)。运行：

```bash
python scripts/audit_medrax_reuse.py /home/dbw/research-notes/medrax-reuse-20260916/metadata.jsonl
```

本地目录 `/home/dbw/research-notes/medrax-reuse-20260916/` 保存固定源记录 `source.json`、元数据、去除答案后的结构行与 `structure_summary.json`。只发布审计脚本和汇总，不在仓库中复制整个 benchmark。此报告是定向排重与任务审计，非声称穷尽全部缓存论文的系统综述。
