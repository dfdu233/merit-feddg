# Med-DEFER v0.6：论文定位与真实开放生成实验

调研与实现日期：2026-09-05。以下是研究假设和工程实现，不是有效性结论，
也不构成“达到 ICLR 录用标准”的承诺。没有复制或移植论文实现。

## 1. 目标不变，问题重新表述

让冻结的医学通用 VLM 在生成过程中借助不同能力的冻结小专家，
同时避免把专科模型在狭窄训练域上的偏差带入通用模型。

学术问题不是“如何再搭一个医学 agent”，而是：
**在未知域上，何时允许异构专家介入一个尚未提交的生成决策，才能取得正向介入收益？**
专家答得准，不等于它与主模型组合后更好；介入对象必须是组合系统的实际生成结果。

## 2. 最接近的论文与不能声称的创新

| 工作 | 已覆盖的关键内容 | 对本项目的约束/启发 |
| --- | --- | --- |
| [GSCo / MedDr，官方项目](https://github.com/sunanhe/MedDr)，[论文](https://www.nature.com/articles/s41551-026-01653-3) | 医学通用模型与分类、报告生成专家协作，模块化组织 | “通用模型+小专家”本身不是创新；旧代码的固定分数加权不是 GSCo 的忠实复现 |
| [CCD](https://arxiv.org/abs/2509.23379)，[方法全文 v1](https://arxiv.org/html/2509.23379v1) | 放射科专家的结构化临床信号进入生成时的 token-logit 调节，冻结主模型 | 最直接的医学对手；不能声称首次医学专家辅助解码。当前代码没有实现 CCD，必须额外进行忠实对比 |
| [FUDGE，NAACL 2021](https://aclanthology.org/2021.naacl-main.276/) | 用部分序列上的属性判别器调整生成概率，并组合多个属性 | 部分序列控制不是新发现；本项目不训练它的二值属性判别器 |
| [GeDi](https://arxiv.org/abs/2009.06367) | 小语言模型通过条件分布指导大模型的逐步生成 | “小模型指导大模型”和概率重加权都已有先例 |
| [PPLM](https://arxiv.org/abs/1912.02164) | 保持语言模型冻结，利用属性模型控制生成 | 冻结主模型、可插拔控制也不能单独列为创新 |
| [VGS-Decoding](https://arxiv.org/abs/2603.20314) | 原图/受损图分布比较，按 token 视觉依赖调节医学生成 | 需要区分主模型自身视觉对比与外部专家证据；本项目的 image-minus-null 不等于 VGS 复现 |
| [FedDG，CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_FedDG_Federated_Domain_Generalization_on_Medical_Image_Segmentation_via_Episodic_CVPR_2021_paper.html) | 医学分割中的跨源域训练、频域扩展和联邦域泛化 | 借鉴未知域评估与多源稳健性，但这里没有实施其频域算法或联邦训练；不能沿用 FedDG 的实验证明 |
| [DomainBed / In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434) | 域泛化方法比较和模型选择协议的重要性 | 目标标签不得用于阈值、温度、专家选择、候选构造；源域模型选择必须明确 |

调研结论：有价值的候选主张是“异构证据接口 + 生成前缀上的有界介入 +
基于实际组合增益的跨源域资格”。这是**待验证的组合设计**，不是已确认独创的算法。
要投稿，应补足 CCD/GSCo 等强对照，而不是把接口模块数量写成贡献数量。

## 3. 为什么不再把双阶段 OOD 作为主线

旧失败：主模型高置信犯错时没及时调用专家；后续重复使用最初 yes/no；
缓存改善没有进入真实输出；OCT 适配错误时，更多样本没有解决语义问题。

v0.6 主线只有一个资格判断：某个专家在这个模态、能力、任务上，
用同样的生成规则介入后，是否在各个源域都呈现正收益。
不再串联熵、二值错误概率、路由置信度和两次 OOD 折扣。
v0.5 的双阶段 OOD 和真实多中心闭集实验保留，作为独立对照，不混称新结果。

对专家 e、源域 d，实际生成完整回答得到连续差值：

    gain_i = token_F1(guided_answer_i, reference_i)
           - token_F1(beam_only_answer_i, reference_i)
    m_ed = mean(gain_d) - 1.645 * sample_std(gain_d) / sqrt(n_d)
    q_e = min_d m_ed

至少两个源域、每域至少八个独立图像/患者单位，且 q_e > 0 才可介入。
符合模态、能力、任务的专家中选 q_e 最高者；无合格者选 NONE。
不用 q_e 再乘证据强度，避免重复缩弱以及“校准用一个剂量、目标用另一个剂量”。
温度、强度、beam 数、预算均预先固定。修改任一项必须重新产生源域结果。

**统计边界**：这是 mean-minus-SE 的保守启发式，不是分布无关的置信保证，
更不是未知目标域的无伤害定理。源域卡的正值也不能作为独立验证集性能报告。
如果以后搜索大量超参数/专家，应增加源域内部嵌套验证及多重选择控制。
不按“canary 必须恰好挽救一例”反复调整目标病例。

## 4. 真实生成流程

    图像 + 问题 + 元数据能力需求
        → 医学 VLM 给出 K 个短 token 续写
        → 合格小专家评估“当前前缀 + 各个新续写”
        → 异构证据映射到这些续写的支持程度
        → 有界修正 + 主模型合理性约束
        → 提交选中续写的原始 token IDs
        → 下一段，直到 EOS 或 token 预算耗尽

默认 K=3、每段最多12 tokens、最多60新 tokens、最多3次专家调用。
专家从第一段就可以参与，不依赖主模型是否低置信。

评分为主模型生成 token 的平均 transition log probability，
加上 `strength * evidence_confidence * (signed_support - mean_support)`。
单个增量绝对值不超过 `2 * strength`；原始平均对数概率比最佳差超过2的续写
不能被专家抬升为胜者。这些都是实现约束，不是医学正确性保证。

关键边界：

- 是 **block-level candidate reranking during decoding**，不是逐 token logit 引导。
- 短段可能是不完整临床断言；没有声称实现完备的 claim parser。
- 候选由主模型动态产生，不来自标签或预设疾病选项。代码内部 `closed_set=True`
  仅表示本轮有限个 beam，不代表数据任务退回闭集分类。
- K 个续写都错时专家无法创造缺失答案；需分析候选覆盖而非无限放大 guidance。
- 当前使用数据提供的模态/能力任务元数据；不声称主模型已学会动态工具规划。
  一个病例选定一个专家，在其续写阶段使用；不是每段重新学习路由。
- 不重分词已提交前缀，避免专家修正后的 token 历史被破坏。
- 缓存图像输入和专家图像特征；尚无跨段 KV-cache 复用。
  beam 和重新 prefill 可能更慢。报告实际延迟，不能拿少调用等同于更快。

## 5. 异构专家接口：已通用的部分和未完成的部分

`OpenExpertPool` 懒加载，`ExpertAdapter` 可以通过配置中的 `factory: package:callable`
导入。原生插件实现：

```python
def infer_claims(self, *, image, claim, generated_prefix):
    # 调用真实专家；只处理本轮 claim.propositions。
    # 返回 NativeEvidence，expert_id 必须等于配置中专家键名。
    # segmentation: masks；detection: boxes；generation: generated_text；
    # retrieval: references/检索证据；并提供每个 candidate_id 的支持值。
    ...
```

`NativeEvidence.provenance["candidate_support"]` 必须覆盖本轮所有 candidate IDs；
`score_semantics` 是 `probability` 或 `logit`。mask、box 或文本本身不是诊断答案：
没有由适配器明确定义的语义映射就弃用，不能猜测其含义。
源域资格必须绑定专家模型、适配器、能力和任务，不能把分类资格通用于分割。

自有模型不必先上传 Hugging Face，可在专家配置中使用 `checkpoint_path`：

```yaml
my_segmenter:
  id: local/my-segmenter
  checkpoint_path: /data/checkpoints/segmenter.pt
  factory: my_medical_package:build_adapter
  factory_kwargs:
    expert_id: my_segmenter
  modalities: [pathology]
  capabilities: [segmentation]
  tasks: [open_vqa]
```

路径交给工厂的 `model_id` 参数；工厂/权重必须实际存在，本项目不会生成假的权重。
自有权重记录文件大小与修改时间，工厂源文件记录 SHA256；这用于恢复一致性，
不是对不可信文件的防篡改证明。外部插件的依赖也应由研究者锁定。

默认实际调用 CONCH 与 BiomedCLIP，都是对比视觉语言证据；
它们输出 image-minus-null 的语义匹配分数，温度固定为0.07。
CONCH/BiomedCLIP 的文本上下文有限，长问题/前缀可能截断，需检查对应失败样本。
接口测试覆盖分类、分割、检测、检索、生成；**这不等于五种真实医学专家均已验证**。
当前没有新增大型分割/检测权重下载，也没有虚构现成的通用 mask→医学语义映射。

## 6. 一键跑真实开放数据

```bash
git pull --ff-only
# 已获 CONCH 权限；沿用既有 HF_TOKEN，不要把 token 写入仓库。
./run_open.sh --mirror cn --source-per-group 16 --target-limit 16
```

新服务器可加 `--install-system`；GitHub 连通性差时可用已有安装选项
`--conch-source <你信任的镜像git源>`。已装好依赖与下载验证过的资产可加
`--skip-bootstrap`。修改配置可用 `--config configs/open_generation.yaml`。
这个 bootstrap profile 只准备 OpenMed 3B、CONCH、BiomedCLIP、PathVQA；
不继续下载 CheXagent 或其它大型专家。已验证完成的同 revision 资产复用。
`HF_TOKEN` 的存在不等于模型访问获批；首次访问仍需遵守 CONCH 条款。

[PathVQA 实际数据集](https://huggingface.co/datasets/flaviagiammarino/path-vqa)：
保留官方 train/test，排除原始 yes/no 问答；validation 此版本不用。
每张图最多一个问题，以图像哈希稳定抽样，train 再分两个明确标注的 proxy groups。
原始答案类型只用于界定开放问答子集，不用于按难度/正确性选病例。
推理 JSONL 不含答案；参考答案另存；检查 source/target 的 ID、图像像素、患者/图像组重叠。
不能据此排除预训练污染或来自同一本教材的更高层级重叠。

**这是实际数据上的开放生成机制实验，不是跨医院 DG 实验。**
跨医院/扫描协议等独立域，需提供自己的 `source.jsonl`、`target.jsonl`、`references.json`：

```bash
.venv/bin/python -m merit_feddg.cli open-study \
  --source /data/study/source.jsonl --target /data/study/target.jsonl \
  --references /data/study/references.json \
  --config configs/open_generation.yaml --artifacts artifacts --output runs/hospital-open
```

每条推理记录必须恰好含：`id,image,question,modality,capability,task,domain,domain_kind,role,
group_id,image_sha256`；均为非空字符串。image 可相对清单路径；image_sha256 通过
`merit_feddg.open_data.pixel_digest` 计算；group_id 用患者/切片ID并跨域一致。
`domain_kind=independent` 只在有真实机构/设备/协议来源证明时填写。
源域和目标域名称不得重叠。当前 pilot 强制每个独立单位只有一个问题。
参考文件为 `{sample_id: [reference_answer, ...]}`。它不传入生成器或专家。

## 7. 实验对照与输出

- 原始 greedy medical VLM。
- `beam_only`：完全相同 K、block 长度与分段策略，剥离搜索本身的贡献。
- `ungated:conch`、`ungated:biomedclip`：实际生成的固定专家介入。
- `robust`：源域贡献资格的稀疏版本，可能合法地全部选择 NONE。
- `reversed_support`：相同选择和调用预算，颠倒候选支持向量。
  **不等于**跨病例 shuffled-image 对照，也不假定永远能破坏证据对应。
- `shuffled_image`：主模型仍看原图，专家改看同模态/能力/任务的另一目标病例图像；
  按像素哈希确定无自配的循环置换，不读取答案。仅一个可用病例时跳过该对照。
  共享相同最大调用预算，但生成长度/EOS 改变后实际调用次数可能不同。

源域每个 baseline/专家均生成完整答案；目标域各方法也是实际生成，
不是对冻结缓存做分数反事实。病例级结果写入原子文件，中断最多重跑未完成病例。
重跑复用的身份绑定代码、依赖、模型 revision/本地文件状态、配置、图像内容和问题。
目标参考变动只重算评估，不改生成；源域参考变动重新计算资格。
更换温度、主模型或适配器使缓存失效；不会仅按文件名误用旧结果。
不同数据规模/参考答案版本保存到不同报告目录，但共享完全相同输入的生成缓存；
移动经过像素校验的同一图像也不必重新推理。
不要在两个进程中同时写入同一输出目录。

查看 `runs/open-generation/latest.json`，定位实际目录：

- `result.json`：EM/F1、配对 F1 增益及域内分层 bootstrap 区间、改善/损害数量、
  调用数、延迟、GPU 已分配内存峰值、逐域 F1。
- `qualification.json`：逐源域实际增益和拒绝原因所需统计；没有目标标签拟合。
- `case-index.json`：本实验每个病例/方法对应的共享缓存路径（`case-cache/`）；
  缓存包含每段实际候选、token IDs、base scores、专家修正、提交位置、
  模型驻留状态以及缓存身份。
- `annotation-blinded.json`：按图像核验支持/不支持/矛盾的医学断言与关键遗漏。
  `annotation-key-private.json` 单独保留，不给盲评标注者。

EM/F1 不是医学事实指标；例如 `no tumor` 与 `tumor` 的词重合分数仍可能很高。
不报告 `1-accuracy` 为开放式幻觉率。不把 test 的词重合改善写成幻觉下降。
延迟包含输入预处理和该次运行发生的专家懒加载，但不包括主模型首次装载；
冷热混合不可用来证明效率，需要后续统一预热和同硬件独立测时。

## 8. 面向 ICLR 的下一步与停止条件

1. 先检查真实新 trace：首段改变是否进入最终答案；真实专家是否优于合理的损坏证据控制。
2. 比较 greedy 和 compute-matched beam；只超过 greedy 而未超过 beam 不能归功于专家。
3. 若所有专家未合格，报告 NONE 和源域零/负收益，检查桥接语义与候选覆盖；
   不靠目标标签降低阈值。保留小样本负结果，不强行放量。
4. 用至少两种**真实不同能力**的小模型和两个医学主模型验证插件泛化；
   目前默认两个对比模型不足以支撑“跨能力普适”结论。
5. 加入真实独立源/目标域的开放任务、leave-one-domain-out 外层循环；
   保留 PathoROB 的独立多中心闭集辅助实验，但不要把两者拼成不存在的开放式跨医院结果。
6. 以盲评医学 claim-level factuality 为主要幻觉证据；报告拒绝、遗漏与质量-成本曲线。
7. 忠实补充 CCD、GSCo、视觉对比解码等强基线，使用相同主模型或明确披露不兼容项。
8. 所谓“域鲁棒收益保证”需要额外统计假设及证明；目前只有可检验的设计假设。

这次升级解决的是实验是否能忠实检验目标，不是预先保证实验成功。
