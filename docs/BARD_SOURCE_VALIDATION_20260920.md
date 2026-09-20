# BARD / HuatuoGPT 实机验证：当前没有超过 Baseline 的证据

本次基于用户指定的 `39462b7c22ac7bb96658c7866b6a0be206eed72b`，在独立分支 `experiments/bard-source-validation-v1` 验证。没有修改 main、PR15 的算法分支，没有恢复旧 C 方法，也没有启动完整 TEST。用户追加优先 HuatuoGPT 后，两张5090承担 Huatuo 主实验，宿主机两张卡复核两个关键样本。

**结论：当前实现能抑制部分专家干扰，但没有证明有效纠错。暂不放量到完整 TEST。** 以下小样本结果不是数据集整体准确率，更不能作为论文提升结论。

## 实测结果与划分纠正

最初从历史 source 缓存取得6条 VQA-RAD，随后以官方 parquet 的问题与图像字节核验，发现其中1条实际来自官方 TEST（test index393）。该条已隔离，不能再声称未暴露，不能算入 TRAIN；没有据此修改算法或参数。错误来自把历史 proxy-source 标签当成官方划分，具体记录见 `reports/bard-source-validation-v1/vqa-official-split-audit.json`。因此本报告主结果为 **5条官方 VQA-RAD TRAIN +10条官方 SLAKE TRAIN**，实际执行16条。

SLAKE采用历史冻结队列前8条，再加入该128条缓存中全部两个四节点样本；VQA采用当时已完成原生专家采集的样本，经官方划分核验后保留5条。选择没有使用答案或得分，但样本存在覆盖和可用性偏差，不能外推。原生专家请求、图像字节、问题、空生成前缀、专家ID和输入包哈希均保留；本实验是**真实 Huatuo receiver 对冻结原生专家输出的重放**，不是重新运行 Huatuo router 和全部专家的端到端评估，也没有冒充匹配成功的旧缓存导入。

所有方法使用相同图像、问题、专家输出和生成设置：HuatuoGPT-Vision-7B，BF16，greedy，repetition_penalty1.2，min_new_tokens1，max_new_tokens1024。先前64-token预检有一条长答案被截断，已用1024预算完整复核。主实验所有104个输出均生成至EOS；每例的无专家分支与原生 generate 逐token一致。

下面统一报告既有 ANCHOR `answer_token_recall` 内容诊断，采用第一参考答案，不以格式合规计分。它是词汇召回率，**不是严格准确率或语义正确性判定**：可能低估中英同义答案，也可能高估包含额外错误内容的答案。原冻结CLOSED/OPEN混合指标另保留在64-token预检文件，避免把SLAKE非二元CLOSED标签当成二元题。

| 方法 | VQA-RAD TRAIN，n=5 | SLAKE TRAIN，n=10 |
|---|---:|---:|
| Generalist Baseline | 60.00% | 30.00% |
| joint_all | 46.67% | 45.00% |
| isolated_mean | 53.33% | 45.00% |
| isolated_geomedian | 53.33% | 30.00% |
| BARD f=1 | 60.00% | 30.00% |

BARD的15/15条答案去除大小写和标点差异后与Baseline相同。12/15条因少于4个节点直接回退，剩余3条也未产生实质纠错；两条SLAKE中记录到的commit只是标点变化。SLAKE mean相对Baseline有两条得分改善，BARD保留0/2；只看两个真正满足四节点条件的SLAKE样本，mean得分50%，BARD与Baseline均为0%。所有Huatuo主实验的joint与isolated实际送达专家集合相同，joint无丢包；这部分对比没有LLaVA canary的上下文容量混淆。

一个内容层面的反例是肺部颜色：参考为黑色，Baseline给出白色，mean给出黑色，geometric median和BARD给出白色。干净条件下，median在首个内容token就选择与Baseline相同的错误方向；此时甚至尚未触发commit阻止修改。这说明失败不只来自最后的保守门限，稳健中心本身也可能丢失有用的少数方向。

## 单节点污染：稳定并不等于正确

两个四节点SLAKE样本，每个固定污染一个专家，分别跑四个节点，共8个干预条件。每一步在当前共同已提交前缀上，重新计算真实receiver residual，并将指定节点替换为 `-8*r_e`；其他节点重新执行正常模型计算。三个方法自由生成到EOS，不是仅统计原轨迹上的单步假想选择。

| 方法 | 保留clean答案（忽略标点大小写） | 污染后内容召回率 | 有害得分翻转 |
|---|---:|---:|---:|
| isolated_mean | 4/8 | 0% | 4/8 |
| isolated_geomedian | 6/8 | 25% | 0/8 |
| BARD | 7/8 | 12.5% | 0/8 |

BARD的clean保留率较高，但clean答案本来就错；median的两次变化反而纠正了错误，BARD只保留一次。不能把“保住错误答案”当成医学鲁棒性成功。独立VQA TRAIN canary中，mean在四种污染中有一次答出明显错误器官；median与BARD均保留正确原答案，但该case的BARD clean没有任何内容修改。

这些是receiver分数层的特定sign-reversal攻击，不是任意恶意文本或所有Byzantine攻击。没有声称joint-all完成同类污染对照：联合上下文没有可单独替换的某专家residual，强行套用会改变威胁模型。各次干预也不是8个独立病例，不能据此给总体显著性结论。

## 机制分析与创新性边界

- **多数受同一receiver偏差影响。** 独立expert_id不意味着独立正确性。异构专家常测量不同属性，诚实输出也不一定回答当前问题；把这些影响向量当成围绕正确更新聚集的观测，目前缺少依据。上述颜色反例支持继续诊断这种机制，而非调整输出格式。
- **Mean中的锚定代数上会抵消。** 忽略不影响argmax的标量中心项，`log p0 + mean(log pe - log p0) = mean(log pe)`。不能仅凭“residual”名称就声称额外保留了Baseline约束。真正限制偏离的是最后的commit。Geometric median还涉及p0加权中心化几何，不能直接把mean的等式推广成完全相同算法。
- **四节点条件不是临床正确性定理。** `n-f`个严格正margin与排序后第f个margin为正是等价条件；在当前实现里“worst-case margin”没有增加独立约束。`3f+1`不是将任意相关专家转换成可靠共识的证明，也没有证明污染后最终答案必定等于clean答案。
- **已有方法需要正面对照。** 几何中位数稳健聚合在[RFA](https://arxiv.org/abs/1912.13445)已有；隔离上下文后聚合及解码级防御在[RobustRAG](https://arxiv.org/abs/2405.15556)已有。BARD将这些构件用于医学专家receiver residual有研究动机，但仅组合这些构件和共识规则不足以确认机制创新。这里是基于代码和上述来源的判断，不是完整查新结论。
- **不能靠大量回退掩盖问题。** 历史SLAKE128条缓存中仅2条具备4个原生节点。应报告有效覆盖率、内容级纠错保留和在原本答错病例上的表现，而不只报告clean输出保持率。

当前最值得解释的问题是：为什么有益专家影响成为少数方向，而多数分支继承了Generalist错误？在该问题没有证据支撑的解决方案前，不建议继续完整TEST或把f降低后直接挑结果。

## 环境、修改与可复现性

两张5090使用torch2.8.0+cu128、transformers4.37.2及既有eager Qwen2实现。宿主机用原huatuo环境torch2.0.1+cu117。两个四节点样本的10个clean输出，跨机器9/10逐token相同；不同的一例为joint-all，宿主机给出灰色、云端黑色。Baseline/mean/median/BARD全部一致，未将两个runtime的得分混合。

5090主实验PyTorch峰值allocated为**18.67GiB**，32GB卡已实际跑通。本数字针对单卡Huatuo receiver与复用专家包，不是所有原生专家同时驻留的32GB容量保证，也不包括CUDA驱动所有显存开销。主实验实际计数2876次receiver语言模型forward；Huatuo预检、主实验、宿主复核和最终回放合计5272次。未仪表化LLaVA及原生专家采集不在该计数中。

没有改变BARD聚合或commit定义。新增Huatuo原生输入/private-KV适配器与重放入口；给单步fault诊断补充同prefix、同fault的mean/median对照；matched manifest保留source元数据。后者不等于官方split验证，上述历史proxy问题单独审计。

验证：15项BARD targeted tests通过；本次修改文件Ruff通过；所有主实验原生Baseline逐token一致；整理代码后重新执行一个四节点case的17个clean/fault任务，与整理前逐token17/17一致。main和PR15未合并或改写。

原始输出保留于工作目录 `runs/bard-source-v1/`，未提交患者图像、原始专家大包或模型权重。核心结果：

- `reports/bard-source-validation-v1/huatuo-native-budget.json`：1024-token主结果、逐case指标、隔离的TEST暴露。
- `reports/bard-source-validation-v1/runtime-parity.json`：宿主与5090对比。
- `reports/bard-source-validation-v1/final-code-parity.json`：最终代码17任务重放一致性。
- `reports/bard-source-validation-v1/cost-and-runtime.json`：分阶段真实receiver forward计数。

复现入口：`python scripts/run_bard_packet_probe.py --bundle <frozen-native-packet.json> --output <new-directory> --shard-index 0 --shard-count 1 --include-joint`。对无污染条件添加`--clean-only`。输入包必须先从官方划分核验；不能仅凭role=source宣称TRAIN。原生包及图像校验通过后才执行，不读取参考答案；评分严格离线进行。
