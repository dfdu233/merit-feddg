# CRES：用对照实验约束冻结专家的解码影响

2026-09-15。基于 `d20c928d46ab86b3998ffc54c379ec46712cbfd1`，新增独立实验入口。
工作名：**Control-Referenced Evidence Steering（CRES）**。
这是可运行、可证伪的研究假设，不是已经验证有效的方法或 ICLR 录用判断。

## 先纠正 Fable 讨论中影响选题的判断

1. “主观逻辑/Dempster–Shafer 没用于 agent”不能成立。
   [InfoGatherer](https://arxiv.org/abs/2603.05909) 明确在医疗/法律信息收集任务中用
   Dempster–Shafer 表示不完整和冲突证据；该页面标注 under review，应称预印本。
   [2026 年多智能体辩论工作](https://arxiv.org/html/2608.03648v1) 也讨论 Subjective Logic。
   不应以“老理论首次移植”立论。
2. [ConSol](https://arxiv.org/abs/2503.17587) 已将 SPRT 用于 LLM 一致性采样停止，
   还需要校准检验参数。若没有明确的两个假设和有效似然比，累加工具分数不是 SPRT。
3. “最小证据能让答案翻转”证明的是决定性/依赖性，不证明正确性。一个错误标签也能
   构成最小决定性证据。它可做解释或压缩，但不能直接作为接受依据。
4. 同前缀配对很有必要，但不是从未出现的设计：CAD/VCD 已在每步共享生成前缀。
   配对得到的模型响应差异也不是医学任务的因果收益 ΔU。
5. 普通 conformal 不自动在未知跨域偏移下保持保证；source minimax 也不约束任意
   unseen target。无标签下不能凭分位数给一个未经验证的似然差加上正确率保证。
6. 参数冻结、source 上拟合小 head、source-label calibration 是三种不同实验条件。
   CRES 当前全部不做：不训练、不用 source/target 标签校准，也不把原生分数转为信任概率。

一个简单不可辨识性论证：若两个环境给出相同图像、问题、专家与 VLM 分数，但参考
标签不同，仅观察这些模型分数的规则会作相同决策，而真实 ΔU 的符号可能相反。
因此严格免训练的方法应声明具体的归纳假设，并用标签隔离的离线实验检验。

## 当前代码和数字到底支持什么

远端 main 和列出的分支没有 Fable 日志中的 `licensed_revision.py`，不能当作已提交成果。
本次选择最新 soft-guidance 分支作为基底，不覆盖 main 或其他研究分支。

[已提交的诊断](https://github.com/dfdu233/merit-feddg/blob/d20c928d46ab86b3998ffc54c379ec46712cbfd1/docs/SOFT_GUIDANCE_PILOT.md)
用同一离线 scorer 重评旧预测：VQA-RAD 451 例 generalist 48.47%、compact 53.96%，
SLAKE 2,094 例 34.68%、39.49%。这些是旧预测的新统一评分，不是新 soft-guidance 的全量成绩。
两个历史损害案例被 native spatial 修复，也同样被 deletion-only 修复；所以尚不能证明
原生几何提供了增量信息。旧 near-tie token 漂移必须保留审计，不能改叫精确复现。
[全量运行文档](https://github.com/dfdu233/merit-feddg/blob/d20c928d46ab86b3998ffc54c379ec46712cbfd1/docs/SOFT_GUIDANCE_FULL.md)
仍记录 running；本次没有服务器访问或更新的完整医学实验结果。

因此主问题收敛为：**当异构专家既提供有效结构又引入接口扰动时，如何只传递可被
结构对照区分的那部分解码作用？** 第一种实际接口是 predicted segmentation geometry。

## 文献与实际代码依据

| 工作 | 原始来源 / 开源实现 | 与本方法的边界 |
| --- | --- | --- |
| CAD，NAACL 2024 | [论文](https://aclanthology.org/2024.naacl-short.69/)，[group_decode_fileio.py](https://github.com/xhan77/context-aware-decoding/blob/1281d7fcd0e7e49d786c78d3170a3048dd9f6dc2/group_decode_fileio.py)：`decode` 中各分支 score 乘 assigned_weight，再 all_reduce，并共享所选 token | 双分支、同前缀、线性 logits 组合都不能算本次首创。CRES 改的是对照集合、允许的残差方向和步长约束。 |
| VCD，CVPR 2024 | [官方仓库](https://github.com/DAMO-NLP-SG/VCD)，[vcd_sample.py](https://github.com/DAMO-NLP-SG/VCD/blob/master/vcd_utils/vcd_sample.py)：`diffs=(1+cd_alpha)*next_token_logits-cd_alpha*next_token_logits_cd`，另有 plausibility cutoff | 借鉴受控分支比较。这里保持原图，只破坏专家几何与图像 patch 的对应；没有复现 VCD 的图像噪声或 cutoff。 |
| DnR，CVPR 2026 | [论文](https://arxiv.org/abs/2511.11005)，[process/uq.py](https://github.com/EavnJeong/Draft-and-Refine-with-Visual-Experts/blob/d72196b1b498e982ecf6eb94e91724707c15ebf2/process/uq.py)：mask 图像、重新回答、比较 CLIP 文本嵌入；`compute_uq_relevance_fidelity` 等 | 利用率不是正确率。CRES 不再用整句利用率为医学修改作证，而在 token 分布中比较真实/错位专家作用。 |
| Semantic Entropy，Nature 2024 | [官方代码](https://github.com/jlko/semantic_uncertainty/blob/a8d9aa8cecd5f3bec09b19ae38ab13552e0846f4/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py)：`get_semantic_ids`、`logsumexp_by_id`、`predictive_entropy_rao` | 衡量回答语义分布，不直接估计采用证据的收益。CRES 不冒称计算 semantic entropy，也不把低熵当作高正确率。 |
| Sufficient Context，ICLR 2025 | [论文](https://arxiv.org/abs/2411.06037)，[作者团队说明](https://research.google/blog/deeper-insights-into-retrieval-augmented-generation-the-role-of-sufficient-context/) | 上下文充足、模型能正确使用、上下文带来收益是不同命题；不能以“decisive”替代这些区别。 |
| RECOMP，ICLR 2024 | [论文页面](https://openreview.net/forum?id=mlJLVigNHp) | 压缩和允许不增强已有先例；本次不把最小证据包或允许空证据作为单独新意，也不声称复现其学习式压缩器。 |
| Relative Entropy Policy Search，AAAI 2010 | [原论文](https://ojs.aaai.org/index.php/AAAI/article/view/7727/7588) | 借鉴信息约束的指数倾斜；CRES 仅计算当前 token 分布，不学习策略、价值函数或奖励。KL trust region 本身是已有思想。 |
| Conformal Risk Control，ICLR 2024 | [论文](https://openreview.net/forum?id=33XGfHLtZg)，[core/get_lhat.py](https://github.com/aangelopoulos/conformal-risk/blob/main/core/get_lhat.py)：读取 calib_loss_table 并以 n/(n+1) 校正风险 | 实现明确需要 calibration losses。本次没有这些量，故不提供 conformal/shift-aware coverage 承诺。 |
| InfoGatherer / ConSol，2026 / 2025 预印本 | [InfoGatherer](https://arxiv.org/abs/2603.05909)，[ConSol](https://arxiv.org/abs/2503.17587) | 反驳“DST/SPRT 尚未进入 LLM-agent”的无先例表述，不把预印本当已录用顶会。 |
| HEAL，2026-09 预印本 | [论文](https://arxiv.org/abs/2609.09206) | 已有通过反事实 Difference-in-Differences 分离视觉/语言信息的工作；“反事实”“因果解码”本身不是新意。本次不使用 head 筛选或 value-vector 校准。 |

表中的代码机制经实际文件读取核查。引用不等于本项目已复现实验，也不保证完整穷尽
相关工作；尤其不能声称本方法是最早或全球最新。

## 实现：原生对应关系 → 对照残差 → 有界分布更新

所有分支在相同原图 I、相同问题 q、相同已提交前缀 h 上取分数。非分割证据固定为
当前 compact 实际送达的那批条目；移除分割文本后，不趁预算空余加入新条目。
首先用 existing `SpatialEvidenceBridge` 将专家 masks 作用于 VLM 自己的 patch features。
因此没有训练跨模型 bridge，也没有将 expert class logits 直接加到 token logits。

设 p0 为删除分割文本后的基分布，pe 为真实原生几何条件分布。两个固定对照分别将
全部 region 在完整非 padding patch 矩形内做水平/垂直半幅循环平移；原图不变。
同一次平移共同作用于所有 masks，保留区域值的多重集、权重、条目数和相互重叠，
固定 padding/边界 cells。循环边界仍不是真实解剖，因此它是压力对照，不是 iid null。
若形状恒定、过窄或生成重复对照，控制族不可用，CRES 返回 deletion 基分支。
无可用空间专家时所有干预臂返回 fresh compact，留在全量分母。

对每个 comparator c∈{p0,pc1,pc2}，定义去掉常数偏移的 log-ratio：

$$d_c(v)=\log p_e(v)-\log p_c(v)
 -\mathbb E_{w\sim p_0}[\log p_e(w)-\log p_c(w)].$$

令 l(v)=min_c d_c(v)，u(v)=max_c d_c(v)，保留共同方向的最保守幅度：

$$r(v)=\begin{cases}l(v)&l(v)>0\\u(v)&u(v)<0\\0&\text{otherwise}.\end{cases}$$

这是 **确定性的控制包络**，不是分位数置信下界。加入基分布作为 comparator 很关键：
若真实证据无作用但错位对照本身制造大扰动，不能把这种“坏对照”差异当作有用信息。
如果真实与任何对照产生相同 token 分布，r 全零；复制同一个对照不会增加信心。
增加 comparator 只会压缩每个残差坐标的幅度，但不保证最终重归一化分布变化也缩小。

最终分布沿指数倾斜路径更新：

$$p_\alpha(v)\propto p_0(v)\exp(\alpha r(v)),\qquad
\alpha_t=\max\{a\in[0,1]:D_{KL}(p_a\Vert p_0)\le\epsilon\}.$$

默认 ε=0.05 nats/token，是预声明研究超参数，不是从正确率推导的安全阈值。
固定对照臂 strength=0.5；48 次纯数值二分确定 α，不做模型反向传播或参数拟合。
这回答了“uncertainty 能否控制 alpha”：控制不确定性先削弱 r，剩下的步长再由
允许分布偏移的预算决定，不把 arbitrary entropy/native confidence 冒充 trust。

当前精确实现对应：

```python
# merit_feddg/control_evidence.py
differences -= (differences @ probability)[:, None]
lower, upper = differences.min(axis=0), differences.max(axis=0)
r = np.where(lower > 0, lower, np.where(upper < 0, upper, 0.0))
# Every branch next_scores() receives the same tuple(prefix).
scores, audit = constrained_scores(base, r, max_strength=1.0, kl_budget=0.05)
token = int(np.argmax(scores))
```

## 可以证明与不能证明的部分

可证明：logit 常数平移不变；缺少对照不自动启用；对照复制不增益；残差坐标随控制
集合扩张不增幅；以及指数族恒等式
`d KL(p_alpha || p0)/d alpha = alpha * Var_p_alpha(r) >= 0`，所以二分搜索返回可行强度。
严格零预算路径直接调用原生产 `propose`，保留其精确 tokens。

不能证明：残差等于 Δaccuracy；更多视觉依赖必定更正确；KL 小则 greedy token 不变；
任意 domain shift 下非劣；对照一致就是高置信。代码特意包含“错误但一致的专家仍会
改变答案”的反例测试。若按 soft distributions 采样且每个前缀都满足约束，可以用链式
法则讨论总 KL；当前 greedy 输出不能套用这个结论保证医学风险。

## 八臂实验和成败判据

| 臂 | 作用 |
| --- | --- |
| generalist | 同环境、同 uniform prompt 的冻结底座 |
| compact | 同缓存专家、完整文本包，对应必须保住的有效 incumbent |
| deletion | 只删分割文本，保留实际送达的其他证据；排除“仅删除即可提升”解释 |
| native_fixed | 既有原生几何接口 + 固定 0.5 强度 |
| null_fixed | 错位原生几何 + 同强度；检查接口扰动是否也造成所谓收益 |
| cres | 基分布与两个几何对照的共同方向残差 + KL 约束 |
| kl_only | 同样 KL 约束，用原始 evidence-base 差，不使用控制残差 |
| residual_unbounded | 同样残差、最大 strength 1，KL 预算放宽到 1e6；去掉实质 trust region |

默认统一 64-token 简答，CE/OE 不决定新的输出契约。旧 answer_type 只用于复核原缓存
对应的原提示，以及独立离线评分。所有新对照重新生成，旧 48.47/53.96 不能直接作为
uniform 实验的分母。可事先选择 historical prompt 配置作协议消融，但不能混用结果。

成功必须同时满足：CRES 超过 deletion、native_fixed 和真实 compact incumbent；减少
损害的同时保留旧语义证据收益；控制错位/重复的消融符合假设；在至少另一个完整数据集
和更强 generalist 上仍成立。当前 native backend 只接 LLaVA-Med；尚未实现 MedGemma/
Qwen 的原生通道，不能称多底座已验证。分类/检索/生成仍沿用既有文本，CRES score core
可接通用条件分支并不等于这些 native adapters 已完成。

输出完整 mixed/CLOSED/OPEN 分数，与 generalist、compact、deletion 的逐例改善/损害、
revision coverage、条件损害率、图像聚类 bootstrap、α/KL、控制可用率与成本。
零修改时 conditional harm=null，不是零风险成功。避免用 residual magnitude 直接排
risk–coverage 并称为校准置信度；若要 sweep ε，须预声明独立配置、完整清单报告，不按
目标结果选最优点。VQA-RAD/SLAKE 已多次参与开发，不再称未见 target。

反证同样明确：如果 null≈real 或 r≈0，只能说明当前接口没有可区分信号；如果 CRES
仅胜 generalist、没胜 compact/deletion，核心主张不成立；不能把全部回退包装成成功。
本地统计和单元测试也不能补上真实模型的效果证据。

## 文件、运行和验证

- `merit_feddg/control_evidence.py`：对照构造、残差与 KL 步长、共享前缀解码。
- `merit_feddg/control_study.py`：复用原生包/生产 session、fresh comparators、八臂运行。
- `scripts/run_control_evidence.py`：完整清单、源缓存/原图 SHA、模型/code fingerprint、锁和续跑。
- `scripts/evaluate_control_evidence.py`：独立 references、scorer SHA pin、完整结果才能评分。
- `configs/control_evidence.json`：固定参数；`tests/test_control_*.py`：数学、几何、运行器回归。
- [服务器 Codex 执行 prompt](CRES_SERVER_PROMPT.md)。

仍使用生产前缀 replay，每 token 至多四个 score API 调用，而且一次 score API 调用
本身可能重放多个 tokens。报告 score_calls 不是 forward 次数。全臂 wall time、单臂
decode 和 startup 分开，专家从缓存读取不等于部署调用零成本。这里未擅自扩展旧
semantic-only PersistentScores 到 spatial KV cache；需真实前缀 logits/token 对齐后另做。

本地六组针对性回归 **65 passed in 4.06s**，包括 21 个新增测试。Ruff、两个 CLI help
通过；新增算法与实际 native transport 的 fake-model 全路径可验证，
没有医学模型权重/GPU，因此没有生成新的真实医学准确率。全仓首次尝试因部分本地
快照缺失的旧配置停止，不能声称全仓回归通过。后续补齐相关依赖配置后运行针对性回归。
