# 统一向量证据接口与 training-free 动态 gate

本轮基于 `4b3d43f`。论文主线恢复为：（1）通用模型与异构专家的非文本统一证据接口；（2）无需拟合的样本自适应证据采用 gate。此前 permissions/uncertainty 的文本实验只作工程诊断，不能证明这两项机制有效。

## 实际代码与算法

### 统一表示需要训练；gate 不训练

`tensor_evidence.py::compile_tensor_evidence` 将注册概念的分类数值、2D 分割、检测输出编译为 TensorPacket，保留数值含义、作用范围及原图坐标。未知概念拒绝并审计，不随机分配 embedding，也不把分类分数伪装为病灶定位。生成文本和检索尚不支持这个非文本接口，主 runner 显式排除它们。

`tensor_bridge.py::NativeTensorBridge` 现在暴露两个接口：

- `encode_evidence(visual, packet) -> [1, N, width]`：任务数值编码、概念/范围/分数语义 embedding、掩码形状以及区域视觉内容。
- `fuse_evidence(visual, evidence)`：共享 attention 读取器及视觉残差融合；空证据严格返回原张量。

区域视觉内容来自 `pooled = weights @ h[0]`。统一向量无需把不同专家自己的隐藏层特征假定为同一语义空间。现有检查点参数名称和张量形状保持兼容。模块 `self.gate` 仍是学习得到的融合强度，不能充当逐样本可靠性；动态采用 gate 位于其上游。

数学路径：`o_i -> A_type(o_i,H) -> E_i`，然后仅对已接纳证据集合执行 `H' = H + tanh(g_fusion) * Delta(H,E)`。当前采用为二值决定，以一次专家返回的可编码结果为单位，不是逐 token 连续加权。单个返回可能包含多个原生记录。

### 新增的 gate 是可证伪的启发式

代码：`vector_gate.py::assess_visual_contrast`；生产接入：`NativeSession.assess_vector_evidence` 与 `CapabilityRuntime.execute`。

1. 在同一原图、问题、已提交前缀和已接受向量集合下，分别生成“不加新证据”和“加入新证据”的短候选 `y0`、`yi`。没有候选答案列表或 CE/OE 分支。
2. 用同一个冻结通用模型作为证据自由的 verifier：不注入任何专家向量，不加入专家文本。分别对原图 `x` 和相同尺寸、相同图像通道均值的纯色对照 `x0`，计算两个候选各自的平均 token log probability。
3. 定义 `S(y) = mean_logp(y | x,q,prefix) - mean_logp(y | x0,q,prefix)`，采用条件为 `S(yi)-S(y0) > min_gain`。默认 `min_gain=1e-6` 是固定数值平局容差，没有数据拟合过程。
4. 无变化、无正增益、非有限分数或会挤掉已接纳向量时，拒绝新证据。保留原状态和已提交 token。探测 token 永不提交；正式答案按原有自由生成接口重新生成。

这是根据图像相对于低信息对照的支持差异作采用判断，不把熵下降、专家高分或专家与基线一致当作正确性的充分条件。它允许与原答案不同的候选胜出，但也可能由于通用模型视觉能力不足而拒绝正确专家。

**训练边界**：专家与通用模型冻结；桥接必须已有匹配当前 base VLM 的训练权重；采用 gate 没有 optimizer、可学习参数、拟合数据或参考答案输入。因此只能称“training-free admission gate”，不能称“整个系统 training-free”。本轮不创建训练/校准子集，也不运行已有桥接训练脚本。

## 与公开代码的联系及区别

| 工作 | 可核查源码 | 本轮采用/区别 |
|---|---|---|
| Prismer | [PerceiverResampler](https://github.com/NVlabs/prismer/blob/4f27ab37e8ff5436707c5badd0ec744980608edd/model/modules/resampler.py) | 固定查询读取异构特征已有先例；本轮沿用项目内独立实现，不能把共享 attention 本身作为原创贡献 |
| Osprey | [MaskExtractor / MaskPooling](https://github.com/CircleRadon/Osprey/blob/be8f465df412ebfd29841452c75cad3d4c56f35b/osprey/model/layer.py) | 区域池化已有先例；当前统一接口还保留非空间分数及作用范围，但尚未证明新任务迁移 |
| VCD，CVPR 2024 | [作者代码 vcd_sample.py](https://github.com/DAMO-NLP-SG/VCD/blob/master/vcd_utils/vcd_sample.py)，约 139–153 行；[论文](https://arxiv.org/abs/2311.16922) | 作者源码计算 `(1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd` 后作逐 token 采样。本轮不移植该解码规则，而是独立实现“证据生成两个候选、无专家输入的视觉对照 verifier、结果级采用 gate” |

VCD 源码核查于 2026-09-10，链接为可变 master；这里只核对视觉对照及采样数据流，不声称复现作者实验。纯色对照也不同于 VCD 的扩散噪声。不存在将自然图像的文献收益直接移用为本医学方法收益的依据。

## 完整清单实验

三个实验臂共享相同 manifest、模型、桥接检查点、图像路由、原生专家持久化缓存、提示和 64-token 自由生成预算：

| 方法 | 向量通道 | 动态采用 |
|---|---|---|
| generalist | 空证据，原图 | 无 |
| tensor_all | 实际可编码原生记录 | 全部采用，仍受原生记录预算约束 |
| tensor_gate | 同一编码器和融合权重 | 本轮 visual-contrast gate |

调用仍按现有兼容工具顺序和上限进行。此 gate 是**调用后采用**，不节省已执行的专家首次推理。多专家决定依赖当前已接纳集合，可能有顺序效应；目前不宣称全局最优子集或与排列无关。

在已有临床模型环境中，从仓库根目录执行：

```bash
export MERIT_TENSOR_BRIDGE=/absolute/path/to/trained-base-matched-bridge.pt
python -m merit_feddg.matched_evaluation \
  --protocol vector \
  --config configs/matched_vector_gate.yaml \
  --manifest /absolute/path/to/existing-full-manifest.jsonl \
  --output runs/matched-vector-gate \
  --artifacts artifacts
```

manifest 沿用已有完整清单，包含 `id/image/question/image_sha256`，不得包含答案。题型元数据在载入时移除。上述路径均需替换为真实本地路径。

桥接缺失、不匹配、融合强度为零或没有任何活动专家绑定时直接报错，避免再跑一个名为向量协作但没有向量干预的实验。如果尚无训练好的桥接，先利用已有官方训练资源和现有 `tensor_train.py` 流程准备权重；不能拿测试答案训练，也不能用随机桥接绕过这个检查。本轮没有擅自分拆数据或训练权重。

## 成本与局限必须报告

- 记录每次 gate 的候选 token、两候选视觉支持、gain、采用原因、耗时，以及 verifier 查询数。`estimated_replayed_forward_steps` 按生产 `next_scores` 的前缀重放估计；它不是整个系统的实际 profiler 统计。
- 默认每个候选最多 8 tokens；二者相同就拒绝并跳过 verifier。因此延迟出现的有用信息可能被漏掉。该长度是预先声明的计算预算，不是答案长度上限。
- 不同候选可有不同长度；使用平均 log probability 消除直接的总长度累加，但没有消除表达方式、EOS 与内容复杂度偏差。
- 图像对照是强分布变化；纯色图像并不代表真实无视觉证据的反事实。视觉对比得分不是真实医学正确率，也不是校准置信度。
- verifier 与回答模型共享参数和视觉盲点；证据自由的验证减少直接循环注入，却不能消除共同偏差。
- 单次 gate 最多要对多个不同前缀反复执行模型。重放开销会随前缀增长；共享专家缓存使逐臂时间也不公平。必须分报 gate 成本，不能只报专家 forward 数。
- 目前检查点概念表固定、只支持注册的 2D 分类/分割/检测。新架构的同语义输出可以通过绑定适配，未知疾病语义不保证零样本泛化。

首次真实实验应比较 all 与 gate 的逐例纠错和错误引入、采用率、无法编码率和开销。无工具/全拒绝案例应回到对应原图基线；不得将前一轮文本 permissions 的收益移作本轮的实验结论。图像级配对统计不需要将数据集拆成多个集合。

## 本轮验证

机制测试覆盖：统一编码/融合一致性、空向量恒等、旧权重路径、真实运行时采用/拒绝状态、相同前缀探测、没有候选 token 泄入正式解码、分数非有限时拒绝、视觉对照恒等与正负增益、已有向量不被挤掉、三臂配置一致。

本环境没有实际医学模型与已训练桥接，未执行完整 GPU 实验。测试通过仅支持实现行为，不支持临床质量或论文创新性已获验证。

本轮完整回归：`python -m pytest -o addopts='' -q`，615 passed、17 skipped；修改的 Python 文件 Ruff 检查通过，CLI 帮助与 diff 检查通过。
