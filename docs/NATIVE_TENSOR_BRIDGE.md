# 原生异构输出的非文本桥接实验

用户提供的补丁基于 `e1fbe3e`，对应“按输出类型编码 → 共享读取器 → 门控注入”。
这是需要训练的新实验分支；尚未训练真实医学桥接权重，也没有新增医学性能结论。
原有默认配置不启用它。这里的检查点必须配合同一个基础 VLM 使用，隐藏维度相同
并不意味着不同基础模型的特征语义兼容。

## 与已核对项目源码的区别

| 项目 / 源码位置 | 已有机制 | MERIT 原代码与本次改动 |
| --- | --- | --- |
| [Prismer `VisionTransformer`](https://github.com/NVlabs/prismer/blob/4f27ab37e8ff5436707c5badd0ec744980608edd/model/modules/vit.py)、[`PerceiverResampler`](https://github.com/NVlabs/prismer/blob/4f27ab37e8ff5436707c5badd0ec744980608edd/model/modules/resampler.py) | 任务专属卷积编码专家预测，汇聚到 64 个 latent，再与 RGB 表示连接 | 原代码没有连续证据 token；新增逐类别/逐区域编码及固定查询读取器。该基本架构已有先例，不算独立创新 |
| [ProxyCLIP `custom_attn`](https://github.com/mc-lan/ProxyCLIP/blob/a0e38d1989c990651fda0d8fbf8aad41bde955b5/open_clip/transformer.py) | 专家特征产生空间相似矩阵，用来聚合 CLIP 的 Value；不同网格插值 | 本次使用预测掩码/框的覆盖权重池化 VLM 表示；没有复制 Proxy Attention，也不把整图疾病概率伪造成空间关系 |
| [Osprey `MaskExtractor` / `MaskPooling`](https://github.com/CircleRadon/Osprey/blob/be8f465df412ebfd29841452c75cad3d4c56f35b/osprey/model/layer.py) | 掩码池化视觉特征并编码区域信息 | 借鉴区域内容与空间范围共同编码的思路；本次还支持非空间分类分数、分数语义和作用范围 |
| [MedRAX agent](https://github.com/bowang-lab/MedRAX/blob/main/medrax/agent/agent.py) | 工具调用后把结果放入工具消息继续推理 | 原来的 `EvidenceItem → JSON/视图 → VLM` 属于相近接口范式；MERIT 另外已有同前缀干预与源域价值估计 |
| [VILA-M3](https://github.com/Project-MONAI/VLM-Radiology-Agent-Framework/tree/main/m3) | 医学专家反馈和专家增强训练，是直接医学对照 | 本次把分类数值和空间证据送入内部视觉表示；不经专家报告文本，不证明天然更优 |

上表为补丁作者提供的源码对照记录，不代表合并者本轮逐行复核了所有上游函数。
本轮另核对了 Prismer、Osprey 和 VILA-M3 的官方仓库页面。
MedRAX 与 VILA-M3 的系统差异延续补丁中的审查记录；
没有把未核对的 MoVA/BRAVE/Eagle 内部代码声称为本次直接移植。
实现是独立编写的 PyTorch 模块，未拷贝上游源码或权重。尤其 Prismer 源码标注
NVIDIA 非商业许可，不应直接将其文件并入本仓库 Apache-2.0 代码。

## 实际新增的计算路径

1. `tensor_evidence.py::compile_tensor_evidence`：显式名称绑定解析概念/作用范围 ID。
   分类分数保留 `relative_similarity`、`uncalibrated_independent_sigmoid` 或
   `probability` 语义。数值输入包括分数、存在标记、可选 [0,1] 不确定性及其存在
   标记、框、面积、空间标记。不把未输出的类别当作阴性。
2. 分割 RLE 先恢复原图坐标，再经过与 VLM 相同的确定性方形 padding；框采用
   patch 覆盖面积。空掩码、不明确坐标、未知概念拒绝并审计。当前仅支持 2D。
3. `tensor_bridge.py::NativeTensorBridge`：三类数值编码器、概念/范围/分数语义
   embedding、掩码形状编码，以及 `R @ H` 的区域视觉内容。`H` 为 VLM 原有视觉
   投影输出，不要求专家隐藏特征与它处于同一个语义空间。
4. 共享 attention 读取器生成固定数量 latent，再由原视觉 token 读取；最终
   `H' = H + tanh(g) * delta(H, evidence)`，保持原视觉 token 数和文本位置不变。
   当前读取器没有单独的 question embedding；问题影响上游调用和后续语言推理。
5. `llava_generalist.py` 在真实生成、同前缀分块和 `next_scores` 重放时使用局部
   `mm_projector` hook。异常也会移除 hook。该 probe 仅供单请求使用。
6. `NativeSession` / `CapabilityRuntime` 的 `evidence_style: tensor` 使用此路径，
   不把工具证据序列化到文本。采用量由实际可编码记录决定；生成/检索专家被排除。
   配置 `request_scope_check` 时仍执行已绑定的问题范围筛选。

专家来源留在审计中，不进入可学习 embedding。类目逐条编码，故已知概念的类别
数量可变；这不等于语义理解已经对新疾病泛化。没有证据或推理时门控为零，桥接
返回原视觉张量。非零门控没有自动识别错误专家的保证，也没有 token KL 上界。
当前与 `BoundedNativeSession` / 文本格式空对照的组合会明确报错，避免运行成空证据。
agent 控制器当前依据原图、问题和已提交前缀选择工具，不读取 tensor latent；
已有源域价值策略仍可基于状态/历史控制调用。

## 源域训练

复用已兼容的 LLaVA-Med Python 环境及本地权重；需要 PyTorch，不必升级服务器的
Torch/Transformers。训练文件通过 `python -m merit_feddg.tensor_train --help` 查看。

创建 `generalist` YAML（也可复制现有配置中的 generalist 段）并加入：

```yaml
generalist:
  backend: llava_med
  id: microsoft/llava-med-v1.5-mistral-7b
  checkpoint_path: ${LLAVA_MED_CHECKPOINT}
  source_path: ${LLAVA_MED_SOURCE}
  vision_tower_path: ${LLAVA_MED_VISION}
  dtype: float16
  device_map: auto
  deterministic_image_padding: true
```

基础模型必须使用 `image_aspect_ratio=pad`、方形 CLIP 预处理、patch-only 特征、
flat patch merge。代码逐项检查，避免将掩码坐标与另一种预处理混用。

缓存每个源病例的实际 `EvidenceItem`，JSONL 每行恰有以下字段：

```json
{"id":"source-001","split":"source","domain":"hospital-a","image":"/absolute/source.png","prompt":"What finding is visible?","answer":"source reference answer","evidence":[{"evidence_id":"c1","expert_id":"cxr_findings","capability":"classification","scope":"cxr_findings","payload":{"score_semantics":"uncalibrated_independent_sigmoid","findings":[{"finding":"Lung Opacity","score":0.71}]},"summary":"","confidence":null,"provenance":{}}]}
```

示意数值不是实验结果。真实缓存可从现有 source 工具 trace 的 `native_evidence`
提取；须确保属于当前病例，并使用与推理相同的作用范围筛选，不能放入标注掩码。
`configs/tensor_contract.example.json` 只是五个概念的示例，**不是完整 XRV 类别表**。
先扩展为实验预声明的完整类目契约，或显式选定子集；训练时任何未注册/无效输出
都会报错，避免悄悄训练空证据。source 标记和数据来源声明仍需研究者真实填写。

```bash
python -m merit_feddg.tensor_train \
  --generalist configs/your_local_generalist.yaml \
  --contract configs/tensor_contract.example.json \
  --source artifacts/tensor_source_train.jsonl \
  --output artifacts/tensor_bridge.pt \
  --epochs 3 --learning-rate 0.0001 --seed 0
```

训练冻结基础 VLM，通过答案 token 的 teacher forcing 损失训练桥接模块；原图/
问题位置标签设为 -100，专家不参与反向更新。检查点携带契约、训练步数和源域记录，
旁置报告记录损失和数据/检查点哈希。训练损失不是临床有效性或跨域鲁棒性证据。
当前单病例一批、无梯度检查点优化；真实 7B 反传仍需要 GPU 显存。

## 在现有框架启用

在实际传给 runner 的**完整配置**中增加：

```yaml
generalist:
  # 保留上述其他 generalist 字段
  deterministic_image_padding: true
  tensor_bridge_checkpoint: /absolute/artifacts/tensor_bridge.pt
capability_value:
  generation:
    evidence_style: tensor
    visual_views: 0
```

不要把上述片段直接覆盖成完整 generalist 配置。原有 `NativeSession` / value-study
runner 会读取它；无需替换 `CapabilityPool`。检查点文件哈希参与运行 provenance，
修改桥接权重不会复用原先身份的缓存。现有 policy 必须针对新证据通道重新拟合。
文本/裁剪策略的收益值不能直接当作 tensor 通道的收益值。

也可通过现有后端直接进行单例诊断：

```python
from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
from merit_feddg.generalist_factory import load_generalist

probe = load_generalist(generalist_spec)  # 已配置本地训练检查点
config = ValueGenerationConfig(evidence_style="tensor", visual_views=0)
session = NativeSession(probe, image_path, question, question, config)
block = session.propose(NativeState(items=tuple(native_evidence_items)), length=16)
```

同概念新专家通过 `probe.tensor_bridge.bind_expert([...])` 增加本地标签别名绑定，
不改变权重；保存扩展后的契约以便复现。`concept` / `scope_id` 必须已存在。
绑定本身不证明新专家与旧专家同样可靠，分数分布改变必须另行评估。
未见概念、3D 掩码、任意 hidden feature、新的 detector 后端不在本次实现范围。

## 怎样形成可辩护的创新点

原框架较值得保留的是“同前缀的真实工具增益 + 源域保守选择”，本次补的是它原先
缺少的非文本输出通道。可以检验的组合假设是：**不依赖专家 ID 的类型化证据接口，
在专家替换/组合变化时，结合通道专属的源域效用选择，能否减少错误证据的传播。**
这是研究假设，不是已确认的文献空白，也不是本次 CPU 测试已经证明的性质。

最低实验集：

| 要验证的贡献 | 必须隔离的对照 |
| --- | --- |
| 数值/空间信息优于文本反馈 | 同一专家输出、同一病例：原图、JSON、裁剪、tensor |
| 复杂读取器值得使用 | 同样 token 预算和训练预算：简单投影/拼接、共享读取器 |
| 专家替换能力 | 同任务换未参与训练的骨干，只增加已知概念绑定；报告适配成本 |
| 输出/组合泛化 | 分类、掩码、框分别及联合；删去专家、调换顺序、改变类别数量 |
| 模型真的读证据 | 固定图像，交换同类专家输出、打乱区域对应、内容空对照 |
| 可靠调用 | 不选择工具、全调用、均值价值、保守价值；报告收益与受损比例 |
| 跨域泛化 | 真实医院/设备来源划分，患者隔离，目标标签只用于最终评估 |

必须将桥接训练源域与价值策略校准/验证病例分开。严格 LODO 时，每一 fold 都要
重新训练桥接、建检索库、拟合价值策略；只把回归器留一域不能代表整条学习管线
留一域。本次没有替你构造这些真实实验数据或完成新的 fold-isolated GPU 训练。

CPU 验证覆盖数值语义、原图/crop 映射、微小框、未知类别、空掩码、同契约新专家、
记录顺序、零门控、真实 PyTorch 梯度、冻结主模型的 teacher forcing、hook 清理和
LLaVA 协议下的生成前缀保持。协议测试使用小型模型替身；不是实际医学权重实验。

补丁作者报告的测试计数不作为本次合并验证结果；本地复验结果单独记录在 STATUS.md。
没有加载真实 LLaVA-Med、XRV 或其他医学模型权重。

## 2026-09-09 合并审查

输入补丁 SHA256：`60a118fb3a5fb6ae3624efe0e687e4381c4ea7bed8e253b1efe8a0abd2a1a39c`。
本地全量验证：591 passed，66.88 秒；Ruff、diff 检查及训练入口 --help 通过。

- 与原有 uncertainty 路径共存，均默认关闭；不将可训练桥接声称为 training-free。
- 新检查点记录基础 VLM/视觉塔/源码的现有 provenance 指纹，生产加载必须匹配；
  只有相同隐藏维度不足以通过。它使用项目既有 provenance 定义，不是新增全权重哈希证明。
  旧的无绑定检查点不准用于生产加载，需重新产生带绑定的训练产物。
- `--study diagnose` 的旧文本通道诊断不支持 tensor，明确报错；通过上面的 NativeSession
  示例或 value-study 执行真实 tensor 路径。不能把旧诊断输出误称非文本实验。
- tensor 暂不解释 native_uncertainty 集合，禁止与 uncertainty_from_probe 同时开启。
- 源域训练仍需另外审计患者隔离及标注来源；JSON 的 source 字段不证明无泄漏。
- 可学习标量门控不是当前输入的域可靠性估计。域泛化仍需要独立门控和真实跨域评估。
- 与 Prismer/Osprey 的模块先例相比，新颖性不能仅靠“掩码池化 + 共享注意力 + 残差”。
  需要实验证明同契约专家替换、域偏移下的证据可用性及训练/校准完全隔离后的收益。
