# 简单可插拔医学专家协作：内容与支持范围保持

## 结论与实现范围

研究目标是一个冻结通用模型使用可替换专家的协作接口，而不是增加多个医生角色、训练一个通用正确率预测器，或为每个失败病例增加专属工作流。最小控制循环保持为“选择一种能力 → 获取完整观察 → 回到原图回答”。

本次更新建立在 `implementation/plug-observe-v1` 的 `7381f9687782bc6c6b4a00402a6c608c906c35ef` 上。该版本已经有原子区域观察、原生 artifact 与模型 content 分离、四臂实验、可扩展 CapabilityPool 和冻结动作选择。本次不将这些已有能力列为新实现，更不将它们称为论文首创。[R1]

实际修改集中在已有 `plug_observe.py` 和 `plug_run.py`：完善观察身份的内容/范围绑定、为检测框补通共享区域接口、纠正局部观察被误认为全图观察的去重、增加可选的可逆共享字段布局。没有新增推理模块、模型、Agent 角色、训练目标、临床阈值或依赖包。历史 semantic/compact、空间算子、旧 gate、旧启动脚本和实验结果不变。

## 1. 文献调查与可复用机制

| 工作 | 核实发表信息 | 已有机制与源码证据 | 本实现的选择 |
|---|---|---|---|
| AgentOccam | ICLR 2025 | 调整观察/动作表示；`AgentOccam/env.py::observation` 保留结构节点并生成简洁模型视图 | 借鉴“表示先于复杂控制”。不移植其网页 DOM 裁剪到医学字段。[1, C1] |
| MedRAX | ICML 2025 | 医学专用工具与多模态 LLM 协作，无额外训练；`medrax/agent/agent.py` 注册工具并循环处理 | 证明医学工具插件已存在。复用现有 CapabilityPool，不重新创建 LangGraph 框架。[2, C2] |
| MedAgent-Pro | ICLR 2026 | 指南检索、疾病级规划、患者级工具执行与反思；`Task_level.py` 明确区分中间分割/测量和最终判断 | 是近邻，不声称“分割+规划+核验”为新贡献；不增加层级规划和代码生成器。[3, C3] |
| ViperGPT | ICCV 2023 | 视觉模块组合；`image_patch.py` 保存区域与原图父坐标关系 | 借鉴几何身份保持，不运行 LLM 生成的任意代码，不把屏幕方位当患者方位。[4, C4] |
| VEP | ACL 2025 | 专用小视觉模型的符号化结果供通用 VLM 使用 | 语义证据是必要强基线，专家转文本并不新；本次不宣称复现其完整训练/评测代码。[5] |
| GSCo | Nature Biomedical Engineering 2026 | 专用诊断与相似病例作为通用模型上下文 | 医学大小模型合作并非空白；基础模型与专家的开发含训练，不能将其全部流程等同于严格免训练接入。[6] |
| LangChain ToolMessage | 官方工程文档，非论文 | content 交给模型，artifact 保留其他程序需要的完整输出 | 借鉴工程分层；不安装框架，不把 content/artifact 区分当学术首创。[7] |

论文发表信息核对官方会议/期刊记录；实现细节来自实际读到的作者源码。没有运行这些项目的医学模型权重，没有声称所有已读代码都是最终会议论文的逐项复现。

### 1.1 源码证明：简洁视图不应破坏对象身份

AgentOccam 的官方实现先处理节点，再返回文字与结构化节点：

```python
DOM_str = translate_node_to_str(node=DOM_root_node, mode="concise")
return {"text": DOM_str, "image": self.obs["image"], "node": DOM_root_node}
```

这说明可以保留完整结构而不把一切都送入语言上下文。医学数据不适合照搬网页元素删除策略，因此本次只做已有原生值的共享字段表示。[C1]

MedRAX 的工具执行器注册工具，并将返回值放入 ToolMessage；其当前文件采用 `content=str(result)`。这提供调用层参考，但不能代替医学 mask、坐标、来源和分数含义的原生存储。[C2]

ViperGPT 的 ImagePatch 通过 `left + parent_left` 等操作保持子区域位置。医学插件同样需要坐标变换，但必须在适配器里明确约定像素/归一化、xyxy 和原图参照，不能自动猜测。[C4]

## 2. 根因与研究假设

不同专家解决的任务不同。分类分数、提示条件下的 mask、相似病例与一般医学知识不能被当成同一种“最终诊断投票”。统一的是调用与表示协议，而不是强行让原生分数进入同一正确率空间。

令观察为 `e = (content, support, source, type)`。同样内容若属于不同区域、不同来源或不同任务范围，不能被当成同一观察。来源与适用范围也不能在压缩或下游阅读时被重新赋义。

本次要检验的假设是：部分协作失败来自内容与支持范围的错配，以及为保持它们而引入过多冗余上下文。可复用、少参数的接口可能比多层评审更适合冻结模型。这是假设，不是现有 TRAIN 结果已经证实的原因。

原 evidence-agent 完整 TRAIN 运行零候选，证明的是候选送达不足；它不能证明观察依赖建模提升医学正确率。此前的 compact_rows 收益应继续归属旧方法。仅修复身份哈希或补一个 box 接口属于工程基础，不足以单独构成 ICLR 贡献。[R2]

## 3. 实际代码改动

### 3.1 完整观察身份

旧观察引用只绑定 case、expert ID、evidence ID、payload 和 parent。新引用为：

```python
ref = digest([case_id, raw, parent, support])
```

`raw` 包含 scope、原始 summary、原生 confidence 与 provenance。因而相同 payload 在不同空间支持、不同 scope 或来源 revision 下不会误去重。`validate_observation` 在打包前从原生 artifact 重建预期内容，检测意外修改或脱离绑定的模型视图。

这是完整性检查，不是医学真伪检查，也不是对恶意插件的安全证明。插件代码仍属于本地信任边界；有能力篡改整个对象并重算 hash 的组件不在这一检查的防护范围。

### 3.2 同一套语义内容，两种布局

`content_encoding: plain` 仍是默认。合法父引用下，普通 answer prompt 保持既有表示；没有改变所有历史方法的提示词。

`content_encoding: shared` 复用仓库已有 `compact_evidence.factor_shared`，提取多条内容中完全相同的嵌套字段。所有类别、低分、否定、单位、未知值、数值类型、缺失和 null 均保留；没有摘要模型，没有根据问题删字段，没有固定截断。

共享布局只作用于已有模型可见视图。密集数组本来就不作为文本像素传入，所以“可逆”指投影后 JSON 内容，不意味着模型读取了完整 mask，更不意味着跨模型 latent 对齐。

父引用统一使用同一套 O0/O1 短标识，planner 与 answer 不再分别维护一套可能脱节的引用。原生字段含 `shared_fields`、`entries`、`columns` 或 `rows` 时，沿用已有碰撞保护并回退普通布局，避免把医学数据字段误当序列化指令。

共享布局不是保证更短的压缩器。小包可能因为 framing 或字段结构反而更长。每次均使用当前模型的实际 tokenizer 预算进行准入；真实 token 数和候选覆盖必须在服务器测量。旧 protected 包仍然受到上下文容量约束，不能保证任何规模的 seed 都能放入。

### 3.3 检测与分割共用空间入口

已有分割 mask 映射不变。检测插件只需将原生坐标转换成以下协议：

```json
{
  "detections": [
    {
      "label": "native structure name",
      "box": [0.2, 0.2, 0.7, 0.8],
      "coordinate_system": "original_image_normalized_xyxy"
    }
  ]
}
```

核心不推测坐标单位，不把空检测变成全图 ROI，也不把检测框当确诊。有效框进入既有 `observe_region` 原子操作，从原图截取局部窗口，调用已有冻结读取器，再将带父引用的观察送回原图回答。

检测框与分割 mask 共享现有 region_limit。只有新 adapter 已有合法权重、原生输出和坐标转换时，才算接入了真实专家。本次没有新增或下载具体检测模型，也没有为某疾病写专用处理。

### 3.4 局部与全图调用不混淆

已有去重只比较 expert/capability/scope，可能把某次 ROI 输出当成已取得该模型全图结果。新判断明确检查是否根观察、support 是否 current_image、是否携带区域字段或已知的 prompt-box/crop 元数据。局部结果不会无故阻止一次全图工具调用；相同全图观察仍避免重复获取。

这不是专家独立性判定：同一模型不同范围的输出仍可能相关，系统没有对它们进行独立概率乘积或多数投票。

## 4. 保持四臂、固定预算和旧方法

实验仍为 `incumbent / plug_read / plug_static / plug_agent`，没有添加第二层 gate 或更多角色。原版配置保持不变，新增 `configs/plug_observe_shared.yaml` 只开启共享布局，调用和 token 上限与原配置相同。

`incumbent` 继续由旧 renderer 在相同中性提示下重生成；历史 CE/OE 条件化分数不能直接用作本次比较基准。`plug_read` 检查接口效应，`plug_static` 检查固定能力链，`plug_agent` 检查冻结动作选择。静态和动态实际调用数可能不同，必须报告实际成本，不因为上限相同就声称计算量完全一致。

更改后的代码身份与配置参与 cache identity；旧结果不能伪装成新运行。无需改旧模型环境、无需引入 LangChain/LangGraph、无需重新训练。

## 5. 下一轮实验与 ICLR 判据

第一阶段只做工程可达性：每类有效观察都能保持内容/支持绑定，被实际 tokenizer 正确计量，并产生真实候选。候选恰好等于 incumbent 仍是一次有效生成；fallback 不是候选。上下文不足必须明确区分 protected 超限、新包超限、父引用未送达、生成未启动、模型空输出。

第二阶段用同一完整 TRAIN 清单、相同权重和工具预算比较 plain/shared。既报告最终指标，也报告 token 使用、送达的原生条目、局部观察覆盖、真实候选率和改善/伤害。完整审计 artifact 不因为模型视图缩短而删除。新布局的额外 framing 也属于开销，不能省略。

第三阶段检验机制而非修补病例：冻结核心接口后，更换未参与开发的同类专家；区分专家自身能力变化与协作收益。用相同工具池、相近预算的 plain ReAct、ReAct+类型检查/来源保留、MedRAX/MedAgent-Pro 移植基线进行比较。移植基线要明确不是原论文完整配置复现。[1–4]

绑定假设的消融应该固定相同数值/文字与预算，再比较绑定正确、绑定元信息移除、离线随机置换绑定。错配只用于受控实验，不能进入正式推理。若正确绑定与错误绑定几乎没有差别，则该机制没有解释力，不应通过追加更复杂模块维护原假设。

重点检查自然的错误专家、正确但不适用的证据、正确却反对初答的证据、相似病例混成患者事实，以及多个派生输出被误当独立支持。只测试人为恶意文本或单个 bad case 不足以支持泛化。

哈希检查、检测框兼容和共享字段布局均不应单独标为算法创新。可争取的贡献是：在固定、无需重新适配的协作机制下，对未见专家仍有效的内容—范围保持，以及对相关失效机制的系统实验。是否足够 ICLR 新颖性取决于这些证据；当前没有医学效果或理论安全保证。

## 6. 验证状态与限制

本地完成新增 `tests/test_plug_bindings.py` 的 40 个 CPU/模拟后端测试、两个运行模块与测试文件的 py_compile、新 CLI help。测试使用合成图像、合成插件与 fake generation callback，不加载医疗权重，不访问目标标签。

覆盖：身份范围/来源区别、内容篡改、父引用、精确类型往返、布局键碰撞、实际预算回调、错误坐标、空检测、检测→原子读取→候选通路、部分范围去重、禁止目标标注、同一地区预算及旧默认表示。

完整当前仓库没有在本地进行全量医学依赖回归；新增 CI 将运行新旧 plug/agent 测试以及带 CPU torch 的全库测试。以实际 CI 结果为准，不用新增测试通过替代全库通过。医学 GPU 运行、真实 tokenizer 压缩量、检测器预测质量、跨医院泛化、可靠性 gate 和知识库覆盖尚未验证。

## 参考资料

[1] Yang et al. AgentOccam. ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/f2c6e459b95694a24ac69c469a4ee746-Abstract-Conference.html

[2] Fallahpour et al. MedRAX. ICML 2025. https://proceedings.mlr.press/v267/fallahpour25a.html

[3] Wang et al. MedAgent-Pro. ICLR 2026 official poster. https://iclr.cc/virtual/2026/poster/10008810

[4] Suris et al. ViperGPT. ICCV 2023. https://openaccess.thecvf.com/content/ICCV2023/html/Suris_ViperGPT_Visual_Inference_via_Python_Execution_for_Reasoning_ICCV_2023_paper.html

[5] Li et al. Visual Evidence Prompting. ACL 2025. https://aclanthology.org/2025.acl-long.205/

[6] He et al. Towards generalizable AI in medicine via Generalist-Specialist Collaboration. Nature Biomedical Engineering, 2026. https://www.nature.com/articles/s41551-026-01653-3 ; bibliographic record https://pubmed.ncbi.nlm.nih.gov/42067582/

[7] LangChain, ToolMessage artifact official API documentation. https://reference.langchain.com/javascript/langchain-core/messages/tool/ToolMessage

[C1] AgentOccam/env.py, commit c078ba629212ea7cee35b2718dc8df72c05e57c7. https://github.com/amazon-science/AgentOccam/blob/c078ba629212ea7cee35b2718dc8df72c05e57c7/AgentOccam/env.py

[C2] MedRAX medrax/agent/agent.py, inspected blob c4b658bdaa1034b23939c81ffd36bdcef5d46ef3. https://github.com/bowang-lab/MedRAX/blob/main/medrax/agent/agent.py

[C3] MedAgent-Pro Task_level.py, inspected blob 7d661ed089106bdafe8b5d7b58040ee9e7aad7f2. https://github.com/jinlab-imvr/MedAgent-Pro/blob/main/Task_level.py

[C4] ViperGPT image_patch.py, inspected blob 9f8cf2777afa0d7ab9ac5b01958e9fa48af5b790. https://github.com/cvlab-columbia/viper/blob/main/image_patch.py

[R1] MERIT plug-observe base. https://github.com/dfdu233/merit-feddg/tree/7381f9687782bc6c6b4a00402a6c608c906c35ef

[R2] MERIT frozen experiment report. https://github.com/dfdu233/merit-feddg/blob/45e450a638c8519eb7ed3de15755ccce5ee43bd3/docs/results/evidence_agent_v1_vqarad_train_2026-09-12/README.md
