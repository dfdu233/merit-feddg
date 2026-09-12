# 简单、可插拔的医学专家协作：调研与实现

## 研究目标与边界

目标不是创建更多 Agent 角色，而是在冻结通用模型、冻结既有专家的条件下，让原生输出保持其任务含义和支持范围。新增模型通过现有 `CapabilityPool` 的 `factory` 和 `infer(request) -> CapabilityResult` 接口接入；新增模块不训练、不校准、不拟合温度或权重，也不根据测试病例更新记忆。

本次代码建立在 `45e450a638c8519eb7ed3de15755ccce5ee43bd3` 上。所有旧源码、配置、实验结果和入口保持不变；新方法使用独立入口和结果目录。所谓 training-free 指协作过程不新增训练，不意味着基础模型或专家没有被预训练。即插即用指复用既有接口、预处理适配和协作逻辑，不保证任意专家在任意图像上有效，更不保证增加专家永不降分。

当前产物是可测试的研究实现，不是已经验证有效的医学系统，也不能从结构图或单元测试推导 ICLR 新颖性成立。

## 1. 已有结果与根因定位

最近完整 TRAIN 实验包含 1,793 个问题、313 个图像聚类；只有 80 个问题取得区域，四个新增分支均没有可用候选。动态分支执行 70 次 crop，却只得到 8 次 inspect。旧 compact_rows 的诊断口径增益约 +1.88 pp，属于原有方法。[R1]

执行版本的 `agent_run.py::synthesize` 先构造 `native + added`，检查旧证据与全部新增证据是否同时送达，不满足便返回 None。底层打包同时约束字符数和真实 token 数。因此，不应把 248 次合成回调统一记为 248 次模型生成失败；具体超限分支还需原始 transport 日志确认。[R2]

另一问题是控制粒度：crop 和 inspect 分别消耗规划决策，但单独创建 crop 文件并没有产生新医学观察。规划视图又未提供有用的区域名称和坐标。新实现不以病例专用规则补洞，而改变共享的观察接口。

## 2. 文献与官方实现对照

| 工作 | 发表 | 已覆盖机制 | 本次采用与未采用 |
|---|---|---|---|
| AgentOccam | ICLR 2025 | 观察与动作空间对齐，避免依赖额外角色/搜索 | 采用可读区域卡与完整观察动作，不复制网页动作或疾病规则。[1] |
| MedRAX | ICML 2025 | 冻结医学工具协作、LangGraph 循环、工具注册 | 复用已有插件接口；不能声称首次医学多工具 Agent。[2] |
| MedAgent-Pro | ICLR 2026 | 疾病级规划、患者级执行、指南 RAG、专业工具与反思 | 作为最近邻。此次不新增疾病级计划库、Coding Agent 或分层评审器。[3] |
| ViperGPT | ICCV 2023 | 可组合视觉模块、带原图坐标的 ImagePatch | 借鉴带身份的空间输入；不运行 LLM 生成的任意代码，不照搬其屏幕坐标语义。[4] |
| VEP | ACL 2025 | 将视觉专家输出作为外部证据交给 VLM | 保留 semantic/compact 强基线；专家转文本不是本次的新贡献。[5] |
| GSCo | Nature Biomedical Engineering 2026 | 通用模型使用专用预测和相似病例上下文 | 原生属性和源病例并非同一任务的投票。全文框架包含开发/训练过的模型，不能整体称为未训练系统。[6] |
| LLMCompiler | ICML 2024 | 显式依赖与可并行工具调度 | 采用“依赖由程序执行”的原则，但本次刻意不加并行调度器，避免单 GPU 资源竞争。[7] |
| BEACON | 2026-08 预印本 | 冻结模型的病理 WSI 证据获取、EIG 与停止决策 | 说明“免训练主动获取证据”也有近邻。本次不加入假设概率/EIG 模块；未核实为已接收顶会。[8] |
| LangChain/LangGraph | 官方软件文档，非论文 | ToolMessage content/artifact 分离、可组合工作流 | 借鉴数据平面与模型上下文分离，不把软件能力包装成新算法，不新增依赖。[9] |

### 直接核对的源码

AgentOccam 的 `AgentOccam/configs/AgentOccam.yml` 明确配置观察输入、允许动作，并关闭 critic 和 judge。论文启发是合适的接口可以优于不断堆叠控制组件，不是所有任务都应取消规划。[1]

MedRAX 的 `medrax/agent/agent.py` 注册工具、绑定模型并循环执行；它把执行结果以 `ToolMessage(content=str(result))` 返回。该入口证明工具流程已有成熟实现，但不能用字符串化结果代替医学 mask 和坐标的程序存储。[2]

ViperGPT 的 `image_patch.py::ImagePatch` 保存父图偏移、边界和中心坐标。医学区域需沿用这种显式空间身份，但患者左右必须有相应元数据，不能由屏幕坐标自动推出。[4]

LangChain 官方 `ToolMessage.artifact` 明确是程序可访问、但不发送给模型的补充数据。这给出了降低审计信息上下文开销的直接工程依据。[9]

## 3. 实现只保留一个观察循环

```
冻结模型选择一个完整观察动作
        ↓
已有专家 infer 或预测区域的 crop + read
        ↓
原生内容 + 支持范围；原始 artifact 单独保留
        ↓
原图上的自由回答
```

新增只有两个运行源码模块：`plug_observe.py` 与 `plug_run.py`。前者不导入任何学习器、优化器或外部 Agent 框架；后者复用既有 CapabilityPool、NativeSession、模型加载器、manifest 和分片检查。没有第二套疾病词典、训练式门控、置信度融合器或长期测试记忆。

### 3.1 插件沿用原协议

已有的 classification、segmentation、detection、retrieval、generation 都可返回原生 EvidenceItem。新插件通过 add-only 的 `--expert-registry` YAML 加入，不能悄悄覆盖已有专家 ID。接口中的 scope、modalities、tasks 和 requires_region 是模型输入契约，不是根据目标答案定制的路由。

主循环只检查这些输入契约；不再使用旧 question_type 疾病/题型模板来筛新插件。需要 ROI 的插件只在有真实预测区域时可用；不制造全图框。分类、生成和整图病例检索不再依赖先出现 segmentation。

“任意新模型零配置”不可实现：权重加载、预处理、类别映射、坐标转换仍由可信的薄适配器负责。薄适配器不是可训练 adapter。未提供正确输出类型的模型不能被假装兼容。

### 3.2 原生内容与支持范围绑定

观察表示为 `(content, support, artifact)`。模型读取原始 payload 的全部标量/文本、scope、专家身份、原生 summary、必要空间支持。未知分数和低分项不被 top-k 删除，也不强行换算为统一正确率。

只移出两种已经明确编码的稠密数组值；编码和尺寸仍保留，并注明模型没有直接看到 mask 像素。原始数组、长哈希、路径和完整 provenance 留在 artifact。未知 native payload 字段默认保留，不凭名称删除可能有医学意义的数据。

由 crop 产生的描述保留 parent、预测区域名称、原图坐标和 scope；被标为同一 generalist 的派生观察，不算独立专家票。retrieval 采用 source_analogy_only，源病例答案仍属于源病例。内容说明不能保证模型永不越权；这正是需要实验评价的部分。

### 3.3 原子观察，不是机械禁止停止

`observe_region` 一次完成 crop 准备和冻结 VLM 读取；规划器看得见区域名称与位置。不会在“文件刚保存、尚无观察”时再支付一次停止判断。STOP 仍可在完整观察之间选择；非法动作是错误，不自动伪装成安全弃权。

局部读取不再重复原始全局医学问题，使用统一的可见解剖/外观观察提示。最终回答读取原图和相应空间支持。该改动是任务表示假设，不能保证局部描述正确；关系、数量、真实体积等新能力仍需要相应专家，不能从普通 mask 自动得出。

### 3.4 真实预算与候选可达性

`pack_observations` 使用已有后端的真实 token 计数（包含图像 token 预算），不用字符长度代理模型上下文。每个观察是不可分割的整体，不能为了装入而截断否定词、单位或分数列表。

初始证据是 protected；若初始内容自身超限，明确记录 `protected_context_exceeds_budget`。新观察按固定顺序尝试准入：一个大包不适配，不会阻止后续小包；每次排除都记录原因和预算。派生观察的 parent 没有送达时，也不会单独投递派生结论。

最终分别记录 `no_new_observation`、`no_new_packet_fits`、生成是否真实开始、空输出和实际候选。保留旧答案不是临床 gate 成功。新候选允许与旧答案相同，但必须是模型真实读到新证据后生成，不能用 fallback 冒充 candidate。

此处的新提示和表示没有临床真实性判别器；观察准入与真假判断不是同一件事。

## 4. 保留旧方法与新实验可比性

所有旧方法文件不改，既有 `compact_rows`、`semantic_all`、`compact_all` 和 Agent-v1 结果目录不覆盖。新增 .py 会改变旧 runner 的全源码缓存指纹，因此应使用独立 worktree，而不是把新文件直接复制进正在运行旧实验的目录。

新入口读取完整旧 no-gate 结果，用于取得原生专家缓存和原有图像模态路由；不将旧答案文字放进新提示。新方法所有题统一自由生成，answer_type/参考答案不进入工具与规划器。

为了与这个要求一致，四个实验臂都在相同中性提示下生成：

| 实验臂 | 用途 |
|---|---|
| incumbent | 原有 compact/semantic 实现，同样的中性提示，重新生成基准 |
| plug_read | 只更换模型视图；不调用新专家，隔离接口效应 |
| plug_static | 固定顺序执行相同可用能力，强基线 |
| plug_agent | 同一工具与预算，由冻结模型选择合法观察 |

原来按 CE/OE 附加不同后缀的历史分数不能与这张新表直接相减。旧 checkpoint、证据和算法均保留，但这个新协议的 baseline 需要真实重生成。原始历史输出继续留在原目录。

无独立“全未知 gate”实验臂：不再用永远保留 incumbent 的退化门控宣称可靠性改进。当前只比较候选生产、接口和能力使用；论文级 gate 仍未完成。

### 成本与来源

新调用与继承成本分开记录。历史 seconds 包含旧回答生成，仅可作为源端准备成本上界，不应直接累加为准确部署延迟。基准模型初始化单列；同进程共享 warm weights，实验顺序影响冷/暖加载，不能把简单均值写成独立冷启动性能。

新 retrieval 只有提供经过隔离检查的 source manifest 后才开启。新调用保留插件原生源端文本和来源标记；不会继承旧实验用于消融的“隐去源答案”设置而把 RAG 变成只有指针。历史 seed 仍按其原有呈现策略重建，新增部分与基线内容变化须独立报告。

source manifest 沿用现有 train/external 许可、图像/组/患者/study 重叠检查。缺失患者 ID 时没有患者级保证；近重复仍需独立排查。此版本不自动下载或构造医学知识库。

## 5. 实验与创新性判据

研究假设是：一部分协作失败来自原生内容与支持范围脱离，而不只是专家准确率低。它尚未被本轮单元测试证明。

先比较 plug_read 与同提示 incumbent，再比较固定完整观察与额外整图观察，最后比较 agent 与 fixed。`observation_view: whole` 是独立配对对照，必须使用同一完整清单、单独输出和同样预算；它记录实际观察窗口为整图，不伪装成局部证据。

未见专家实验需冻结核心代码、提示、预算和源库后再加入新 ID；只允许预处理/坐标的薄适配。应同时报告新专家原生任务质量，避免把更强模型的贡献归于协作机制。还需与 MedRAX/MedAgent-Pro 风格的同骨干匹配实现和“ReAct + schema 检查 + 可读原子操作”强基线比较。

后续 scope 绑定消融应保持相同实际内容和 token 预算，只改变绑定关系或移除冗余描述，并报告内容送达差异。仅增加 source 字段、使用 LangChain artifact 或合并 crop/inspect 都不足以单独证明创新；需要真实医学质量、伤害/救回和跨专家泛化结果。

临床正确、操作成功、模型接收和回答使用是不同事件。当前工程 tests 不提供临床安全保证；动态停止与模型低 entropy 也不等于正确。

## 6. 运行与验收

先在独立 worktree 复用本地权重与环境，不升级、不重新下载。变量必须指向同一完整官方 TRAIN 清单和已完成的 no-gate 基准；不能用停跑快照代替完整结果。

```bash
python -m pytest tests/test_plug_observe.py tests/test_evidence_agent.py -q
python -m merit_feddg.plug_run --help
python -m merit_feddg.plug_run \
  --base-run "$BASE" --incumbent compact_rows --manifest "$MANIFEST" \
  --artifacts "$ARTIFACTS" --output "$OUT" --check-only
```

去掉 `--check-only`，添加 `--canary-cases 2` 进行同一清单上的调度 canary。检查区域/整图插件是否真实运行、候选是否进入生成、protected 是否超限；不能因为“没有异常”就继续放量。如果前两例没有适用动作，应报告原因并检查完整清单的能力覆盖，不能按得分挑选有利病例。

确认后移除 canary 参数恢复同一完整清单。只在两张卡均明确授权时，用 `--shard-index 0/1 --shard-count 2` 分工；全完成才 `--merge-only`。不在推理时读取 references。

```bash
python -m merit_feddg.agent_evaluate --run "$ROOT" \
  --manifest "$MANIFEST" --references "$REFERENCES" \
  --anchor-root /home/dbw/ANCHOR --output "$ROOT/evaluation.json"
```

旧 evaluator 的诊断和 ANCHOR 指标分别保存，不更换评分器，不把 token recall 称临床准确率；报告任务不能使用该 VQA scorer。输出的逐动作 events 还需单列专家调用，不能仅用语言模型 calls 数代替所有专家工作。

目前本地验证为 CPU 合约与模拟后端，不包含真实 LLaVA-Med 或其他医学权重。GitHub CI 应另行检查当前提交结果；CPU CI 也不是医学 GPU 效果复现。没有完整真实效果前，保持草稿 PR，不自动合并。

## 参考来源

[R1]: https://github.com/dfdu233/merit-feddg/blob/45e450a638c8519eb7ed3de15755ccce5ee43bd3/docs/results/evidence_agent_v1_vqarad_train_2026-09-12/README.md
[R2]: https://github.com/dfdu233/merit-feddg/blob/45e450a638c8519eb7ed3de15755ccce5ee43bd3/merit_feddg/agent_run.py
[1]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/f2c6e459b95694a24ac69c469a4ee746-Abstract-Conference.html
[2]: https://proceedings.mlr.press/v267/fallahpour25a.html
[3]: https://iclr.cc/virtual/2026/poster/10008810
[4]: https://github.com/cvlab-columbia/viper
[5]: https://aclanthology.org/2025.acl-long.205/
[6]: https://www.nature.com/articles/s41551-026-01653-3
[7]: https://proceedings.mlr.press/v235/kim24y.html
[8]: https://arxiv.org/abs/2608.05757
[9]: https://docs.langchain.com/oss/python/langchain/messages

代码核查入口：
- AgentOccam: https://github.com/amazon-science/AgentOccam/blob/c078ba629212ea7cee35b2718dc8df72c05e57c7/AgentOccam/configs/AgentOccam.yml
- MedRAX: https://github.com/bowang-lab/MedRAX/blob/main/medrax/agent/agent.py
- MedAgent-Pro: https://github.com/jinlab-imvr/MedAgent-Pro/blob/main/Task_level.py
- ViperGPT: https://github.com/cvlab-columbia/viper/blob/main/image_patch.py
- LangGraph 工作流边界: https://docs.langchain.com/oss/python/langgraph/workflows-agents

发表状态以正式会议/期刊页面为准，预印本单列；main 链接仅代表本次所读版本，不冒充固定最终会议源码。新实现不包含复制的上游模型权重或数据。
