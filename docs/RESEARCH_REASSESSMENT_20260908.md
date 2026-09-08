# 2026-09-08：负面结果复盘与有边界的增量修复

## 1. 保留研究目标

主体是冻结医学通用 VLM 在生成中调用异构专用小模型，接收其原生证据后继续生成。
域泛化是专家证据的适用性/采纳门控，不改造成一个单独的图像扰动方法或 KV 平滑论文。
当前实现是工程与机制探索，尚不能宣称临床收益、通用性已验证或达到 ICLR 接收标准。

## 2. 实验支持什么、不支持什么

依据远程同步的 `4d05a4a` STATUS.md：6 个 source 病例，0 target，全部 proxy 域。
详细 runs 文件在服务器路径，未随本次 Git 同步；这里不假装读过全部原始轨迹。

| 观测 | 较直接的解释 | 不能据此得出的结论 |
| --- | --- | --- |
| 复制同一图像的拼图也会把 axial 改成 coronal | 输入布局/缩放/提示本身构成混杂因素 | 多图读入成功就是视觉协作成功 |
| 问器官、参考 breasts，肺/心分割后答 heart | 工具输出标签范围不能回答所有器官识别问题 | 分割模型不准确或域风险高 |
| crop 与等面积 control 都答 heart | 尚无区域特异性收益 | 模型利用了正确的病灶位置 |
| gross specimen 被 CONCH 当组织切片 | 路由范围错误；组织外观目录不含目标任务答案 | 增强专家信任就会改进 |
| gamma 下敏感性仅约 0.009–0.018，回答仍错 | 弱扰动稳定不是正确性 | 放宽阈值可证实域泛化 |
| 专家说 consolidation/atelectasis 但缺左心后区 | 疾病概念和空间定位是不同能力 | 用 F1 衡量完整医学事实可靠性 |

最初原图经预处理进入模型；改成左右拼图会改变视觉编码器实际看到的布局和分辨率。
“拼图构建前像素没丢失”不等于“模型输入表达不变”。保留服务器兼容修复与旧对照，
但本轮 profile 默认只输入原图，不默认启用 panel/crop。

## 3. 重新核实的相关工作（2026-09-08）

### KVSmooth：不是当前错误的一剂通用药

[CVPR 2026 论文](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_KVSmooth_Mitigating_Hallucination_in_Multi-modal_Large_Language_Models_through_Key-Value_CVPR_2026_paper.html)
与 [arXiv 正文](https://arxiv.org/html/2602.04268v1)：注意力熵引导缓存平滑，主要针对
生成轨迹漂移。论文可核实；本轮未定位并验证官方可运行仓库，不伪称已复现。
借鉴其诊断先于干预的思路，但它不能证明一开始错误的组织/器官判断可以被平滑修好。
把视觉 Key 直接与文本位置的 Key 平均，还需要处理位置编码、坐标映射和语义空间；
不能把分享对话中的建议公式直接当可用实现。

### VEP：文本并非天然无效

[ACL 2025 论文](https://aclanthology.org/2025.acl-long.205/)将小视觉模型的输出符号化
供大模型使用。它是最直接的协作基线；不能因为新方法用了文字就预判其无效，
也不能仅靠“输出结构化”宣称创新。本轮未核实其官方完整可运行仓库。

### Draft and Refine：借鉴问题相关性，不照搬事后重写

[官方仓库](https://github.com/EavnJeong/Draft-and-Refine-with-Visual-Experts)
标注 CVPR 2026 Highlight；已查看
[process/uq.py](https://github.com/EavnJeong/Draft-and-Refine-with-Visual-Experts/blob/master/process/uq.py)。
其遮挡后重生成、回答嵌入相似度计算有真实开源实现，不能称为无额外前向的免费指标。
借鉴“问题相关证据”和按模型处理输入的原则，不把多次事后修订改成我们的主流程。

### HaloProbe：内部信号需要验证混杂因素

[论文](https://arxiv.org/abs/2604.06165)明确讨论 token 位置和重复实体对注意力统计的
影响，并使用训练得到的因素分解。它不是可直接拿来作免训练可靠性证书的熵阈值。
本轮以作者论文为依据，没有单独核实其会议录用状态或官方可用权重。

### MoBE：不能再声称没人考虑专家的未知域问题

[官方仓库](https://github.com/BioMedIA-MBZUAI/MoBE-A-Test-Time-Modality-Generalization-Method)
标注 MICCAI 2026，提供专家权重链接；已查看
[mobe.py](https://github.com/BioMedIA-MBZUAI/MoBE-A-Test-Time-Modality-Generalization-Method/blob/main/mobe.py)。
它研究模态专家的测试时路由和统计适配，与我们的异构原生能力接口不同。
因此“域泛化首次进入专家协作”不是可辩护的主张。应验证我们在固定接口下的
专家适用性门控能否保留有效收益、降低跨来源的负迁移。

没有复制以上仓库代码、导入其依赖或声称完成 baseline 复现。

## 4. 简洁形式化

状态 s=(原图 x，问题 q，已生成前缀，既有证据)。
定义 C_i(q) 为任务/概念是否落在专家 i 声明的能力范围，D_i(x) 为已有源域
参考门控对当前图像的适用性判断。两者不是同一个分数：

    eligible_i = C_i(q) AND D_i(x)
    E_i = S_i(x, request)
    adopted_i = relevant_native_observations(E_i, request)
    y_next = G(x, q, prefix, adopted evidence)

控制器只从 eligible 工具中选择或 NONE。诊断分支可以不启用 D，但必须明确
这只是任务范围/证据通道实验。域门控代码不作放宽；行为探针也不转成正确概率。

本轮新增 request_contract：named_concepts 要求问题显式提到标签或声明的别名；
free_query 只允许生成/检索工具。概念来自问题而非目标答案，不推断阳性/阴性。
focused 证据样式只保留这些概念的原生输出。不存在相应输出则留空，禁止回退到
最高分的无关结果。比如问 cardiomegaly，保留其原始低分也不让 infiltration 抢答。

这是一个透明的有限覆盖修复，不是“学到了医学语义”或论文新理论。
泛化的下一步可让主模型产生受约束的证据需求，但必须单独验证需求是否正确，
不能把它猜测的诊断当事实。当前用精确词组/别名，英文/复合问题/隐含需求仍有限。
支持其中一个概念不代表能回答整个复合问题；系统只能提供相应的局部证据。

## 5. 这轮明确不做的事

- 不改 LLaVA-Med 的 KV、attention heads 或模型参数；不添加无依据的锚点。
- 不把方法改为纯二分类，不把目标改成只做 CXR。CXR 是有现成小模型的首轮验证。
- 不声称词表过滤解决 gross→histology 路由错误；图像类型仍须独立审核。
- 不重命名 proxy 为医院、不根据 target 提升调门控、不把零调用写成成功。
- 不删除之前的空间专家、非文本接口、行为探针与原始实验结果。

## 6. 服务器 Codex 执行任务

读取本文件、STATUS.md、configs/request_scoped_pilot.yaml。
检查工作树并保留本地修改，安全拉取；复用 huatuo、LLaVA-Med 与已有专家，不升级/下载。
先重跑上一轮固定 6 个 source 病例，使用下面新配置而非旧 panel 配置：

```bash
cd /home/dbw/merit-feddg
PY=/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
# 选择实际空闲的 GPU；这里不自动占用卡。
"$PY" -m merit_feddg.llava_run --study diagnose \
  --config configs/request_scoped_pilot.yaml --skip-download \
  --source-manifest /实际路径/上一轮source清单.jsonl \
  --target-manifest /实际路径/隔离target清单.jsonl \
  --references /实际路径/references.json \
  --llava-source /实际路径/llava-med-1.5 \
  --output runs/request-scoped-source
```

占位路径必须以服务器文件为准。diagnose 入口仍需 target 清单作隔离检查，不生成 target。
对照共享同一次专家执行，比较 native_text、scoped_text、focused_text；不生成拼图。
新配置不开行为探针，不训练 value policy，不采集/更新 target 统计。
诊断为公平比较仍执行范围外专家的原始分支；focused 分支不采纳它。
部署运行则在调用前拒绝。不得从诊断重放时间宣称节省了在线调用成本。

检查输出中的 request_coverage 与逐分支 request_scope；每个拒绝必须有原因。
若 named concept 没有对应原生标签，属于接口覆盖不足，不允许猜测补标签。
逐例审阅：图像类型、问题所需能力、真实专家输出、证据是否相关、回答是否完整正确。
保留原 6 例结果，不用修改问题或删掉困难病例来制造改进。

第二批另建预先冻结、按真实数据问题分层的 source confirmation 清单：
显式疾病概念问题、专家实际支持的解剖定位问题、范围外问题均保留；不按模型得分挑选。
若可用数据没有足够任务匹配的题目，报告数据覆盖不足，不自己把原题改成容易题。
报告全体和覆盖子集，拒绝率、回答受损率及专家使用后的医学收益同时呈现。

只有确认存在实际有效证据后，再用现有 ApplicabilityGate 比较无 DG/有 DG，
所有阈值 source-only 校准，患者隔离，真实来源留出。需要至少多少来源仍按
docs/INPUT_APPLICABILITY.md 的 LODO 支持要求，不按结果降低数量。
新接口改变了干预分布，必须重新采集源域 memory，不复用旧词面误差卡冒充同一策略。

运行测试/Ruff；更新 STATUS.md，提交代码修复及真实小结果摘要并推送。
原始医学图片、权重、凭证和大型缓存不入库。
