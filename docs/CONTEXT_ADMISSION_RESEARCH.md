# 多模型协作中的上下文污染：文献、假设与冻结实验

## 问题与边界

研究问题不是“如何相信专家”，而是：在问题 q、原图 x、原生能力定义 c 和实际返回的证据 e 下，哪些 e 应进入回答上下文？相关不等于有用，有用不等于正确，冲突也不等于错误。假设翻转是：从“已路由到的专家输出默认可作为回答依据”改成“输出仍须证明其用途处于该专家原生能力范围内”。这只是待检验假设，不是已成立的新算法贡献。

冻结的三个问题：

1. 哪种适用性判断能减少污染而不阻止有价值的纠错？
2. 同一个通用模型兼任判断者有哪些偏差？
3. 不训练、不设数值准入阈值时，如何排除全部拒绝和 fallback 造成的假改善？

## 调研方法

按上下文过滤、判断者失败、多医学模型协作三个相互质疑的视角检索，优先核对正式会议页面和作者仓库。区分论文机制、源码实际行为与本轮实现；未核实的会议归属不作为顶会证据。检索覆盖至本轮服务器日期 2026-09-19；这不是证明不存在其他相关工作的穷尽性检索。

## 证据关系图

| 方向 | 代表证据 | 对本问题的启示与冲突 |
|---|---|---|
| 无关内容干扰 | [Shi 等，ICML 2023](https://proceedings.mlr.press/v202/shi23a.html)；[I³C，NAACL 2024](https://aclanthology.org/2024.naacl-long.379/) | 先识别无关条件已有先例，不能把 judge-before-answer 当原创。 |
| 相关性与效用过滤 | [Self-RAG，ICLR 2024](https://openreview.net/forum?id=hSyW5go0v8)；[RECOMP，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/bda88ed2892f5e61c9a9bf215c566913-Abstract-Conference.html) | 相关性、支持性和效用需要区分；正式方法涉及训练，不能声称本轮复现完整方法。 |
| 清洁上下文 | [S2A，2023 预印本](https://arxiv.org/html/2311.11829v1) | 重建上下文可减弱干扰，但也可能去掉有益提示；同时保留原污染上下文会削弱过滤目的。 |
| 医学外部证据依赖 | [RULE，EMNLP 2024](https://aclanthology.org/2024.emnlp-main.62/)；[MMed-RAG，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a559a5a8aa5ae6682ced009ad97cdb16-Abstract-Conference.html) | 外部证据会破坏本来正确的回答；其校准或偏好训练不符合本轮限制。 |
| 医学工具输出审核 | [MedRAX，ICML 2025](https://proceedings.mlr.press/v267/fallahpour25a.html)；[MedAgent-Pro，ICLR 2026](https://openreview.net/forum?id=ZOuU0udyA4) | 通用模型协调专家、再判断工具输出已经存在。本轮只做更窄的用途准入，不认证医学真实性。 |
| 冲突与判断者偏差 | [What Evidence Do Language Models Find Convincing?，ACL 2024](https://aclanthology.org/2024.acl-long.403/)；[Conflicting Needles，EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1742/)；[Self-Preference，EMNLP 2025](https://aclanthology.org/2025.emnlp-main.86/) | 表面相关、顺序和表达可影响信任；同模型判断不能视为独立验证。 |
| 医学多模态冲突 | [CoRe-MMRAG，ACL 2025](https://aclanthology.org/2025.acl-long.1583/)；[MMKC-Bench，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39385) | 发现冲突与正确解决冲突是两件事；同一提示中要求“先不看证据”不等于真正隔离上下文。 |

### 官方代码核对及采用范围

- [Self-RAG relevance prompt](https://github.com/AkariAsai/self-rag/blob/1fcdc420e48f50a7d7ab1ece5494221b93252e99/data_creation/critic/gpt4_reward/chatgpt_relevance.py)：借鉴分类式、问题条件化的相关性判断；该数据创建脚本存在提示与解析不一致，未直接复制，不采用其训练或加权推理。
- [RECOMP](https://github.com/carriex/recomp/tree/51d4432151efb3275257a9407dc71d1e5ec6634d)：训练压缩器允许无有用信息时为空；本轮不压缩或改写原证据。
- [I³C](https://github.com/wzy6642/I3C-Select)：核对 run.py/prompt.py；其条件选择支持先过滤的先例，不等价于医学专家能力约束。
- [MedAgent-Pro](https://github.com/jinlab-imvr/MedAgent-Pro)：Examiner.py 已用原图和工具结果请求模型审核；不采用其疾病任务评估启发式，也不声称审核标签是正确性置信度。
- [CoRe-MMRAG](https://github.com/iLearn-Lab/ACL25-COREMMRAG)：scripts/our_methods.py 在同一上下文组织检查。本轮物理分离判断与最终回答会话，而非仅要求模型忽略已看到的内容。
- [CRAG](https://github.com/HuskyInSalt/CRAG/tree/de7c2961ae624a1483a138c5798e1f6d0c4fb0e0) 和 [FILCO](https://github.com/zorazrw/filco/tree/4fe80c6a9c5dbd6aab8b12c8c0f31533050d0ec7)：检查到训练或数值阈值，未移植；后者空过滤后回退完整上下文也不符合本轮过滤审计目标。
- [SCARLet，EMNLP 2025](https://aclanthology.org/2025.emnlp-main.33/) 的[代码](https://github.com/ylXuu/SCARLet/tree/871c3756c98bf7e40d5e94c8fb6e4127a613fc9f)使用真值和消融计算训练归因，提示证据存在组合效应，不是可直接用于无真值推理的 Gate。

## 最小实现与预注册比较

核心保持原 MERIT 的模型、路由、专家输出和回答生成。只改变送达视图：

1. 以历史 compact 真正送达的原生包为固定全集 E0，不能因过滤腾出空间而引入原先未送达的包。
2. 通用模型在独立会话中看原图、问题、专家能力定义、单个原生包；不看 Baseline 答案或参考答案。
3. relevance 对照输出相关/无关/未知；scope 输出直接回答依据/仅辅助操作/不相关/未知。使用有限类别解码，不拟合温度或设数值准入阈值。
4. 仅直接依据进入新的回答会话；保持原始证据不变。拒绝内容、审计理由和分类字母都不进入回答提示。
5. 全部保留时复用 compact，全部不保留时复用 generalist，并明确记录。这是精确缓存复用，不是新候选或 Gate 成功。

五臂为 generalist、compact、relevance、scope、scope_complement。最后一臂仅使用被 scope 排除的包，是反证性诊断而非部署策略。没有为本实验训练任何模型，也没有按测试集选择规则。

采用现有完整 VQA-RAD TRAIN 1793 行，复用同清单基准。它包含与 test 共用图像，不能据此声称独立 holdout 收益。历史专家为 CheXagent 文本、CXR 解剖分割和 BiomedCLIP 解剖分类，不代表新扩充专家池或其他骨干。原始生成预算 64 tokens，与正在其他会话开展的 1024-token 正式测试并非同一生成协议，不混表。

先完成旧 24 例 revision 的终态审计；空输出保留空，预算挤出证据的臂记 unavailable，不伪造输出。另行固定当前评分器，对所有臂统一离线评分；历史评分身份不修改。新 Gate 通过 CPU 回归和 5 例真实 parity 检查后执行双卡完整 TRAIN，复用完成病例。评分必须等待精确 ID 全集完成。

主要观察是相对 compact 的分数变化、改善/伤害、原收益保留和原伤害恢复，同时报告准入分布、真实调用、候选生成、缓存复用、空输出、额外成本和按图像聚类的置信区间。任何全部准入/拒绝只能说明行为，不能证明可靠判断。

## 反证与残余限制

这只是包级而非条目级过滤：一个被准入的多标签包仍可能携带无关条目。独立逐包判断会忽略多包组合效用；同模型也可能共享错误和先验偏见。有限类别选择不是校准概率。缺少人工用途标签时不能报告 Gate 正确率。额外判断消耗模型调用，并且当前并发任务使 wall-time 不是独占硬件基准。

R1 暂不能由文献保证：需要观察收益保留与伤害减少能否同时发生。R2 的偏差不能仅凭换提示消除，本轮通过不展示原答案降低自我一致性诱因，而不声称独立 verifier。R3 由五臂、真实送达审计及成本记录作有限检验，仍缺少更强的随机等长、人工适用性和组合效用对照。本轮先完成这一问题，不叠加 uncertainty、训练 Gate、新专家或 RAG。
