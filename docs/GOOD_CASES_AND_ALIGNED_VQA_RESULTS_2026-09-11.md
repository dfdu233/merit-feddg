# 典型医学专家纠错案例与 VQA-RAD/SLAKE 对齐结果

日期：2026-09-11

本文完成两件事：整理当前最有代表性的专家纠错案例；将远程评测产出的原始答案重新交给 merit-feddg 本地 matched evaluator 评分，与本地新方法结果形成同口径表格。

## 对齐规则

本次没有引用远程目录中预先计算的分数作为最终数值，而是读取各方法的 `answers.jsonl`，重新执行本地规则：

- 完整测试集：VQA-RAD 451 题（251 CLOSED、200 OPEN），SLAKE 2,094 题（836 CLOSED、1,258 OPEN）。
- `Strict mixed`：CLOSED 按本地 leading Yes/No strict 规则计 0/1，OPEN 按 reference answer-token recall，最后按全部样例加权平均。
- `内容诊断`：CLOSED 使用冻结的 MedHEval v11 content-aware parser，OPEN 使用相同的 answer-token recall。
- evaluator：`medheval-decoded-eval-v11-explanatory-binary-source-audited`，源码 SHA-256 为 `47587c200cef7b6b182f6788b85c1c021d1cd83ad6878ffcdcd8f5ae3f14f7c3`。
- 本地 matched generalist 与远程 Greedy 原始文本逐条相等：VQA-RAD `451/451`，SLAKE `2094/2094`。因此两个实验的基线输入与输出已经实证对齐，不是仅凭名称判断。
- 不读取目标答案进行生成、门控或阈值选择；答案只在离线评分阶段使用。

SLAKE 的 `CLOSED` 不全是 Yes/No 问题，因此 `Strict mixed` 会把一部分非二元 CLOSED 题按本地兼容规则计分。为避免把解析效应误称为准确率，SLAKE 必须同时报告 `内容诊断`、`CLOSED 内容诊断` 和 OPEN recall。

## 典型案例一：CONCH 将肌层纠正为肿瘤组织

![TCGA 食管肿瘤病理切片](assets/good_cases_2026-09-11/pathology_tumor_tcga_case044.png)

| 字段 | 内容 |
|---|---|
| 数据集 | PathoROB Tolkach ESCA |
| 医疗中心 / 病例 | `VALSET3_TCGA` / `case_044` |
| patch | `tumor.3719` |
| 标准类别 | `tumor tissue` |
| 通用模型 | OpenMed/Qwen2.5-3B-MedVL |
| 通用模型输出 | `muscularis propria`，错误；confidence `0.1829` |
| 专家 | MahmoodLab/CONCH，revision `f9ca9f877171a28ade80228fb195ac5d79003357` |
| 专家输出 | `tumor tissue`；confidence `0.6873` |
| Med-DEFER 输出 | `tumor tissue`，正确 |
| gate weight | `0.1680`（融合权重，不等同于概率置信度） |
| OOD | pre-call `0.0`，post-call `0.0` |
| 打乱证据对照 | `regressed tumor tissue`，错误 |

该例同时满足“通用模型错、完整方法对、打乱证据错”。因此纠错依赖正确的专家证据内容，而不是仅因发生了专家调用。

## 典型案例二：跨医院食管黏膜纠错

![WNS 食管黏膜病理切片](assets/good_cases_2026-09-11/pathology_oesophageal_mucosa_wns_case018.png)

| 字段 | 内容 |
|---|---|
| 医疗中心 / 病例 | `VALSET2_WNS` / `case_018` |
| patch | `sh_oes.3685` |
| 标准类别 | `oesophageal mucosa` |
| 通用模型输出 | `muscularis propria`，错误；confidence `0.1837` |
| CONCH 输出 | `oesophageal mucosa`；confidence `0.8353` |
| Med-DEFER 输出 | `oesophageal mucosa`，正确 |
| gate weight | `0.0564` |
| OOD | pre-call `0.8571`，post-call `0.4286` |
| 打乱证据对照 | `muscularis propria`，错误 |

该例说明较小的专家融合权重也能改变一个接近决策边界的通用模型答案。但 pre-call OOD 很高，不能把此例解释为门控已经校准良好。

48 个 PathoROB 跨中心 patch 上，Generalist accuracy 为 `18.75%`，CONCH always-on 为 `54.17%`，Full Med-DEFER 为 `33.33%`，shuffled evidence 为 `27.08%`。Full 相对 shuffled 有 3 个 rescue、0 个 harm，但 paired sign-test `p=0.25`，属于机制信号而非统计显著结论。

## 典型案例三：BiomedCLIP 纠正 CT 器官识别

![SLAKE CT 肝脏识别样例](assets/good_cases_2026-09-11/slake_ct_liver_0250.jpg)

| 字段 | 内容 |
|---|---|
| 数据集 / ID | SLAKE / `slake-official-test-0250` |
| 问题 | What organ is the gray part on the left of the image? |
| 标准答案 | `Liver` |
| Generalist | “... is the spleen.”，错误 |
| 专家 | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`，revision `9f341de...` |
| 能力 | whole-image major-anatomy classification |
| 专家前三项 | liver `0.3996`；spleen `0.3399`；breast `0.3267` |
| Compact-all | “... is the liver.”，正确，评分 `0 → 1` |
| Compact-verified | 恢复为 “spleen”，错误 |

这些数值是固定目录内的 relative similarity，不是校准后的疾病概率。该例能够清楚归因给分类专家，因为 `biomed_anatomy` 是唯一被采用的专家；同时它暴露了 answer verifier 将正确修订错误回退的问题。

## 典型案例四：胸片左右定位被纠正，但不是纯分割收益

![SLAKE 胸片肺炎定位样例](assets/good_cases_2026-09-11/slake_cxr_pneumonia_0514.jpg)

| 字段 | 内容 |
|---|---|
| 数据集 / ID | SLAKE / `slake-official-test-0514` |
| 问题 | What part of the lung is the pneumonia located in? |
| 标准答案 | `Lower Left Lung` |
| Generalist | `right lower lobe`，左右侧错误；open recall `0.667` |
| 分割专家 | `xrv/pspnet-chestxray`：Left Lung、Right Lung、Heart masks |
| 生成专家 | `StanfordAIMI/CheXagent-2-3b`：输出 `Lower Left Lung` |
| Compact-all | `left lower lobe`，open recall `1.0` |

这个案例必须标为“胸片分割 + CheXagent 联合纠错”，不能标成分割模型单独纠错。`semantic_all` 在未采用 `cxr_anatomy` 的情况下也能答对，而且 CheXagent 直接给出了标准位置；现有反事实证据说明分割不是必要条件。

## VQA-RAD：远程原始答案按本地 evaluator 重评分

所有数值均为完整 451 题；括号中是相对同表 Greedy 的百分点变化。

| 方法 | Strict mixed | 相对基线 | 内容诊断 | CLOSED strict | CLOSED 内容 | OPEN recall |
|---|---:|---:|---:|---:|---:|---:|
| Greedy / matched generalist | 48.251% | — | 48.473% | 60.956% | 61.355% | 32.307% |
| ICD | 46.936% | -1.315 | 47.158% | 58.964% | 59.363% | 31.842% |
| VCD | 42.453% | -5.798 | 44.892% | 49.004% | 53.386% | 34.231% |
| DoLa | 46.549% | -1.702 | 46.549% | 57.769% | 57.769% | 32.468% |
| MMedPO | 49.906% | +1.655 | 49.906% | 60.159% | 60.159% | 37.039% |
| MedRAG | 39.481% | -8.770 | 39.481% | 46.614% | 46.614% | 30.530% |
| Ours semantic-all | **51.351%** | **+3.100** | 53.790% | 63.347% | 67.729% | 36.297% |
| Ours compact-rows | 50.187% | +1.936 | **53.957%** | 61.355% | **68.127%** | 36.172% |
| Ours compact-all | 50.409% | +2.158 | **53.957%** | 60.956% | 67.331% | **37.172%** |
| Ours compact-verified | 49.992% | +1.741 | 52.209% | 61.753% | 65.737% | 35.232% |

按本地 Strict 主列，当前新方法最佳臂是 semantic-all，较相同 Greedy 基线提高 3.10 个百分点；MMedPO 提高 1.66 点。Compact-verified 低于三个未仲裁证据臂，说明当前 verifier 丢失了一部分有效修订。

## SLAKE：远程原始答案按本地 evaluator 重评分

所有数值均为完整 2,094 题。由于 SLAKE CLOSED 含非二元答案，论文主分析不应只解释 Strict 一列。

| 方法 | Strict mixed | 相对基线 | 内容诊断 | CLOSED strict | CLOSED 内容 | OPEN recall |
|---|---:|---:|---:|---:|---:|---:|
| Greedy / matched generalist | 32.914% | — | 34.681% | 42.344% | 46.770% | 26.648% |
| ICD | 34.563% | +1.648 | 33.273% | 49.522% | 46.292% | 24.622% |
| MMedPO | **48.350%** | **+15.436** | **46.535%** | **67.464%** | **62.919%** | **35.648%** |
| MedRAG | 21.339% | -11.575 | 27.166% | 26.316% | 40.909% | 18.033% |
| Ours semantic-all | 33.704% | +0.790 | 39.339% | 36.364% | 50.478% | 31.936% |
| Ours compact-rows | 33.516% | +0.602 | 39.486% | 36.005% | 50.957% | 31.862% |
| Ours compact-all | 33.620% | +0.706 | **39.590%** | 36.124% | **51.077%** | **31.956%** |
| Ours compact-verified | 32.508% | -0.406 | 36.329% | 38.756% | 48.325% | 28.356% |

SLAKE 的解释分成两层：

- 完全复现本地 `Strict` 时，compact-all 仅比 Greedy 高 `0.706` 点，compact-verified 低 `0.406` 点。
- 对非二元 CLOSED 使用 v11 内容解析后，compact-all 比 Greedy 高 `4.908` 点；但这仍是自动内容诊断，不是临床专家评分。
- MMedPO 在两个口径下均明显最高。当前结果不支持声称新方法在 SLAKE 上超过全部对比方法。

## 尚不能进入完整对齐表的方法

当前检查到的 SLAKE VCD、DoLa、OPERA、PAI、AvisC、VISTA 文件只有 1,536 条 fine-grained 子集结果，不是冻结的 2,094 条完整 test。VQA-RAD 的 OPERA、PAI、AvisC、VISTA 也只有 200 条 OPEN 子集文件。即使它们来自同一模型和生成框架，也不能把缺失样例补零或与完整测试表直接合并。

VQA-RAD 的 VCD 与 DoLa 各有 451/451 完整答案、相同 full-test manifest 和通过的生成结构审计，因此已在上表中使用本地 evaluator 重评分。不同方法固有的解码参数、MMedPO 权重以及 MedRAG 检索上下文属于方法定义，不会被强行改成 Greedy；对齐的是数据、样例顺序、基础评测任务、输出后处理与最终评分器。

## 可复核产物

- 机器可读对齐表：[`results/aligned_vqarad_slake_local_rescore_2026-09-11.json`](results/aligned_vqarad_slake_local_rescore_2026-09-11.json)
- 机器可读案例记录：[`results/typical_good_cases_2026-09-11.json`](results/typical_good_cases_2026-09-11.json)
- VQA-RAD 本地完整评测摘要：`runs/matched-verified-packets-anchor/72f36cd3907456c90166699cd0a7fe007de3eff937f6696e04f41a0dae5e5082/evaluation-summary.json`
- SLAKE 原始完整输出：`runs/slake-matched-verified-packets-anchor/a57a640a1eb9e395da0523c9e999f3b28e67ece1edfc4c5861e9883f0855e465/`
- PathoROB 完整结果：`runs/pathorob-real-per-center12/result.json`

原始 run 文件包含完整 masks 和逐 token trace，体积可达数 GB，因此文档与机器可读记录只提取可审计字段，不复制密集 mask 数组。
