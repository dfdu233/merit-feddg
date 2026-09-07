# v0.10 阶段 0–4 服务器执行结果

执行日期：2026-09-07

## 结论先行

阶段 0–4 已按 `docs/SERVER_CODEX_V010.md` 完成。阶段 3 的真实 source 诊断提供了继续做
source-only legacy/scoped 拟合的理由，但最终两套 **robust** 策略在已观察的 200 个 source 状态上
都没有选择任何工具。这个结果说明当前证据通信有局部改善信号，但在预声明的跨 source 支持门槛和
LODO 过估计惩罚下，还没有足够稳定、可迁移的正效用支持执行式工具调用。

全程没有运行 `value-stage evaluate`，没有生成或评分 target，也没有利用旧的 32 个 development
target 选择配置。所有“改善/伤害”均只指连续 Token-F1，不能解释为医学事实正确率、幻觉率或真实
医院域泛化。

## 阶段与状态

| 阶段 | 状态 | 主要结果 |
| --- | --- | --- |
| 0 环境/资产/旧结果/GPU | 完成 | 复用现有 huatuo、LLaVA-Med、CheXagent、CONCH、XRV 和 v0.9 数据；无下载 |
| 1 v0.9 只读审计 | 完成 | 发现小样本支持、路由错误、检索普遍负增益和 XRV anatomy 任务错配 |
| 2 静态检查与真实 canary | 完成 | 447 passed、2 skipped；8/8 block-NONE 一致；29 个工具事件无错误 |
| 3 64-source 通信/组合诊断 | 完成 | 64/64、233 个工具事件、0 错误、64/64 block-NONE 一致 |
| 4 legacy/scoped source 拟合 | 完成 | 两套 robust 策略均为 0 次 source 调用；未运行 target |

## 实际代码、环境和资产

- 基础提交：`dbc4af5c865283b014528a3d492c1e0be8e5f70e`。
- 本次服务器修复尚未提交：
  `merit_feddg/capability_value_study.py`、`tests/test_capability_value_study.py`。
- Python：`/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python`，Python 3.10.20。
- Torch / Transformers：2.0.1 / 4.37.2。
- LLaVA-Med 源码：
  `/home/dbw/ANCHOR/data/medheval/code/baselines/Med-LVLMs/llava-med-1.5`。
- LLaVA-Med checkpoint：
  `/home/dbw/ANCHOR/hf_cache/llava-med-v1.5-mistral-7b`（约 15 GB）。
- vision tower：
  `/home/dbw/ANCHOR/hf_cache/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1`。
- CheXagent：
  `/home/dbw/merit-feddg/artifacts/models/StanfordAIMI--CheXagent-2-3b`（约 12 GB）。
- GPU：`CUDA_VISIBLE_DEVICES=0`，NVIDIA GeForce RTX 4090，49140 MiB。
- 全部运行设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、
  `HF_DATASETS_OFFLINE=1`，没有使用用户 token、镜像或联网下载。

本次 source 配置及数据沿用 v0.9 成功运行：

- 基础配置：`runs/native-v09-value-full-gpu1/config-llava-ccc4803d0175.yaml`，
  SHA256 `bc334e094ccbd5676b721f3ae21d0405b04c4bd46df051c1e959ffb56b0b8175`。
- source：`runs/native-v09-value-full-gpu1/data/both-source16-target16-seed17/source.jsonl`，
  64 例，SHA256 `86c12522b81962eb9d7a359eaef769b3e8c8fd65b8f8a35b6752025fae20b67a`。
- target：同目录 `target.jsonl`，32 例，只用于拆分/指纹准备，
  SHA256 `c18584edf13dd6435983f3883dc4f311c80ac6a32591dec3e462406dca209555`。
- references：同目录 `references.json`，
  SHA256 `aa7bce3d336063eb7337055ee2845ea838373745411810d41de6b878380a22c1`。

这些 source 域是 PathVQA/VQA-RAD 的数据集/哈希代理分区，不是医院域。没有患者 ID 或真实医院
元数据，专家预训练是否暴露这些来源也未能排除。因此本轮不构成真实域泛化验证。

## 阶段 0–2：审计与 canary

v0.9 审计保存在 `runs/native-v010-audit/V09_AUDIT.md`。主要负面发现：

- v0.9 的 293 条 source 干预来自 64 个独立图像组；CheXagent 虽有局部正均值，但没有任何域达到
  预声明的每域 8 个独立组支持要求。
- source retrieval 在四个代理域的初始平均 Token-F1 增益都为负。
- 旧 development target 中存在把新生儿临床照片路由成 CXR 并调用 CheXagent 的明确错误；其
  词面增益不能作为专家有效性证据。
- XRV anatomy 的旧 target overlay 几何上像胸廓结构分割，但对 mass/pneumothorax 等问题不相关，
  且两例生成空答案。单工具结果始终是 LLaVA-Med + 工具证据，不是小模型独立准确率。

首次阶段 3 尝试在 11 例后发现总显存升至约 48.5 GiB，但病例记录的实际 PyTorch 峰值仅约
17.4 GiB。定位为诊断重放后 allocator cache 没有逐病例释放，不是模型装不下。代码现在在每个
未命中缓存的病例成功或异常退出后执行病例状态重置、Python GC 和 `torch.cuda.empty_cache()`，
同时保留专家模型权重。针对性测试验证每例释放 CUDA cache。

修复后的真实 8-source canary：

- 结果：`runs/native-v010-canary-cleanup/llava/latest-diagnostics.json`。
- 8/8 source 完成；29/29 工具事件执行，0 错误；1 个合法空/不可用证据。
- block-NONE 8/8 与 baseline token 完全相同。
- `target_generations=0`、`policy_fitted=false`。
- 单例 PyTorch peak 15.92–23.68 GiB；任务退出后 GPU 回到约 20 MiB。

静态验证为 447 passed、2 environment skips；Ruff、Bash syntax 和 `git diff --check` 均通过。

## 阶段 3：64-source 真实诊断

主索引：`runs/native-v010-source-diagnostics/llava/latest-diagnostics.json`。
实际 run：`runs/native-v010-source-diagnostics/llava/33f84d09e0bdb3a1`。
生成配置：`runs/native-v010-source-diagnostics/config-llava-7bac6ab921ea.yaml`，
SHA256 `77947288e65dd518f1dffc0dfcc7a05a338fe9ec063e2eb24a175348eaa48a9e`。

执行完整性：

- 64/64 独立 source 图像组完成，64/64 block-NONE token 完全一致。
- 233 个工具事件全部执行，0 runtime error；231 次采纳，2 次为同一病例 pair 重放中的合法空检索。
- PyTorch 单例峰值 15.92–24.03 GiB；全部结束后 GPU 回到约 20 MiB。
- `target_generations=0`、`policy_fitted=false`。

### 路由后覆盖

| 工具/条件（initial） | 代理域独立组支持 |
| --- | --- |
| source retrieval | 四域各 16 |
| CONCH tissue | PathVQA proxy-0: 8；proxy-1: 9 |
| CheXagent description | PathVQA proxy-0: 1；VQA-RAD proxy-0: 6；proxy-1: 4 |
| XRV findings | PathVQA proxy-0: 1；VQA-RAD proxy-0: 5；proxy-1: 4 |
| XRV anatomy | VQA-RAD proxy-0: 1 |
| Biomed anatomy | VQA-RAD proxy-0/1 各 1 |

只有病理 CONCH 和病理检索达到两个 source 域各至少 8 个独立组。CXR 总例数并不能替代逐域、
逐 scope、逐历史条件支持；本轮没有降低 `min_cases_per_domain=8`。

### 呈现与工具结果（Token-F1）

下表为所有适用 initial 独立组的描述统计：

| 分支 | n | 平均增益 | 改善 | 伤害 | 零变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| duplicate original 控制 | 64 | -0.05095 | 0 | 17 | 47 |
| CheXagent question/native | 11 | +0.01771 | 2 | 1 | 8 |
| CheXagent question/scoped | 11 | +0.01771 | 2 | 1 | 8 |
| CheXagent EvidenceNeed/native | 11 | +0.00894 | 1 | 1 | 9 |
| XRV findings/native | 10 | -0.04175 | 2 | 2 | 6 |
| XRV findings/scoped | 10 | +0.00646 | 3 | 1 | 6 |
| CONCH/native | 17 | +0.00109 | 3 | 1 | 13 |
| CONCH/scoped | 17 | +0.00254 | 3 | 1 | 13 |
| retrieval/native（带答案） | 64 | -0.03097 | 7 | 14 | 43 |
| retrieval/scoped（去答案） | 64 | -0.03062 | 6 | 14 | 44 |

关键解释：

- 原图复制本身就造成显著退化，尤其 VQA-RAD initial；不能把所有多图/overlay 分支伤害归因于
  小模型观察。
- scoped 对 XRV findings 的改善主要来自压缩未校准的完整 finding 列表。例如
  `vqarad-b9c14f55bccd14627fe3-8a45357ac421` 的问题只是图像类型，native 把大量疾病名写入回答，
  增益 -0.49655；scoped 缩短后仍为 -0.20000。它是伤害缓解，不是医学收益。
- `vqarad-3e22b6de97ac77ab0384-df6e19978bff` 上 XRV scoped 从 -0.08139 到 +0.07719，
  但参考答案是“左侧心后区密度增高”，工具输出为 atelectasis/infiltration；需要医学事实审查，
  不能凭词面分数宣布纠正幻觉。
- `pathvqa-train-14925` 是人工看过的真实 CXR；CheXagent 输出 Cardiomegaly，与参考中的
  cardiomegaly 有词面一致性，增益 +0.08696。它仍只是一个独立组。
- CheXagent 的 question/native 与 question/scoped 在本批次产生相同主模型回答；EvidenceNeed
  请求没有稳定优于原问题请求。
- 去掉 retrieval 的 source answer 没有消除总体负效应。检索证据仍会把其他患者/图像的答案带入
  当前病例；scoped 有时避免直接复制，也会丢失有用上下文。
- CONCH 的 scoped 改善很小。最大样例 `pathvqa-train-15641` 的参考文本含糊，generic tissue
  catalog 产生的词面改善不能视为病变识别。

### 组合结果

- CONCH→retrieval：17 个 initial 组，联合平均增益 -0.00612；2 改善、2 伤害、13 不变。
- PathVQA proxy-0 initial：n=8，联合 -0.02174，`beats_best_single=0/8`。
- PathVQA proxy-1 initial：n=9，联合 +0.00776，只有 1/9 优于最好单工具。
- CXR anatomy→CheXagent 只有 1 个 initial 组，联合增益 0。
- 因此没有稳定的两工具互补证据，也不支持训练或宣称复合动作策略。

原始分支、观察和审查表：

- `.../33f84d09e0bdb3a1/diagnostics/source-diagnostics.json`
- `.../33f84d09e0bdb3a1/diagnostics/diagnostic-summary.json` / `.md`
- `.../33f84d09e0bdb3a1/diagnostics/evidence-audit.json`
- `.../33f84d09e0bdb3a1/diagnostics/evidence-audit-key-private.json`
- `.../33f84d09e0bdb3a1/diagnostics/routing-audit.json`
- `.../33f84d09e0bdb3a1/diagnostics/routing-predictions-private.json`

`evidence-audit.json` 和 `routing-audit.json` 的人工字段仍为空。上面的正确性判断只是 Codex 初筛；
关键 CXR、病理和路由病例必须由有能力的医学评审完成标注。

## 阶段 4：source-only legacy/scoped 策略

阶段 3 的 XRV scoped 伤害缓解、CONCH 续写小幅改善，以及 retrieval 的稳定负信号，足以进行一次
预声明的 source-only 配置对照。没有进行 4 维 encoder、ridge 或惩罚强度搜索。

### Legacy

- policy：`runs/native-v010-value-legacy/llava/d297275731080256/value-policy.json`。
- 配置：`runs/native-v010-value-legacy/config-llava-80cb95c67f31.yaml`，
  SHA256 `2eb01eeb11565ef6e21649eb78e7ff3f975a47a7f87ce4c2263ae09ed1980750`。
- 292 条真实 source 干预；292 executed；0 runtime failure；5 empty；64 独立组。
- 35 个精确 action/state/history 条件中仅 5 个支持充分，均为病理 CONCH/检索条件。
- 133 维 robust policy 在 200 个 observed source states 上选择工具 0 次。

### Scoped

- policy：`runs/native-v010-value-scoped/llava/8fe97651e41d06d4/value-policy.json`。
- 配置：`runs/native-v010-value-scoped/config-llava-ee21cc032544.yaml`，
  SHA256 `38f8473116bf6d9b9b3429b3975c9563a43c7beec118f74ba36a7e6ecc28bf72`。
- 294 条真实 source 干预；294 executed；0 runtime failure；3 empty；64 独立组。
- 39 个条件中同样仅 5 个支持充分，仍只覆盖病理 CONCH/检索。
- 133 维 robust policy 在 200 个 observed source states 上选择工具 0 次。

两次 source collection 的 initial 分支均值如下；它们是各自整套 profile 的结果，不是单因素因果
估计：

| 工具 | Legacy | Scoped | 解释 |
| --- | ---: | ---: | --- |
| CheXagent（11 组） | +0.01771 | +0.00894 | scoped 的 EvidenceNeed 请求未胜过原问题请求 |
| XRV findings（10 组） | -0.04175 | +0.00646 | 主要为冗长 evidence 的伤害缓解 |
| CONCH（17 组） | +0.00109 | +0.00254 | 正信号很小且不均匀 |
| source retrieval（64 组） | -0.03097 | -0.02815 | 两套都为负，scoped 没有解决跨病例污染 |

两套策略的 supported 条件均被 source-LODO 过估计惩罚压到非正效用：legacy 惩罚约
0.1258–0.3638，scoped 约 0.2402–0.3547。CXR 条件因为逐域支持不足，根本不会被 robust gate
执行；这比降低门槛强行产生调用更符合当前证据。

去掉惩罚的 mean 消融在同一训练 source 上会选择工具：legacy 为 36/200 状态（8 个 initial），
scoped 为 37/200（11 个 initial）。这是训练集内重评分，且 scoped 仍包含 1 个受损选择，不能作为
泛化效果，也不能据此自动进入 target。

## 最小下一步

当前最小且有信息增益的下一步不是 target 调参，也不是增加更多 pair：

1. 先完成 `routing-audit.json` 的问题/答案盲模态标注，以及 `evidence-audit.json` 中关键 CXR 和
   病理病例的 observation correctness、question relevance、answer factuality、unsupported claims
   和 clinical harm。
2. 无标签地补充有明确真实来源的 CXR source，使 CheXagent/XRV 的每个实际 scope 在至少两个
   source 域各达到 8 个独立患者/图像组；记录小模型预训练暴露为已知/未知，不创造医院 ID。
3. 在相同 source 上只复验 scoped 文本与 legacy 文本；保留 retrieval 带答案/去答案单因素对照。
   不增加新的组合，不降低支持门槛。
4. 只有事实审查确认“观察正确且相关但 LLaVA-Med 仍读取失败”时，才设计小型 evidence adapter/
   projector；当前结果还不能为 LoRA、复合动作或新 target 试参提供依据。

任何新 test/target 运行应另行确认，并与本轮 source、历次 development/test 做像素及患者/组级
去重。本交接没有执行阶段 5。
