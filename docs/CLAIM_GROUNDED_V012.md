# v0.12：命题级结构化证据与内容控制资格

日期：2026-09-07。本文既是算法说明，也是远程服务器 Codex 的执行交接。

## 1. 本轮要验证的科学问题

目标仍是让冻结医学通用 VLM 使用可插拔专用小模型，并只在其 source-domain
证据有可靠增益时使用。v0.12 不把“能调用模型”当作方法成功，先检验：

1. 专家原生输出能否以保守、统一、可审计的结构进入主模型；
2. 输出变化是否来自专家医学内容，而不是工具消息格式；
3. 该内容在每个真实 source 域是否同时优于空格式对照和未干预主模型；
4. 一个待提交临床命题是否具有合格、范围匹配、必要时带空间覆盖的证据。

形式化地，同一图像、问题和前缀下记录：

```text
content_gain = quality(real evidence) - quality(format-null evidence)
output_gain  = quality(real evidence) - quality(generalist)
control_gain = quality(format-null evidence) - quality(generalist)
```

恒有 `content_gain = output_gain - control_gain`。资格卡只能使用 source 结果，并且
选择组、独立确认组的每个真实 source 域上，content gain 与 output gain 都必须为正。
这避免“真实证据只比一个很坏的格式对照稍好、但仍低于主模型”被错误放行。

## 2. 目前实际代码流程

```text
原始图像 + 原始问题
  -> 现有模态/任务/问题类型 applicability 过滤
  -> 一个原生专用工具（分类、分割、检索或生成）
  -> typed-clinical-evidence-v1
       identity: expert/evidence/capability/scope
       claim_query
       observations: type/content/native score semantics/spatial scope
       limitations: non-exhaustive, uncalibrated, source-only, etc.
  -> 两类 real arm
       direct: 结构化证据直接作为短上下文
       bounded: 原图基线与证据分支的有界 token guidance
  -> 每类 real arm 都有相同工具身份的 format-null arm
  -> source-only 选择 direct 或 bounded+strength
  -> 独立 source groups 确认；不合格则 NONE
```

`structured_evidence.py` 对真实 generation 路径生效：

- CONCH/BiomedCLIP 的相似度保存为 `visual_match`，不转成疾病概率；
- XRV finding 保存原始 score semantics，低分不解释为阴性；
- XRV/MedSAM mask 保存为预测 anatomy/foreground region，不解释成 lesion；
- retrieval 明确标记 `source_image_only` 和 `not_query_evidence`；
- CheXagent 文本标记为 `unverified specialist_statement`；
- 超过字符预算时删除完整 observation，不截断可能含否定词的句子。

默认 evidence study 改为 `evidence_style=graph`、`request_style=need`、单原图、无 overlay。
旧 capability/value/diagnose 入口不被替换。

## 3. 提交前命题验证契约

`claim_verification.py` 新增 `ClaimCommitVerifier`，动作只有：

- `COMMIT`：有 source-qualified、scope-matched 的充分支持；
- `REVISE`：合格证据明确矛盾或支持/矛盾冲突；
- `ABSTAIN`：无合格证据、覆盖不足或缺空间支持。

重要安全约束：

- 多专家不通过乘法或投票制造虚假独立置信度；当前保留最强支持和最强矛盾；
- `spatial_required=true` 的命题没有 box/mask 时不能提交；
- 否定命题没有显式 `coverage_semantics=exhaustive` 时不能提交；
- proxy 域不能构造 qualified certificate；target outcome 不能进入证书。

这个 verifier 目前是经过单元测试的**语义契约**，尚未接入 LLaVA-Med 的 live
claim-boundary generation。当前真实服务器实验只验证 typed graph 与内容控制资格。
若 source 证据内容仍无正向信号，不应先接复杂 live verifier。

## 4. 与 v0.11 的实质差异

| 问题 | v0.11 | v0.12 |
| --- | --- | --- |
| 专家呈现 | 按问题裁剪的原生 JSON | typed evidence graph，显式语义与限制 |
| 资格目标 | guided - generalist | real - format-null，同时要求 real - generalist > 0 |
| 可选通信 | 只选择 bounded strength | source 可在 direct 与 bounded 中选择 |
| direct 对照 | 无独立 null | 新增 direct-format arm |
| 阴性/空间命题 | 文本警告 | verifier 中成为 fail-closed 规则 |
| 默认专家请求 | 原问题 | capability-specific evidence need |

旧 v0.11 policy 的含义与新 schema 不同，代码指纹也不同，不能复制或改名复用。

## 5. 服务器 Codex 执行步骤

先保留服务器本地修改，禁止 `reset --hard`：

```bash
cd /home/dbw/merit-feddg
git status --short
git pull --ff-only
```

检查代码和既有离线资产：

```bash
MERIT_LLAVA_PYTHON=/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python \
bash run_llava_med.sh --study evidence --check-only --chexagent on
```

若实际 huatuo 在 `/opt/miniconda3/envs/huatuo/bin/python`，只替换解释器路径。
不要重新下载已有 LLaVA-Med、CheXagent、CONCH、BiomedCLIP 或 XRV 权重。

运行 source-only canary：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
MERIT_LLAVA_PYTHON=/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python \
bash run_llava_med.sh --study evidence --evidence-stage source \
  --source-per-group 2 --target-limit 1 --chexagent on \
  --output runs/claim-grounded-v012-canary
```

Canary 必须确认：

- `target_generations == 0`；
- baseline、Block-NONE、零强度 token 完全一致；
- 专家只执行一次，随后复用于 direct/direct-format/bounded/format；
- 每条 calibration record 同时包含 intervention、gain、output_gain、control_gain；
- typed graph 中没有把 similarity 写成 diagnosis、把 mask 写成 lesion、把 source answer
  写成 query truth；
- direct-format 和 bounded-format 中不存在真实 observation；
- tool/runtime error 为 0，KL 仍在预算内；
- proxy 域全部拒绝是预期行为，不是 DG 成功或实验失败。

若 canary 通过，再复用冻结的 64-source manifests：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
MERIT_LLAVA_PYTHON=/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python \
bash run_llava_med.sh --study evidence --evidence-stage source \
  --source-manifest runs/native-v09-value-full-gpu1/data/both-source16-target16-seed17/source.jsonl \
  --target-manifest runs/native-v09-value-full-gpu1/data/both-source16-target16-seed17/target.jsonl \
  --references runs/native-v09-value-full-gpu1/data/both-source16-target16-seed17/references.json \
  --source-per-group 16 --target-limit 16 --chexagent on \
  --output runs/claim-grounded-v012-source
```

这组数据仍是 proxy 域，只能诊断 evidence bridge，不能执行正式 target DG 结论。
服务器 Codex 跑完 source 后先报告以下配对结果，不自行降低阈值或启动旧 target 调参：

1. 每个 expert/scope 的 direct、direct-format、bounded、format 输出增益；
2. direct content gain 与 bounded content gain；
3. real 与 null 文本实际不同的病例数；
4. improved/harmed/unchanged，并逐例人工判断医学方向；
5. 每个 typed observation 是否与问题、模态、scope 匹配；
6. policy 的 selected intervention、content/output confirmation 和拒绝原因；
7. 总时延、专家执行次数、显存峰值和缓存恢复情况。

## 6. 停机条件与下一步

本轮 source 结果满足以下任一条件时停止扩量：

- real 与 format-null 基本相同：说明证据仍未影响生成；
- content gain 不正或 real 仍低于 generalist：说明桥接/专家不适配；
- improvement 只来自 Token-F1 词形而非医学正确性；
- 正确证据与 shuffled/wrong-scope 无差异；
- typed graph 丢失否定、位置或 native score semantics。

只有至少一个 scope 在真实医学审查下呈现可重复的正 content gain，下一步才把
`ClaimCommitVerifier` 接入 claim-boundary hidden draft：计划命题 -> 获取/定位证据 ->
生成内部候选 -> COMMIT/REVISE/ABSTAIN。正式 DG 还需要真实医院、中心或独立数据来源，
以及 localization、claim factuality、worst-domain harm 和临床严重度指标；Token-F1 只能
保留为辅助指标。
