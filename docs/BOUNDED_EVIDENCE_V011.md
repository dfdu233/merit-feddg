# v0.11：有界异构证据桥与服务器执行交接

日期：2026-09-07。基于 `8bdd256`，保留服务器兼容/缓存修复。
这是可执行的增量研究实现，不是声称已减少幻觉的完成版论文方法。

## 1. 为什么先改证据桥

最新 v0.10 源域结果中，复制原图作为第二图也造成明显退化，说明存在输入格式混杂；
紧凑 XRV 证据比原生长文本更好，CheXagent 原问题请求优于重构请求；retrieval 仍负面，
两套 LODO 保守策略都选 NONE。因此继续叠加控制器、训练价值回归或降低资格门槛，
不能解决证据进入生成器后产生的副作用。

本次先验证一个可证伪的假设：**同一条真实专科观察，限定其对生成分布的影响范围和
持续时间，是否比直接上下文注入更有效、是否优于只有工具格式的控制？**
若这一步失败，不把新增代码量包装成创新，也不继续盲目增加代理层。

## 2. 顶会代码的实际关系

| 来源 | 参考点 | 本次实际处理 |
| --- | --- | --- |
| [ThinkOmni，ICLR 2026](https://github.com/1ranGuan/thinkomni) | 解码时不同分布的引导及生成接口 | 阅读官方实现后独立实现同一医疗 VLM 的有/无证据分布桥；不是复制其三模型公式或完整复现 |
| [DeepScan，CVPR 2026](https://github.com/YChenL/DeepScan) | 在生成过程中使用视觉工具 | 保留现有能力注册与精确前缀接口；本次不复现其训练/奖励方法 |
| [I2MoE，ICML 2025](https://github.com/Raina-Xin/I2MoE) | 区分异构信息的互补和冗余 | 不要求分割、分类、生成给出同一答案；目前只有同专家重复观察去重，不声称实现其交互分解 |

具体审阅的 ThinkOmni 版本：
[`b3d9f94a8d3967c44afb5958710ecc3c5df15a26 / thinkomni/inference_utils.py`](https://github.com/1ranGuan/thinkomni/blob/b3d9f94a8d3967c44afb5958710ecc3c5df15a26/thinkomni/inference_utils.py)。
未把第三方源码直接复制或 vendoring 进仓库；不声称本次使用了这些论文的新权重。
更多域鲁棒/证据论文、适用与不适用之处见
[完整调研](DOMAIN_ROBUST_EVIDENCE_RESEARCH_20260907.md)。

## 3. 目前准确的算法流程

```text
原始医学图像 + 问题
  → 现有图像路由/能力范围匹配（不看 reference）
  → 原生专科工具：分类观察 / 分割结构测量 / 生成描述
  → 范围保留、紧凑呈现、同专家精确去重、有 token 有效期的证据包
  → 在相同已生成 token 前缀上：
       基线分支：原图 + 原问题
       证据分支：同一原图 + 原问题 + 紧凑证据
  → 裁剪分布残差 → KL 预算内引导 → 只提交一个 token
  → 更新前缀/预算；证据过期或无预算后由原模型继续
```

两分支共享同一冻结 LLaVA-Med 权重，不加载第二份主模型，不需要专家与 VLM 共享词表。
专家仍输出其原生观察；词表对齐来自主模型对证据的条件化，而不是把所有专家变成答案分类器。
始终只传一张原图：分割的 mask/测量保存在原生产物，紧凑结构摘要进入证据分支，
不把 overlay 当成第二幅图。工具生成内容并非事实或可靠标签。

在实际前缀 `h` 上，使用生产生成接口返回的下一词分数：

```text
p0 = softmax(base_scores(image, question, h))
pE = softmax(evidence_scores(image, question, packet, h))
r  = clip(log(pE) - log(p0), -b, b)
pλ = softmax(log(p0) + λ r)
```

在 `[0, source_selected_strength]` 内用二分求满足以下约束的最大 λ：

- 当前 token：`KL(pλ || p0) <= token_kl`；
- 此实际生成轨迹的累计 KL 不超过 `case_kl`。

默认最大强度 0.5，残差裁剪 2，单 token KL 0.02，病例累计预算 0.32，证据 16 token 后失效。
固定实验分支默认用 0.5；校准分支在预声明 `[0.25, 0.5]` 中选择或回退 0。
这些是待验证的工程初值，不是理论最优值。KL 约束不等于医学安全：小分布改动也能翻转
greedy 结果，病例轨迹上的 KL 累计也不是已证明的临床错误上界。

无证据/零强度/预算耗尽路径调用原生产生成代码，不附加提示，不做无意义的双分支前向。
逐 token 轨迹保存实际前缀 hash、证据 ID、token、λ、KL、剩余预算和证据过期原因。
当前证据有效期按 token 距离，不是一个已经可靠识别的 claim。

## 4. 异构模型与当前覆盖

继续使用已有 registry，不新增大权重：

| 专家 | 原生能力 | 输入范围 | 本入口传递 |
| --- | --- | --- | --- |
| CONCH | 组织外观目录匹配 | 组织切片 | 原始概念/相似度，不当作校准诊断概率 |
| BiomedCLIP anatomy | 大器官目录匹配 | 已配置的 X-ray/CT/MRI/gross 图像 | 紧凑概念观察，仅适用器官类问题 |
| XRV Findings | 胸片病征分类 | CXR | 紧凑原生分数，不合成阴性诊断 |
| XRV Anatomy | 解剖分割 | CXR | mask 衍生的空间观察，原始 mask 保留 |
| 本地 CheXagent | 生成式专科描述 | CXR | 回答原问题的短描述，明确为模型观察 |

这是注册能力覆盖，不等于每种模态都已有有效结果；检测仍缺本次真实实例。
新 study 暂排除 retrieval：避免确认组经检索库返回 source 答案形成数据泄漏，也响应负面结果。
旧入口的 retrieval、动态多工具路径未删除。新 session 可接收多个证据包，但新实验先做
initial 单工具证据桥，不把多个包能运行宣称为多能力有效协作。

## 5. 源域校准：现在做到哪里

不训练 VLM、不训练专家、不新增 ridge/LLM 控制器。仅拟合很小的强度配置：

1. 每个实际 source 图像/患者组确定性分为选择组与独立确认组；同组不能同时进入两半。
2. 对每个合法工具，在同一图像、问题、初始前缀执行一次，复用这次真实原生输出比较
   直接上下文、真实证据有界引导、空内容格式引导。各强度比较相同 source 病例。
3. 以最终自由文本连续质量增益，选择使最差 source 域均值最大的强度；并列选较小强度。
4. 固定选择后在另一半独立组确认；每个域选择/确认均正增益才有资格，否则回退 NONE。
5. 最少 2 个真实来源域，每域每一半至少 8 个独立组。不是每域一共 8 个；随机切分后可能
   需要多于 16 个/域才达到两半要求。只计算该专家实际适用的样本，不能用所有问题数充数。
6. `proxy`/未核验域一律只生成诊断，不认证为真实域；`dataset` 只代表独立数据来源，
   不能直接称跨医院。域字段的事实真实性仍需研究者核验，代码不能鉴定医院元数据真伪。

这是最差源域均值选择加独立确认，**不是 LODO、conformal coverage、CVaR、原生特征 OOD，
也不是已经完成的域泛化算法**。它把域问题纳入源域证据使用校准，不承诺未知目标安全。
多视图稳定性与域敏感分量抑制仍为后续设计，未伪造为现有指标。

默认质量仍为 Token-F1，只用于连续桥接诊断，绝不能解释成医学幻觉率。
沿用 `InterventionScorer` 支持带版本指纹的本地连续评分插件；JSON 同时报告配置评分和
词面指标，导出盲评表。医学收益必须另做 supported/contradicted/unsupported claim 评估。

## 6. 服务器 Codex：照此执行

### A. 保留环境并检查

先 `git status --short`。有修改先保存/人工合并，不执行 `reset --hard`，不要重建环境。

```bash
git pull --ff-only
bash run_llava_med.sh --study evidence --check-only
```

脚本自动发现现有 huatuo；如需指定使用 `--python /实际存在的路径/bin/python`。
用户较早提供 `/opt/miniconda3/envs/huatuo/bin/python`，新实验记录为
`/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python`，请实测。
主模型仍 `/home/dbw/ANCHOR/hf_cache/llava-med-v1.5-mistral-7b`；最新配置的源码目录为
`/home/dbw/ANCHOR/data/medheval/code/baselines/Med-LVLMs/llava-med-1.5`。
它与早期提供的 `Mitigation/llava-med-1.5` 不同，按现有可运行环境核查，可用 `--llava-source` 覆盖。
不要把 Windows 本地路径搬到 Linux。

此 study 强制离线并拒绝 `--install-deps`；模型、数据集不重复下载。
若缺本地资产，停止并报告确切缺项；不要直接重新 bootstrap 整套环境。

### B. 仅 source 工程 canary

```bash
bash run_llava_med.sh --study evidence --evidence-stage source \
  --source-per-group 2 --target-limit 1 --chexagent on \
  --output runs/bounded-evidence-canary
```

`both` 默认 PathVQA+VQA-RAD，每 source proxy group 2 例；仍是实际图像和自由文本问题。
`--target-limit` 只控制准备/隔离检查的清单大小，这一步生成 target 数量必须为 0。
若 CheXagent 的本地 checkpoint 确实缺失，先报告，不盲目下载。

产物入口：`runs/bounded-evidence-canary/llava/latest-evidence.json`。
其指向目录包含 `source-result.md/json`、`source-interventions.json`、`evidence-policy.json`、
完整 `case-cache/` 和 `provenance.json`。相同命令从病例级缓存恢复；中断中的未完成病例
可能需重做，但已有权重绝不删除。代码/模型/配置/运行环境改变会产生新实验指纹。

必须检查：

- Baseline、Block-NONE、零强度 token 一致；生产 next_scores 对原生成 token 的回放一致。
- 单图、相同实际 prefix、KL 不超预算；空证据/过期不重复加载专家或额外前向。
- Source 运行错误为 0；原生空证据记录为真实空结果，不把异常记成零收益。
- 记录真实工具适用数/未覆盖模态；至少一个工具分支实际执行，否则未验证证据桥。
- 比较 direct/guided/format，检查逐例医学含义，不把词面变化当作正确性变化。
- 当前 proxy 数据政策全部拒绝属于预期限制，不是 canary 失败或方法成功。

canary 仅验证接线，不因“正例数”通过就宣称统计优势。若分数接口不一致、OOM、
invalid mask、runtime error，停止并定位；不得跳过样本/放宽断言制造完成。

### C. 固定 source 配置后再决定目标评估

完整 source 可使用 `--source-per-group 16 --target-limit 16` 和新的 output；若复用旧研究
的精确 source/target 队列，三个参数必须同时给出：
`--source-manifest PATH --target-manifest PATH --references PATH`。
保持独立图像/组/像素检查、真实来源说明，禁止把 hash 域改名成医院。

完整 source 审查后，服务器 Codex 先报告，不自行开展目标超参数搜索。显式目标命令示例：

```bash
# 与 source 的全部模型/配置/队列/output 参数完全一致，只改 evidence-stage
bash run_llava_med.sh --study evidence --evidence-stage evaluate \
  --source-per-group 2 --target-limit 1 --chexagent on \
  --output runs/bounded-evidence-canary
```

这是很小的接线演示，不是正式论文样本量。评估匹配已冻结策略，否则拒绝。
报告含 generalist、source_calibrated 及每工具 direct/bounded/format，保存生成轨迹、配对
词面指标、延迟、显存及盲评模板。空格式保留同类 schema/工具标识，但不是严格 token
长度匹配；不能排除全部格式混杂。旧开发 target 已被多次查看，正式结论需新的冻结留出集。

## 7. 成本、验收与下一步

新路径没有梯度训练成本，但 source 需多次生成，有真实推理成本。旧版 LLaVA 接口
每个受引导 token 分别沿 base/evidence 的生产 KV-cache 路径强制重放已提交前缀，默认最多
16 次引导；源域还做基线一致性回放。不能改成一次性完整前缀 prefill：FP16 下后者与
生产 KV-cache 解码并非数值等价，近似并列的 token 可能翻转 argmax。
它可能慢于直接注入，不能把“少了控制器”写成已加速。源域复用证据时间不是线上延迟；
目标逐方法计时包含实际工具执行和路由，报告 warm/cold 与总设备显存应另外注明。

本地验证命令（服务器请用兼容 Python，不强制升级）：

```bash
python -m pytest -o addopts='' -q
python -m ruff check .
bash -n run_llava_med.sh
git diff --check
```

本地 CPU 测试不能验证医疗权重效果。交付保留三个下一步而非无限加模块：

本次本地结果：488 项测试全部通过，Ruff、三个 Bash 入口语法和 diff 检查通过。
环境 Python 3.10.9、Torch 2.6.0 CPU、Transformers 4.57.1；不是远程 huatuo 的医学权重运行。

后续顺序：

1. 桥有效才测真正域偏移：使用同任务真实来源域，验证域稳定观察与不稳定观察对生成的差异。
2. 在正向单工具基础上接入真实 ROI/条件工具链，验证联合增益超过最优单工具，非仅多次调用。
3. 若正确证据不优于空格式或直接注入，先修范围/桥；若收益成立再优化 KV 复用与预算调度。

有界解码、registry、独立校准分别不是新创新。ICLR 需要可迁移的异构证据使用规律、
真正跨域证据和足够强的对照；当前只是为这些检验提供可信实现，不能承诺投稿录用或性能。
