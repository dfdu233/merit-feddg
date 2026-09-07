# 输入适用性感知的医学小模型协作：实验说明

主体仍是 NativeSession + CapabilityRuntime：按需选择分类、分割、检测、检索、
生成专家，获取原生证据，并在保留生成前缀和原图的条件下继续回答。
新增适用性模块只过滤传给原有有限动作控制器的工具，不改变证据接口或解码规则。
ROVER 是独立诊断入口；本次没有将区域对比解码变成框架主体。

## Motivation 与当前实现的边界

之前 CONCH 接到肉眼标本，说明粗模态不等于适用性；整卡拒绝变成全 NONE，
说明需要输入级分析；source retrieval 没有实际调用，导致资格支持样本不足。
本次沿用图像模态路由（区分 histopathology / gross specimen），再估计每个专家
在当前图像附近的风险。路由本身仍可能错误，必须检查 routing.json。

同一图像对不同专家具有不同风险。距离只用来找源域参考病例，不直接当作正确率。
每个参考点都绑定实际观测误差；源域按来源均衡聚合，防止大域吞没小域。
默认参数是预先给定的试验值，不是通过目标数据调出的最优参数。

**两种误差必须区分：**

- 提供 `--native-losses`：使用外部真实测量的专家原生任务误差，例如 1-Dice、
  分类错误或经过验证的连续损失。文件格式为 `source_id -> scope_key -> [0,1] loss`。
  只接受 source ID，必须记录任务标注及评分过程，不得用模型自信度冒充误差。
- 未提供：采集同一原图/问题、空前缀下的原始回答与强制单工具回答，使用
  `max(0, Q(base)-Q(tool))`。这是当前证据接口下的回答受损程度，**不是专家
  原生能力或医学幻觉率**。默认 Q 为 Token-F1，也可用既有自定义连续评估器。
  零伤害可能只是工具没有作用，不意味着专家有收益。报告同时保留 gain。

## 核心算法

1. 对每个 source 病例枚举所有兼容、无需人工 ROI 的工具，强制各调用一次。
   不由控制器决定是否采集，保留空证据结果；运行错误停止，不当成负例。
2. 通过 `CapabilityPool.domain_embedding(expert, image)` 获取该专家的冻结特征。
3. 分专家、模态、任务、能力、scope 建立源域参考库；一患者/图像仅计一次支持。
4. 对每个来源取距离阈值内最近 k 个参考点，计算距离加权的局部实际误差。
5. 对比六个方法：generalist、原始 agent、global、distance、local、robust。
   - global：所有源域的平均风险；不提取当前图片特征。
   - distance：只检查邻域覆盖，不使用误差标签。
   - local：各个有支持来源的局部风险均值。
   - robust：局部最差风险 + 源域留一验证的低估残差分位数。
6. 只有风险不高于预先设置的阈值，才将工具提供给原有动态控制器。
   控制器仍可选择 CONTINUE；各方法使用相同生成/工具预算。

风险、来源支持数、拒绝原因写入真实生成 trace。原图、结构化证据、空间叠加图、
多工具历史与单工具不可重复规则继续使用现有实现。
所有新方法默认不开启旧整卡资格检查，以便独立比较输入适用性；这是明确的新实验
分支，不是偷偷放宽旧 v0.12 资格条件。已有入口不变。

## 特征支持与成本

- CONCH、BiomedCLIP/其检索工具：已有原生视觉 embedding。
- XRV findings：新增冻结 DenseNet 分类头之前的空间特征池化。
- 自定义 adapter：实现 `domain_embedding(image) -> finite 1D vector` 即可接入，
  与其 `infer(request)` 分类/分割/生成能力独立。
- XRV anatomy、CheXagent 等尚未验证原生 embedding 的适配器，不会偷偷使用
  BiomedCLIP 替代；采集报告 `unsupported_native_embedding`。核心能力仍保留。
  本轮默认只选已经实现原生特征的三项工具，不声称全能力适用性已覆盖。

提取原生特征可能需要加载专家并运行其视觉编码器。拒绝的是后续证据调用，
不是保证“拒绝前完全不加载专家”。GPU 和延迟必须实际测量。

## 域、样本与泄漏约束

- 默认仅接受 `domain_kind: independent`，这个字段须有真实来源依据，不得将
  原 hash 分组改名成医院。不得使用目标标签选阈值、邻居数或残差分位数。
- `allow_proxy: true` 仅用于显式工程诊断；memory/result 保留 proxy 标记。
- 默认 min_domains=2，LODO 残差需要每次留出后还有至少两个有支持域：至少
  **3 个源域**。在源域上嵌套排除查询域进行 robust 评价，通常至少 **4 个源域**。
  每域、每 scope 还需足够独立且在距离范围内的病例；总样本数不能替代局部支持。
- 源域自评会排除查询域、患者、RGB 像素，并重新计算排除查询域后的残差。
  但这不是整个模型/检索流水线的独立重训 LODO；检索参考库保持共享源域设置。
- 没有查询邻域覆盖时拒绝。robust 只有部分域能形成 LODO 残差时也拒绝。
  这可能导致零调用，必须如实报告，不能当成已证明泛化。
- 初始单工具误差并不等于后续多工具条件收益；本模块只估计输入适用性。
- 源域范围外的任意分布变化没有理论保证；经验残差不是置信区间。

## 服务器 Codex 执行指令

保留本地修改并 `git pull --ff-only`，阅读本文件。复用现有 huatuo Python、
离线模型和真实数据，不安装 research extras、不下载新权重。
先盘点真实来源及每专家支持。当前若仍只有两个 hash 域，先做显式 proxy 工程
诊断，报告 robust 缺支持；不要虚构医院或为了通过门控按结果降低阈值。

```bash
cd /home/dbw/merit-feddg
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python
# 使用实际存在的环境；旧环境也可能在 /opt/miniconda3/envs/huatuo/bin/python
SOURCE=/实际/source.jsonl
REFS=/实际/references.json
LLAVA_SRC=/实际/llava-med-1.5
CFG=configs/applicability_pilot.yaml

"$PY" -m merit_feddg.applicability_run --stage collect \
  --source-manifest "$SOURCE" --references "$REFS" --config "$CFG" --check-only

"$PY" -m merit_feddg.applicability_run --stage collect \
  --source-manifest "$SOURCE" --references "$REFS" --config "$CFG" \
  --llava-source "$LLAVA_SRC" --output runs/applicability-pilot

# 默认 source 交叉排除评价，不生成 target。
"$PY" -m merit_feddg.applicability_run --stage evaluate \
  --source-manifest "$SOURCE" --references "$REFS" --config "$CFG" \
  --llava-source "$LLAVA_SRC" --memory runs/applicability-pilot/memory.json \
  --output runs/applicability-pilot --limit 4
```

`--limit` 按 ID 固定排序，不能保证域平衡。先制作固定、按来源/模态平衡且与模型
正确率无关的小 source manifest，再采集。check-only 只验证元数据，不加载 GPU。
需要工程 proxy 配置时复制 YAML 到本地 run 目录，显式设置 allow_proxy=true，
保留其他预设参数。不要修改原始数据 provenance。

仅在 source 诊断确认后，冻结 memory/config，才通过 `--query-manifest` 指定真实
独立 target。查询和 source 必须无患者、像素、来源名称重叠。references 中的
目标答案仅用于生成结束后的评分，传给检索和模型的只有 source references。

采集缓存含源域答案指纹、模型、专家设置、整个 Python 实现指纹、图像、生成配置。
完整病例可复用；中断病例重新执行。memory 与模型/代码/接口/配置不一致直接报错。
因此改动代码后不能复用旧 memory 冒充同一个策略。模态路由也缓存并单独输出。

输出：memory.json、source-interventions.json、routing.json、predictions.json、
result.json/result.md。检查真实工具数量、原生特征维度、分专家输入风险、调用率、
逐例医学事实变化及耗时。现有报告比较相同预算上限，但不自动保证实际调用率相等；
论文仍需要 source-only 选阈值形成调用率—质量曲线，以及最佳单专家等强基线。

完成后将真实执行结果摘要和必要修复提交 STATUS.md 并推送；权重、原始图像、
大型缓存和凭证不要提交。禁止把单元测试通过表述成医学性能或 ICLR 创新已成立。
