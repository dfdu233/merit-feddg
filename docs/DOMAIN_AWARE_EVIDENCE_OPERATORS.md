# 输入适用性与原生证据操作：服务器实验指令

## 改了什么

主体仍是冻结 LLaVA-Med + CapabilityRuntime 按需调用异构专家，并保持生成前缀。
没有新增训练、投影器、区域对比解码或事后全文重写。

1. `ValueGenerationConfig.visual_mode` 可选 overlay（原有默认）、crop、control_crop。
   crop 使用原生预测掩码/框映射到原图后的区域，增加一张原始像素裁剪图。
   原图仍在。control_crop 使用同一预测区域的宽高，移到最远角落，作为位置对照。
   它不保证无关，尤其全肺区域；须检查实际框和重叠，不能称为错误病灶真值。
2. XRV findings/anatomy 支持 `behavior_probe`：off（兼容默认）、audit、reject。
   对原图及 gamma=0.95/1.05 的图像分别执行原生模型，记录额外三次前向及耗时。
   此阶段有意重跑原图以比较同一种原生输出，未做推理复用优化。
   分类比较逐标签最大绝对分数变化，绝不对独立 sigmoid 作 softmax。
   分割比较同一裁剪坐标网格中非空掩码的最大 1-IoU；全空不算可靠证据。
   各视图的类别顺序和坐标变换必须相同。仅光度变化，无翻转或几何扰动。
3. source-only diagnose 增加 `evidence_operators: true` 开关。
   文本、叠加图、裁剪、等面积位置对照共享同一次专家输出及精确前缀。
   crop/control_crop 共享文本和提示格式，避免把不同提示当空间能力收益。
   所有新增字段纳入已有配置/缓存身份；旧默认行为不变。

## 算法含义与边界

专家提供原生输出，类型化接口决定输出如何进入主模型；输入适用性决定能否采用。
当前行为探针在控制器选定工具、工具执行后、证据采用前运行，不会遍历全部专家。
已有 ApplicabilityGate 仍在控制器前检查源域参考支持；新模块不取代或自动放宽它。

audit 只记录行为，不改变证据，不宣称域风险认证。reject 必须显式提供
`behavior_max_sensitivity`（[0,1]），不得按目标表现选阈值。
超过阈值、无信息或不支持探针会拒绝采用，但工具及探针成本已经发生。
诊断通道比较禁止 reject，以免把证据过滤与呈现效果混在一起。

当前探针只验证 XRV 分类/解剖分割；CONCH、检索、CheXagent 及其他工具返回
unsupported，不会用 BiomedCLIP 隐式替代。无需原生特征即可对 XRV anatomy
作行为诊断，但这不等于它已有源域风险参考库。其他专家能力仍可照常调用。

弱 gamma 也不是已证明的医学标签保持变换，必须抽查微小病灶可见性。
稳定错误仍可能出现；全空掩码不等于器官不存在；解剖定位不等于病灶定位。
本轮没有拟合新的联合风险模型，没有任意未知域保证，也没有临床效果结果。
模型不更新参数，但使用源域标签选阈值属于源域校准，不能叫完全无监督。

## 服务器运行

从仓库根目录执行。先检查工作树，若有本地修改不要覆盖或自动 stash/pop。
复用 huatuo 与已有权重，先制作与结果无关的固定 source 小清单，至少包含
真正的胸片解剖/定位问题及胸片 findings 问题，另加其他模态检查不兼容工具。
不要把 PathVQA 所有图片标为组织切片。选 4–8 例工程 canary 不是统计验证。

```bash
cd /home/dbw/merit-feddg
git status --short
git pull --ff-only
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# 使用实际空闲 GPU，不抢占其他实验。
export CUDA_VISIBLE_DEVICES=0
PY=/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python

"$PY" -m merit_feddg.llava_run --study diagnose \
  --config configs/evidence_operators_pilot.yaml \
  --skip-download --check-only

"$PY" -m merit_feddg.llava_run --study diagnose \
  --config configs/evidence_operators_pilot.yaml --skip-download \
  --source-manifest /实际路径/source-canary.jsonl \
  --target-manifest /实际路径/target.jsonl \
  --references /实际路径/references.json \
  --llava-source /实际路径/llava-med-1.5 \
  --output runs/evidence-operators-source
```

已有入口需要 source/target/reference 三个文件进行隔离检查；diagnose 分支只生成
source，不拟合策略、不生成 target。不能把同一个 manifest 同时作两种角色。
check-only 是环境/权重检查，不替代 cohort 审计。所有真实路径由服务器检查后填写。
若当前环境无法满足入口依赖，报告缺项，不升级 torch/transformers 或下载新权重。

结果在 `runs/evidence-operators-source/llava/<fingerprint>/diagnostics/`，
入口摘要为 `runs/evidence-operators-source/latest-diagnostics.json`。
查看 source-diagnostics.json 和 diagnostic-summary.json；保存所有真实 probe 记录。
首轮关闭 continuation 和双工具组合节约时间；通过后可在单独配置中打开。

## 给服务器 Codex 的执行任务

读取本文件及 STATUS.md，检查最新实现后运行上述 source-only canary。
模型、图片、环境离线复用，不下载新权重。先确认路由、真实分割输出和区域输入。
对比同病例文本/叠加图/crop/control_crop，记录：

- baseline 与 Block-NONE 完全一致；视觉上下文变化创建新会话，前缀不被改写；
- 空掩码、错误坐标不创建视图；列出实际有效空间干预的病例数量；
- anatomy 是解剖不是病灶；对照裁剪是否与预测区域重叠；
- 每个真实探针的敏感性、缺失/空输出、三次附加前向和耗时；
- 专家原生质量（有真实标注时）、医学答案变化、词面指标分别报告；
- source provenance 如实标注，proxy 不改名为医院；目标答案不进入决策。

本轮只验证证据通道与行为诊断。不得把零调用/稳定性/词面增益写成医学改善。
若发现真实空间输入没有进入主模型，先修复并做 parity，不扩大样本。
后续在独立 source confirmation 上选定探针阈值，再与已有局部风险比较 off/audit/reject。
为后续域泛化实验准备真实来源和专家预训练来源清单；未具备来源支持时明确阻塞。
只提交代码修复、测试和真实结果摘要，更新 STATUS.md 并推送；不提交权重、原图和凭证。
