# 不确定性证据接口：服务器执行说明

这是设计的第一步，不是完整域泛化方法或性能提升版本。
主模型、专家、路由和既有 ApplicabilityGate 不变；未新增训练或样本关键词。

## 已实现

- `uncertain_evidence.attach_alternatives` 接受同一专家、同一请求的原生输出集合。
- 始终包含原始观察；一致语义保留，数值取包络，冲突或重复对象对应不明确则不合并。
- `evidence_style: uncertainty` 接入真实 NativeSession；禁止额外图像绕过证据接口。
- `uncertainty_from_probe: true` 使用已有 XRV 分类探针的真实输出，显式付出三次额外前向。
  空间探针暂只报告敏感性，不伪造校准掩码集合。其他专家没有集合时显示 unknown。
- 源域 diagnose 新增 typed_text 和 uncertainty_text 对照，共享原始工具输出。
- uncertainty_point 使用同一编译器和预算，但移除替代观察，作为点观察对照。
- 固定目录分类输出使用 label-keyed value/range 表，共同的类型、极性、空间范围和
  分数语义只记录一次；这是无损结构压缩，不删类别、不改原始浮点值。
- 范围是观测到的数值范围，不是医学置信区间；共同观察不是共同真相。

不采用候选答案二值估计，不把原生分数转成临床阳性，不按问题词表筛标签。
已有默认配置不变。新配置关闭 request_scope_check，避免将上一轮别名规则作为主方法。
现有工具元数据/旧路由仍有局限，本轮没有声称已解决其语义覆盖问题。

## 执行

复用服务器原有 huatuo、LLaVA-Med 和专家权重；不安装或升级包。
读取 STATUS.md 和 UNCERTAINTY_PRESERVING_COLLABORATION_DESIGN.md。
先运行测试，再使用已冻结的 source manifest。旧 canary 只检查兼容，新的确认集须事前固定。

```bash
cd /home/dbw/merit-feddg
PY=/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$PY" -m pytest -q
"$PY" -m merit_feddg.llava_run --study diagnose \
  --config configs/uncertainty_source_pilot.yaml --skip-download \
  --source-manifest /实际路径/source.jsonl \
  --target-manifest /实际路径/隔离target.jsonl \
  --references /实际路径/references.json \
  --llava-source /实际路径/llava-med-1.5 \
  --output runs/uncertainty-source-pilot
```

必须替换占位路径；target 清单仅用于隔离审计，不生成 target 答案。
检查 uncertainty.status、observation_count 和 native_alternatives：XRV 分类有真实多次观察，
其余工具必须诚实标注 unknown；不得手工制造有利候选或改动专家原始分数。
typed_text 使用既有 graph/top-k 实现，uncertainty_text 不裁剪集合，因此其差异还包含
呈现长度/裁剪策略混杂。优先比较 uncertainty_point 与 uncertainty_text，二者仍有
有效载荷长度差异，需记录长度；typed_text 只能用于接口探索，不能单独归因算法收益。
预算不足时整包略过，不截断不确定范围；必须检查 presented_memory 是否实际非空。

## 未实现及后续门槛

尚未实现真实来源的原生误差校准、未知域覆盖保证、临床语义蕴含、
自由文本的硬性证据约束、空间对象跨模型匹配。不得把本版本称为完整设计已落地。
现有域门控仍独立运行；本 source diagnose 不拟合它。
现阶段可验证的是“观测不确定性传输”而非“门控证明专家正确”。

报告全体病例的正确性、完整性、超范围属性、长度、调用和延迟；Token-F1 仅作辅助。
不因负面结果扩展别名、挑掉困难病例或修改 target。若无正向机制信号，保留结果并停机。

2026-09-09 的 6 例兼容 canary、必要上下文修复和负面医学结果见
[UNCERTAINTY_PILOT_RESULTS_2026-09-09.md](UNCERTAINTY_PILOT_RESULTS_2026-09-09.md)。
