# 完整清单、统一自由生成的匹配实验

所有题目使用同一图像入口、原问题、通用提示和生成预算。不按 CE/OE 建立不同 prompt、语法、候选答案集或 logits 约束。题型标签在加载 manifest 时被移除，不能进入路由、推理或缓存身份；只可在生成后独立统计。

## 输入与执行

使用已有完整官方测试清单，不进行 source/confirmation/proxy 等新划分。每行至少包含 `id`、`image`、`question`、`image_sha256`；其中 image 指向运行环境中可读的图像。生成清单不得带答案字段。`image_sha256` 可沿用已有像素身份；运行器另外读取实际文件字节摘要绑定缓存。

先检查 `configs/llava_med_capabilities.yaml` 中本地模型位置，按现有项目流程准备专家权重。然后从仓库根目录运行：

```bash
python -m merit_feddg.matched_evaluation \
  --protocol text \
  --manifest /absolute/path/to/existing-full-manifest.jsonl \
  --config configs/matched_permissions.yaml \
  --output runs/matched-permissions \
  --artifacts artifacts
```

此命令需要真实权重与 GPU 环境；此处未执行。它不会创建数据划分或拟合策略。四个方法均遍历相同完整清单：

| 方法 | 证据表达 | 生成设置 |
|---|---|---|
| generalist | 无专家证据，重新生成基线 | 与其余方法同一提示、64-token 预算、自由生成 |
| point | 原生点值的 scoped 表达 | 相同 |
| uncertainty | 原生观测变化范围 | 相同 |
| permissions | 保留共同测量及有限关系 | 相同 |

专家原生结果在方法之间以及恢复运行时共享持久化缓存；扰动仅在需要的方法中启用。为避免混淆，该实验使用 all_evidence 而不训练调用策略。保留所配置的兼容性与调用上限，并不是每个样本调用所有专家。

缓存以实现源码、解析后配置、模型身份、实际图像字节和完整规范化清单为依据。结果位于输出目录中的身份子目录，包含四份方法结果、routing、protocol 和逐例缓存。source-case 检索排除，精度卡禁止加载。

## 评估顺序

1. 检查 `protocol.json`：`dataset_partitioned=false`，`answer_type_used_for_generation=false`，`output_grammar=unconstrained_for_all_questions`。
2. 检查逐例 `trace` 中 decode 的 `evidence_transport`：实际 presented 列表、omitted 原因、上下文预算。不要用 tool 事件的 adopted 数量代替它。
3. 比较三个证据方法是否拿到同一专家原生输出、实际呈现了哪些子集。表达长度会改变装入的子集，这属于机制差异，需要单独报告。
4. 生成完成后，使用既有固定参考答案和同一离线评估器比较全部方法。自由回答可能不符合旧严格二值解析器；应如实报告该指标的局限，不能因此对某类题加生成约束。
5. 报告每臂质量及证据丢弃情况。共享缓存与顺序执行的 warm timing 不构成公平速度比较；性能实验需独立冷启动协议，不能直接相减当前耗时。

原补丁记录的验证为 597 项通过、17 项跳过（共 614 项）；这不是合并环境的新结果。
本次合并验证另见 STATUS.md。真实医学结果仍待上述完整运行。

合并时增加了专家模型 provenance 和 experts 等子目录代码的缓存身份绑定，避免改动
专家权重或适配器后误用旧缓存。读取 JSON 清单/缓存显式使用 UTF-8。
provenance 沿用项目现有定义，并不表示对每份权重新增全字节哈希验证。
