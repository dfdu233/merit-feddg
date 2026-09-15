# Native uncertainty / verifier：固定 TRAIN 首轮实测

2026-09-15。本报告是 12 题、10 个图像簇的 TRAIN 机制试验，**不是 VQA-RAD 全量测试集结果，也不是临床准确率证明**。

## 身份与范围

- 独立分支 `implementation/evidence-uncertainty-v1`，基于 `d20c928`，未修改旧方法或旧结果。
- 运行目录 `/home/dbw/merit-feddg-uncertainty/runs/uncertainty-train-v1`。
- 实验 identity：`d9d2246b3dcc0b78be0ca8fb97d9e2f86c4a635f191f14e27c7dbb9be4120a5d`。
- 完整 TRAIN manifest：`/home/dbw/merit-feddg/runs/vqarad-official-protocol-v3/data/train/manifest.jsonl`，1793 行；完整 incumbent：`/home/dbw/merit-feddg/runs/vqarad-official-train-v3/transport-fixed/0379506d53573b3a4ba8eb1c1e3648604b22fdf781949f780b6b1091d3f8dbd6`，`shards_complete=true`，ID 集合一致。
- 仅选定调度的 12 题完成；`pilot-complete.json` 明确 `full_manifest_complete=false`，没有伪造完整 TRAIN 标记。
- 问题正向语法 + 旧 modality=cxr + 官方原生属性；每个属性最先出现的 4 张不同图像。范围仅是全图存在性：Effusion、Cardiomegaly、Pneumothorax，不支持侧别/病因/程度。推理不加载 references。
- Actor：已有 LLaVA-Med v1.5 Mistral 7B，fp16，原图/回答协议与原 TRAIN 配置；12/12 历史 incumbent token 完全一致。
- Source：已有 XRV DenseNet121 `res224-all`，原生独立 sigmoid，关闭 operating-point normalization；SHA256 `56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899`。
- 独立 judge：已有 `OpenMed/Qwen2.5-3B-MedVL`，bf16；微调训练数据与评测数据的重叠未独立排除。不是已验证的医学 verifier。
- XRV/actor 的原始预训练数据与当前图像的逐患者重叠亦未独立排除；manifest 没有可用于完整患者隔离验证的患者 ID。本实验只能声明 TRAIN 探索，不能声明完全去污染或患者级独立验证。
- 两张已授权 RTX4090 分片执行；宿主机 GPU0 使用 `merit-runner`，GPU1 在容器可见。无需训练、升级依赖或下载权重。

## 工程验证

新增/相关 19 项测试通过；全库初次 **890 passed in 18.39 s**，最终复核 **890 passed in 19.57 s**。CLI、编译与 diff 检查通过。

双卡各先真实 canary 一例，再接续剩余同一固定清单。新增 XRV 证据确实送达，原有效证据集合没有变化；固定 0/1 端点及固定 .5/CAD token 与旧实现一致，每例都检查。源 U 范围约 0.0097–0.9263，有真实非零动态权重和候选改变，没有因缺失 uncertainty 全部 fallback。

报告中的 self-judge **不是工程成功**：虽然模型调用实际完成，其输出不符合 A/B/TIE/UNCERTAIN 协议，12 对均记为无效；没有用关键词兜底把失败改成有效判断。该分支与约束输出的跨模型 judge 不构成公平的判别能力排名，只作接口失败诊断。

## 主结果

沿用冻结 `medheval-decoded-eval-v11-explanatory-binary-source-audited`。这些 12 题均为二元问题。下表“正确”只是该评分器的判对数；既有解释性否定解析缺陷仍存在。另列的严格 leading yes/no 只检查显式回答，不把未显式回答自动叫医学错误。

| 方法 | ANCHOR 判对 / 12 | 相对原 incumbent 改善 / 伤害 | 输出文本改变 |
|---|---:|---:|---:|
| 原 incumbent，无新增 XRV | 11 | 0 / 0 | 0 |
| 新增完整 XRV 标签表文字 | 4 | 0 / 7 | 10 |
| 仅所问 XRV 属性文字（适用性对照） | 5 | 0 / 6 | 11 |
| 固定 Blend 0.5 | 7 | 0 / 4 | 7 |
| 固定 CAD 1.5 | 6 | 0 / 5 | 11 |
| ACD | 8 | 0 / 3 | 7 |
| Source-only = 1-U | 5 | 0 / 6 | 8 |
| Source-ACD = (1-U) rho | 9 | 0 / 2 | 7 |
| 同属性平均强度常数（source-only 对照） | 7 | 0 / 4 | 7 |
| 同属性循环打乱 U（source-only 对照） | 6 | 0 / 5 | 8 |

Source-only 不优于常数/打乱，尚不支持“源熵识别了哪些证据更值得采纳”。Source-ACD 比 ACD 多保留一题，但仍不如 incumbent，且本轮没有专门的 source-ACD 同均值/打乱控制，不能把这一个差异归因于 source 信息。常数匹配的是**同属性、按病例平均**强度；答案长度变化使逐 token 加权均值不必相同。

严格 leading yes/no：incumbent 9/12 有显式开头，其中 8 例匹配参考；source-ACD 6/12 有显式开头，其中 4 例匹配参考。它不是完整语义重评，说明不能只看 ANCHOR 一个数字。12 题/10 簇的 bootstrap 信息有限，不支持稳定推广结论。

## 最重要的机制观察：低 uncertainty 不保证证据被正确理解

一个 TRAIN Pneumothorax 病例的原生阳性分数为 **0.003228**、U=**0.03136**，参考为 no。原答案也是否定；加入所问属性文字后却回答存在气胸，source-only 同样如此。另一个 Cardiomegaly 病例 p=**0.001618**、U=**0.01734**，也从否定变成肯定。

这些是源输出方向与语言解读方向不一致的具体反例。低熵只表达原生分类器在其标签空间内的确定性，不能说明接收模型会忠实解读数值和否定性。把 1-U 直接加大，可能放大的是错误解读后的条件分支。XRV 原生二元 argmax 在这 12 题中匹配参考 9 题（仅离线诊断，不作为诊断阈值），而全部属性文字分支只被 ANCHOR 判对 5 题；不能把所有伤害简单归咎于分类器本身。

## Verifier 结果

预先固定 source-ACD 为候选，原 incumbent 为保留选项。输入不含参考答案、源分数、alpha 或候选来源。每对双顺序，只有两次都选候选才替换。

| 分支 | 模型调用 | 原始输出/问题 | 替换 | 最终 ANCHOR 判对 |
|---|---:|---|---:|---:|
| Qwen 独立视觉 judge | 24 | 23 次 TIE、1 次 A；无双顺序一致替换 | 0 | 11/12 |
| 同 Qwen + 同尺寸空白图 | 24 | 24 次 TIE | 0 | 11/12 |
| 同 LLaVA-Med 自验证 | 24 | 15 次空输出，其余未遵循标签格式；12 对无效 | 0 | 11/12 |

原候选池内可救回病例为 **0**，会伤害 incumbent 的为 **2**。因此最终 11/12 只是保留原答案，**不是 verifier 改善，也不能说明它识别了两个坏候选**。视觉与空白图最后决策一致，没有建立视觉验证机制证据。所有调用均保留，不把空输出计为正常拒绝。

## 实际成本（累计计算时间，不等于双卡墙钟时间）

- 源模型推理：12 次，合计 **0.794 s**；源模型加载 **1.378 s**。
- 全部多臂病例累计 **246.773 s**，其中 parity 复核 **51.400 s**，不要再重复相加。
- Actor + judge 四次加载累计 **80.118 s**（含双卡 canary 后重启），单列启动开销。
- 继承旧 incumbent 的历史成本 **7.405 s**，不冒充本轮免费计算。
- 当前 incumbent 生成合计 **11.631 s**；source-only **22.282 s**；source-ACD **15.970 s**。
- 跨模型视觉 judge 双顺序累计 **7.406 s**；空白图 **5.490 s**；self-judge **6.329 s**。

源概率一次计算可复用，不必每个 token 重新运行分类器。显存空闲并不是新增复杂 verifier 必然划算的理由。

## 命令与原始产物

```bash
cd /home/dbw/merit-feddg-uncertainty
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 scripts/with_gate_kb_env.sh \
  scripts/run_uncertainty_train.py --stage prepare \
  --output runs/uncertainty-train-v1 \
  --gpu-uuid GPU-3846413a-4238-d307-b1f3-10c2dfbe002c

# 容器 GPU0 = 物理 GPU1；宿主机 GPU0 用 shard 1 和对应 UUID。
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 scripts/with_gate_kb_env.sh \
  scripts/run_uncertainty_train.py --stage generate \
  --output runs/uncertainty-train-v1 --shard-index 0 \
  --gpu-uuid GPU-3846413a-4238-d307-b1f3-10c2dfbe002c

scripts/with_gate_kb_env.sh scripts/evaluate_uncertainty_train.py \
  --run runs/uncertainty-train-v1
```

Prepare 拒绝覆盖已有目录；以上命令记录实际流程，不应重做已存在的 prepare。生成可按相同身份续跑，evaluator 拒绝覆盖旧评分。宿主机 UUID：`GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023`；双卡 tmux 名为 `uncertainty-gpu1-pilot` / `uncertainty-gpu0-pilot`，现均已完成。

本地原始数据：`frozen.json`、`sources.json`、`cases/*.json`、`evaluation.json`、`pilot-complete.json`、`gpu0/1-canary.log`、`gpu0/1-pilot.log`；均在上述运行目录。GitHub 仅上传源码、测试、调研、脱敏聚合报告，不含图像、病例原文、模型或凭证。

## 决策

不根据这 12 题改 alpha，不启动全量测试集。当前首先需要检验数值证据的正/负含义是否被 faithful 地送入语言分支，再研究 U 是否能预测采纳收益。跨模型 judge 需要先通过具有真实好/坏候选、顺序稳定性和图像依赖的能力测试；不可通过放宽解析把全平局或无效自验证改写为成功。文本专家语义熵、分割不确定性和检索不确定性仍未实现，不能计入本轮成果。
