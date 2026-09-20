# MERIT：有限、可审计的 Training-free 算法实验链

## 交付状态

基于 `0be2c3cdc3a89f6119ac73d163c6436e3070798e`，只新增独立代码，不覆盖旧方法、结果或 main。
本次实现的是 A/B/C 实验算子、原生 Huatuo 接口、独立评分进程、D/E 放量判定、持久化控制器及测试。
**没有在本次实现环境运行真实 Huatuo、医学图像或服务器 GPU；没有新的医学实验成绩。**
CPU 验证的实际范围与源码哈希见 `reports/algorithm_chain_cpu_validation.json`。

起点证据来自 `docs/results/huatuo-pathway-restore-train-v1/`：最后层 vector 恢复没有修复原有损害；
mass 的数值无定义不是医学拒绝信号。因此不继续 epsilon 修补或遍历最后层强度。
以下 A/B/C 是新的、尚待实证的研究设计，不是已证实的最优方案，也不保证必然找到有效算法。

## 一、实验决策图

```text
A：首次决策分歧 + 两个预声明中层边界的方向性干预诊断
  ├─方向性证据通过 → B：固定一个全局边界，完整生成残差恢复候选
  │                     ├─开发筛选通过 → D
  │                     └─失败 / 证据不足 → C
  └─失败 / 证据不足 → C：同证据布局敏感残差投影解码
                        ├─开发筛选通过 → D
                        └─失败 / 证据不足 → STOP_NO_CANDIDATE

D：只验证已冻结的一个候选；使用未参与前序选择的独立 source 队列
  ├─通过 → READY_FOR_SCALE → 写入 scale_plan.json，不自动运行 TEST
  ├─明确失败 → STOP_FAILED_CONFIRMATION
  └─证据不足 → E：仅启用预先声明、与此前不重叠的一次扩展
                ├─D+E 合并通过 → READY_FOR_SCALE
                ├─明确失败 → STOP_FAILED_CONFIRMATION
                └─仍不足 → STOP_INCONCLUSIVE

任意节点：技术失败 → BLOCKED_TECHNICAL（原算法、原代码最多显式重试一次）
          缺少独立数据 → BLOCKED_DATA；达到前向预算 → BLOCKED_BUDGET
```

这里的 pass 是**节点目标通过**：A 的 pass 只是定位证据，B/C 的 pass 只是开发筛选，
只有 D/E 的 pass 才能批准更大规模的冻结研究评估。没有“失败就降低门槛直到通过”的循环。
D 开始后不能因结果不好切换到另一个候选，也不能把已经看过的确认数据重新命名为未见数据。

## 二、A：首次决策分歧与方向定位

在相同原图、相同已提交 token 前缀下运行无专家和原 compact 两分支。最多检查前 16 个
决策位置；第一次 greedy token 不同时固定竞争 token `a`（无专家）和 `b`（compact）。
若前缀内未分歧，记录 no_divergence；它不证明完整答案必然相同。

只诊断两个相对层位置：`floor(L/3)-1` 与 `floor(2L/3)-1`。
对于 28 层 Huatuo 是 **零起始层号 8、17**；不是从所有层/头中选最高准确率。
在每个边界比较：

- restore：只替换当前最后 query 的 block-output residual 为无专家分支对应向量；
- roll：相同残差差值沿 hidden 维循环平移一半后加入，保持投影前差值范数，作为方向对照。

记录原始 logit 差、原协议处理后的 logit 差、概率差、argmax、实际干预范数、前向数量。
重新 prefill 必须复现缓存分支的 greedy 决策；logit 数值差只记录，不能冒充逐位相同。
首次 token 差异可能只是措辞差异，A 不声称自动识别了事实声明。

推理端不读取答案。独立评分后，在原 compact 有改善或损害的病例上计算：

`directional_effect = -(score_compact - score_baseline) * (margin_restore - margin_roll)`

这里 margin 为 `p(a)-p(b)`。正值表示相对方向对照，更支持有益的分支方向。
这是 source 开发诊断，不是单例正确性估计，也不能用于 inference 的逐例标签选择。
两个边界分别做图像组 bootstrap，并分配比较误差预算；下界为正且覆盖足够时选择一个
**全局边界**进入 B。全局开发选择需要如实披露：training-free 不等于没有开发数据选型。

## 三、B：中层 residual relay 恢复

A 选定一个边界后，不再按题型、模态或标签改变层号。每个生成步：

1. 无专家参考分支消费当前已经提交的同一 token 前缀，提取该边界最后 query 的 residual。
2. 专家分支消费相同前缀，在同一边界替换该 residual，再由剩余层处理并决定下一 token。
3. 同一个 token 提交给两个私有 KV 流；专家 prompt token 的状态与后续层仍保留。

**这不是“纯视觉通路恢复”**：替换的是一个中层边界上的完整 residual，不是整个模型的输出，
也不是把 Baseline 完整答案重新贴入 prompt。正确专家可能需要改变该 residual，B 因而也可能
破坏有用纠错，必须同时统计原有收益保留与损害修复。

matched control 为 `relay_roll`，与 B 同为两流生成。只超过破坏性的 roll 对照不够：
还必须同时超过原 Baseline 和原 compact。D/E 中继续使用同一个控制，不临时换对照。

## 四、C：布局敏感协作残差的正交投影

C 不依赖 A 的层定位结果。固定原图、问题、实际已送达的原生专家包，只构造两个控制：
切换已有 compact 行/列编码；反转整个专家包顺序。**不改变包内的标签排序、数值、否定、
限定语或医学内容**。复用已有 `compact_records` / `compact_prompt`，必须逐字符重现中央
compact prompt；任一控制超预算即记录技术失败，不能改变实际送达专家集合。

四个私有 KV 流为：无专家、原 compact、控制 1、控制 2；每步共享已提交的 token 前缀。
令 `l=log p`，`W=diag(p0)`，所有向量先按 p0 中心化：

```text
r = lE - l0
N = [l_view1 - lE, l_view2 - lE]
r_perp = r - N (N^T W N)^+ N^T W r
p_new = softmax(l0 + r_perp)
```

小型 2x2 Gram 求解用 float64，原词表 log probability 用 float32；`rtol=1e-6` 是数值秩
判定，不是医学 Gate 阈值。奇异/重复控制可退化为低秩；没有布局变化时不伪装成去干扰成功。
为了保持原 repetition penalty 的语义，融合结果在处理器前重新定到中央分支的 logsumexp
偏移；这是明确的解码设计，必须通过原生关闭干预 parity，不应静默改变原生生成限制。

C 的 matched control 为 `layout_average`：三种证据布局的平均 log probability，也执行
同样的四次前向，排除仅因额外计算或普通集成产生的收益。

**布局稳定不等于正确，布局变化也不保证只包含干扰。** C 可能删除真实信息或保留稳定的
错误。此版本只实现上下文布局方向，不伪称实现了医院域泛化、图像扰动专家重推理或因果证明。

## 五、预声明门槛与放量范围

完整可执行配置位于 `configs/algorithm_chain_v1.json`。这些默认值是研究筛选约定，
不是已有结果拟合出的阈值，不是临床风险承诺；必须在运行前固定，不能在结果出现后放宽。

| 检查 | 默认约定 |
|---|---|
| A | 最多 16 个决策位置；至少 8 个有信息的图像组；改善和损害至少各 3 例 |
| B/C 开发 | 至少 32 个图像组；候选比 Baseline、compact 均提高至少 0.005；超过 matched control |
| 纠错结构 | 至少修复 1 个原有损害；保留至少 75% 原有改善；原 Baseline 满分病例损害比例至多 10% |
| D/E 覆盖 | 至少 64 个独立图像、至少 2 个预声明 source 队列，每队列至少 24 个图像 |
| 统计门槛 | 三个配对差值区间下界均大于 0；各队列相对 Baseline 的下界不低于 -0.02 |
| 风险/保留 | 损害比例 Wilson 上界不超过 0.10，收益保留 Wilson 下界不低于 0.75 |
| 时间 | 新候选同步解码平均不超过 10 秒/例；不是把缓存 Baseline 计为零来计算加速比 |
| 总预算 | 最多 200,000 次语言模型前向，含 canary、失败尝试；每节点最多一次显式技术重试 |

bootstrap 默认 4,000 次、seed 0。D 和 D+E 仅两个预声明完整队列观察，按两次观察、
三个整体对照、两个比例及各队列对照分配 `alpha=0.05`。这些经验区间不构成严格的临床
有限样本保证，最小样本数也不保证足够功效；区间宽时进入 E 或停止，绝不强行宣布成功。

比例界要求确认样本一图一问。开发诊断可包含同图多问，均值差按图像组 bootstrap；
当前实现拒绝用同图多问的 IID Wilson 比例界批准放量。不要为了过门在看到分数后删题。
需要多问确认时应在新预声明协议中实现适合的组相关风险界，不改变官方 TEST 的完整评测集。
两个 source 队列不自动等于两个医院域；放量许可只针对当前冻结接收模型和已验证适用范围。

所有计划病例留在分母，技术失败的预测及评分为 null。全回退、零改答、缺少原有改善/损害
覆盖不能通过。C 失败则本链结束；它不会自动添加更多专家、训练裁判或继续网格搜索。

## 六、运行与文件契约

入口：`python scripts/run_algorithm_chain.py --help`。
阶段命令：`policy`、`freeze`、`run`、`status`、`retry`；worker 和 score 为控制器内部子进程。
`run --once` 只推进一个节点，默认 run 按冻结图自动推进到终态或阻塞。

本轮已有 SLAKE128 / VQA87 结果均属于已被观察的开发数据，不能只改目录名作为新确认集。
计划必须列出所有此前使用过的像素哈希注册表与官方 TEST 像素哈希注册表。代码重新计算
实际 RGB 像素哈希，检查跨角色/队列/测试重叠；**无法凭空证明外部历史暴露注册表完整，
也没有患者级身份信息来证明患者隔离**。这两项需由数据维护者审计并保留限制说明。

每个 source_run 应复用既有原生入口的 `protocol.json`、`complete.json`、`cases/<id>.json`，
包含真实缓存 generalist / compact token 与专家包，且原生成 manifest 不带参考答案。
原历史 prompt/transport 必须严格匹配，不能用相同答案代替输入等价证明。

计划的顶层结构：

```json
{
  "schema": "merit-algorithm-chain-v1",
  "policy": "此处放 configs/algorithm_chain_v1.json 的完整对象，而非文件名",
  "development": [{"name":"slake-dev", "dataset":"slake", "split":"train", "source_run":"实际完整源队列目录", "generation_json":"源后端核验过的生成参数 JSON"}],
  "confirmation": [],
  "extension": [],
  "exposed_pixels": "所有此前观察图像的 SHA256 字符串数组 JSON",
  "test_pixels": "官方测试图像 SHA256 字符串数组 JSON",
  "runtime": {
    "factory": "anchor.corrected_sgta.models_oe:HuatuoOEAdapter",
    "kwargs": {}, "checkpoint_dir":"/home/dbw/models/HuatuoGPT-Vision-7B",
    "gpu_uuid":"已授权且实际可见的 GPU-UUID", "import_roots":["/home/dbw/ANCHOR"],
    "pins": {"绝对文件路径":"从实际文件计算的 SHA256"}
  },
  "scorer": {
    "import_roots":["/home/dbw/ANCHOR"],
    "references":{"slake-dev":"参考答案 JSON 的绝对路径"},
    "pins":{"评分源码及参考文件绝对路径":"从实际文件计算的 SHA256"}
  }
}
```

这是结构说明，不是可直接执行的虚构资产配置。runtime pins 至少包含 adapter 源码、原生
动态模型源码/配置及真实模型权重；scorer pins 必须包含实际导入的两个冻结 ANCHOR 评分函数
源码和所有参考文件。可用 `storage.digest(path)` 计算实际 SHA256，不填写占位哈希。
新增确认/扩展队列使用同样对象结构和对应 references 项，在 freeze 前完成声明。
没有新确认集时可让 A/B/C 开发运行，随后明确停在 BLOCKED_DATA，绝不把旧数据自动切成验证集。

```bash
# 在现有兼容服务器环境和独立工作树中执行，不升级 Torch/Transformers。
PYTHON=/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python
$PYTHON scripts/run_algorithm_chain.py freeze --plan /实际路径/plan.json --output runs/chain-v1
$PYTHON scripts/run_algorithm_chain.py run --output runs/chain-v1 --once
$PYTHON scripts/run_algorithm_chain.py status --output runs/chain-v1
# 完整持久执行：包装器只接受已冻结计划，不替用户选择 GPU。
bash scripts/run_algorithm_chain_detached.sh "$PYTHON" /实际绝对路径/runs/chain-v1
```

不得把示例目录当已在本次会话核验过的服务器资产。GPU 使用物理 UUID 检查可见性及已有
计算进程，不终止他人任务；这是启动检查与本链 flock，不是对所有外部程序的设备独占保证。

生成子进程不接收 scorer 或 references 路径，独立评分子进程禁用 CUDA。评分严格复用
`evaluate_medheval_answers.evaluate_rows` 的 CLOSED 判断和 `answer_token_recall` 的第一参考
OPEN 口径。生成器不按模态/closed-open 改 Gate 或候选；原 benchmark_prompt 原样保留。
原生前两例 canary 检查 historical/native/off 以及 audit parity，至少一例必须真实送达专家。
固定前两例均无专家时技术阻塞，不基于有利输出另挑病例。

`plan.json` 固定代码、权重、评分器、图像/缓存及全部数据角色。
节点产物在 `A-attempt-1/` 等目录：label-blind input-job、完整 predictions、独立 scores、
不可覆盖的 decision、逐例缓存、前向预算、日志、退出状态。
`state.json` 是唯一可更新状态指针，异常退出可恢复；失败与已花费预算不会抹除。
每次前向计数持久写盘会增加实验计时，该实现不宣称达到最优部署速度。
`READY_FOR_SCALE` 只发布冻结候选及证据链，不在源码中内置任何“拿 TEST 分数再选方法”的路径。

## 七、实现对应关系与复核

- `decoding.py`：实际 A/B/C 前向算子、私有 KV、处理器、匹配对照。
- `native.py`：现有 Huatuo 原生序列化/证据重建、只读专家缓存、评分进程。
- `statistics.py`：标签仅在离线分组、对照统计和全局节点判定中使用。
- `policy.py`：有限决策图、阈值、一次性扩展与重试预算。
- `storage.py` / `cli.py`：身份、泄漏检查、持久调度与放量清单。

代码复用依据：本仓库 `huatuo_pathway._Stream/prepare_pair`、`compact_evidence`、
`run_huatuo_pathway.restore_context` 和 `evaluate_huatuo_pathway`，均在指定基点核读。
线性代数 API 依据：https://docs.pytorch.org/docs/stable/generated/torch.linalg.pinv.html 。
`pinv` 的数值截断不能解释为医学可靠性筛选。

研究背景中的 activation patching 和视觉信息经文本位置转发，不证明本次候选一定有效。
尤其 B 的 residual 不是已分离的纯视觉部分；C 的布局控制也不是已验证的真实域控制。
如 A/B/C 全部失败，应保留完整负结果与明确停止原因，再提出新的假设，不在本链内临时换算法。
