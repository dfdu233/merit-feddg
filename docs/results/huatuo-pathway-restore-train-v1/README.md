# Huatuo baseline visual-pathway restoration: existing TRAIN diagnostic

2026-09-20 UTC. **本轮不支持“恢复最后层视觉路径能减少原 MERIT 的损害”这一效果主张。** SLAKE128 全队列已处理和离线评分；restore_vector 无技术失败但未修复原有损害，restore_mass 在 63 例遇到无定义比值。VQA-RAD87 的固定首例 canary 遇到相同问题，未启动完整队列。没有训练参数，也没有运行官方 TEST。

## 冻结范围与代码

实现基础 `7cd12f041264ce3ab8ed0e0c53031b04ab4d8396`；最终生成代码为
`b51dc3f13b5911c7a0c699767110b06e81c9781f`。后续结果、评估脚本与测试提交不改变生成源码。
分支 `experiments/huatuo-pathway-restore-train-v1`；未合并 main 或 PR #13。

固定 `--layers -1 --modes restore_mass restore_vector`，只干预最后 decoder 层当前最后 query。
仍使用 `visual = A_visual @ V_visual`，vector 模式补偿完整视觉向量差，mass 模式补偿视觉 mass 比值对应的残差。
未加入 epsilon、阈值、专家、裁判、过滤器、训练模块或其他层/头选择。
参考分支消费 receiver 已提交的相同 token IDs；轨迹分叉后它不是独立 greedy Baseline。

本机真实问题及修复：

- `6658fbb`：按生成 scores 步数提取原生 token，排除原生起始 BOS；保留 BOS 对 repetition penalty 的影响。它不计入输出步数或 min_new_tokens。
- `0c2cbbd`：保持源后端独立的输入长度上限与 KV 生成上限，逐例精确比较历史 prompt 哈希和 evidence transport。
- `4aadb81`：模型配置 JSON 整数键规范化，修复 canary 到 run 的身份比较。
- `b51dc3f`：不改变 mass 数学定义，将无定义 mass 的真实失败及已发生的成本写入逐臂结果，继续其他臂和完整固定队列。canary 仍严格失败停止。

## 数据覆盖与输入一致性

SLAKE：原队列 128/128 已处理，generalist/compact/vector 均 128/128 有完整预测；mass 65/128 有完整预测，63/128 明确技术失败。所有计划样本均保留。原序列和前缀没有改变。128 例的原图字节、像素哈希、缓存身份和冻结依赖均核验；历史 prompt 哈希与整个 transport **128/128 完全匹配**。

另外独立检查了原始 TEST 清单哈希及图像像素集合：SLAKE128、VQA-RAD87 都为零 TEST 图像重叠。这是图像层面重新审计；runner 自身只继承历史审计。没有 TEST 推理、评分或重新划分；患者级隔离未知。

VQA-RAD 原完整 87 例队列及依赖通过静态核验，但本轮 0/87 完成逐例产物：首例 canary 的 mass 模式失败，86 例未尝试，vector 尚未执行。见 `vqarad87/blocked.json`。不将其旧报告分数作为本轮成绩。

## 同口径结果

评分严格复用源队列的冻结 ANCHOR CLOSED parser + OPEN first-reference token recall。离线进程才读取参考答案。两个 scorer 的哈希与历史冻结报告相同；重新计算的 generalist/compact 逐例分数均 128/128 对齐。不是临床准确率，OPEN 指标可能对额外错误内容不敏感。

| 臂 | 完整预测/计划 | 全 128 例分数 | 说明 |
|---|---:|---:|---|
| generalist | 128/128 | 58.6068% | 原缓存复用、重新评分 |
| compact | 128/128 | 57.4349% | 原缓存复用、重新评分 |
| restore_mass | 65/128 | 无完整可估计值 | 完成子集 53.2308%；不能与上面直接比较 |
| restore_vector | 128/128 | 57.0443% | 真实双分支干预 |

mass 缺失的 63 例保持 null，不伪造答案或评分。完整计划分母的理论评分范围为 [27.0313%, 76.2500%]（将缺失分数分别界定为 0 和 1，仅为界限）。四臂共同完成的 65 例：generalist 60.1538%，compact/mass/vector 均 53.2308%。这是显式标注的共同完成子集，不能证明全队列无损。

原 compact 相对 generalist：14 例评分改善、14 例损害。vector 修复 **0/14** 损害，保留 **14/14** 改善；相对 compact 为 0 改善、1 损害、127 不变。原 generalist 满分而 compact 非满分的 12 例，vector 修复 0/12。mass 可评价的原损害 8 例中修复 0，另外 6 例失败；可评价的原改善 3 例全部保留，另外 11 例失败，不能算保留或丢失。两模式均无 generalist/compact 都非满分而新方法满分的新增候选。

vector 相对 generalist 为 -1.5625 个百分点，按图像配对 bootstrap 95% CI [-8.7891, +5.6641]；相对 compact 为 -0.3906 个百分点，CI [-1.1719, 0]。固定 seed=0、2000 次，128 个像素图像组。mass 的配对结果只在完成子集上计算，完整分母和失败另列。

唯一 vector 评分下降发生于中文开放题，token recall 从 1 到 0.5；这不能直接等同于医学事实错误。原始参考及输出对照只保存在本地 `runs/pathway-slake128-v5/offline-diagnostics.json`。本轮未发现评分改善，更没有独立核实的医学事实纠正。

## 真实干预与新增成本

实际 kernel 为 `Qwen2Attention` eager，未全局替换 kernel；Qwen2 28 query heads / 4 KV heads，保持 V 偏置，O 差分投影不重复加偏置。原生展开确定视觉 span；两分支原始像素与视觉 embeddings 严格相同。KV 独立，正常/异常均移除 hooks。

| 项目 | restore_mass | restore_vector |
|---|---:|---:|
| 实际发生补偿的病例（含随后失败） | 76 | 96 |
| 完整预测且实际干预的病例 | 33 | 96 |
| 干预步数 | 231 | 839 |
| 已记录零残差步数 | 0 | 0 |
| 相同输入 bypass | 32 | 32 |
| 完整输出 token 序列改变数 vs compact | 1 | 7 |
| reference forwards | 294 | 839 |
| receiver forwards（包含失败尝试） | 423 | 968 |
| 生成 tokens（mass 含失败前缀） | 360 | 968 |
| 同步双分支解码秒数 | 47.5635 | 71.2656 |

额外预处理共 17.2677 秒；本次最终 SLAKE canary+run 两次加载共 42.3958 秒；audit 0.8939 秒、off 对照 0.8937 秒。原生 parity 生成耗时未独立记录，早期失败尝试和 VQA 失败尝试也不包含在上述解码合计中。它们不是零成本，不计算部署加速比。

两模式观测峰值 allocated 20,621,275,648 bytes（19.21 GiB），reserved 31,388,073,984 bytes（29.23 GiB）。reserved 包含 allocator 保留池，不等同于激活大小。GPU UUID、导入路径、版本、模型配置、权重实际 SHA256 和 generation JSON 随包提供。

逐例 receiver/reference visual mass 及 delta_norm 的统计在 JSONL 中；完整逐步观测留在本地。没有保存的干预后完整 attention 不作重建或补造；mass 的目标是 reference mass，不声称 attention 重新归一化。零 mass 原因可能涉及原生低精度下溢，但本轮没有额外对照来证明原因。

## 验证、限制与复现

103 项当前环境相关 CPU/接口测试通过，0 失败、0 跳过；编译、CLI、差异检查通过。最终 SLAKE 固定前两例均通过 historical/native/off token parity、audit parity，且均真实执行专家 attention。63 次 mass 失败都记录 hooks 已清除。早期失败目录均保留，不混用身份。

本轮结果未支持所设干预能修复负迁移。vector 保留了已有纠错，却没有减少原有损害；mass 数值无定义阻止了可靠全队列输出。没有非视觉/随机路径对照，不能归因于视觉路径特异性，也不能据此否定所有其他层的机制。不要为改善成绩更改层、kernel 或 mass 公式。

查看 `commands.md` 获取本机可直接运行的启动、恢复和评分命令。完整结果、分项、逐例改善/损害、失败分母、bootstrap、内部统计、耗时及出处分别位于 `slake128/summary.json`、`slake128/per-case-results.jsonl`、`slake128/provenance.json` 与 `validation.json`。

数据致谢：SLAKE，Bo Liu、Xiao-Ming Wu 等，*A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Question Answering*，ISBI 2021；本地数据卡标注 CC BY 4.0，[项目页面](https://www.med-vqa.com/slake/)。VQA-RAD 本地数据卡标注 CC0 1.0，[数据页面](https://osf.io/89kps/)。仅发布匿名标识、模型生成短答案和数值；不上传原图、参考答案、患者映射、原生专家报告、模型权重、激活张量或凭证。
