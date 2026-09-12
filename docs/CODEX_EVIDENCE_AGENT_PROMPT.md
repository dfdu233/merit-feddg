# 给服务器 Codex 的实验执行 Prompt

请在现有服务器上实际执行 MERIT 的 evidence-agent-v1 工程验收和训练集实验，不要只给建议。
先阅读仓库 `docs/EVIDENCE_AGENT_V1.md`、`docs/VERIFIED_PACKETS.md`、`STATUS.md`，再检查代码和真实资源。

## 约束

1. 新分支是 `implementation/evidence-agent-v1`，基于 `implementation/evidence-revision-v2` 的提交 `cec7bb9f93c4869a8197e285da8203cbef04b054`。不要直接切换、覆盖有未提交修改的工作目录。先 `git status`，优先 fetch 后建立新 worktree。记录实际新分支 commit。禁止 reset --hard、clean -fd、强推、自动合并或覆盖旧实验。
2. 保留全部旧方法、配置和结果，尤其 semantic_all、compact_rows、compact_all；本次只走新入口。所有模型冻结，不训练 bridge、router、adapter、gate，不做标签校准、测试集阈值选择或跨病例策略学习。不要用“无梯度”冒充“无校准”。
3. 复用现有解释器、LLaVA-Med 源码、权重与数据。新模块不要求安装 LangGraph/LangChain。不要升级共享环境、重新下载模型或数据。需要相对路径时，在新 worktree 中为已有 artifacts/upstream 等建立不覆盖现有文件的链接。不能定位的资源记为真实阻塞，不伪造目录或自动替换模型。
4. 只使用明确授权的 GPU。先核对 dbw 容器可见设备与物理卡对应关系，用带超时的检查，CUDA 健康检查失败就停止 GPU 阶段。不要重启宿主机、驱动、容器，不杀他人任务，不在宿主机擅自启动双卡。
5. 所有生成和路由只读取无标签 manifest；references 仅在离线 evaluator 使用。不得使用目标真值 mask、目标报告或目标答案查询知识库。知识库严格 train/external，保留患者/图像去重审计。不要把相似病例答案当当前患者事实。

## 第一步：确认基础状态

运行 git 状态与版本检查，阅读新 CLI 的 --help。运行新增测试 `python -m pytest tests/test_evidence_agent.py -q`；在已有依赖允许时再运行仓库完整 pytest。不要为了测试安装/升级大批依赖；区分测试失败与缺失依赖。

从真实 runs 目录找最近、完整且能对应现有训练集清单的 no-gate semantic/compact 结果。优先使用既定 compact_rows，但不要根据新测试集分数重新选择它。BASE 必须有完整 protocol.json、shards_complete=true 和 compact_rows.json，每例有 generation_config、text、token_ids、原生 evidence、input_modality。确认 MANIFEST 与 BASE 完整 ID 集合一致；不要拿停跑的部分结果或不同协议混用。

如果当前完整结果在服务器未推送版本产生，先核对接口和协议，不要强行导入。能用原有 runner 在同一完整 TRAIN manifest 复现兼容基准时可执行，单列复现开销，旧结果保留；不能建立兼容性则报告原因。不要重切分数据或把已用于调试的测试集改称未见目标域。

## 第二步：预检和真实 canary

使用已确认的实际路径定义 BASE、MANIFEST、ARTIFACTS、OUT。OUT 必须是新目录。运行：

```bash
python -m merit_feddg.agent_run --base-run "$BASE" --incumbent compact_rows \
  --manifest "$MANIFEST" --artifacts "$ARTIFACTS" --output "$OUT" --check-only
```

记录 identity 和输出 ROOT。先不提供 source manifest，因为不存在的 KB 不能算作已经启用 RAG。已有经过明确审核的小型 source bank 才在所有命令一致加入 --source-manifest。

随后运行同一完整 manifest 的 `--canary-cases 2`。这只是调度停止点，不能评分为完整实验，不能改变 manifest 或样本选择。逐题检查 graph_noop 的文本和 token IDs 与 incumbent 完全相同；任何不一致立即停止新方法实验，定位预处理/提示词/版本/证据送达差异，禁止放宽 parity 检查。

检查已有分割 raw mask 是否真正形成 region -> crop -> inspect；原图用于最终合成，crop 只用于局部观察。空 mask、坐标不明、没有模型、预算不足应记录 unavailable，而不是伪造全图框或诊断阴性。检查 region_control 与真实 crop 是否不同；重合时不声称有位置对照作用。检查同源结果没有被当作独立多票。

默认 gate 是 audit_only；agent_committed 返回 incumbent 是设计预期，不是 gate 成功。需要看 agent_candidate、graph_static、region_control、full_image_control 的真实输出。若没有可用预测分割，报告覆盖率不足，不要移除保护或把 CAM 改称病灶 mask。

## 第三步：全量训练集确认

canary 通过后，移除 --canary-cases，在完全相同配置、清单和输出根下继续，复用已完成病例。默认单个授权 GPU。只有两张卡都明确授权时才做相同身份的 2 个调度分片；确认各进程设备，完整后 --merge-only。不得填补缺失行或只汇报先完成病例。Runtime/OOM 是工程失败，不算医学负证据。

不要一开始打开昂贵的 PE/SE/DSE/KLE，也不要在此阶段增加训练式 gate。external_compare 只是复用现有比较器的独立对照，开启须另存配置/输出和完整记录；不能因为 audit_only 不改答案就偷偷换成更激进的规则。

## 第四步：离线评价

只在根 protocol.json 表示完整后运行：

```bash
python -m merit_feddg.agent_evaluate --run "$ROOT" --manifest "$MANIFEST" \
  --references "$REFERENCES" --anchor-root /home/dbw/ANCHOR \
  --output "$ROOT/evaluation.json"
```

ANCHOR 必须是原有冻结版本。该命令不改原评分器；新增 lexical 指标只作诊断，不能称临床准确率。SLAKE 非二元 CLOSED 不能统统映射成 No；报告生成不能用本 VQA lexical scorer 替代事实指标。

汇报七个实验臂完整表：incumbent、graph_noop、graph_static、region_control、full_image_control、agent_candidate、agent_committed。分别给出相对 incumbent 的提升/伤害、文本变化、候选覆盖、gate 决定、工具真实运行/送达、无区域/缺资源/预算失败、所有模型调用和耗时，给出按图像聚类的配对 bootstrap。额外说明原有 generalist 与 incumbent 的差异，不把旧收益归因于新 Agent。

## 第五步：提交实验报告而不是宣传结果

输出 commit/config/manifest/model/KB 身份、设备和环境、完整命令、工程检查结果、全量表、典型救回/伤害、机制是否生效、训练集观察与未来 holdout 计划。列明哪些专家/检索资源尚不存在、哪些模块只完成工程验证。正确但不适用、稳定错误、正确但反对初答、重复同源证据都应检查。

原始患者图像、受限数据、权重、API key 和凭证不要提交 GitHub。可整理脱敏汇总文档放在新的结果目录；任何代码修复用独立提交，不改旧实验记录。若新方法不如 incumbent，如实报告并保留有效旧方法，不做测试集调参，不宣称 ICLR 创新性或医学收益已经成立。
