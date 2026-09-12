# Codex 执行交接：Plug-Observe

在 dfdu233/merit-feddg 的 `implementation/plug-observe-v1` 上执行工程验收和完整官方 TRAIN 实验，先读 `docs/PLUG_OBSERVE.md`。不是只解释代码。不要修改旧方法/评分器，不要根据 test 分数调参。

先检查真实 HEAD、git status 和未提交改动，优先创建独立 worktree。不要 reset --hard、clean -fd、强推、自动合并 PR、覆盖旧配置或 runs。记录真实提交。旧 semantic_all/compact_rows/compact_all 和 Agent-v1 必须保留。

本次新增入口是 `python -m merit_feddg.plug_run`，不需要 LangChain/LangGraph，不新增训练、校准、拟合权重或测试时优化。复用现有 Python 环境、已取得的权重和数据，不升级共享依赖、不自动重下模型。只使用明确授权的 GPU；有超时的 CUDA 检查失败就停止，不能重启机器/驱动或杀他人进程。

找到与完整 TRAIN manifest 对应、已完成的 no-gate BASE，优先已选定的 compact_rows。BASE 有 protocol.json 和 compact_rows.json，shards_complete=true，样本 ID 完全一致且包含原生 evidence。新协议所有题统一自由生成：旧 CE/OE 提示产生的历史分数不能直接作新比较基准。新代码会重生成相同中性提示下的 incumbent，旧源结果保持不变；参考答案和 answer_type 不进入新 Agent。

赋值真实的 BASE、MANIFEST、ARTIFACTS、OUT 后执行：

```bash
python -m pytest tests/test_plug_observe.py tests/test_evidence_agent.py -q
python -m merit_feddg.plug_run --help
python -m merit_feddg.plug_run \
  --base-run "$BASE" --incumbent compact_rows --manifest "$MANIFEST" \
  --artifacts "$ARTIFACTS" --output "$OUT" --check-only
```

check-only 只检查输入与协议，不表示 GPU 成功。记录打印的 output_root 为 ROOT。依赖缺失不升级共享环境；分别报告新测试、旧测试和完整回归结果，不能删除测试制造通过。

用相同参数去掉 --check-only，加入 --canary-cases 2。canary 是完整清单的调度停止点，不另切数据集，不汇报整体医学效果。检查：
1. incumbent 确实使用原有 renderer 和统一新提示生成；plug_read 不调用新专家。
2. 模型视图保留原生分数、否定、单位和 scope；原始 mask/哈希留在 artifact。
3. observe_region 是一次完整 crop+read；planner 看到名称和坐标，不是匿名 crop ID。
4. whole-image 分类/检索不依赖 mask；新 expert ID 沿用 infer 协议。
5. 真实新增观察被送达后，answer 模型调用确实开始并产生候选。相同答案也可是真候选，fallback 不可冒充。
6. 明确区分 protected_context_exceeds_budget、no_new_packet_fits、no_new_observation、空模型输出和真实生成。零候选不代表医学 gate 可靠；不要跳过这些问题直接放量。
7. 原图保留；患者方位未知就保留未知，不能把屏幕左右推为患者左右。

没有 source manifest 时 RAG 会明确关闭。已有审核过的 source-only 库才能通过 --source-manifest 加入，并保证预检、canary、全量、合并都一致。全清单图像/患者/study 与源库隔离；未知患者 ID 明确标注限制。源病例内容只属于源病例。不要自行下载大知识库或伪造 source 数据。

要接入新的已下载专家，可用 --expert-registry 指向只含 experts 的 YAML；只准加新 ID，不覆盖旧模型 ID。模型配置声明自身模态/任务/输入与原生能力，不为具体坏例增加疾病专属规则。尚无适配器/权重就报告未接入，不虚报覆盖。

canary 确认真实候选路径可达后，移除 --canary-cases，继续同一完整 TRAIN 清单。默认单张授权 GPU；只有两卡均授权时才采用 --shard-count 2 和 --shard-index 0/1，完成后 --merge-only。所有缓存绑定代码、配置、图像、专家和源库身份，不填补缺失行，不混不同身份结果。

全量完成才离线评价：

```bash
python -m merit_feddg.agent_evaluate --run "$ROOT" \
  --manifest "$MANIFEST" --references "$REFERENCES" \
  --anchor-root /home/dbw/ANCHOR --output "$ROOT/evaluation.json"
```

报告 incumbent、plug_read、plug_static、plug_agent 四臂。区分接口效应、额外工具收益和动态选择收益；额外整图观察对照使用 observation_view: whole 的独立配置和同一完整清单。不要把它与原图或局部 ROI 的语义混用。

至少汇报真实候选/内容送达覆盖、改善和伤害、开放题与闭式指标、逐专家/原子操作、规划停止、未知与超限、真实调用和耗时、按图像聚类的配对区间。单列历史源端准备成本：旧 seconds 包含旧回答生成，只是上界；共享 warm 模型与新生成成本不能冒充冷启动部署时间。

本版本没有训练式或临床真伪 gate，不承诺新增专家永不降分；局部读取仍可能错。新方法不如 incumbent 就如实报告，不能改 test 阈值、删除坏例或把单元测试当 ICLR 创新性。报告真实 commit、配置/manifest/模型/知识库身份、命令、日志、能力覆盖和剩余阻塞。受限图像、患者记录、权重、密钥不要上传 GitHub；必要兼容修复单独提交并保留 diff。
