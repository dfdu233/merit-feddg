# Codex 执行交接：范围绑定与可逆共享视图

在 `dfdu233/merit-feddg` 的 `implementation/plug-observe-bindings` 分支执行工程验收，再按条件进行完整 TRAIN 实验。先读 `docs/PLUG_OBSERVE_BINDINGS.md`、`docs/PLUG_OBSERVE.md` 和原 Codex handoff。本分支基于 `7381f9687782bc6c6b4a00402a6c608c906c35ef`；先核实真实 HEAD/PR，不从旧会话猜版本。

## 保护与运行条件

先检查 git status、工作树、授权 GPU 和现有虚拟环境。创建独立 worktree，不 reset --hard、clean -fd、强推、自动 merge、不覆盖原 configs/runs。不升级共享依赖，不重下权重，不重启 GPU/容器/宿主机，不杀他人进程。只有明确授权 GPU 可用；CUDA 检查设置超时，失败就停止 GPU 阶段。

整个协作过程免训练：不训练 router/adapter/bridge/gate，不温度拟合，不用 test 标签选择规则，不新增跨病例反思。原 semantic_all/compact_rows/compact_all 保留。新方法仍只有 incumbent、plug_read、plug_static、plug_agent 四臂，没有新增临床真伪 gate。

## 工程检查

```bash
python -m pytest tests/test_plug_bindings.py tests/test_plug_observe.py tests/test_evidence_agent.py -q
python -m merit_feddg.plug_run --help
```

在现有依赖允许下运行全库 pytest，分别报告当前新旧测试与全库结果。不能删除失败测试或偷偷安装新版 torch 修饰结果。查看 PR 的真实 CI 状态。本地 40 个新增合成测试不是医学模型验证。

必须检查：相同 payload 在不同 scope/ROI 下 ID 不同；原生分数、否定、单位、null 和缺失在共享布局反展开后完全一致；改动 content 后不能绕过绑定检查；未知坐标不能生成 ROI；空检测不产生全图框；派生观察必须保留真实父引用。

## 两种布局、同一完整清单

找到完整合法的 TRAIN BASE 和相同 ID 集合的 MANIFEST，优先使用已经固定的 compact_rows，不依据 test 重新选择。BASE 有完整 protocol、generation_config、原生 evidence 和 routed modality。references 只用于离线评分，不进推理。

给 BASE、MANIFEST、ARTIFACTS、OUT_SHARED 赋真实路径，执行：

```bash
python -m merit_feddg.plug_run \
  --base-run "$BASE" --incumbent compact_rows \
  --manifest "$MANIFEST" --artifacts "$ARTIFACTS" \
  --config configs/plug_observe_shared.yaml --output "$OUT_SHARED" --check-only
```

保留打印的 output_root；check-only 不代表 GPU 运行成功。plain 对照使用 `configs/plug_observe.yaml` 和另一个 OUT_PLAIN。除了 content_encoding 和输出目录，保持同一清单、模型、插件、预算和 source corpus。新协议统一中性自由生成，会生成新的同协议 incumbent；历史 CE/OE 结果不能直接当作新结果比较。

随后用相同参数去掉 --check-only，添加 --canary-cases 2。canary 仅是完整清单的调度停止点，不创建新子集，不报告医学总体表现。遇到两个调度样本恰好没有有效空间输入时，应补做单独标记的工程测试来验证通路，不能以“无动作但正常退出”验收区域协作，也不能按参考答案挑样本。

确认原图 final answer 调用真正开始、返回合法 token_ids；回答与 incumbent 相同可以是真候选，fallback 绝不能计为候选。明确记录 protected_context_exceeds_budget、no_new_packet_fits、parent_not_presented、生成未启动和空输出。没有真实候选时不要继续全量，不通过放宽阈值或删医学字段制造通过。

检测插件只接入已经有权重/适配器的模型。其 `payload.detections` 每项必须包含原生 label、归一化原图 xyxy box 和 `coordinate_system: original_image_normalized_xyxy`。坐标转换由 adapter 承担。检测测试用合成插件不代表已增加真实医学检测能力。

## 专家与知识库

`--expert-registry` 只添加新 ID，不覆盖旧检查点。scope 与模态来自模型能力说明，不为个别病名/失败问题写专属规则。区域输出不能被标为临床确诊。记录实际执行、返回、送达与生成覆盖率，不把配置注册率当能力覆盖率。

没有审核过的 source manifest 时 RAG 保持关闭。使用 `--source-manifest` 时，对 plain/shared 的预检、canary、完整运行、合并保持一致，检查图像/患者/study/近重复隔离和数据许可。未知患者 ID 要明确说明。源病例结论保持源图归属，不写成当前患者事实。

## 全量与评价

canary 确认通路有效后移除 --canary-cases，在相同完整 TRAIN 清单继续。默认单 GPU；只有两卡都授权时使用两分片及 --merge-only。不可混用身份不同的缓存、填补缺失行或覆盖旧结果。

完整合并后离线调用原 `agent_evaluate` 和冻结 ANCHOR。记录诊断分数与原评分器口径；不要把 token recall 命名为医学准确率。按图像聚类计算配对区间，分别统计改善/伤害及候选送达。plain/shared 应另外报告实际 tokenizer 输入数、模型调用、静态/动态实际动作、停止原因、预算不足；共享布局对小包可能不省 token，必须如实报告。

对每个新专家，区分原生能力强弱与协作收益。当前框架没有保证加入专家永不退化，也没有临床风险保证。结果不如 incumbent 就保留旧方法，先报告原因，不自动再堆一个 gate。禁止在本轮引入在线训练或根据 test 调参。

最后给出 commit、配置/manifest/模型/source hash、设备、完整命令、逐阶段结果与成本、少量有真实依据的成功/失败案例，以及未解决问题。未跑项目不填造数值。不要把受限图像、患者原始数据、权重或密钥提交 GitHub。必要兼容修复单独提交并保留差异；不自动合并本 PR。
