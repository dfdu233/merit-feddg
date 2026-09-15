# 给服务器 Codex 的完整执行 prompt

请验证并运行 dfdu233/merit-feddg 的 `implementation/control-referenced-steering-v1` 分支。
任务是严格 training-free 的 CRES 空间证据对照残差解码实验；先读
`docs/CONTROL_REFERENCED_STEERING.md`、本文件和该分支 PR。不要凭方法名字默认有效。

1. 先只读检查 `/home/dbw/merit-feddg` 与 `/home/dbw/merit-feddg-soft-guidance` 的 branch、
   HEAD、dirty files、tmux/GPU 任务以及已完成实验。特别检查旧 full soft-guidance 的
   `complete.json` / `evaluation-summary.json` 是否已产生；汇总真实新结果，但不要暂停
   正在运行的其他工作。拉取新分支，在独立 worktree `/home/dbw/merit-feddg-cres` 工作，
   记录实际 SHA，不覆盖/合并 main，不改旧工作树。此分支基于 d20c928。
2. 使用已有 Python/模型环境和已分配给 dbw 容器的 GPU。核查容器编号到物理 GPU 的
   实际映射；不因文件中出现旧双卡安排就扩展本轮占用。不升级共享依赖、不下载模型，
   不读出凭据、不启训练。遇到资源冲突或数据缺失，完成可做的 CPU 审计并明确报告。
3. 在新 worktree 使用现有环境运行下面六组测试及 Ruff/compile。父分支还有完整旧
   测试；能运行则执行全仓回归，区分现有缺依赖/缺数据和本次新增回归，不能删除断言
   来得到通过。修复局部工程 bug 可以继续，方法/ε/strength/输出提示不得按测试成绩修改。

```bash
python -m pytest tests/test_control_evidence.py tests/test_control_study.py \
  tests/test_soft_guidance.py tests/test_spatial_evidence.py \
  tests/test_verified_packets.py tests/test_full_soft_guidance.py -ra
python -m ruff check merit_feddg/control_evidence.py merit_feddg/control_study.py \
  scripts/run_control_evidence.py scripts/evaluate_control_evidence.py \
  tests/test_control_evidence.py tests/test_control_study.py
```

4. 从原 `runs/` 中定位 VQA-RAD 451 和 SLAKE 2094 的完整无标签 manifest，及其匹配的
   verified-packets source run：必须含 protocol.json、routing.json、case-cache/compact_rows。
   可参考父分支 `scripts/run_soft_guidance_pilot.py` 的 SOURCES，但要验证磁盘实际文件。
   本轮读取 cached native experts，不复用旧答案作新 prompt 的对照。核对每例实际图像
   文件 SHA、路由 group_id、source cache identity、旧 prompt/evidence delivery audit。
   如旧 image_sha256 的语义不是 file SHA，调查生成清单的代码并记录映射后再修复适配，
   不直接重写 hash 或跳过检查。references 只给独立 evaluator。
5. 默认 `configs/control_evidence.json`：uniform 64-token 简答、ε=0.05、max_strength=1、
   fixed_strength=0.5、ablations=true，全部八臂。CE/OE 采用完全相同的新增推理提示。
   这与旧 ANCHOR 提示的数值不可直接比较，所以 generalist/compact 都 fresh regenerate。
   不新增数据划分；`--max-cases` 仅暂停同一全量任务，已完成病例续跑复用。

```bash
# 先将以下三个变量绑定到已核实的绝对路径；路径不能指向 references。
python scripts/run_control_evidence.py --source-run "$CRES_SOURCE_RUN" \
  --manifest "$CRES_MANIFEST" --artifacts "$CRES_ARTIFACTS" \
  --output runs/cres-vqarad --check-only
python scripts/run_control_evidence.py --source-run "$CRES_SOURCE_RUN" \
  --manifest "$CRES_MANIFEST" --artifacts "$CRES_ARTIFACTS" \
  --output runs/cres-vqarad --max-cases 2
```

6. 运行器会打印具体 ROOT。检查两个记录均完整、fresh zero parity 通过、原生 geometry
   operator delta 非零、两种 controls 的 SHA 不同且确实改变 regions、各分支使用相同
   prefix、未读取参考答案、CRES token KL≤0.05。若头两例没有可用空间证据，按原完整
   manifest 顺序继续 scheduling，直到出现真实空间调用；不要按参考答案选样本。对照
   退化属于 unavailable；runtime/OOM/NaN/parity failure 不能算 abstention 或收益。
7. 在读取本轮成绩前 pin 原有 ANCHOR scorer。先估算实测每例成本；若工程检查通过且
   GPU 可用，tmux 中移除 `--max-cases`，续跑同一完整 VQA-RAD 和 SLAKE。不要用 CRES
   输出调整 ε/strength，不对失败例改疾病规则。遇到工程错误停止该作业，保留日志后排查。

```bash
python scripts/evaluate_control_evidence.py --run "$CRES_RUN_ROOT" \
  --anchor-root /home/dbw/ANCHOR --freeze-scorer
# 只有该完整 manifest 的 complete.json 成立后才执行评分：
python scripts/evaluate_control_evidence.py --run "$CRES_RUN_ROOT" \
  --anchor-root /home/dbw/ANCHOR --references "$CRES_REFERENCES"
```

8. 报告全部八臂的完整 N、CLOSED/OPEN/mixed 分数，对 generalist、compact、deletion 的
   paired delta、图像聚类 95% CI、改善/损害、revision coverage、harm_given_revision、
   controls unavailable、残差为零比例、α/KL 分布、score API 调用与真实 wall/startup 时间。
   不将 score API calls 等同底层 forward 次数；零覆盖条件风险为 undefined，不能称安全。
   分别回答：是否超过 deletion？是否超过已有 compact？是否仅 KL 限幅就够？真实
   几何是否优于错位几何？减害时是否保住语义证据收益？
9. 不从当前两个已用于开发的数据集挑阈值或宣称 target-free 泛化。若信号弱、全面回退
   或无增量，就明确否定当前假设，不自动增加新的 gate/模型/训练。更强底座目前 native
   adapter 尚未实现，需要另行等协议验证，不能只替换模型名声称完成。
10. 将无患者原文的 aggregate summary、版本/环境/实际命令、测试结果与失败诊断提交
    到同一新分支，推送更新 PR；不上传病例图片、原始 masks/answers、权重、凭据，不
    自动合并。最终报告区分 CPU fake-model 验证、真实 scheduling 检查和完整临床任务评分。
