# 给服务器 Codex 的执行交接：v0.10 能力通信优化

这份文档可以直接作为执行任务。先完整阅读本文、`V010_CAPABILITY_COMMUNICATION.md`、
服务器 `STATUS.md`，再运行。默认授权范围是本仓库代码核查、已有真实数据/模型的 source 实验、
生成报告；不得自动扩大到全量下载、核心环境重装或新的 target 调参。

## 可直接复制给服务器 Codex 的任务

> 在 `/home/dbw/merit-feddg` 中按 `docs/SERVER_CODEX_V010.md` 执行阶段 0–4。
> 目标不变：医学 VLM 在 decode 过程中使用可插拔原生小模型能力，并考虑跨域条件效用。
> 先检查工作树与 v0.9 结果，保留兼容性修改；复用已有 LLaVA-Med、工具权重及环境。
> 使用真实 source 数据做通信/模态/组合诊断，不使用 target 标签选择配置，不降低 DG 门槛来制造调用。
> 出现模型错误、baseline token 不一致、证据不可核验或缺少必要数据时，停止相应阶段并具体报告。
> 最后给出 source 的逐域证据、调用/呈现细节、建议选择及反例；事实正确性未经人工核验时如实说明。
> 本轮不要自动执行新的 target 性能验证，不把词面 F1 当成医学幻觉率。

## 阶段 0：保护现场并确定真实运行环境

1. `git status --short`、`git log -3 --oneline`、`git remote -v`。若有未提交修改，先查看差异。
   不 `reset --hard`、不清缓存、不覆盖模型适配器、不盲目 `stash pop`。需要合并且冲突时先询问用户。
   工作树干净后用 `git pull --ff-only`。不要把实验数据、访问 token 或大权重提交进 Git。
2. 查阅上一轮目录：

   `runs/native-v09-value-full-gpu1/llava/b8013cc570604684/`

   包含 `source-interventions.json`、`value-policy.json`，以及
   `evaluations/b0a71687490f275a/result.json` / `predictions.json`。
   source 分支正文在上一轮输出父目录的 `value-case-cache/`；按缓存键/记录对应，不能把目录中的
   所有 JSON 混成一轮实验。原始文件只读，输出新的核查报告。
3. 从上一轮 `prepared.json` 找实际生成配置，从其中验证模型、source 路径和 Python 版本。
   旧目录或文件缺失时，明确报告缺失；不要伪造“已核对逐例轨迹”。
4. 已知位置只是发现线索，均在远程服务器，不在 Windows 本地：

   - 模型：`/home/dbw/ANCHOR/hf_cache/llava-med-v1.5-mistral-7b`
   - 用户原始 Python：`/opt/miniconda3/envs/huatuo/bin/python`
   - 新 launcher 优先探测：`/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python`
   - 另一已有环境：`/home/dbw/.venvs/llava15-official-431/bin/python`
   - 用户原始 LLaVA 源码：`/home/dbw/ANCHOR/data/medheval/code/baselines/Mitigation/llava-med-1.5`
   - 当前默认配置源码：`/home/dbw/ANCHOR/data/medheval/code/baselines/Med-LVLMs/llava-med-1.5`

   两个源码目录可能只有一个存在/兼容。以成功的 v0.9 运行配置、import 检查为准，不猜选。
   检查 vision tower 本地路径，不让 Transformers 临时联网再下载。
5. 检查 GPU 空闲显存和其他任务。上轮总设备显存曾达 46.3/48.5 GiB；不能只看本进程
   PyTorch 峰值 23.9 GiB 就假设有余量。用已授权空闲设备的 `CUDA_VISIBLE_DEVICES`，不结束别人的任务。

核心限制：不运行 bootstrap；不执行 `pip install -U transformers torch`；不全盘安装 research extra；
不重新下载 7B 主模型；不删除 `.incomplete`、模型目录或已完成实验缓存。

## 阶段 1：读旧结果，形成三类待验证原因

创建新的 `runs/native-v010-audit/` 存放核查结果。重点查：

- 每个工具在每个 source 域的独立图像支持数；不是 293 条干预全部算独立样本。
- 被路由为 CXR 的图像是否确为胸片，尤其 gross specimen 误路由；只看图判模态，不看答案猜。
- CheXagent 的 source 正收益是否局限于某类问题；CONCH/retrieval 两例 target 受损仅作为开发期失败分析。
- XRV anatomy 导致的大退化：实际 mask、原图、叠加图、坐标映射、prompt 和输出长度。
- “单工具”必须写成“LLaVA-Med + 该工具证据”，不能偷换为工具独立任务准确率。

把此前反复查看的 32 个 target 标为 development/diagnostic cohort。分析旧样本不等于新的测试验证。

## 阶段 2：环境检查和真实小样本 canary

以下是默认路径仍可用时的最短命令；如不同，传 `--python`、`--config`、`--llava-source` 等已验证路径。
wrapper 默认 `--skip-download`，先验证资产。缺失时停止并列明缺什么，不取消 offline 约束自动大下载。

```bash
cd /home/dbw/merit-feddg
bash run_diagnostics.sh --chexagent on --check-only

# pytest 使用已安装测试依赖的项目 Python；不要为此替换模型推理环境。
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check merit_feddg tests
bash -n run_diagnostics.sh run_llava_med.sh run_value.sh
git diff --check

# 默认 PathVQA + VQA-RAD，真实非 yes/no 问题，每 source proxy group 2 例。
# target-limit=1 只供准备/拆分校验，诊断不会生成或评分 target。
bash run_diagnostics.sh --chexagent on --output runs/native-v010-canary
```

若 `.venv/bin/python` 没有 pytest/Ruff，先查既有开发解释器；不因此升级医学模型环境。
如果默认数据快照无法离线读取，优先使用上一轮的真实 manifest，见下一节。

canary 必查：

- baseline 与 block-NONE token 完全相同；异常会停止，不允许继续拟合。
- 每个真实工具事件的 request、scope、模态与实际图像相符；错误为零不等于临床正确。
- 同一原始专家输出在多种呈现间复用，原生 payload 不变。
- 叠加图与原图复制明确标记，原图始终是第一张；没有伪造 ROI。
- `target_generations=0`、`policy_fitted=false`；不产生新的 value-policy.json。
- 合法空证据如实记录；没有 CXR 样本时写“未覆盖”，不能写成 CheXagent 无收益。

canary 验收看流程，不要求“至少救一例”或“零 harm”才通过；这么小的样本不能证明效果。

## 阶段 3：source 通信与组合对照

先完成 canary，再扩大 **source** 到上一轮真实 64 组或改进的预声明 source 队列。
不要连续增加 target 来寻找显著性。沿用成功运行配置以保留远程兼容性参数。

下面变量应由你从服务器文件确认后赋值，而不是让用户重新手工提供：

- `SERVER_PYTHON`：实际兼容 LLaVA 的 Python。
- `BASE_CONFIG`：上一轮 prepared.json 指向的成功配置，或核对后的本版本配置。
- `SOURCE_MANIFEST`、`TARGET_MANIFEST`、`REFERENCES`：成套真实清单及独立参考答案文件。
- `CUDA_VISIBLE_DEVICES`：有足够显存且允许使用的设备。

```bash
bash run_diagnostics.sh \
  --python "$SERVER_PYTHON" --config "$BASE_CONFIG" --chexagent on \
  --source-manifest "$SOURCE_MANIFEST" \
  --target-manifest "$TARGET_MANIFEST" --references "$REFERENCES" \
  --output runs/native-v010-source-diagnostics
```

完整重跑相同命令会复用已完成的病例缓存，不重新下载权重。只支持病例级恢复：
中断时正在运行的病例可能重做，不能承诺每个 token/每条分支都续算。
代码、模型、配置、source 或参考答案改变时会产生新指纹；不要改缓存键强行复用旧模型输出。
这一诊断包含多种呈现重放，首次运行会明显慢于单个在线方法，不将其总耗时当在线延迟。

默认 pairs 在 runner 中预声明：CXR anatomy→CheXagent、CONCH→source retrieval。
若想增加/反转 pair，先在单独 source 配置的 `capability_diagnostics.pairs` 中声明，限制在少数有
任务依据的组合；不基于 target 标签筛选。不兼容的 pair 自动记录跳过，不强制调用。

产物索引：`runs/native-v010-source-diagnostics/llava/latest-diagnostics.json`，其 summary 指向：

- `diagnostic-summary.json/.md`：按 source 域、初始/续写状态、独立组汇总。
- `source-diagnostics.json`：NONE、各种呈现、两工具的真实输出与原生观察。
- `evidence-audit.json`：图像、问题、响应、呈现观察、事实/相关性/伤害的空白审查字段；含基线。
- `evidence-audit-key-private.json`：审查条目对应方法；评审时单独保管。
- `routing-audit.json`：只含图像和待填的模态，先盲于问题/参考答案标注。
- `routing-predictions-private.json`：原路由输出，用于标注完成后计算错误。

当前审查表不是完全双盲：原生观察中仍能看到工具身份，也未导出渲染后的 overlay PNG。
看几何时，应从 `source-diagnostics.json` 的 raw EvidenceItem 与同一原图调用
`native_evidence.make_visual_evidence` 重建；不要把 JSON 内的框坐标当成看过分割图。

### 结果怎样决定下一步

| 证据 | 判断与行动 |
| --- | --- |
| 原图复制就伤害明显 | 怀疑多图格式/训练分布不适应。先用文本，不把所有退化归咎于小模型。 |
| 原生文本差、scoped 文本更好 | 证据通信有改善可能；按 source 域验证，不直接宣称幻觉减少。 |
| 去掉检索答案退化/归零 | 说明此消融去掉了有用信息，不能声称检索无效；继续测试有明确来源的紧凑答案呈现。 |
| 原生观察本身不正确/不相关 | 检查模态路由、适配、任务匹配；暂不训练更强的调用策略。 |
| 正确相关观察仍导致错误回答 | 主模型能力读取/对齐瓶颈。提出小型 adapter 训练方案及成本，不临时堆更多提示。 |
| ΔAB 优于 NONE 但不优于最好单工具 | 无组合必要性证据，不把两次调用写成互补协作。 |
| 单工具 ΔA 不好而 ΔAB 稳定好 | 候选复合动作有意义；下一阶段实现和校准有限序列，而非降低单步 DG 阈值。 |
| source 好、真实留出域差 | 检查完整管线的域迁移和预训练暴露；不能用 hash 分区包装医院 DG。 |

所有改善/伤害默认只指连续 Token-F1。`kidney→renal` 类词面变化不能直接算医学收益。
关键病例请有能力的医学评审看原图、问题及观察；Codex 的文字判断只能作初筛，不能冒充专家标注。

### source 覆盖要求

先按模态/任务、来源、独立患者或图像组建立 source 覆盖表，再无标签地补样。
优先补有明确真实来源的 CXR 支持，同时保留病理作为第二能力/模态验证；不要将框架改成只调 CheXagent。
不是每域“总数达到 8”就够：实际工具/scope/历史条件必须达到独立支持要求。
也不能只补能答对的病例。核查 CheXagent/XRV 的预训练数据是否包含这些来源，保留暴露/未知记录。
需新数据许可、医院元数据或大下载时向用户说明；不要创造医院 ID。

## 阶段 4：有源域证据后，拟合可执行策略

本次提供可直接运行的 scoped 配置，但没有假定它胜过 legacy。仅当阶段 3 表明值得验证，
固定同一 source、工具列表、预算、主模型/prompt 后进行 source 拟合对照：

```bash
# 确保变量已在阶段 3 确认；两个输出目录分开，不能混用策略。
bash run_value.sh --skip-download --python "$SERVER_PYTHON" \
  --config "$BASE_CONFIG" --chexagent on --evidence-profile legacy \
  --source-manifest "$SOURCE_MANIFEST" --target-manifest "$TARGET_MANIFEST" \
  --references "$REFERENCES" --value-stage source --output runs/native-v010-value-legacy

bash run_value.sh --skip-download --python "$SERVER_PYTHON" \
  --config "$BASE_CONFIG" --chexagent on --evidence-profile scoped \
  --source-manifest "$SOURCE_MANIFEST" --target-manifest "$TARGET_MANIFEST" \
  --references "$REFERENCES" --value-stage source --output runs/native-v010-value-scoped
```

这两个命令只拟合 source；不会生成 target。条件收益来自真实分支，不用二值错误估计。
scoped 同时改变生成专家子请求、文本裁剪、检索答案展示和默认视图；解释原因应依赖阶段 3
的单因素诊断，不能仅拿两套组合配置差异作某一模块的因果证明。

若担心 133 维对小样本过拟合，可在独立 source-only 配置中把 `capability_value.encoder.dimensions`
从 16 改为 4（37 维总特征），做预声明的小范围比较；不要在 target 上试维数/惩罚强度。
现有 LODO 是策略回归校准，不是自动嵌套超参选择，也没有隔离整个检索库，必须明确这一限制。
本阶段不降低 `min_cases_per_domain`，不把 `value_mean` 的正预测当作已证实正收益。

本次未实现复合动作策略训练、LoRA/projector 或完整检索 LODO。如果需要这些，应先根据诊断
报告说明理由、数据/成本，再给用户一个有边界的下一步，而不是把未写的功能标为已完成。

## 阶段 5：另行确认的新测试集与 ICLR 证据链（本轮不自动运行）

冻结最终 source 配置和策略，指定未参与这几轮讨论的新患者/图像 test。仅更换随机种子不能
保证不重叠，必须与历次 source/development/test 核对图像像素、患者/组 ID，记录真实域来源。
保持同工具菜单、调用预算、token 预算和主模型；必须有 best-single、固定任务规则和无 DG 对照，
而非只打赢 generalist。模态 oracle 只能单列为诊断上限，不能混进部署方法。

正式结论需要医学事实指标（支持/不支持/矛盾命题）、每域表现与 harm、配对区间、成本；
真实域泛化、多能力联合必要性和换主模型/新插件迁移实验也需实际运行。
当前代码和负面实验不够支持“达到 ICLR 要求”，本轮目标是建立能定位瓶颈并推进机制的证据链。

## 最终向用户交付

保存一份 `runs/native-v010-audit/HANDOFF_RESULT.md`，至少包括：

1. 实际代码版本、解释器/模型/数据清单路径、启动命令和 GPU；哪些阶段完成/未完成。
2. 数据独立性、真实域/代理域、预训练暴露状态；每工具逐域独立支持。
3. 能力观察正确性、任务相关性、证据呈现、模态路由分别发现了什么；提供原始病例定位。
4. 单工具/组合的逐域增益与受损例；词面分数与事实判断明确分开。
5. source 拟合是否仍全 NONE，为什么；不把回退包装成性能胜利。
6. 推荐一个最小的下一步，并明确还需要人工标注、真实域数据或实现什么，而非列一长串新模块。

更新 STATUS 时保留历史负面结果。提交只包含代码、配置和去敏摘要；不提交模型、患者图像、
token、完整医疗数据或大运行缓存。是否推送服务器新改动遵守用户当次授权。
