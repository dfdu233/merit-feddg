# Codex：一次只验证“重新观察”这个 Bit Flip

实际执行，不只解释。读取 `docs/BIT_FLIP_OBSERVATION_MAP.md`；分支 `implementation/bitflip-region-reencoding-v1`，基于 `02c9397852e876b2a6aaba2f1a2f9f8c60d8b2e6`。先检查真实远端 HEAD、工作树与未提交修改，创建独立 worktree。不要 reset/clean/强推/自动 merge；不修改 main、旧 semantic/compact、旧 CRES 参数和结果；不恢复四小时已停止的旧任务，不合并 PR7。

## 目标

只验证：预测区域从原始像素重编码的效果，能否超过同区域低细节参照及同细节错位区域，并超过删除空间文本基线。固定单位融合、固定视觉 token 数、普通单路回答。没有新 alpha、uncertainty、judge、知识库或新模型训练。

## 环境与测试

复用已有虚拟环境/权重/源码，不升级共享依赖，不下载模型，不动其他用户进程。GPU 必须用户明确授权，核对容器编号和 UUID，CUDA 测试外层设置超时；不重启设备或宿主机。

```bash
python -m pytest tests/test_region_reencoding.py -q
python scripts/run_region_reencoding.py --help
python scripts/evaluate_region_reencoding.py --help
```

现有环境允许时运行全库测试，记录真结果，不删除失败测试。CPU/fake-backend 测试并不证明真实 processor/权重已经验证。

## 源输入

寻找服务器上已存在的完整 official TRAIN manifest 和对应完整 ungated compact_rows 源 run，包括 protocol.json、routing.json、原 case-cache、原始预测 masks 和 trace 内 evidence_transport。必须同一完整 ID 集合；模型/图像/路由 donor/旧证据 hash 全通过。不要拿 `cres-four-hour` 的不完整测试输出冒充 source。

给 SOURCE、MANIFEST、ARTIFACTS、ANCHOR、OUT 设置真实路径，OUT 新建目录。参考答案只给 evaluator，生成阶段不读 references。不会重新划分数据，max-cases 是调度预算。

```bash
python scripts/run_region_reencoding.py \
  --source-run "$SOURCE" --manifest "$MANIFEST" --artifacts "$ARTIFACTS" \
  --output "$OUT" --check-only
```

预检不加载模型，不能叫 real canary。之后对授权 GPU 执行（UUID 用实际授权值，不猜测）：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_region_reencoding.py \
  --source-run "$SOURCE" --manifest "$MANIFEST" --artifacts "$ARTIFACTS" \
  --anchor-root "$ANCHOR" --output "$OUT" --gpu-uuid "$AUTHORIZED_UUID" \
  --max-cases 2 --max-seconds 600
```

该时间限制在病例间检查，不强制中断 GPU kernel。需要硬停止用外层 timeout 并报告未完成成本。日志 ROOT 是身份目录。runner 在模型分配前 pin 现有评分器。单个 worker，锁防并发；不自动全量。若前两例没有适用 mask/分辨率条件，明确报告 coverage，不把 fallback 当通路成功，也不按答案挑成功样本。只有进一步明确预算授权后才继续同一调度清单。

## 真实 canary 必查

1. 原 compact、deletion 是新统一短答协议下真实输出，不能与不同旧提示分数混比。
2. identity hook 与 deletion 每个 token 完全一致，未通过立即停止，不能放宽数值容差或换答案格式。
3. 支持样本真实执行 `native_real/low_real/native_displaced/low_displaced` 四次局部编码，各自最终仍读取原图、同 prompt、同其他证据。没有拼双面板，没有第二个 image token。
4. 检查 mask 坐标、crop 像素尺寸、padding、错位框 IoU。错位不保证医学无关。原始图像哈希没有改变；原图长度 N 不变，区域外 feature 按位不变。
5. 逐候选 extra_vision_encodes=1、projector_calls=1，不以外层 score-call 数冒充所有模型 forward 数。计入原图 prefill、额外编码、生成、模型加载和失败成本。
6. 真/低细节像素差为零时不叫获取到新细节。重新回填可能失效或导致 OOD；不能只看热图改了就宣布有效。
7. 第一个实现只支持固定-grid、deterministic pad 的 LLaVA-Med。遇到未支持 processor / EXIF / 多图行为应失败并单独修复兼容性，不能声称通用部署完成。

## 离线评价

执行后用 ROOT 指向实际完整身份路径、REFERENCES 指向原完整 TRAIN refs：

```bash
python scripts/evaluate_region_reencoding.py --run "$ROOT" \
  --anchor-root "$ANCHOR" --references "$REFERENCES" --partial-diagnostic
```

小样本只能输出 partial-evaluation，不创建全量标记。相同前缀的所有七臂必须齐全；评分器/输出不覆盖。总分仍是 ANCHOR CLOSED + OPEN token recall，非临床正确率。报告全样本与适用样本、图像簇数量和选择局限。

重点不是是否高于 compact 一点，而是：
- native_real - deletion：是否超出删除空间文本收益；
- native_real - low_real：是否真有新像素细节贡献；
- native_real - native_displaced：区域是否有对应价值；
- 2x2 interaction：差分作用，不能单独当医学因果证明；
- 改善/伤害、答案内容与实际成本。

未超过删除或匹配对照就如实停止并保留旧方法，不自动调整 beta/KL/ROI规则，不新增 verifier。使用相同数据做开发后，后续规则冻结再做未见专家/来源评测；不把已看测试前缀当未见集。

## 交接与提交

给出实际 commit、清单/源缓存/权重身份、所有命令、测试与真实调用记录、覆盖与不可用原因、配对表和下一步是否值得继续。不要提交权重、病例原图/原文、患者元数据或凭证。新的实验报告只提交脱敏汇总到独立 experiment 分支，不自动合并实现 PR，不把纯工程通过写成 ICLR 创新已成立。
