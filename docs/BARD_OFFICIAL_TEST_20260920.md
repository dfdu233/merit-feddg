# BARD / HuatuoGPT 官方 TEST 启动记录

用户在source预检后明确授权：直接在TEST上用当前四张显卡评测。新授权覆盖先前“不启动TEST”的阶段性决定；旧C方法不恢复。

完整范围为官方VQA-RAD451条与SLAKE2094条，共2545条。通过官方VQA-RAD test parquet的题目/图像字节、SLAKE test JSON的qid/题目核验；参考答案只保存在宿主机独立离线评分文件，不传给推理进程。

固定HuatuoGPT-Vision-7B、greedy、1024-token预算、repetition_penalty1.2、min_new_tokens1、BARD f=1。五个clean方法为Generalist、joint_all、isolated_mean、isolated_geomedian、bard。不根据TEST结果改方法或参数。本轮测量与source预检一致：复用冻结原生专家包、重新运行Huatuo receiver与原生Baseline；并非重新运行router和所有专家的端到端采集。

四个常驻模型进程按完整队列索引模4分配；每例五个方法均在同一设备，避免跨硬件拼接Baseline。cloud GPU0/1负责分片0/1，宿主GPU0/1负责分片2/3，共637/636/636/636例。宿主GPU0的其他训练保持运行。

输入包、队列和代码均绑定哈希。每例原生generate先与无专家receiver逐token比较，失败即停。保存每个arm、原生control和真实receiver语言forward计数；完成部分可断点续跑。每worker设置100000次forward保护，当前新运行最大400000次；这不是重置历史累计预算，触及限制时保留结果并重新核对累计账目。

云端只传输冻结compact-native编译器实际使用的信息。原始6.5GB输入包中包含本来不会进入模型文本的数组值；仅去掉这些编码数组中的data/counts，1273条云端输入的各专家compact_records逐项完全一致，传输输入约11MB。完整原始掩码仍留宿主机，源文件哈希及等价性证明保留。没有改变呈现形式、置信度、专家数量或模型可见标量。

原始专家节点覆盖：VQA-RAD为0/1/2/3/4节点=99/127/51/172/2例；SLAKE为441/771/160/674/48例。因此仅50/2545例在打包前可能满足f=1四节点条件；最终须报告实际送达数量和结构性回退，不能将大量回退当成有效协作。

运行入口（每台使用对应队列和Python环境）：

```bash
python scripts/run_bard_packet_probe.py \
  --queue runs/bard-official-test-v1/queue.jsonl \
  --output runs/bard-official-test-v1/results \
  --shard-index 2 --shard-count 4 \
  --clean-only --include-joint --resume --max-forwards 100000
```

云端改用`cloud-queue.jsonl`及分片0或1。输出`results/<dataset--case>/{00..04}.json`；日志`logs/{host0,host1,cloud0,cloud1}.log`。已完成后续分片不可与不同输入/代码静默拼接。

15项BARD targeted tests通过，改动runner的Ruff及编译检查通过。启动时四个进程已实际生成；本文件不是完成报告，不声称全量得分。合并必须核对2545条及五臂共同覆盖，并分别报告官方标注评分、内容诊断、纠错/有害翻转和四节点有效子集。
