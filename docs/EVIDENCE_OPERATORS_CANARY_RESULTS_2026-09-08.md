# Evidence operators canary：直观结果

日期：2026-09-08
性质：固定 6 例 source-only 工程 canary，不是临床评测或域泛化证明。

## 一句话结论

空间证据已经能够真正送进 LLaVA-Med 并影响输出，但当前没有观察到医学收益；
相反，额外的双面板格式和不适用的专家证据都可能把答案带偏。

## 直观病例

| 病例 | 参考答案 | 原始 LLaVA-Med | 加入的内容 | 加入后输出 | 直观判断 |
| --- | --- | --- | --- | --- | --- |
| 胸片解剖 | breasts | liver | XRV 解剖文本 | heart | 仍然错误 |
| 胸片解剖 | breasts | liver | 左肺预测 crop | heart | 仍然错误 |
| 胸片解剖 | breasts | liver | 等面积、零重叠 control crop | heart | 与预测 crop 相同，没有定位特异收益 |
| 胸片解剖 | breasts | liver | 解剖 overlay | 描述 left lung / overlaid regions | 证据进入了生成，但没有回答问题 |
| 左心后区阴影 | increased opacity in the left retrocardiac region | right-sided pleural effusion | XRV findings | atelectasis | 疾病方向更接近，但没有左心后区定位，不能判正确 |
| 左心后区阴影 | 同上 | right-sided pleural effusion | CheXagent | consolidation | 同样缺少空间定位，不能判正确 |
| 胸片病理 | cardiomegaly | right-sided pneumothorax | XRV findings | infiltration / lung opacity | 错误证据未纠正答案 |
| 胸片病理 | cardiomegaly | right-sided pneumothorax | CheXagent | enlarged pulmonary artery | 错误 |
| 脑 MRI 方位 | axial | axial | 仅复制原图形成双面板，无新证据 | coronal | 原本正确变错误，证明格式本身可造成退化 |
| 大体病理 | prostate | generic tumor | CONCH tissue catalog | connective tissue / smooth muscle | 专家作用域不匹配，未回答器官 |

## 空间证据是否真的进入模型

是。修复前，原生双 image-token 路径会直接输出 EOS；修复后把原图和辅助视图
逐像素组成左右面板，通过一个 image token 输入冻结视觉塔。最终运行中：

- 6/6 duplicate-original 格式对照产生非空文本；
- anatomy 的 crop、control crop、native overlay、scoped overlay 和 text duplicate
  五个空间/格式分支全部产生非空文本；
- 左肺 crop 像素框为 `[541,107,921,846]`；等面积 control 为
  `[0,461,380,1200]`，两框重叠为 0；
- crop、control、overlay 和原图的像素摘要均不同。

这只能证明视觉干预到达生成器。crop 与 control 给出同样的错误答案，说明目前
没有证据表明模型利用了预测位置本身。

## 真实分割与扰动

XRV anatomy 输出了非空 512x512 Left Lung、Right Lung、Heart 掩码，前景比例分别
为 0.22076、0.23244、0.06831。它们是预测解剖区域，不是病灶或人工真值。

gamma=0.95/1.05 探针共增加 9 次 XRV 前向：anatomy 敏感性 0.0118836，两个
findings 病例为 0.0086983 和 0.0180077。数值只描述光度扰动响应，不能证明答案
正确、专家适用或临床不变。

## 耗时

- 6 例完整诊断回放：50.42 秒；
- 记录的工具事件：12.29 秒；
- 峰值 PyTorch allocation：23.47 GiB；
- anatomy 病例单次 baseline 回放：0.385 秒；crop/control：约 0.56 秒；
  overlay：约 0.96--1.00 秒；
- 三组扰动探针各自耗时：0.191、0.063、0.073 秒。

presentation 分支复用了同一次专家输出，因此这些回放时间不是部署时完整在线延迟。

## 当前判断

1. 不能声称“不弱于原始通用模型”：MRI 已出现明确的正确到错误反例。
2. 本轮没有拟合或运行门控；`behavior_probe=audit` 只记录，不拒绝证据。因此这些
   错误不是门控决策造成的。
3. 已观察到三种风险：双面板格式敏感、专家输入/任务作用域不匹配、专家证据本身
   不足以回答空间化医学问题。
4. 所有数据域都是 proxy；不能把结果归因为或解释为真实医院域泛化成败。
5. `target_generations=0`，未调整阈值，也未把词面提升、空检索或零调用视为成功。

## 原始输出

- `runs/evidence-operators-source-final2/llava/12a7597d747e9f7a/diagnostics/source-diagnostics.json`
- `runs/evidence-operators-source-final2/llava/12a7597d747e9f7a/diagnostics/diagnostic-summary.json`
- `runs/evidence-operators-source-final2/llava/12a7597d747e9f7a/diagnostics/evidence-audit.json`
- `runs/evidence-operators-source-final2/llava/12a7597d747e9f7a/provenance.json`

完整执行结论及失败修复历史见 `STATUS.md`。
