# Uncertainty-preserving source canary, 2026-09-09

性质：固定 6 例 source-only 框架兼容 canary，不是临床评测、独立确认集或域泛化证明。
未运行 target，未拟合策略/门控，未放宽阈值，所有域仍是 proxy。

## 代码审查结论

新代码形成了两条默认关闭的实验分支。

1. 免训练 uncertainty 接口把同一专家、同一请求下的有限原生观测编译成数值包络；
   原观察必定参与，语义不一致的对象不猜测对应关系。它传输的是实测扰动范围，不是
   临床置信区间、正确率或未知域覆盖保证。
2. native tensor bridge 将分类分数、2D 掩码和框按显式概念/作用域契约编码，读取
   LLaVA 视觉 token 后以门控残差回注。它需要 source-only teacher forcing 训练，
   当前没有真实桥接权重，故本轮没有用随机初始化冒充实验。

较可辩护的组合创新假设是：专家 ID 无关的类型化输出契约，加上通道专属 source
效用选择，能否在专家替换和域偏移时减少错误传播。单独的掩码池化、共享读取器或
残差门控都有已有架构先例。当前 tensor 门是全局可学习标量，不是样本/域可靠性；
读取器也没有独立问题 embedding，训练入口只验证声明为 source，不能替代患者隔离、
真实来源划分和独立验证。因此这些仍是待实验假设，不是已经成立的创新结论。

## 运行与必要修复

复用已有 huatuo（Torch 2.0.1、Transformers 4.37.2、XRV 1.5.4）、LLaVA-Med、
CONCH、BiomedCLIP、CheXagent 和 XRV 权重；全程 offline，无安装、升级或下载。

首轮发现内部 `native_uncertainty` 的三份替代 payload 被 legacy native/scoped 对照
直接序列化，导致 LLaVA 输入扩展到 2786 tokens，超过 2048。修复后，旧对照只接收
原始点观测，只有 uncertainty 分支接收包络。随后完整 18 类表示仍超出上下文；最终
把每个类别重复的类型、极性、空间范围和分数语义提升为共享字段，并用 label-keyed
表保存值/上下界。两个最终 findings 包分别为 2086 和 2083 个预算字符，18 类及原始
浮点值均保留。

最终运行：`runs/uncertainty-source-pilot/llava/a726dadab20bfb9a`

- source cases: 6；target generations: 0；policy fitted: false。
- baseline/Block-NONE exact parity: 6/6。
- XRV findings empirical packets: 2/2 非空，每包 18 类、4 个观测（工具原输出 +
  3 次 probe forward）。其他专家没有替代观测时为 `unknown`，空检索保持空。
- 9 次额外 XRV forward 共 0.450 秒；敏感性为 0.0118836（anatomy，仅审计）、
  0.0086983 和 0.0180077（findings）。
- 六例 generation time 合计 85.81 秒；tool-event time 19.21 秒；峰值 PyTorch
  allocation 23.71 GiB。

原始文件：

- `diagnostics/source-diagnostics.json`：逐病例、逐分支回答、完整证据和工具事件。
- `diagnostics/diagnostic-summary.json`：source-only 汇总。
- `diagnostics/evidence-audit.json`：待人工医学审核的盲化条目。
- `provenance.json`：模型、专家、环境和输入来源。

## 直观病例

### 左侧心后区密度增高

- reference: `increased opacity in the left retrocardiac region`
- baseline: `The image shows a right-sided pleural effusion.`
- uncertainty point: `The image shows a low-lying mass in the right lower lobe ...`
- uncertainty range: 与 point 完全相同。
- Token-F1: 0.1333 → 0.1818，但肿块判断和右侧定位均不符合 reference。

这是词面重合增加，不是医学收益，也没有 range-specific 作用。

### 心脏增大

- reference: `cardiomegaly`
- baseline: `right-sided pneumothorax`
- point observation: 泛化为“related to findings of a chest X-ray”。
- range observation: 改成右下肺 `mass`。
- point/range Token-F1 均为 0。

区间确实进入生成并改变了答案，但方向更错误；这证明传输，不证明可靠使用。

## 结论

工程门槛已通过：真实扰动观测被完整、非空、在上下文预算内送入生成。科学门槛未过：
0/2 XRV 病例有 range-specific 医学收益，1/2 出现不利语义变化。当前不能声称优于或
不弱于通用专家，也不能归因于域门控，因为本 canary 没有拟合或应用校准域门控。
下一阶段应先冻结独立 source 确认集并完成人工医学标注；tensor 分支则必须先构建
患者隔离、来源可审计的 source evidence cache，再训练桥接权重和做同预算对照。
