# Med-DEFER / MERIT-FedDG

## v0.6：真实开放生成实验入口

新版主线是**生成期间的短段专家介入**，不再把闭集候选分数变化当作开放生成结果：
医学 VLM 提出动态续写 → 源域合格专家提供证据 → 有界选择 → 提交原始 token → 继续生成。
没有二值错误预测器，也不再要求双阶段 OOD。资格判断依据**源域上组合后实际回答的连续增益**。

```bash
git pull --ff-only
./run_open.sh --mirror cn --source-per-group 16 --target-limit 16
```

沿用已有 HF_TOKEN；CONCH 需要已获访问权限。新服务器可加 `--install-system`。
只下载/复用 OpenMed 3B、CONCH、BiomedCLIP 与真实 PathVQA 数据；不下载 CheXagent。
断点复用已完成资产和与配置/代码/模型/输入身份匹配的病例生成结果。

完整说明见 [论文调研、算法、实验协议与边界](docs/OPEN_DECODING_RESEARCH.md)，
配置见 [open_generation.yaml](configs/open_generation.yaml)。
输出位置由 `runs/open-generation/latest.json` 指定。

**当前边界**：默认两个实际专家都是对比视觉语言模型；分割、检测、检索、生成已进入
统一 native-evidence 插件接口和测试，但不代表这些真实专家已经全部完成医学验证。
PathVQA 是真实开放问答，train 哈希分组不是医院域；EM/F1 不是医学幻觉率。
短段 beam reranking 不是逐 token 引导，没有跨段 KV 复用，不能预先宣称更快。
还需要服务器上的实际医学模型结果、独立域评估及强论文基线，才能判断 ICLR 研究价值。

## v0.5 及更早版本：保留的闭集与机制对照

**A medical generalist keeps authorship; qualified specialists can intervene before it commits a clinical decision.**

This repository now implements **Med-DEFER**, a research prototype for claim-level conditional
computation in a medical VLM under unseen-domain shift. At the start of a clinical claim, a
controller can choose `NONE` or one compatible specialist. In the implemented closed-set path,
this is pre-commit candidate-space deferral: the medical VLM scores the answer candidates, the
controller optionally calls a source-qualified specialist, and the bounded guided argmax is
locked before any explanation is generated. The main method considers the specialist even when
the generalist is confident;
the old uncertainty-only trigger is retained as an ablation because it missed a real
high-confidence error. Specialist evidence performs bounded candidate or phrase guidance and
does not independently compose the final response.

The original MERIT layer-residual method remains intact as a baseline because the first public
experiment did not show expert-specific residual recovery: MERIT tied wrong-route and shuffled
controls, while the OCT adapter scored zero. The new method therefore makes three changes:

- **Pre-commit deferral:** the first closed-set clinical decision is made before any answer is
  emitted. True token-by-token open-claim deferral remains a separate experimental stage; it
  requires a fresh semantic `ClaimSpec` and must never silently reuse the original candidates.
- **Capability-aware evidence:** classification, retrieval, segmentation, detection and generation
  experts share an envelope that preserves scores, masks, boxes, text and provenance.
- **Domain-robust trust:** the geometric mean of source reliability LCB and lower-tail
  cross-domain performance is discounted by label-free OOD distance and image quality. Hard
  qualification thresholds reject unsafe experts without repeatedly shrinking valid evidence.

The academic hypothesis is **domain-robust claim-level specialist deferral**, rather than generic
medical-agent orchestration. Target labels are never used for routing or trust fitting.

## What is implemented

- A real closed-set path that uses medical-VLM sequence likelihoods, invokes at most one lazy
  specialist before committing the answer, and isolates later explanatory text from that answer.
- A Transformers-compatible `MedDeferLogitsProcessor` retained as a token-level research
  prototype; it has not yet demonstrated an open-generation benefit.
- A `NONE`-or-one claim controller with first-claim, uncertainty-ablation, capability, route,
  latency and trust policies.
- Lazy expert construction, one-call-per-claim caching and an auditable per-claim trace.
- A native evidence contract for concept scores plus masks, boxes, generated text and provenance.
- Real source-task multiclass qualification using macro-F1, calibration, LCB and lower-tail CVaR.
- Two-stage source-only OOD in the qualified PathoROB evaluator: frozen BiomedCLIP features
  before a selected call and expert-native frozen features afterward. Empirical distances are
  cross-fitted by leaving out an entire source center (or one sample when center folds are
  impossible), then deployment references are refit on all source observations.
- Semantic evidence bridges for classification, retrieval, segmentation, detection and
  generation; unsupported native evidence fails closed.
- Qwen2.5-VL layerwise image-minus-null concept likelihoods.
- CheXagent-2-3B, CONCH and LO-VLM concept-evidence adapters.
- A compact medical-VLM modality router with metadata and oracle controls.
- Top-1 specialist routing; no expert voting in either proposed method.
- MERIT bounded residual restoration in the generalist's own evidence span.
- Federated source-client reliability calibration using aggregate sufficient statistics only.
- GSCo-context, broad-specialist, direct-logit-fusion, wrong-route and shuffled-expert comparisons.
- Deterministic CPU smoke study and real-model JSONL evidence caching.
- Dataset/model registries, license gates, leakage audit and Windows/Linux bootstrap scripts.

## Recommended real-domain experiment

The retained v0.5 real-domain validation uses the public PathoROB Tolkach ESCA dataset rather than binary
yes/no VQA. It contains six histopathology tissue classes and four explicit medical-center
domains. This repository runs a custom four-fold leave-one-medical-center-out (LOCO) protocol;
it is not the official PathoROB APD leaderboard protocol. No target label is given to
qualification, OOD estimation, routing or inference.

After accepting the CONCH terms and exporting a read token, run a small but real 48-patch pilot.
Its 12 patches per center are selected by stable ID hash without consulting class labels:

```bash
git pull
export HF_TOKEN='hf_your_read_token'
./run_pathorob.sh --mirror cn --limit-per-center 12 --install-system \
  --conch-source 'git+https://ghfast.top/https://github.com/Mahmoodlab/CONCH.git@141cc09c7d4ff33d8eda562bd75169b457f71a62'
```

`--mirror cn` covers Hugging Face and PyPI, not arbitrary GitHub repositories. The optional
`--conch-source` value above follows the proxy form suggested by `ghfast.top`; replace it with a
currently reachable proxy, internal mirror or local package path when needed.

The command reuses completed model downloads, fetches the roughly 317 MB dataset, materializes
label-blind per-center samples, audits slide leakage, extracts model evidence once and runs:

- Generalist, Specialist and routed-fusion controls;
- uncertainty-only Med-DEFER;
- Med-DEFER without domain generalization and with mean-domain trust;
- full multiclass-LCB + worst-domain-CVaR + native-OOD Med-DEFER;
- equal-budget shuffled-evidence and wrong-capability falsification controls.

Results are written to `runs/pathorob-real-per-center12/result.json` and `result.md`. The default
48-patch run is an engineering pilot whose size is fixed before its labels are inspected. Its
post-hoc mechanism flags are descriptive only: they must not be used to tune thresholds, select
a method, or decide a larger run on the same target centers. A confirmatory sample size and seed
must be registered before evaluation; an inspected pilot needs a disjoint final cohort. Existing
evidence is reused only after its manifest, configuration, extraction contract and model snapshot
fingerprints all match. A stale or incomplete cache is re-extracted automatically; use
`--force-extract` for an explicit rerun.

Because this pilot sampling is label-blind, a 12-patch center need not contain all six classes.
Every fold therefore reports target support for all frozen classes, separates classes absent from
the sample from classes structurally unavailable at that center, and computes fixed-taxonomy
macro-F1 without dropping zero-support classes. Qualification also requires a configurable minimum
aggregate source support for every frozen class; a missing class makes the main expert fail closed
instead of assigning it fabricated trust.

The matched comparison deliberately uses one frozen real-model evidence cache. Accuracy and
causal shuffled-evidence comparisons are real; reported selected-call rates are counterfactual
until the same fold is run through the live lazy loader. They must not be reported as measured
latency savings.

This stage measures multiclass accuracy, fixed-taxonomy macro-F1, ECE, worst-center accuracy,
rescue/harm, expert-call rate, and a direct full-vs-shuffled slide-cluster bootstrap interval and
paired sign test. The latter is descriptive mechanism falsification, never a target-tuning signal. It
deliberately does not rename classification error as an open-ended hallucination rate.
Open-report hallucination is a separate second phase requiring dynamic claim beams and
claim-level factuality annotations.

## Fast validation of the new path

No model download is needed to verify conditional execution, caching and domain gating:

```bash
pip install -e .
merit-feddg med-defer-simulate \
  --config configs/smoke.yaml \
  --output runs/med-defer-smoke/result.json
```

The result records generalist and guided accuracy, the dense-call counterfactual, actual sparse
calls, abstentions, and every claim trace. It is a mechanism test, not a medical result.

`run_all.sh` also runs `med-defer-compare` on the frozen real-model evidence cache and writes
`runs/<name>/med-defer/result.json`. That comparison estimates which cached expert calls would
have been selected, so its call count is counterfactual.

`guided-generate` is the real lazy closed-set integration: it computes Qwen/MedVL candidate
likelihoods, calls the controller before output, locks the guided candidate, then optionally asks
the generalist for a separate explanation. Existing `ConceptExpert` implementations plug in
through `LazyConceptExpertProvider`. The `MedDeferLogitsProcessor` remains available for research
on dynamic open claims, but its phrase bias is not the validated v0.5 method and must not be used
to claim that open-text hallucination has been reduced.

The same path is exposed as a command. Edit `examples/guided_case.json`; it intentionally has no
label. The source cache supplies only frozen source-domain reliability statistics:

```bash
merit-feddg guided-generate \
  --case examples/guided_case.json \
  --source-cache cache/medical-small-public-paper.predicted.jsonl \
  --model-config configs/medical_small.yaml \
  --compare-config data/medical-small-public-paper/compare.yaml \
  --artifacts artifacts \
  --output runs/live-case-001.json
```

During this command, the generalist and router load normally, but a specialist is instantiated
only after the pre-commit controller selects it. The returned `answer` is exactly the guided
candidate argmax; any free-text `explanation` is a separate field and cannot overwrite it. The
output lists loaded experts, call counts and the decision trace. This generic command still uses
the legacy source-cache trust card; strict fingerprinted two-stage OOD is currently validated in
the PathoROB evaluator, not yet across every live heterogeneous adapter.

## Linux server: full installation

### Legacy proxy-domain regression runner

This older path is retained for regression and compatibility checks; it is not the primary
medical-center DG result. After accepting the gated CONCH terms and exporting a read token, it
installs dependencies, downloads the compact medical model pool and three datasets, converts
compatible public samples into a unified manifest, audits image-level leakage, extracts
evidence once, and runs both predicted-router and oracle-router comparisons:

```bash
export HF_TOKEN='hf_your_read_token'
./run_all.sh --mirror cn --preset canary --install-system
```

The default `medical-small` profile uses OpenMed Qwen2.5-3B-MedVL as the medical
generalist and a roughly 750 MB BiomedCLIP model as an independent modality router. It
does not download Qwen2.5-VL-7B or MedM-VL. To reproduce the larger generic-generalist
baseline explicitly, pass `--model-profile research-2d`.

`--mirror cn` uses the Tsinghua PyPI mirror and `hf-mirror.com` for this process only;
individual Hugging Face downloads automatically retry against the official endpoint. It
does not modify global pip or apt configuration. Use `--mirror auto` outside mainland
China. If system packages are already installed, omit `--install-system`.

The `canary` preset prepares two examples per generated domain so the complete path can
be checked before a long run. After it succeeds, run the complete compatible image set:

```bash
./run_all.sh --mirror cn --preset paper
```

Completed model and dataset snapshots are verified locally and skipped without contacting
Hugging Face. Interrupted snapshots resume into the same directory. Pass `--force-download`
only when an intentional refresh is required. Evidence caches are likewise reused when newer
than the generated manifest; deterministic benchmark preparation preserves the manifest's
timestamp when its contents have not changed, so rerunning the command does not accidentally
invalidate the evidence cache. Pass `--force-extract` to recompute it. Outputs are written
below `runs/medical-small-public-canary/` or `runs/medical-small-public-paper/`.

The public runner partitions unique images deterministically into two source clients and
one target partition, never splitting QA rows from the same image across domains. These
are explicitly **proxy domains for mechanism testing**, not hospitals or scanners, and the
generated report sets `strict_hospital_dg_claim_allowed: false`.

`smoke` is only a CPU/CI code-path check and intentionally downloads no real assets. On a new Debian/Ubuntu GPU server, the compact model-and-dataset installation is:

```bash
git pull
export HF_TOKEN='hf_your_read_token'
./bootstrap.sh --profile medical-small --include-gated --install-system
```

The command installs Linux libraries, creates `.venv`, installs PyTorch and the research dependencies, installs the official CONCH runtime, downloads every registered model and dataset with resumable Hugging Face snapshots, verifies each local payload, and runs tests. Accept the CONCH repository terms in the browser before using `--include-gated`. If system packages are already present, omit `--install-system`.

The optional `research-2d` profile checks for at least 80 GiB of free space. Change the guard with `--min-free-gb N` only after checking the actual filesystem. PyTorch is installed from PyPI by default; when your driver requires a specific wheel channel, copy the index URL from the [official PyTorch selector](https://pytorch.org/get-started/locally/) and pass `--torch-index URL`.

The default assets include OpenMed MedVL, BiomedCLIP, CheXagent, CONCH,
LO-VLM, VQA-RAD, PathVQA and OCT-summary. The optional `research-2d` profile adds the
generic Qwen-7B and SLAKE baseline. Interrupted Hugging Face downloads
resume when the same command is run again. A successful snapshot receives a local
completion marker containing its file-count, byte-count and path/size fingerprint;
unchanged snapshots are skipped on later runs. Required files and minimum payload counts
prevent partial sharded models or a SLAKE snapshot without `imgs.zip` from being accepted.
Tokens, weights and medical data stay outside Git.

To install only public assets, omit `--include-gated`; pathology experiments still require approved CONCH access. Hugging Face documents the browser approval and server-token flow in its [gated model guide](https://huggingface.co/docs/hub/models-gated).

## Smoke and smaller profiles

The smoke profile downloads no private data and validates the complete experiment path:

```powershell
.\bootstrap.ps1 -Profile smoke
```

```bash
./bootstrap.sh smoke
```

Open small models and the OCT dataset:

```powershell
.\bootstrap.ps1 -Profile open-small
```

Compact medical 2D research assets:

```powershell
$env:HF_TOKEN = "your_read_token"
.\bootstrap.ps1 -Profile medical-small -IncludeGated
```

Linux users should use the full server command above, not the PowerShell script.

Run the compact asset audit first if storage is limited:

```bash
merit-feddg asset-plan --profile medical-small
merit-feddg download --profile medical-small --root artifacts --dry-run
```

CONCH is gated and non-commercial. Accept its Hugging Face terms, use an institutional-email account, and add `--include-gated`. The downloader records skipped and failed assets without concealing partial success. No model weight or medical image is committed to Git.

## Model pool

| Role | Default | Interface | Online use |
|---|---|---|---|
| Medical generalist | OpenMed Qwen2.5-3B-MedVL | layerwise phrase likelihood | produces the final evidence and answer |
| Independent router/control | BiomedCLIP ViT-B/16 | zero-shot modality similarity | route and broad-specialist control |
| CXR specialist | CheXagent-2-3B | generative phrase likelihood | specialist lens |
| Pathology specialist | CONCH | contrastive concept score | specialist lens |
| OCT specialist | LO-VLM 247M | generative phrase likelihood | specialist lens |
| CXR validator | RAD-DINO | patch features | offline occlusion validation only |

Qwen2.5-VL-7B and MedM-VL-2D-3B-en remain in the optional `research-2d` registry for
larger baseline studies. MedM's official LLaVA-derived runtime is not silently mixed into
the Transformers adapter; integrate it through its upstream package if selected.

## Real experiment

### 1. Prepare one manifest per institution

```bash
merit-feddg make-manifest \
  --root /data/hospital_a \
  --output data/manifests/hospital_a.jsonl \
  --modality cxr \
  --domain hospital_a
```

Add the task-specific `candidates` and integer `label` fields. Every row follows:

```json
{"id":"A-0001","image":"/data/A/0001.png","domain":"hospital_a","modality":"cxr","prompt":"Which finding is supported?","candidates":["pneumothorax","pleural effusion"],"label":0}
```

Audit patient/sample leakage before using any target labels:

```bash
merit-feddg audit-split \
  --manifest data/manifests/hospital_a.jsonl \
  --manifest data/manifests/hospital_b.jsonl \
  --manifest data/manifests/hospital_unseen.jsonl \
  --held-out hospital_unseen
```

### 2. Extract evidence once

Merge the audited manifest files, then run models sequentially:

```bash
merit-feddg extract \
  --manifest data/manifests/all.jsonl \
  --config configs/medical_small.yaml \
  --artifacts artifacts \
  --output cache/real-evidence.jsonl
```

For the routing ceiling, repeat with `--oracle-router`. Real routing never sees the ground-truth `modality` field. The extractor stores numerical evidence and metadata only—not image pixels or weights.

After editing the source/target names in the comparison config, installation, model/data download, evidence extraction and comparison can be launched together:

```powershell
.\study.ps1 -Manifest C:\data\all.jsonl -RunName cxr-path-oct-lodo -IncludeGated
```

On Linux:

```bash
./study.sh /data/manifests/all.jsonl cxr-path-oct-lodo --include-gated
```

If the environment or any registered asset is missing, `study.sh` invokes the full bootstrap first. Add `--install-system` on a new Debian/Ubuntu server.

### 3. Compare every method on exactly the same cache

Edit the domain names in `configs/real_compare.example.yaml`, then:

```bash
merit-feddg compare \
  --input cache/real-evidence.jsonl \
  --config configs/real_compare.example.yaml \
  --output runs/real-lodo
```

The output contains resolved configuration, per-sample predictions, route metrics, client summaries, result JSON and a Markdown comparison table.

## Med-DEFER method

For the first clinical decision, the medical VLM assigns every real task candidate a
length-normalized sequence log-likelihood. The controller considers one compatible specialist
before emitting an answer. In the qualified PathoROB path, its exact model/adapter/task
fingerprint must have a valid source qualification artifact. The expert direction is centered,
normalized and norm-capped;
the medical VLM score remains the base score. Entropy-only selection is an ablation, not the main
trigger.

Qualification is fit on real source-center multiclass probabilities. Trust combines a
cross-center performance LCB, the worst source-center CVaR, image quality and an exponential OOD
discount. OOD uses a cheap frozen feature before loading the expert and the selected expert's
native frozen feature afterward. The expert-predicted class—not the target label—selects any
post-call class-conditional OOD reference. The pre-call check instead uses the nearest known
source class, so a high-confidence wrong generalist class cannot itself block expert deferral.
A second check after the call can suppress the evidence.

The bridge contract can pass semantic candidate propositions with a current question and
generated prefix. Dynamic open-text claim generation is not yet the validated path: when no new
`ClaimSpec` exists, the system must choose `NONE` rather than reuse an initial yes/no question.
The complete algorithm, experiment matrix and falsification criteria are in
[the research design](docs/MED_DEFER_DESIGN.md).

## Legacy MERIT method

For clinical concept (c), the specialist exposes only causal visual evidence

\[
e_s(c)=S_s(I,c)-S_s(I_{null},c).
\]

At generalist layer (l), the analogous evidence is

\[
g_l(c)=\log p_l(c\mid I,q)-\log p_l(c\mid I_{null},q).
\]

Cosine agreement with (e_s) selects early generalist evidence. If agreement drops at the final layer, MERIT adds a norm-bounded version of

\[
\sum_l w_l g_l-g_L
\]

to the generalist output. Route confidence, expert reliability and erasure magnitude gate the intervention. Scaling a specialist's logits cannot scale the correction because only centered cosine agreement is used.

## FedDG protocol

Each real medical center is a domain. Source centers qualify the frozen expert on its native
multiclass task and fit feature references; the target center is held out in full. In a federated
deployment, centers can transmit per-domain metric summaries plus fitted reference statistics,
while retaining images and labels locally. The original count-only beta-LCB implementation
remains a legacy baseline.

Evaluation uses leave-one-medical-center-out separation. Target labels are joined only after every
prediction has been frozen. The real multiclass runner reports accuracy, fixed-taxonomy macro-F1
(including every frozen class), per-center class support, ECE, worst-center accuracy, rescue/harm,
call rate and direct full-vs-shuffled paired uncertainty tests. Open-ended
hallucination metrics are intentionally deferred until claim-level reference annotations exist.

## Required comparison matrix

| Method | Purpose |
|---|---|
| Generalist | untouched medical-VLM candidate decision |
| Specialist | frozen specialist alone |
| Routed fusion | direct specialist-injection control |
| Uncertainty-only | reproduces the trigger that missed a high-confidence error |
| Med-DEFER without DG | pre-commit deferral without domain trust |
| Mean-domain trust | tests whether lower-tail robustness is necessary |
| Full Med-DEFER | bounded pre-commit guidance with geometric LCB/CVaR trust and two-stage OOD |
| Equal-budget shuffled evidence | causal sample–expert correspondence control |
| Wrong capability | must fail closed |

Legacy GSCo, MERIT/MERIT-FedDG, wrong-route and oracle-routing comparisons remain available for
the earlier proxy benchmark. The token-level logits processor is a Stage-2 prototype, not a
PathoROB comparison method. A final, pre-registered experiment falsifies the mechanism if genuine
evidence does not outperform equal-budget shuffled evidence or if rescues do not exceed harms;
pilot target labels must not be used to tune the method first.

## Relation to the previous experiment

The earlier corrected ANCHOR-CLIPCEIL/FedDG-ERM audit produced identical semantic predictions to Task-only on all 128 samples. Therefore frequency-space image augmentation remains in `continuous_frequency_mix` strictly as a baseline. It is not part of MERIT's claimed mechanism, and no result is inferred from that failed experiment.

## Scope and safety

This is research software, not a medical device. It must not be used for clinical diagnosis. Verify every upstream model and dataset license independently. Never push protected health information, model weights, access tokens or local manifests containing patient paths to GitHub.

## References

- Kömen et al., [Towards robust foundation models for digital pathology](https://www.nature.com/articles/s41467-026-73923-2), Nature Communications, 2026; [PathoROB dataset](https://huggingface.co/datasets/bifold-pathomics/PathoROB-tolkach_esca).
- He et al., [Towards generalizable AI in medicine via Generalist–Specialist Collaboration](https://www.nature.com/articles/s41551-026-01653-3), Nature Biomedical Engineering, 2026.
- Liu et al., [FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_FedDG_Federated_Domain_Generalization_on_Medical_Image_Segmentation_via_Episodic_CVPR_2021_paper.html), CVPR 2021.
- Chen et al., [CheXagent](https://arxiv.org/abs/2401.12208), 2024.
- Lu et al., [CONCH](https://www.nature.com/articles/s41591-024-02856-4), Nature Medicine, 2024.
