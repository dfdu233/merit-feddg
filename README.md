# Med-DEFER / MERIT-FedDG

**A medical generalist keeps authorship; qualified specialists are conditionally called inside decoding.**

This repository now implements **Med-DEFER**, a research prototype for claim-level conditional
computation in a medical VLM under unseen-domain shift. At the start of a clinical claim, a
controller can choose `NONE` or one compatible specialist. The specialist is loaded and called
only when the generalist is uncertain and the expected correction benefit exceeds latency and
domain-risk costs. Its evidence guides the current claim through bounded phrase-token bias; it
does not replace the medical VLM or compose a final answer independently.

The original MERIT layer-residual method remains intact as a baseline because the first public
experiment did not show expert-specific residual recovery: MERIT tied wrong-route and shuffled
controls, while the OCT adapter scored zero. The new method therefore makes three changes:

- **Decode-time deferral:** expert use occurs at punctuation-delimited clinical claim boundaries,
  not after a complete draft and not by densely running every expert.
- **Capability-aware evidence:** classification, retrieval, segmentation, detection and generation
  experts share an envelope that preserves scores, masks, boxes, text and provenance.
- **Domain-robust trust:** source-client reliability LCB, lower-tail cross-domain performance,
  label-free OOD distance and image quality multiply into a conservative intervention gate.

The academic hypothesis is **domain-robust claim-level specialist deferral**, rather than generic
medical-agent orchestration. Target labels are never used for routing or trust fitting.

## What is implemented

- A Transformers-compatible `MedDeferLogitsProcessor` that invokes an expert while the medical
  VLM is generating and guides only the current clinical claim.
- A `NONE`-or-one claim controller with uncertainty, capability, route, latency and trust terms.
- Lazy expert construction, one-call-per-claim caching and an auditable per-claim trace.
- A native evidence contract for concept scores plus masks, boxes, generated text and provenance.
- Conservative source-only trust using federated LCB, lower-tail/CVaR stability, OOD and quality.
- Qwen2.5-VL layerwise image-minus-null concept likelihoods.
- CheXagent-2-3B, CONCH and LO-VLM concept-evidence adapters.
- A compact medical-VLM modality router with metadata and oracle controls.
- Top-1 specialist routing; no expert voting in either proposed method.
- MERIT bounded residual restoration in the generalist's own evidence span.
- Federated source-client reliability calibration using aggregate sufficient statistics only.
- GSCo-context, broad-specialist, direct-logit-fusion, wrong-route and shuffled-expert comparisons.
- Deterministic CPU smoke study and real-model JSONL evidence caching.
- Dataset/model registries, license gates, leakage audit and Windows/Linux bootstrap scripts.

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
have been selected, so its call count is counterfactual; only `MedDeferLogitsProcessor` performs
genuinely lazy calls inside live generation.

For a real Qwen/MedVL integration, construct `MedDeferLogitsProcessor` with a request factory and
pass it to `QwenLayerProbe.generate(..., logits_processor=processor)`. Existing `ConceptExpert`
implementations plug in through `LazyConceptExpertProvider`; a segmentation or detection adapter
can return `NativeEvidence` directly while retaining its masks or boxes. Concept phrases provide
the initial vocabulary bridge. A learned spatial-to-token projector is deliberately not assumed.

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
only after the decoding controller selects it. The output lists `loaded_experts`, call counts and
claim traces so the sparse-execution claim is directly auditable.

## Linux server: full installation

### One-command public benchmark (recommended)

After accepting the gated CONCH terms and exporting a read token, the following command
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

At each clinical-claim boundary the controller compares `NONE` with compatible experts using
generalist uncertainty, modality confidence, capability match, source-only domain trust, expected
benefit and latency. Only the winner is called. Its concept evidence is centered, normalized and
norm-capped before it becomes phrase-token bias in the medical VLM's active decoding loop.

The domain trust is the product of federated source reliability LCB, lower-tail source-domain
stability, an exponential label-free OOD discount and image quality. A second check after the
expert call can still suppress its evidence. The complete algorithm, experiment matrix and
falsification criteria are in [the research design](docs/MED_DEFER_DESIGN.md).

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

Each hospital is a separate client. Source clients calculate expert correctness counts and squared confidence error. The server aggregates these statistics with a beta prior and a lower-confidence-bound penalty. Raw images, embeddings and per-sample logits stay local. The resulting scalar reliability can suppress unsafe interventions but cannot vote for an answer.

Evaluation uses leave-one-domain-out source/target separation. Target labels are available only after every model, threshold and calibration parameter is frozen. Report average accuracy, hallucination rate, ECE, worst-domain accuracy, rescue/harm, route calibration and output equality with the base model.

## Required comparison matrix

| Method | Purpose |
|---|---|
| Generalist | untouched decoder |
| Broad specialist | tests whether routing is necessary |
| GSCo context | specialist-answer/context collaboration reference |
| Routed logit fusion | tests direct specialist injection |
| MERIT | expert selects the generalist's own residual |
| MERIT-FedDG | source-client reliability-gated MERIT |
| Med-DEFER without DG | claim-level conditional decoding without domain trust |
| Med-DEFER | sparse claim-level decoding with lower-tail domain trust |
| Post-hoc verification | tests whether decode-time intervention is necessary |
| Dense all-expert | computation/latency counterfactual |
| Wrong route | modality specificity control |
| Shuffled expert | causal correspondence control |

Also report oracle routing separately. Stop the research direction if oracle routing does not beat a broad specialist, genuine experts do not outperform shuffled/wrong experts, or gains are merely shorter outputs or parser artifacts.

## Relation to the previous experiment

The earlier corrected ANCHOR-CLIPCEIL/FedDG-ERM audit produced identical semantic predictions to Task-only on all 128 samples. Therefore frequency-space image augmentation remains in `continuous_frequency_mix` strictly as a baseline. It is not part of MERIT's claimed mechanism, and no result is inferred from that failed experiment.

## Scope and safety

This is research software, not a medical device. It must not be used for clinical diagnosis. Verify every upstream model and dataset license independently. Never push protected health information, model weights, access tokens or local manifests containing patient paths to GitHub.

## References

- He et al., [Towards generalizable AI in medicine via Generalist–Specialist Collaboration](https://www.nature.com/articles/s41551-026-01653-3), Nature Biomedical Engineering, 2026.
- Liu et al., [FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_FedDG_Federated_Domain_Generalization_on_Medical_Image_Segmentation_via_Episodic_CVPR_2021_paper.html), CVPR 2021.
- Chen et al., [CheXagent](https://arxiv.org/abs/2401.12208), 2024.
- Lu et al., [CONCH](https://www.nature.com/articles/s41591-024-02856-4), Nature Medicine, 2024.
