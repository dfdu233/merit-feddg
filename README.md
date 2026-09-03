# MERIT-FedDG

**Experts do not vote; they trace what the generalist forgot.**

MERIT-FedDG is a reproducible research scaffold for medical VLM hallucination mitigation under unseen-hospital domain shift. A compact biomedical encoder routes each study to one modality specialist, while a frozen medical VLM owns the final answer. The selected specialist never supplies its diagnosis or logit magnitude to the answer. Instead, its native-image versus null-image evidence identifies clinically aligned visual evidence already present in intermediate layers of the medical generalist; MERIT restores only the generalist's own erased residual.

The repository combines two ideas while keeping their scientific roles separate:

- **GSCo-style generalist–specialist collaboration:** multiple lightweight specialists provide domain expertise.
- **FedDG-style source-only generalization:** institutions remain separate, target domains are held out, and only aggregate reliability statistics leave a source client.

It does **not** claim that modality routing, specialist collaboration, or frequency augmentation alone is novel. The testable contribution is *modality-specific expert-confirmed evidence erasure* across decoder layers.

## What is implemented

- Qwen2.5-VL layerwise image-minus-null concept likelihoods.
- CheXagent-2-3B, CONCH and LO-VLM concept-evidence adapters.
- A compact medical-VLM modality router with metadata and oracle controls.
- Top-1 specialist routing; no expert voting in the proposed method.
- MERIT bounded residual restoration in the generalist's own evidence span.
- Federated source-client reliability calibration using aggregate sufficient statistics only.
- GSCo-context, broad-specialist, direct-logit-fusion, wrong-route and shuffled-expert comparisons.
- Deterministic CPU smoke study and real-model JSONL evidence caching.
- Dataset/model registries, license gates, leakage audit and Windows/Linux bootstrap scripts.

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

## Method

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
