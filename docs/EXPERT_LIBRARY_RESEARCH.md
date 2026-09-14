# Expert library audit and mechanism-first research checkpoint

Audit date: 2026-09-14. This is a targeted field scan, not a completed novelty
survey, an ICLR acceptance prediction, or a frozen paper hypothesis.

## Coverage, not registration count

| Expert/source | Modality | Site | Actual capability | Status / main boundary |
|---|---|---|---|---|
| Existing XRV DenseNet | CXR | chest | finding classification, native CAM | Existing weights; CAM is not a lesion mask |
| Existing XRV PSPNet | CXR | chest anatomy | anatomy segmentation | Existing weights; anatomy does not imply normality |
| Existing BiomedCLIP | multiple 2D modalities | catalog-dependent | contrastive fixed-catalog scoring | Existing weights; no diagnostic probability |
| Existing CONCH | histopathology | tissue | image/text feature comparison | Existing weights; fixed tissue scope, not whole-slide diagnosis |
| Existing CheXagent | CXR | chest | query-conditioned text observation | Existing weights; prose is fallible and needs task-use checks |
| MedMNIST BreastMNIST ResNet18-28 | breast ultrasound | breast | malignant vs normal/benign classification | Downloaded official run-1 checkpoint; strict load and one TRAIN-image inference passed |
| Original U-KAN BUSI | ultrasound | breast | binary predicted foreground | Original author SharePoint links require login; not downloaded |
| U-Bench U-KAN BUSI reproduction | ultrasound | breast | binary predicted foreground | Author-hosted alternative identified and downloading; separate identity, NOT original paper weights |
| U-Bench U-KAN ACDC | MRI | heart | cardiac segmentation | Candidate, not yet integrated/validated; multiclass/slice normalization must be audited |
| U-Bench U-KAN DRIVE | fundus | retinal vessels | vessel segmentation | Candidate, not yet integrated; cannot infer diabetic disease from vessel geometry |
| U-Bench U-KAN ISIC / Kvasir | dermoscopy / endoscopy | skin / bowel | lesion / polyp segmentation | Candidates, not yet integrated; distinct preprocessing and splits |
| MedSAM | supported medical 2D images | prompt-selected region | box-prompted segmentation | Existing optional interface; box must not come from target truth or invented full-image region |
| BiomedParse | supported medical modalities | registered objects | text-prompted segmentation | Existing optional interface remains disabled in frozen canary; exact supported-object list needs audit |
| TotalSegmentator | volumetric CT/MRI | whole-body anatomy | 3D segmentation | Not a drop-in expert for a single VQA PNG; do not fabricate missing volume/spatial metadata |
| RETFound | fundus/OCT | retina | pretrained representation | Backbone weights alone are not a disease classifier; no new training allowed here |

The lower rows are a sourced expansion queue, NOT claims of configured, executed
or medically beneficial tools. Initial downloaded additions focus on one new
modality with two distinct capabilities; their common BUSI lineage explicitly
prevents treating them as independent patient evidence.

Primary repositories: [MedMNIST](https://github.com/MedMNIST/MedMNIST),
[official benchmark and weights](https://github.com/MedMNIST/experiments),
[U-KAN](https://github.com/CUHK-AIM-Group/U-KAN),
[U-Bench](https://github.com/FengheTan9/U-Bench),
[MedSAM](https://github.com/bowang-lab/MedSAM),
[BiomedParse](https://github.com/microsoft/BiomedParse),
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator),
[RETFound](https://github.com/RorschachY/RETFound_MAE).

## Weight and environment audit

No shared dependencies were upgraded. Pinned architecture source was downloaded
into this worktree's ignored `runs/vendor/`, not installed globally or committed.

- U-KAN source: `b20bf63490f01ba45cc2c18834bc0c4975c12715`.
- MedMNIST experiments source: `70b6b3a7ad7afddff1df2a3b735235830fbdb142`.
- MedMNIST metadata inspected at `805b74237fb5bd2db376fd94e50646b6a8905d9d`.
- U-Bench source: `eafbb9fb9039a53c747789aee59894198e77a7a5`.
- U-Bench model release: `e6945d560d2395254c72ce0100cf10638784c578`.

BreastMNIST official release is [Zenodo 7782114](https://zenodo.org/records/7782114),
linked by the authors' repository. Selected `resnet18_28_1.pth` by run number,
before looking at results. HTTP ranges extracted only that stored ZIP member
instead of the entire 1.13 GB archive; CRC32 `e3757ec2` and file size 44,737,899
bytes passed. SHA256:
`4899661aafa59b2b1d5d2c5e44f79a4f5a63feb47745f171f44e99b80c6e8928`.
The archive's whole-file checksum was NOT checked because it was not downloaded.

The original assumption of a one-channel checkpoint was caught by strict
loading: this official checkpoint has `conv1.weight=[64,3,3,3]`. The adapter now
requires an explicit `as_rgb` setting and follows the official loader. Native
28px input → RGB → /255 → normalize(mean=.5,std=.5). No tensor reshaping or
permissive parameter loading was used to conceal this mismatch.

Official [BreastMNIST data](https://zenodo.org/records/10519652/files/breastmnist.npz)
MD5 `750601b1f35ba3300ea97c75c52ff8f6` passed. Only `train_images[0]` was opened
for an adapter smoke, chosen before references. Train shape is (546,28,28).
Strict checkpoint load, finite two-class output, request/result validation and
all-frozen parameter checks passed on CPU. This is training-image engineering
validation, **not held-out model accuracy**. Official labels are 0=malignant,
1=normal/benign; split sizes 546/78/156. Both classification and segmentation
BUSI variants must be excluded from any independence claim involving BUSI.

The same fixed first TRAIN image also passed on authorized container GPU 0:
strict load, all parameters frozen, one valid native classification artifact;
initialization 0.268 s and synchronized first inference 0.154 s. This includes
cold-start effects and is not a throughput benchmark. Reproduce with
`scripts/verify_small_medical.py --kind breastmnist --bundle artifacts/models/breastmnist/bundle.json --image runs/model-downloads/breast-train-first.png --device cuda:0`
through the existing environment wrapper. No labels are supplied to inference.

Original U-KAN BUSI SharePoint returned HTTP 401; the root checkpoint link was
also investigated. The authors' issue tracker contains a weight-link failure
report ([issue 51](https://github.com/CUHK-AIM-Group/U-KAN/issues/51)). No bypass
of authentication was attempted. Instead, the independently published
[U-Bench author checkpoint](https://huggingface.co/FengheTan9/U-Bench/tree/e6945d560d2395254c72ce0100cf10638784c578/U_KAN/busi)
was located. Its advertised LFS SHA is
`2c6ae3fc0c99650585776d6cc304144bb334dde267057c69cd52722db3672e22`.
The MedOtter mirror's source/hash metadata matched this, but download provenance
remains the U-Bench author's release. U-Bench is not the original U-KAN run.

U-KAN/BUSI preprocessing was traced in actual upstream loader code, not guessed
from a model name: OpenCV BGR → resize → ImageNet Normalize → additional /255.
Its held-out identities depend on the source list/seed; patient-level separation
has not been established. Do not compare performance on BreastMNIST as if it
were an independent test set for BUSI-trained segmentation.

## Why adding experts is not yet an ICLR contribution

Four verified construction analyses (not claims about authors' private discovery
history) guide the proposed next question:

| Work | Observation → abstraction → decisive check | Implication here |
|---|---|---|
| [CRITIC, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/fef126561bbf9d4467dbb8d27334b8fe-Abstract-Conference.html) | Initial model output can be wrong → external tool feedback → compare correction with useful feedback | External experts and correction loops alone are already covered |
| [Self-RAG, ICLR 2024](https://arxiv.org/abs/2310.11511) | Indiscriminate retrieval can be unhelpful → selective retrieval/reflection → compare retrieval and generation behavior | Generic relevance gating is not a sufficient novelty claim; their trained reflection differs from our frozen constraint |
| [ViperGPT, ICCV 2023](https://viper.cs.columbia.edu/) | Complex vision questions require compositional functions → explicit visual APIs → execute constituent operations | Segmentation may have operational value without directly answering a clinical question |
| [Cannot Self-Correct Reasoning Yet, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html) | Self-correction sometimes degrades answers → distinguish internal reflection from external feedback → controlled failures | Same-generalist crop/verification calls cannot be counted as independent expert corroboration |

[MedRAX, ICML 2025](https://github.com/bowang-lab/MedRAX) is especially close:
it already integrates specialized CXR tools and multimodal reasoning without
additional training. Its author-maintained code establishes direct competition
for a generic medical-tool-agent claim. The U-Bench and MedMNIST teams provide
reusable model/data artifacts, not evidence that our fusion policy works.
An evidence-graph MedRAX extension was also retrieved; it needs deeper source
and mechanism inspection before making any non-overlap claim.

## Proposed problem freeze — awaiting user confirmation

Recommended phenomenon: **task-use mismatch of heterogeneous observations can
turn useful specialist outputs into harmful answer evidence**. Work side:
appropriate task/source/region context. Non-work side: treating a crop locator,
source analogy or catalog score as equally authoritative patient diagnosis.
Simple alternative: gains/losses may only reflect prompt length or expert quality.

First separating experiment, once scope contracts are authoritatively restored:
hold observation values, model, prompt budget and selected questions fixed;
vary whether operational-only evidence is used as a final answer premise.
Measure actual delivery, candidate changes, paired improvements/harms, and cost,
with matched-layout controls. If effects follow token budget rather than use
semantics, the proposed mechanism is weakened or refuted.

This is a proposal, not a proven mechanism. Preliminary rubric I=2/M=2/N=1/E=2:
multiple external-tool/retrieval studies motivate the issue; a distinguishing
intervention is describable; novelty search is incomplete; the existing runner
can instrument it but source-contract recovery is still required. Do not market
this as an established high-novelty result. Expert-count scaling alone is a
weaker alternative unless a compute/information boundary can be demonstrated.

The problem-freeze question was sent to the user. Mechanism/substrate freezes
and a second-pass collision audit have not been completed. Research-ops continues
resource preparation and previously authorized engineering validation only;
the full stopped TRAIN experiment and any target experiment remain unlaunched.
