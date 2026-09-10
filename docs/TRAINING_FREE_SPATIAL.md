# Strict training-free spatial expert collaboration

This is the current method entrypoint. It supersedes the trained bridge and
source-training recommendations in the historical vector reports. No new bridge,
MLP, embedding, adapter, gate, temperature, or calibration policy is trained.
Published generalist/expert weights are reused frozen. Historical training code
is retained for regression/provenance; its CLI and source-preparation CLI are
retired and the normal model factory refuses trained bridge checkpoints.

## What changed and why

The latest remote result at `9a2564804c3363e90a9db2b6763d19f4ca71da84` reported
an almost inert trained bridge and 0/180 gate acceptance. The new implementation
replaces that dependency with a parameter-free spatial operator. This is a new
protocol informed by that development history, not a claim of an untouched test
set or an improvement in medical accuracy. No new medical GPU run was performed
when implementing it.

### 1. Actual representation and importance

Implementation: [`spatial_evidence.py`](../merit_feddg/spatial_evidence.py),
`spatial_packet` and `SpatialEvidenceBridge.forward`.

Let `H` be the generalist's **existing projected** visual patch tokens, `M` the
expert masks/box coverage in the identical square-padded patch grid, and
`R = M / row_sum(M)`. Region content is `E = R @ H`. For record weights `w`,
`assignment = M.T * w`, `mass = assignment.sum(-1)`, and

```python
evidence = r @ h
assignment = regions.T * weights
mass = assignment.sum(-1, keepdim=True)
result = (h + assignment @ evidence) / (1 + mass)
```

There are no new learned Q/K/V matrices or random concept vectors. The operator
has zero parameters and an empty state dict. Pixels outside all selected regions
and an empty/all-zero intervention preserve the input exactly. It runs in a
scoped forward hook on the existing multimodal projector; the hook is removed
after success or exceptions. This version uses deterministic region aggregation,
**not an implementation of decoder attention-logit bias**.

Weights are an explicit simple baseline: `1 + fraction of label words found in
the question`, or constant one in the equal arm. Identical geometry within an
expert does not duplicate its vote; records are normalized within each expert,
then experts receive equal total mass. Scores from sigmoid/softmax/CLIP are not
mixed into a common confidence. This lexical relevance is not a semantic verifier
or a correctness probability. Different configured IDs are still different
sources; wrappers around the same model should not be advertised as independent
experts. No fitted relevance model or clinical threshold is present.

### 2. Heterogeneity is handled with explicit limits

| Output | Current implementation | Limit |
|---|---|---|
| Binary mask | Existing validated RLE/crop mapping | Predicted foreground is not confirmed disease |
| Soft mask | Lossless compressed float32 payload and checked mapping | No mask-confidence calibration |
| Detection box | Exact fractional patch coverage | No new detector weights are bundled |
| XRV classification | Positive CAM using existing DenseNet feature/head weights | Spatial attribution, not lesion segmentation; raw class scores are retained but not injected |
| Score-only classification | Explicit preflight exclusion / packet rejection | No claim that arbitrary black-box scores are aligned |
| Generated expert report or retrieval | Excluded from the spatial protocol | Text baselines remain separate historical protocols |

The CAM comes from the same forward producing the original scores:
[`native_xrv.py`](../merit_feddg/experts/native_xrv.py),
`classify_with_spatial` and `positive_class_maps`. Its formula is positive
`sum_c classifier_weight[k,c] * relu(feature[c,h,w])`, followed by interpolation
and per-map maximum normalization. A zero map remains zero. No gradient-based
optimization or target labels are needed. A low classifier score is never turned
into negative disease evidence.

Short native class names are audit and relevance identifiers. This method does
not feed their generated descriptions into the answer prompt. Region pooling
does not transfer an arbitrary numeric/semantic fact: two classes with identical
spatial maps do not become different visual embeddings just because their scores
differ. `class_scores_injected: false` is recorded to prevent that interpretation.

### 3. Gate remains a heuristic, now with a full answer horizon

[`vector_gate.py`](../merit_feddg/vector_gate.py) compares continuations with and
without the proposed evidence, then scores both using an evidence-free frozen
generalist on the original and same-size channel-mean image. Positive improvement
in image-relative support is required. The numerical tolerance remains `1e-6`.
All proposals can be rejected; agreement with the original answer is not required.

The new config uses the complete remaining 64-token generation budget, avoiding
the previous eight-token blind spot. For LLaVA-Med the verifier calls
`LlavaMedAnswerSession.sequence_mean_logp`: one teacher-forced forward per image
and candidate, rather than replaying every prefix through generation. At most four
sequence verifier forwards are needed per differing candidate pair. This uses raw
conditional likelihood; it is explicitly logged as a different scoring protocol
from the historical production-prefix replay. It is not claimed numerically
equivalent to that replay or a demonstrated wall-time speedup.

Candidate generation still costs two continuations. Mean-color controls can be
out of distribution, self-verification shares the generalist's blind spots, and
sequence means can be length-sensitive. Identical full-budget candidates remain
a rejection with an explicit reason, not a successful reliability decision.

## Expert coverage

The default spatial protocol uses XRV findings with CAM and XRV anatomy, plus an
optional **BiomedParse v1** expert when its local weights/source are installed.
Score-only BiomedCLIP anatomy and CONCH remain available in the repository's text
paths but are explicitly excluded from this spatial experiment. CheXagent and
retrieval are also excluded; their presence in a YAML is not counted as coverage.

[`native_biomedparse.py`](../merit_feddg/experts/native_biomedparse.py) follows the
official frozen inference interface, preserves soft masks, and never calls the
target-statistics/p-value screen or loads target masks. It pins clean v1 source
to `db5c10782dab2377db4f68bbc03f71c54572e51b`. The current upstream v2 branch is
not substituted: its published 3-D weights are not the dedicated 2-D v1 weights.

Configured object groups span CT/MRI, chest X-ray, pathology, fundus, dermoscopy,
endoscopy, OCT and cardiac ultrasound. For CT/MRI anatomical groups and cardiac
ultrasound, explicit question hints are required; FLAIR and contrast-enhanced T1
groups require the corresponding sequence hints. Unknown context returns
`unknown_anatomy_or_sequence`, before loading the expert. Those hints are only
conservative applicability rules, not validated modality/sequence recognition.
Object prompts come from the pinned published catalog, ordered by question
overlap and capped at eight; omitted prompts are logged. Prompted segmentation
does not establish object presence or arbitrary specialty diagnosis.

HoVer-Net, 3-D TotalSegmentator and video experts are not added to this patch.
They require appropriate pathology/volume/video inputs and independent task
validation. The current VQA-RAD experiment cannot demonstrate those modalities.

## Run with the existing complete manifest

Use the existing environment containing official LLaVA-Med and local weights.
No source JSONL, new data split, calibration set or `MERIT_TENSOR_BRIDGE` is needed.
The script respects `CUDA_VISIBLE_DEVICES` and can use `MERIT_PYTHON` for the
existing Python interpreter. Base model paths remain inherited from
`configs/llava_med_capabilities.yaml`.

```bash
bash scripts/run_vqarad_vector_pipeline.sh /absolute/path/to/full-test-manifest.jsonl
```

This runs `--protocol spatial` with four matched arms: `generalist`,
`spatial_equal`, `spatial_weighted`, `spatial_gate`. They use the same complete
manifest, image, prompt and unconstrained 64-token generation budget. Reference
answers and CE/OE metadata do not enter generation. `--protocol vector` remains
a three-arm naming alias but now also requires the training-free operator.

To add BiomedParse, place the official v1 weights at
`artifacts/models/BiomedParse/biomed_parse.pt`, its pinned clean source at
`upstream/BiomedParse-v1`, and install the upstream inference dependencies in a
compatible environment. Cache its required pretrained tokenizer/encoder assets
before the run. The script starts with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`; missing assets fail explicitly. The adapter does not
install packages, change model weights or download them for you. A missing
optional checkpoint appears in `protocol.json`, not as an executed expert.

```bash
python scripts/audit_spatial_run.py --run runs/matched-spatial/RUN_ID
python scripts/evaluate_matched_vector.py --run runs/matched-spatial/RUN_ID \
  --manifest /absolute/path/to/evaluation-manifest.jsonl \
  --references /absolute/path/to/references.json \
  --anchor-root /home/dbw/ANCHOR \
  --output runs/matched-spatial/RUN_ID/evaluation-summary.json
```

The second command is the existing VQA-RAD/ANCHOR offline evaluator, extended to
read all four method names from the protocol. Only this offline evaluation may
use reference answers and answer-type annotations. It does not alter generation
constraints. The label-free audit reports calls, actual presentation, nonzero
token intervention, gate decisions, changed strings and overhead. String changes
and token deltas alone do not establish medical gain.

## Verification and next experiment

CPU tests cover exact region arithmetic, empty/outside identity, absence of
parameters, source balance, duplicate invariance, label-extension stability,
soft-mask/crop validation, score-only rejection, frozen CAM arithmetic, actual
projector-hook execution/cleanup, BiomedParse interface behavior with a stub,
full-budget gate decisions and likelihood alignment after image expansion.
Final validation: 634 tests passed and 17 optional-dependency tests skipped;
changed-code Ruff checks and launcher shell syntax checks passed.
Real checkpoint compatibility, CUDA memory, runtime speed and medical performance
remain unverified in this environment.

Next use the four-arm run above without splitting the dataset or tuning the gate
on its references. Report correction/harm counts alongside adoption, coverage and
cost. Separate the new-expert benefit from the weighting/gate benefit. A spatial
shuffle intervention is a further research control, not currently a fifth arm.

## Upstream code basis and novelty boundary

- [ProxyCLIP / ECCV 2024 custom attention](https://github.com/mc-lan/ProxyCLIP/blob/a0e38d1989c990651fda0d8fbf8aad41bde955b5/open_clip/transformer.py): external spatial relations with existing visual content. Our region return operator is inspired by this separation, not copied or equivalent.
- [PAI / ECCV 2024 attention.py](https://github.com/LALBJ/PAI/blob/9bbd8bd57a0b0923f996197e4bd3e02cc10b8d58/attention.py): inference-time attention intervention is prior art; this patch does not implement its decoder patch.
- [VCD / CVPR 2024 sampling](https://github.com/DAMO-NLP-SG/VCD/blob/d6568ff81b8fd306a49e630df44f2db5c2300191/vcd_utils/vcd_sample.py): visual-contrast decoding is prior art, not a calibrated evidence gate.
- [DnR / CVPR 2026 expert scoring](https://github.com/EavnJeong/Draft-and-Refine-with-Visual-Experts/blob/d72196b1b498e982ecf6eb94e91724707c15ebf2/process_uq.py): training-free visual experts/refinement must be compared, not claimed as a new category.
- [BiomedParse / Nature Methods official v1 example](https://github.com/microsoft/BiomedParse/blob/db5c10782dab2377db4f68bbc03f71c54572e51b/example_prediction.py) and [soft-mask inference](https://github.com/microsoft/BiomedParse/blob/db5c10782dab2377db4f68bbc03f71c54572e51b/inference_utils/inference.py): existing frozen expert loading/inference, excluding its example's ground-truth evaluation.

No accuracy or novelty guarantee follows from combining these ingredients.
