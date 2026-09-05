# Native capability collaboration during medical VLM generation

## 1. Design target and what changed

The primary problem is not choosing a specialist's answer. It is allowing a medical
generalist to acquire useful **heterogeneous visual capabilities** while answering,
then controlling how narrowly trained tools may contribute on an unseen domain.

Earlier PathoROB results favored CONCH over the generalist and the gated method;
the open block experiment did not establish a beneficial classification-to-text
bridge. Those results do not justify increasing an arbitrary score multiplier.
They motivate removing candidate scoring from the new primary protocol.

The older closed-set experiments and `run_open.sh` remain baselines. The new
`run_capabilities.sh` path does not use `ClaimSpec`, candidate labels, beam reranking,
binary error prediction, or a fixed yes/no vocabulary.

## 2. Minimal online algorithm

```
image + question + committed answer + evidence memory
                        |
                medical VLM controller
              /                         \
     CONTINUE                            CALL
         |                        registered capability tool
   greedy answer block                   |
         |                    validated native observations
         |                               |
         +------------ evidence memory --+
         |
  next decision / EOS / budget exhausted
```

1. Before the first answer block, offer compatible capability descriptors. A
   descriptor declares modality, task, capability, scope and any ROI requirement.
   The legacy manifest `capability` column does **not** limit the tool menu.
2. The same frozen medical VLM produces a short structured `CALL` or `CONTINUE`
   action. It can request evidence even when confident. There is no entropy-only
   gate or learned binary risk predictor. JSON is parsed and whitelist-validated,
   **not** grammar-constrained during token sampling. Malformed actions are traced
   and result in continuation; no arbitrary code/tool name is executed.
3. A selected adapter receives a label-free `CapabilityRequest`: visual evidence
   need, optional region, current answer prefix, and input metadata. It returns
   `CapabilityResult` containing `EvidenceItem` objects, not answer-candidate scores.
4. Accepted observations enter bounded evidence memory. Similarities remain
   similarities; source case answers remain answers about those source images;
   a prompted foreground mask is not a diagnosis. Missing capability/unknown
   concept is never automatically interpreted as a negative medical finding.
5. Greedy generation continues conditioned on image, original question and
   memory. Later requests can use a different capability or a new region.
   Committed output token IDs are preserved exactly, not decoded and retokenized.

Evidence updates rebuild the multimodal prompt and **re-prefill** the committed
prefix. This is evidence-conditioned continuation, not hidden-state injection,
token-logit correction, KV-cache reuse, or final-answer verification. It cannot
retroactively repair already committed claims. First-block evidence acquisition
is therefore a required live trace check, not an assumed property of a 3B model.

Calls, controller calls and answer tokens have explicit budgets. Duplicate
requests reuse evidence; prefix-insensitive reuse requires an explicit adapter
declaration. When no tools are eligible, or intervention budgets are exhausted,
the remaining answer is generated in one greedy call. Latency includes controller
and tool execution, including invalid controller outputs. No speedup is assumed.

## 3. Native capabilities actually implemented

| Capability | Implementation | Evidence and boundary |
|---|---|---|
| Classification | CONCH or BiomedCLIP | Fixed config catalog, native cosine similarities, non-exhaustive vocabulary; not arbitrary VQA |
| Retrieval | BiomedCLIP source-image index | Similar source images with **their own** question/reference; not query-text retrieval or a diagnosis |
| Segmentation | Optional Transformers-compatible MedSAM | Explicit normalized box, original-resolution RLE mask, actual foreground bbox/centroid/area; semantic class remains unknown |
| Generation | Optional `QwenVqaCapabilityExpert` factory | Specialist-generated text observation and actual token usage; unverified, not a fact oracle |
| Detection | Native factory protocol | A real detector must return its own boxes/classes/coordinate semantics; no bundled detector efficacy claim |

The default config activates the first two and downloads no new large models.
Model and feature loading is lazy; compatible tools can share one encoder.
Weights may stay resident after loading, so simultaneous medical VLM + experts
can use more memory than the old serial cache-extraction experiment. Measure
the new run's peak memory rather than reusing the earlier 9.6–10.7 GiB figure.

Full native payloads are stored in traces. The current generalist bridge is
**textual structured evidence**: masks are retained in artifacts, while their
coordinates, centroid and area are passed to the VLM. It does not show a dense
mask as a new image and cannot claim dense spatial reasoning from that mask.
Classification/retrieval currently operate on the whole image and reject ROI
requests. Tools must describe limitations in their descriptor and payload.

## 4. Domain generalization layer: scope-specific source evidence

No new two-stage OOD model is introduced. Offline source qualification is
separate from the online decision about which capability is useful now.

For each `(expert, modality, task, capability, scope)`, run complete real source
answers with the generalist and with that one tool scope offered to the same
adaptive controller. The paired continuous utility is

`delta_i = token_F1(guided_answer_i, reference_i) - token_F1(base_answer_i, reference_i)`.

Only cases with an actual tool invocation provide support for that scope;
zero calls do not certify an expert. Empty native returns still count as actual
attempts. Every compatible source domain must have support. The qualification
margin is the worst source-domain mean minus `penalty * standard_error`.
This is a conservative **heuristic**, not a statistical coverage guarantee.

Cards distinguish insufficient support, observed negative gain, unproven gain,
and qualified. A failing scope does not invalidate other separately registered
scopes of the same checkpoint. The present implementation does not automatically
split a classification catalog into per-class qualifications; declare finer tool
scopes and validate them separately when appropriate. Scope declarations are not
a learned semantic entailment test, and the model may still overgeneralize them.

At target inference, only source-qualified scopes enter the DG tool menu. This
prototype qualifies an evidence-producing capability, not the correctness of
every individual output. It does not estimate per-case native-feature OOD, does
not retrain specialist representations, and makes no unseen-domain guarantee.
Multiple positive single-tool cards also do not prove positive tool interactions.
The no-DG comparison is mandatory: if DG loses useful interventions, report that
failure rather than relabeling it safety or hallucination reduction.

Source-image retrieval excludes the entire query domain, same group, same ID and
same RGB pixels. Thus source utility is evaluated on retrieval from **other**
source domains. Target references never enter the controller, retrieval index,
tool prompts, qualification fitting or generation cache identity. Changing only
target references reruns evaluation, not model generation.

Real cross-domain claims require externally defined hospital/scanner/protocol
domains and source-only selection of all choices. Hash partitions of PathVQA
train are proxies. This code does not implement federated training/communication;
the historical repository name is not evidence that the experiment is FedDG.

## 5. Run and inspect

```bash
git pull --ff-only
./run_capabilities.sh --mirror cn --source-per-group 16 --target-limit 16
```

Add `--install-system` on a new Linux server. Use `--skip-bootstrap` only when
dependencies and the default snapshots are already installed and verified.
The bootstrap uses the existing `open-generation` asset profile: OpenMed 3B,
CONCH, BiomedCLIP and real PathVQA. CONCH remains gated. Do not publish HF tokens.
Custom-config assets must be installed separately; the wrapper does not infer
and download arbitrary plugin checkpoints. Verified snapshots are reused.

The default source is official PathVQA non-yes/no training QA, split into two
label-blind hash groups; target is official test QA. Each selected image has one
question; source/target ID, group and RGB-pixel overlaps are audited. This is a
real free-text task with **no supplied answer choices**, but not cross-hospital DG.

Four comparable outputs use the same medical checkpoint, prompt suffix and
answer budget:

| Method | What runs |
|---|---|
| `generalist` | One greedy medical VLM generation, no tools or controller |
| `all_evidence` | Compatible unprompted tools once before generation, then one greedy answer; bounded by call budget, not an unlimited oracle |
| `adaptive_no_dg` | Dynamic native tool acquisition without source admission |
| `adaptive_dg` | Same adaptive algorithm with source-qualified scope menu |

The static baseline does not invent ROIs. It is not a true all-tools baseline
for prompted segmentation. With such plugins, add a matched-ROI comparator.
Differences from one-shot greedy can also include re-prefill/numerical effects;
fixed-request, same-schedule no-evidence and shuffled-native-evidence replay
controls are still needed to isolate causal evidence benefit in a paper.

`runs/capability-generation/latest.json` identifies the run directory:

- `result.md/json`: EM, lexical F1, paired bootstrap interval, worst-domain F1,
  actual call/abstention rates, ordered capability sequences, controller output
  tokens, latency and peak GPU allocation. Cold/warm costs are currently mixed.
- `predictions.json` and case cache: all generated text, committed token ranges,
  controller actions, full native results, adopted evidence IDs and memory hashes.
- `qualification.json`: every source attempt including NONE, paired source
  gains, expected-domain coverage, scope cards and fingerprints.
- `annotation-blinded.json`: blank clinical claim annotation template; do not
  give annotators the separate `annotation-key-private.json` method key.

The first canary must show actual multi-capability requests, meaningful native
results consumed before relevant text, and no forbidden data flow. There is no
manufactured requirement that a tiny canary must rescue one case: correctness
tests cannot force a scientific outcome. Do not expand merely because tests pass.
First inspect invalid-action rates, NONE reasons, tool relevance and evidence use.
EM/token-F1 is not a medical hallucination measure; assess unsupported claims,
contradictions and important omissions using independent clinical annotations.

## 6. Add an expert without modifying the controller

Provide a class `infer(request: CapabilityRequest) -> CapabilityResult`, then
register its importable factory in YAML. Native items include expert identity,
capability, scope, payload, optional native confidence, summary and provenance.
Identity, scope, finite JSON payloads, normalized ROIs and budgets are validated.
Custom Python factories are trusted **user configuration**; the VLM cannot choose
an unregistered factory or a path to execute.

Example optional generative tool (merge under `experts`; download the chosen
Qwen-compatible specialist checkpoint yourself, using its own license):

```yaml
specialist_report:
  id: your-organization/your-specialist-qwen-vl
  checkpoint_path: /absolute/path/to/downloaded/specialist
  adapter: native_factory
  factory: merit_feddg.experts.native_qwen:QwenVqaCapabilityExpert
  factory_kwargs:
    expert_id: specialist_report
    scope: specialist_observations
    dtype: bfloat16
    max_new_tokens: 96
  modalities: [pathology]
  tasks: [open_vqa]
  capabilities: [generation]
  scope: specialist_observations
  description: Generate specialist visual observations; these remain unverified.
```

Example optional MedSAM tool, after downloading a complete
[Transformers-compatible MedSAM snapshot](https://huggingface.co/wanglab/medsam-vit-base):

```yaml
medsam_region:
  id: wanglab/medsam-vit-base
  checkpoint_path: /absolute/path/to/downloaded/medsam-vit-base
  adapter: medsam
  modalities: [pathology, cxr, oct]
  tasks: [open_vqa]
  capabilities: [segmentation]
  scope: prompted_foreground
  requires_region: true
  prefix_invariant: true
  description: Segment foreground inside an explicit normalized box; not lesion diagnosis.
```

These modality declarations are configuration, not validation of MedSAM on all
listed modalities. Its native interface follows the
[SAM model API](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/sam).
The generation adapter supports Qwen2.5-VL-compatible checkpoints, not every VLM
architecture automatically. Detection and other architectures require a real
native adapter; protocol unit tests are not clinical integrations.

## 7. Academic positioning and honest remaining work

Medical expert-assisted decoding is not new:
[CCD](https://arxiv.org/abs/2509.23379) already integrates clinical expert signals
at token-logit level. [FUDGE](https://arxiv.org/abs/2104.05218) studies modular
generation control; [DomainBed](https://arxiv.org/abs/2007.01434) makes domain
generalization model-selection practice central. This release is **not** a
reimplementation or verified improvement over those methods.

The testable research hypothesis is that scoped native evidence acquisition and
reuse can improve the *generalist's* grounded answers across heterogeneous tasks
and domains, at a measured budget, beyond static evidence concatenation. A tool
registry, JSON controller and source threshold by themselves are not an ICLR
contribution. Needed evidence includes:

- real multi-capability use and plug-in transfer to held-out specialist types;
- independent domain shifts and source-only model/policy selection;
- strong static, dense, expert-alone where meaningful, and matched-budget
  dynamic baselines; no fake arbitrary-QA evaluation of a tissue classifier;
- fixed-request evidence ablations, composition conflicts and useful scope
  retention when another scope lacks support;
- live claim factuality and total compute/latency, not cached score gains.

Current local tests establish interface/execution correctness, including tiny
random Qwen/SAM computation. They do not establish expert usefulness, hallucination
reduction, real medical segmentation quality, or an ICLR-level novelty result.
