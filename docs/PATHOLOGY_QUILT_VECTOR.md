# Pathology expert headroom: frozen Quilt-LLaVA pilot

## Evidence update / one Vector

Base: `b0de4a04852a36a059c0e67c79f848de62361f3c`, preserving admission repair and
request-parser experiments. Those XRV canaries establish engineering execution,
not pathology efficacy. The existing CONCH native `_classify` compares a fixed
10-entry tissue catalog and records `query_used=False`; this is not arbitrary VQA.
No new PathVQA accuracy is claimed by this change.

**Vector:** does a question-conditioned pathology specialist provide useful
information missing from the incumbent, or is the remaining failure in how the
incumbent uses that information? This dated Vector supersedes the old project-
status tail in `skills/cs197-research/SKILL.md` for this branch, not the skill's
research principles.

Affinity location: MedRAX/task-matched tools; CONCH/pathology encoders;
Quilt-LLaVA/pathology visual instruction tuning. Adding a pretrained expert or
caching its predictions is NOT a novel algorithm or a demonstrated Bit Flip.
Freeze uncertainty, authority projection, graph methods, RAG, new segmenters,
MUSK/PathGen replacement, and prompt optimization in this pilot.

## What is implemented

1. `audit_conch_native_parity.py` compares the current catalog adapter with direct
   official image/text encoding, on exactly the same checkpoint/image/prompts.
   It does not claim checkpoint authenticity or medical accuracy. No null-image
   subtraction, enlarged catalog, or template ensemble silently changes CONCH.
2. `run_quilt_worker.py` loads official Quilt-LLaVA once in its own interpreter,
   produces short free-text predictions, then exits. The actor loads separately.
   No second `llava` package is imported into the LLaVA-Med process. The worker
   uses `aldraus/quilt-llava`'s loader, conversation, image processor, tokenizer,
   stopping criterion and prompt-plus-output token layout. It uses the released
   7B model's CLIP-L/14-336 tower, NOT an assumed QuiltNet tower. The actual model,
   tower, source hashes, precision, version and cost are recorded.
3. `CachedQuiltExpert` implements the existing `infer(CapabilityRequest)` contract
   through `factory(model_id=..., **factory_kwargs)`. It only serves exact prepared
   image/question predictions, never controls or cache misses. This is a
   **cache-backed specialist**, not a live on-demand GPU server.
4. A five-arm matched pilot reuses the incumbent's real delivered evidence and
   native renderer. New evidence is a full unverified text observation. The
   final answer remains unrestricted text; no yes/no candidate restriction.

## Five arms and controls

- `compact`: exact historical prompt and token parity required.
- `without_conch`: only previously delivered `conch_tissue` items removed.
- `quilt_alone`: same raw question/short-answer prompt, specialist alone.
- `compact_quilt`: keep all old delivered entries, add matched Quilt output.
- `compact_wrong_image`: same question and prompt, Quilt sees another selected
  microscopic image from a different image/group. This is a NEGATIVE CONTROL,
  not a truthful patient observation; actual source IDs/hashes remain in private
  audit. No `wrong_image` flag is revealed in the model prompt. Do not deploy it.

Why wrong-image rather than random text? It keeps the question/instructions and
expert identical, avoiding question mismatch as the only cue. It is not a
perfect length/content-matched control and not a calibrated reliability signal.
Donors can be reused; this is not an independently sampled benchmark cohort.

## Frozen minimum experiment

Input: unchanged, answer-free official `data/train/manifest.jsonl`, complete
matching `protocol.json` and `compact_rows.json` (or explicit incumbent arm).
Require an existing shared short free-text contract and semantic compact output
with 64 tokens. Do not silently convert CE/OE-specific prompts or use a test run.
If no compatible TRAIN incumbent exists, stop and report that prerequisite.

Schedule a hash-ordered default 12 (maximum 100) microscopy cases based only on
existing input-modality metadata. Preserve the full manifest and report counts
for all image types. This is a development subset, not a new train/test split or
a complete PathVQA score. Existing modality routing can itself be wrong; inspect
image-type coverage independently and do not tune selection on answer outcomes.

Preserve the actor's prompt/settings. A fixed 4096-character study evidence budget
is used for every actor arm (explicit configurable bound). Previously invisible
old evidence is never resurrected. Assert that the kept old packets produce the
EXACT historical prompt and tokens before adding anything. Extra expert evidence
uses extra context; no equal-token/latency claim is made. Stop on new omission,
old displacement, text truncation or parity loss. Do not silently enlarge budgets
mid-run, drop failed cases, or report fallback as an expert benefit.

H1: a real pathology specialist has complementary answers, and matched evidence
improves score/rescue-harm versus compact AND wrong-image evidence. H0: the expert
adds no task-relevant headroom, or gains come only from changing old evidence.

Primary results: source/actor outputs, exact delivery, compact token parity,
CONCH removal effect, matched-vs-wrong-image paired score deltas, closed-item
rescue/harm and open-item continuous score changes SEPARATELY. Independent factual
review remains necessary; token recall and parser success are not clinical truth.
References are read only by the offline evaluator. Existing ANCHOR source is
hashed at prepare and checked at evaluation; no new medical scorer is introduced.

For 12 cases the scheduled worker calls are 24 and actor calls 48 (plus optional
CONCH parity). Sequential processes, not co-resident 7B models. Costs include
expert load, all controls, token usage, actor load and peak CUDA allocation.
Failures preserve partial private logs; no automatic retry with changed settings.

## Scope and re-vector rules

- If adapter/direct CONCH disagree, fix that engineering issue before judging
  CONCH's task ability. Passing parity does not prove CONCH is useful for VQA.
- If Quilt alone has no complementary correct observations, inspect task/domain
  fit; PathGen-LLaVA may be a NEXT separate capacity control. Do not tune a gate.
- If Quilt alone helps but transport fails, examine evidence utilization next.
- If CONCH removal helps, report that this catalog is detrimental in that scope;
  do not claim the entire CONCH model is weak.
- If matched and wrong-image effects are similar, visual expert contribution is
  not established. No extra planner or verifier is added to rescue the story.
- Positive small-TRAIN results justify a frozen larger diagnostic, not ICLR
  novelty, general clinical performance, or target-free calibration claims.

`enforce` admission currently supports only reviewed XRV adapters. This pilot is
an explicitly isolated **legacy-text-transport** experiment, not an extension of
semantic authority guarantees. It refuses an enforce baseline instead of turning
that protection off. The optional plugin is undeclared for authority; never claim
it passed XRV's evidence-level constraints. Gross specimens/non-pathology inputs
are rejected by the cache adapter and excluded from this first capability test.

All components use released weights without new fitting. Training-free is not
proof of pretraining-data independence: audit PathVQA/book/PEIR overlaps and the
exact checkpoint's benchmark finetuning status before publication. Download
weights under official terms; do not commit/re-distribute models or patient data.

## Official code lineage / novelty audit

- MedRAX, ICML 2025: https://proceedings.mlr.press/v267/fallahpour25a.html
  https://github.com/bowang-lab/MedRAX/blob/main/medrax/tools/classification.py
  Task-specific tool integration is prior art, not our method innovation.
- Quilt-LLaVA, CVPR 2024:
  https://openaccess.thecvf.com/content/CVPR2024/html/Seyfioglu_Quilt-LLaVA_Visual_Instruction_Tuning_by_Extracting_Localized_Narratives_from_Open-Source_CVPR_2024_paper.html
  https://github.com/aldraus/quilt-llava/blob/main/llava/model/builder.py
  Reviewed builder blob: `46d779365cf145e3d2c2e54f326a5a47ab924480`.
  https://github.com/aldraus/quilt-llava/blob/main/llava/serve/cli.py
  Reviewed CLI blob: `9898e77e9b0a0743cfff83e71e7d87f1242ead60`.
  https://huggingface.co/wisdomik/Quilt-Llava-v1.5-7b/blob/main/config.json
  Use a pinned local snapshot/source, record hashes, no automatic downloads.
- CONCH, Nature Medicine 2024:
  https://www.nature.com/articles/s41591-024-02856-4
  https://github.com/mahmoodlab/CONCH/blob/main/conch/downstream/zeroshot_path.py
  Prompt ensembling/retrieval are future independent baselines, not new inventions.

Velocity means learning whether there is genuine pathology-expert headroom and
where it is lost, not adding a longer model list. No efficacy results accompany
this code submission.
