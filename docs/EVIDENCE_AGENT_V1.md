# Training-free evidence agent: additive experimental entrypoint

## Status and scope

This branch starts at `cec7bb9f93c4869a8197e285da8203cbef04b054` on
`implementation/evidence-revision-v2`, not the older `main`. Existing runtime,
configs, scorers, model adapters and semantic/compact methods are unchanged.
All new code is opt-in. This is an executable engineering/research scaffold,
not a validated medical improvement or a completed novel reliability algorithm.

The scheduler is a dependency-aware Python state machine, so no LangChain,
LangGraph, server API, or new package upgrade is required. Its controller uses
the existing frozen model's constrained `allowed_texts` interface. LangGraph
can later wrap these functions without changing evidence semantics. Merely
switching orchestration frameworks is not a scientific contribution.

## Implemented

- `evidence_agent.py`: typed actions, dependency checks, immutable raw ledger,
  source-lineage propagation, failure accounting, descendant invalidation,
  bounded legal-action selection, and three-way commit/keep-incumbent policy.
- `agent_regions.py`: existing XRV/BiomedParse/MedSAM-style mask encodings and
  explicit coordinate mappings to original-image ROIs. Missing/empty/malformed
  masks are unavailable, never whole-image replacement boxes or disease absence.
  CAMs are not promoted to lesion masks. Threshold 0.5 is only a fixed geometric
  extraction convention, not a disease or confidence threshold.
- `agent_runtime.py`: predicted ROI -> saved crop -> local frozen-model observation;
  optional crop -> existing source-case retriever. The final candidate reads the
  original image and intact semantic/compact packets, not a new two-panel image.
- `agent_run.py`: checks a completed incumbent run, reproduces its text AND token
  IDs through the original `NativeSession`, then runs matched extension arms.
  A parity failure stops the experiment. Existing evidence cannot be evicted by
  additions. Explicit output roots, complete-case atomic caches, one-worker-per-
  shard locks, strict full-manifest merge, model provenance checks and canary cap.
- `agent_evaluate.py`: separate offline scoring, rescue/harm diagnostics and image-
  cluster bootstrap. Optional unchanged ANCHOR evaluation. No labels in planner,
  expert inference, selection, or generation.

## What is NOT implemented or claimed

No new medical expert weights, new segmentation model training, automatically
assembled external knowledge base, trained/fitted gate, temperature calibration,
cross-case learned memory, arbitrary code generation, or clinical risk guarantee.
No claim that model self-consistency, a CLIP margin, a crop, or a segmentation mask
proves medical correctness. No automatic medically correct claim extraction or
clinical-error detector. Ledger revocation is structural/integrity-based; it is
not a demonstrated causal medical validator. The frozen planner selects ready
local-observation operations, not arbitrary self-invented tools/diagnoses.

Only compatible **predicted segmentation evidence already present in the incumbent**
is used to create regions in v1. A case with no such evidence is honestly marked
as having no region extension. The plugin path does not pretend to cover missing
CT/MRI/ophthalmology models. Expand the specialist pool in a separate legacy
baseline run first; do not silently change it during an agent comparison.

The final experimental gate is `audit_only` by default: it records
`agent_candidate` separately and keeps the incumbent in `agent_committed`.
This all-unknown preservation arm is NOT a gate-accuracy achievement. Optional
`external_compare` reuses the existing fallible `assess_revision` comparator,
but maps its abstention to incumbent preservation, never a bare-generalist
fallback. It is a comparator baseline, not a new uncertainty estimator. All
candidate arms must be evaluated; do not hide harmful candidates behind fallback.

## Experimental arms

| Arm | Meaning |
|---|---|
| incumbent | Exact existing semantic_all / compact_rows / compact_all output |
| graph_noop | Real regeneration using identical native packets and legacy renderer; exact parity required |
| graph_static | Fixed typed region -> observation plan |
| region_control | Same plan/budgets, opposite-window control of the same pixel dimensions |
| full_image_control | Same plan/budgets, repeated whole-image observation instead of a crop |
| agent_candidate | Frozen constrained controller chooses ready actions or STOP |
| agent_committed | Audit-only preservation or explicitly selected external-comparator baseline |

Controls do not flip images. An opposite window may coincide with the expert
window (for example a central/full image box); that is recorded and is not counted
as a valid distinct-location control. All arms share budget caps; the agent may
STOP early, so realized computation is not necessarily identical. Original
incumbent compute, extra compute, control tokens and replay-audit overhead must be
reported separately. Shared warm models are not independent cold-start timings.

A local VLM description depends on the crop AND its upstream segmentation. These
are not independent votes. Source-model lineage is preserved, but different
checkpoints are not assumed statistically independent. Invalidating a crop blocks
its descendants while retaining unrelated evidence. The code does not force
confidence values from heterogeneous experts into a common probability scale.

## Required inputs

`--base-run` must be the **finalized root** of a matched/evidence_revision run,
with `protocol.json` (`shards_complete: true`) and an incumbent output JSON.
Every incumbent case must contain `text`, `token_ids`, raw native `evidence`,
`generation_config`, and actual routed `input_modality`. It must use full-answer
semantic transport without a learned bridge, spatial modification or old gate.
The root must match the entire inference manifest; partial snapshots are rejected.

`--manifest` is label-free JSONL: required id/image/question/image_sha256;
answer_type may be used ONLY to reproduce the exact legacy answer prompt. Optional
modality/task/domain/group and patient/study identifiers are accepted. References,
answers, labels and unexpected fields are rejected. Images are hashed before
model loading. Never use a scored/annotated manifest for inference.

Run from the repository root. Checkpoints/source code paths remain those in the
base protocol; `--artifacts` resolves existing model artifacts. In a new worktree,
use symlinks to already-present local assets as needed; never copy over an existing
path, overwrite old outputs, upgrade shared packages, or redownload checkpoints.

## Commands

Inspect the existing environment before using these commands. Replace paths with
actually discovered paths; do not invent manifest names or rebuild dataset splits.

```bash
python -m pytest tests/test_evidence_agent.py -q
python -m merit_feddg.agent_run --help

BASE=/absolute/path/to/completed/base-run
MANIFEST=/absolute/path/to/the/same/label-free-manifest.jsonl
ARTIFACTS=/absolute/path/to/existing/artifacts
OUT=/absolute/path/to/new/runs/evidence-agent-v1

python -m merit_feddg.agent_run \
  --base-run "$BASE" --incumbent compact_rows --manifest "$MANIFEST" \
  --artifacts "$ARTIFACTS" --output "$OUT" --check-only

# Same full manifest/identity. Stops after two complete cases, saves atomic caches,
# and NEVER publishes partial aggregate results. Remove flag to resume full run.
python -m merit_feddg.agent_run \
  --base-run "$BASE" --incumbent compact_rows --manifest "$MANIFEST" \
  --artifacts "$ARTIFACTS" --output "$OUT" --canary-cases 2

python -m merit_feddg.agent_run \
  --base-run "$BASE" --incumbent compact_rows --manifest "$MANIFEST" \
  --artifacts "$ARTIFACTS" --output "$OUT"
```

There is no training command and no automatic model download. Offline mode is
forced. `--check-only` validates manifests and provenance structure but does NOT
certify GPU health or model execution. Runtime/OOM errors propagate rather than
being disguised as negative evidence. An interrupted incomplete case reruns;
completed cases are reusable only with the exact identity and model provenance.

For two explicitly authorized GPUs, use the same paths/options/output root with
`--shard-count 2 --shard-index 0` / `1`, one worker per device. Then rerun with
`--shard-count 2 --merge-only`. Do not assume container GPU numbers are physical
host numbers; do not touch another person's processes. A missing shard never
produces a completed protocol or imputed rows.

## Optional small source-case bank

No source manifest means retrieval is disabled and logged as such. A source JSONL
record must explicitly contain:

```json
{"id":"source-case-001","image":"/absolute/source.png","image_sha256":"ACTUAL_HASH","question":"Source question or description","modality":"cxr","domain":"external-source","group_id":"source-patient-image-group","role":"source","split":"train","task":"open_vqa","reference":"Optional source-only answer/report"}
```

Use actual data, not this placeholder. `split` must be `train` or `external`.
Include subject_id/study_id where available. Every query image in the FULL
manifest is excluded by pixels/ID/group, not just the current shard; patient/study
exclusion is checked when IDs exist. No guarantee of patient separation is claimed
without those IDs. Near-duplicate image/paper-figure review remains a data-curation
requirement outside these exact-hash checks.

Add `--source-manifest /absolute/curated-sources.jsonl` to ALL run/check/merge
commands. The frozen `source_cases` model from the base configuration is reused.
It retains its same-domain/task/modality exclusion and image/question retrieval
logic. This is a small-corpus reference implementation, not a large-scale FAISS
pipeline. Crop retrieval may be out of the encoder's validated distribution.
Source answers continue to obey the incumbent renderer's
`retrieval_answer_context` policy. When that is false, source-answer text may be
withheld; do not describe a pointer-only result as injected medical knowledge.
Source references are analogies, never observations of the query patient.

## Offline evaluation and reporting

`ROOT` is the identity directory printed by preflight, not its parent `$OUT`.

```bash
python -m merit_feddg.agent_evaluate \
  --run "$ROOT" --manifest "$MANIFEST" \
  --references /absolute/offline/references.json \
  --anchor-root /home/dbw/ANCHOR \
  --output "$ROOT/evaluation.json"
```

Without ANCHOR the command provides only clearly named lexical diagnostics.
Nonbinary CLOSED references are not silently recoded as No. The diagnostic scorer
is VQA-only; it refuses report generation. The official evaluator is imported,
not rewritten. Report-generation experiments need a separate frozen factual
scorer and are not validated by these VQA diagnostics.

Report original vs incumbent, all candidates vs incumbent, metric gains/harms,
changed-answer coverage, useful-region coverage, actual operations, unavailable
reasons, dependent evidence counts, budget failures, compute, and paired image-
cluster intervals. Audit clinically important errors invisible to token recall.
Do not select a gate, threshold, region count or prompt on held-out test labels.
Already inspected development/test data must not be called an untouched target.

## Code lineage and contribution boundary

ReAct supplies the reasoning/action pattern; MedRAX already has a medical
LangGraph tool loop; LLMCompiler already schedules dependent tools. Their ideas
are not claimed as new here:

- ReAct: https://github.com/ysymyth/ReAct
- MedRAX: https://github.com/bowang-lab/MedRAX/blob/main/medrax/agent/agent.py
- LLMCompiler: https://github.com/SqueezeAILab/LLMCompiler/blob/main/src/llm_compiler/task_fetching_unit.py
- Local incumbent and expert contracts: merit_feddg/capability_runtime.py,
  capability_experts.py, compact_evidence.py, matched_evaluation.py.

The next research question is whether data-dependency-aware acquisition and
revision helps with naturally erroneous, irrelevant, duplicate and conflicting
experts under matched compute, beyond these strong existing baselines. No such
medical experiment has been run by this code change.
