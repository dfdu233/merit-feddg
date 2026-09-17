# Codex execution: pathology expert headroom, no new gate

Read `skills/cs197-research/SKILL.md` and `docs/PATHOLOGY_QUILT_VECTOR.md`.
Work on a clean independent worktree of `implementation/pathology-quilt-pilot-v1`.
Keep all dirty worktrees, existing models/configurations/results and active jobs.

1. Run the focused test file and full CPU regression in the existing MERIT
   environment. Distinguish prior missing-dependency/lint problems; do not hide
   failures. Two upstream integration checks were skipped in the authoring
   container; they MUST execute in the real repository. Compile and run CLI help.
2. Locate the actual PathVQA TRAIN manifest and complete shared-64-token compact
   baseline. Do not substitute a test prefix or reconstruct missing outputs. If
   the existing baseline is incompatible, report it; any new baseline must have
   its own frozen identity and keep the old one intact.
3. Run CONCH direct-vs-adapter audit on a fixed eligible TRAIN image. Same existing
   checkpoint/catalog/preprocessing; no new diagnosis labels, threshold or null
   subtraction. Retain the catalog's limited tissue-appearance interpretation.
4. Set up Quilt in a SEPARATE interpreter/source directory following the official
   aldraus/quilt-llava instructions. Do not upgrade shared MERIT dependencies or
   import both llava packages into one process. Obtain official generic
   wisdomik/Quilt-Llava-v1.5-7b and its declared CLIP-L/14-336 dependency with
   authorized access. Freeze model/source revisions and download hashes. No
   automatic clinical dataset download or task finetuning. Inspect model terms
   and training overlap. Do not use PathGen/Mistral weights with the Quilt loader.
5. Prepare the fixed 12-case pilot using the existing model-free stage. Run the
   Quilt worker with exactly one authorized GPU UUID; run both matched and
   same-question wrong-image jobs. The worker exits BEFORE the actor is loaded.
   For a 24GB-class limit start with the declared 4-bit Quilt arm; preserve that
   choice across all cases, report it, and stop on OOM rather than changing it.
6. Run actor stage in existing MERIT environment. It verifies all historical
   prompt/token parity and actual delivery. No new gate, parser, RAG, segmentation,
   templates or decoding. No automatic full benchmark. Missing/corrupt caches,
   empty outputs, truncated/displaced evidence or parity drift must stop with a
   report, not a invented fallback or deletion of the offending sample.
7. Evaluate only after every frozen case completes, using the already-frozen
   ANCHOR scorer and matching TRAIN references. Report all five arms; show closed
   rescue/harm separately from open continuous score changes. Record expert-alone
   headroom, same-case evidence vs wrong-image control, CONCH removal effect,
   real delivery, all calls/cost/peak memory, and image-type coverage. Inspect
   factual correctness independently; do not claim clinical accuracy from recall.
8. Push only reviewed source/runtime fixes, deidentified aggregates and provenance
   hashes to an independent results branch. Never upload model weights, source
   images, raw answers, raw caches, references or credentials. Verify remote SHA.
   No force push, automatic merge or indefinite GPU workload.

## Commands (resolve actual paths; placeholders are not existing resources)

```bash
# Existing MERIT interpreter:
PYTHONPATH=. "$MERIT_PY" -m pytest -q tests/test_pathology_quilt_pilot.py
PYTHONPATH=. "$MERIT_PY" scripts/audit_conch_native_parity.py \
  --config "$EXISTING_CONFIG" --image "$FIXED_TRAIN_IMAGE" \
  --gpu-uuid "$AUTHORIZED_GPU_UUID" --output "$NEW_CONCH_AUDIT_JSON"

PYTHONPATH=. "$MERIT_PY" scripts/run_pathology_quilt_pilot.py \
  --stage prepare --manifest "$PATHVQA_TRAIN_MANIFEST" \
  --incumbent-run "$COMPLETE_COMPACT_RUN" --incumbent-arm compact_rows \
  --anchor-root "$ANCHOR_ROOT" --output "$NEW_RUN" --limit 12

# Different Quilt interpreter, same code checkout; process must finish before actor:
CUDA_VISIBLE_DEVICES=0 HF_HOME="$QUILT_OFFLINE_HF_HOME" \
  "$QUILT_PY" scripts/run_quilt_worker.py --run "$NEW_RUN" \
  --source "$QUILT_SOURCE" --checkpoint "$QUILT_CHECKPOINT" \
  --precision 4bit --gpu-uuid "$AUTHORIZED_GPU_UUID"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. "$MERIT_PY" \
  scripts/run_pathology_quilt_pilot.py --stage generate \
  --output "$NEW_RUN" --gpu-uuid "$AUTHORIZED_GPU_UUID"

PYTHONPATH=. "$MERIT_PY" scripts/evaluate_pathology_quilt_pilot.py \
  --run "$NEW_RUN" --references "$PATHVQA_TRAIN_REFERENCES"
```

No real model run was executed in the authoring container. The worker intentionally
checks the official prompt-plus-output token layout and rejects an incompatible
runtime. Compatibility fixes need new source/frozen identity before rerunning;
never fabricate text by guessing slice offsets. The current pilot cannot prove
performance on gross pathology or whole PathVQA.
