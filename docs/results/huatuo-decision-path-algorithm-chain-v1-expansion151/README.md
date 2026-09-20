# Exploratory TRAIN expansion — GPU1 only, queued

2026-09-20 UTC snapshot. **The additional 151-case experiment is frozen and queued for GPU1. No new model cases have run and no expansion efficacy result is available yet.** See `summary.json` for the exact snapshot time and `runs/chain-v1-expansion151-gpu1/` for live execution state.

The user requested a larger evaluation after the original chain reached `STOP_NO_CANDIDATE`, then explicitly restricted this expansion to host GPU1. This authorizes a separate exploratory evaluation; it does not convert the previous negative result into `READY_FOR_SCALE` or independent confirmation.

## Scope frozen before inference

| Queue | Additional cases / distinct images | Overlap with previous 128 | Overlap with official TEST |
|---|---:|---:|---:|
| Existing native SLAKE TRAIN64 | 64 / 64 | 0 | 0 |
| Existing native VQA-RAD TRAIN87 | 87 / 87 | 0 | 0 |
| Total added | **151 / 151** | 0 | 0 |

The two added queues also have no mutual RGB-pixel overlap. Together with the previous SLAKE128, the planned observation scope is **279 cases**, including 192 SLAKE and 87 VQA-RAD cases. All cases in the two complete cached queues are included; no selection uses answer quality. Protocol/completion identities, source code pins, image bytes/pixels, cached baseline/compact output presence and reference ID alignment were validated. The host execution account could read all 355 necessary input paths.

These are previously exposed research TRAIN queues. They are not independent holdouts; patient-level independence remains unknown. Official TEST images are used only for exclusion checks. Confirmation/extension remain empty, and no TEST generation or scoring is scheduled.

The fixed candidate is C `project`, with unchanged `layout_average` matched control, generation settings, model/expert payloads, score functions and scientific thresholds. The old baseline/compact caches are reused. Existing 128-case results remain preserved separately. The intended report separates new-only, per-queue, per-dataset and pooled 279-case results; the current package does **not** invent the pending scores.

## Separate exploratory execution

The new `freeze-expansion` entry point requires the preserved predecessor stop and fixed C candidate. It verifies prior costs/decisions, prevents predecessor image reuse, preserves generation implementation and scorer pins, and retains attempts/forwards. Only administrative controller/state changes and an explicitly frozen device change are allowed; it does not reopen the original chain.

The successor starts the scheduled **C-attempt-2** with 10164 inherited forwards and 189836 remaining. The previous A=3/C=1 results and technical preflight records are copied byte-for-byte as history. A device change clears only the runtime fingerprint to let the new physical device establish its own provenance; it does not clear history, cost or candidate identity.

After the added cohort is fully generated and independently scored, the existing quality report supplies descriptive gate outcomes. Completion is always `EXPLORATORY_COMPLETE`, even if those descriptive gates pass; the controller cannot promote this expansion into D/E or `READY_FOR_SCALE`. Technical failures remain bounded, explicit and retryable through the existing mechanism.

## GPU1-only correction and actual queue

An initial GPU0 resource waiter was created while both GPUs were occupied. After the user's GPU1-only instruction, its process group was terminated. PID 2075033 no longer exists, and that queued plan has no controller log, no `C-attempt-2` directory, no model execution and no consumed model forwards. Its immutable plan and cancellation certificate remain in `runs/chain-v1-expansion151/`.

The active successor is **only host GPU1 UUID `GPU-3846413a-4238-d307-b1f3-10c2dfbe002c`**, container-visible index 0. No GPU0 or automatic alternative-device fallback exists. Plan identity:

`a505a2d988729d39b0f6c54d561c07574c104229bba59248b0fbfe1c023141fa`.

A detached, exclusively locked resource waiter is actually running, container PID **3475962**, with log `runs/chain-v1-expansion151-gpu1/resource-wait.log`. Its latest verified sample showed 22409 MiB free on GPU1 and another active model job. It checks every 30 seconds, up to 1440 checks (approximately 12 hours), requiring at least 30720 MiB free and no other compute job before invoking the existing detached controller wrapper. It does not kill other jobs or consume a model attempt on known resource contention. A timeout is logged explicitly.

The original wrapper receives exactly the Python absolute path and the frozen output absolute path. Only after that wrapper launches, and runtime/budget/case artifacts grow, can this be called a real model run. At this snapshot, new coverage is **0/151**, new forwards **0**, actual native canaries **not executed**, and new scores **null**. The queue PID is not an inference worker.

## Validation and artifacts

Execution/controller revision: `575385e`; candidate-generation code remains byte-identical to the repaired `9a55426` implementation. **134 CPU tests passed** in 3.61 s, with 0 failures/skips. Tests include prevention of confirmation promotion, inherited-budget preservation, terminal resumption safety and new-device runtime identity. Compilation, CLI help, wrapper syntax and diff checks passed.

Each added queue's fixed first two historical cases includes real expert delivery. Actual native/historical/off/audit parity must still pass during execution; cached prerequisites and CPU tests are not evidence of GPU success.

The public package preserves all 151 planned cases as hashed, explicitly not-started rows. It contains the frozen scope, queue status, null result fields, data audit, validation, provenance and cancellation history. Raw images, questions, answers, reference contents and complete expert payloads remain local.

The directory copies predecessor artifacts for lineage. **Do not count copied `C-attempt-1` or A case files as added results.** Only the new `C-attempt-2` outputs belong to this expansion. The [previous completed report](../huatuo-decision-path-algorithm-chain-v1-repair/README.md) remains the latest observed effect: candidate 57.6302%, Baseline 58.6068%, matched control 58.2161%, on the original 128 cases only.
