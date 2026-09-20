# Full official TEST evaluation: frozen and queued on GPU0

2026-09-20 UTC snapshot. **The user explicitly authorized the complete official TEST datasets. All2545 inputs and genuine cached Baseline/native outputs are verified, but the new C candidate has completed0/2545 cases. No new TEST score or canary pass is claimed.**

| Dataset | Questions | Distinct images | New candidate/control complete |
|---|---:|---:|---:|
| VQA-RAD official TEST | 451 | 203 | 0/451 |
| SLAKE official TEST | 2094 | 180 | 0/2094 |
| Total | 2545 | 383 | 0/2545 |

SLAKE retains all1061 English and1033 Chinese questions. Manifest order and complete ID sets are frozen. No answer-based selection, TEST-to-TRAIN relabeling, new expert, gate, tuning or generation change. Each queue's fixed first two cases must pass the unchanged real native/historical/off/audit canary before continuing that queue. Runtime errors and incomplete cases will not be hidden.

## Fixed method and interpretation

Candidate `project`, matched control `layout_average`, unchanged HuatuoGPT-Vision-7B and decoder source `9a55426`. Greedy generation, repetition penalty1.2, min1/max1024 new tokens, native EOS, private shared-prefix streams. Reuses the existing `algorithm_chain.native.run_job` and a separate CPU-only `score_job`; the original TRAIN source guards and A/B/C/D/E controller remain unchanged.

This is an explicitly requested benchmark measurement following the failed TRAIN screen. It does not retroactively make the method READY_FOR_SCALE. Historical official TEST Baseline/native answers were already available in this workspace; this is not a never-observed independent-confirmation claim. References stay out of the inference job and are read only by the isolated scorer. The scorer remains the frozen CLOSED metric/OPEN first-reference token recall; scores are not verified medical correctness.

## Verified cache compatibility and actual failures

Both full original Baseline/native ID sets, official questions, image file hashes, prompt construction, generation settings, code fingerprints and completion markers were verified. No baseline or expert output was fabricated or regenerated. A TEST-specific adapter exports the actual cached outputs and original raw evidence for the existing worker.

1. TEST attempt1 (`official-test2545-gpu0-v1`): model loaded, strict transport check stopped before any forward. Canonical JSON sorting had changed nested evidence field order, which changes compact array ordinals and prompt identity. Original-cache rendering reproduced the historical hash exactly; the first adapted rendering did not. A regression reproduces this failure. The repaired exporter preserves nested order and verifies all2545 original prompt hashes. Old artifacts remain untouched.
2. TEST attempt2 (`official-test2545-gpu0-v2`): the repaired plan passed static checks, but another training job acquired host GPU0 during preparation. Device ownership check stopped before model load. Zero new forwards. The other task was not stopped.

Both failures and attempts are retained, leaving one of the three TEST execution attempts. Prior chain attempts A=3/C=3 remain recorded; this is a separate user-authorized TEST evaluation, not a reset of C. Historical44791 forwards remain in the cumulative200000 cap, leaving155209. A request to increase the cap has not been approved at this snapshot; no increase was applied. Reaching the cap must be reported as incomplete, not full evaluation.

## Actual resource queue

Host GPU0 UUID `GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023`. At verification it had37578 MiB free, but another `train.py` process PID2684245 occupied10486 MiB. The frozen guard rejects competing compute processes even when memory remains. Its existing narrow34 MiB Nautilus C+G exception is retained. GPU1 is also occupied and is not a fallback.

A detached, bounded resource wait is actually running on the host, PID **2740944**, checking the same `check_device` every30 seconds for up to1440 checks (approximately12 hours). It does not load a model or consume an execution attempt while the check fails. Once the device is eligible, it invokes the frozen TEST runner, which rechecks all pins and device ownership, grants attempt3, performs inference and then isolated offline scoring. A race at final acquisition can still cause the last attempt to fail; no fourth attempt or budget reset is authorized.

A resource waiter is not an inference worker. Current controller state is `BLOCKED_TEST_TECHNICAL`; the external scheduling status is `QUEUED_GPU0_RESOURCE`. If the wait expires, no automatic inference begins. No efficacy result is available yet. Inspect actual case artifacts/state before reporting progress.

## Artifacts and validation

95 CPU tests passed, zero failures/skips, including the observed serialization-order regression. Compilation, CLI help and diff checks passed. Real canaries are still pending; CPU tests are not an algorithm-success claim.

- Local source: `runs/official-test2545-inputs-v2/`.
- Frozen run: `runs/official-test2545-gpu0-v2/`; plan `7b989faafb3e1b5aec5c4811034db44dc8610e7742c3774857683a5fdddba13e`.
- Logs: `controller.log`, `inference.log`, `resource-wait.log`; waiter PID file `resource-wait.pid`.
- `summary.json`, `node-decisions.json`, `costs.json`, `failures.json`, `validation.json`, `data-audit.json`, `plan.redacted.json`, `provenance.json` preserve the actual snapshot and lineage.
- `per-case-results.jsonl` includes every planned hashed case with null scores and `not_generated`, not invented results.
- [Executed commands](commands.md). Raw images, references, predictions, expert payloads and full plans remain local.

See the [completed TRAIN expansion](../huatuo-decision-path-algorithm-chain-v1-expansion151-gpu0/README.md) for prior results. No main/PR merge.
