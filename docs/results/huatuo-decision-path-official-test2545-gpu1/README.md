# GPU1 real TEST preflight: historical cache mismatch, no efficacy result

2026-09-20 UTC. **The complete official TEST evaluation was actually launched on host GPU1, but the first native canary failed historical-token parity before any C candidate/control case completed. Current coverage is0/2545 and there are no new paired TEST scores.** This is a technical stop, not a scientific rejection or passing result.

The user's latest instruction authorized GPU1 and requested continuation through results. The previous GPU0 resource waiter PID2740944 was terminated and verified absent before migration. GPU1 UUID `GPU-3846413a-4238-d307-b1f3-10c2dfbe002c` was idle with48491 MiB free. An immutable device successor preserved both prior failed TEST attempts,44791 historical forwards, the full2545 cases and fixed project/layout_average candidate. Controller PID2809973 and its real worker ran on GPU1; both have now exited, leaving20 MiB device use. No other task was stopped and no old queue remains active.

## Actual canary evidence

| First fixed VQA-RAD case | Generated tokens |
|---|---:|
| Current native model.generate | 47 |
| Current algorithm-off replay | 47 |
| Historical cached Baseline | 121 |

Current native and off replay are exactly token-identical. Both end with native EOS. The historical sequence agrees for16 tokens and diverges at token17. The worker preserved all token traces locally and raised `Native/historical/off canary mismatch`; no candidate generation or offline scoring followed. Compact parity and the other three fixed canaries have not yet been exercised. Zero full canaries passed.

Read-only investigation confirmed the first input's image file hash/path, official question, benchmark prompt, generation settings and adapter/config source hashes. The older Baseline export came from a historical runtime path; these checks do not prove historical kernel/runtime equivalence. The exact underlying cause of the historical divergence is not isolated. It would be invalid to loosen token equality or reuse that incompatible Baseline as if it matched the current generator.

## Coverage, limits and costs

The frozen scope remains VQA-RAD451 + SLAKE2094 (English1061/Chinese1033), all official TEST questions. Every planned case appears in the anonymized per-case file with null scores. Previous full cache/prompt audits remain [here](../huatuo-decision-path-official-test2545/README.md); the earlier queued status is superseded by this stop.

- TEST attempt1: JSON sorting changed native array ordinals; repaired with original nested-order preservation.0 forwards.
- TEST attempt2: competing GPU0 compute, before model loading.0 forwards.
- TEST attempt3: GPU1 actual native/off generation agrees but historical Baseline differs.94 forwards.
- Cumulative **44885/200000** forwards, including every earlier campaign/probe cost. Remaining155115. The frozen TEST attempt ceiling3 is exhausted.

`costs.json` records actual worker/model-load wall and the measured off-replay allocation peak. A failed canary is not a full inference benchmark, and no candidate latency or efficacy estimate is available.

## Concrete repair prepared, awaiting additional authorization

`scripts/refresh_algorithm_test_controls.py` is prepared to regenerate both native controls for all2545 official questions using the same frozen Huatuo model, original prompt and actual expert evidence. It checks exact historical prompt/transport identity before generation; it never reads reference answers. It retains original caches and evidence audits, persists every actual native generation and forward count, checks current native/off parity on the fixed first two cases of each queue, and emits a complete refreshed source only after all cases finish.

The TEST freezer now supports explicit authorized GPU/attempt-budget selection and a completed refresh source, verifies its lineage and adds all refresh forwards before setting the remaining inference budget. It does not change the algorithm, scorer, generation defaults or TRAIN controller. The default TEST ceiling remains3; it has not been silently raised.

An explicit asynchronous question requests authorization for **TEST attempt4 and cumulative1000000 forwards**, including native cache refresh. At this snapshot that authorization has not arrived. **The refresh and fourth attempt have not run.** The new refresh code has passed compilation, but has not been GPU-validated;95 existing/regression CPU tests pass. `proposed-continuation.json` is a reviewable proposal, not an executable frozen run or a claim of success.

Canonical failed GPU1 artifacts: `runs/official-test2545-gpu1-v1/`, plan `ab0ee844ebff03a0622747575027cb43562d7b74761398b27989955fceb8ec20`. All earlier failures remain intact. Raw medical images, references, answers, tokens and evidence remain local. Original TRAIN screen remains failed; no READY_FOR_SCALE or new clinical claim.
