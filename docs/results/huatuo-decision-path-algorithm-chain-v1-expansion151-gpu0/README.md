# GPU0 exploratory expansion completed: 151/151, screening fails retention

2026-09-20 UTC. **Host GPU0 completed all added SLAKE64 + VQA-RAD87 cases and independent offline scoring, with zero inference failures and 4/4 fixed native canaries passed.** The controller reached `EXPLORATORY_COMPLETE`; the frozen C screening verdict is `fail`, because gain retention is 70%, below 75%. No READY_FOR_SCALE, D/E, TEST inference, training or parameter search followed.

The latest user instruction explicitly authorized host GPU0 and superseded the earlier GPU1-only restriction. The previously queued GPU1 run had actually failed its ownership check before model loading when another compute job occupied the device: 0 forwards, 8.767 seconds. That C-attempt-2 failure was preserved; the standard retry granted C-attempt-3 without resetting attempts or the inherited 10164 forwards.

## Complete-denominator results

Scores use the unchanged pinned CLOSED metric and first-reference OPEN `answer_token_recall`, on [the original conditional-chain protocol](../huatuo-decision-path-algorithm-chain-v1-repair/README.md). All added cases and both arms completed. Generalist/compact cached answers were rescored; they were not regenerated. These automatic scores are not verified clinical correctness.

| Queue | N | Baseline | Compact | C projection | Layout-average control | C − Baseline, pp |
|---|---:|---:|---:|---:|---:|---:|
| Added SLAKE | 64 | 52.8646% | 52.9167% | 56.8750% | 52.3958% | +4.0104 |
| Added VQA-RAD | 87 | 60.3448% | 53.4483% | 59.1954% | 54.0230% | -1.1494 |
| All added | 151 | 57.1744% | 53.2230% | 58.2119% | 53.3333% | +1.0375 |
| Original SLAKE | 128 | 58.6068% | 57.4349% | 57.6302% | 58.2161% | -0.9766 |
| Pooled SLAKE | 192 | 56.6927% | 55.9288% | 57.3785% | 56.2760% | +0.6858 |
| All pooled | 279 | 57.8315% | 55.1553% | 57.9450% | 55.5735% | +0.1135 |

The 151 added images have no pixel overlap with the original128, official TEST or each other. **All are historically exposed TRAIN development data.** Pooled279 is descriptive, combines the same fixed C decoder/scorer across nonoverlapping images, and is not a new independent confirmation. Patient-level independence remains unproven. Dataset composition changes the weighted mean; SLAKE192 and VQA-RAD87 are separately available in `group-results.json` (the VQA-RAD dataset total is the same added87).

| Scope | Candidate vs | Delta, pp | Image-bootstrap 95% interval, pp | Improved / harmed / unchanged |
|---|---|---:|---|---|
| Added151 | generalist | +1.0375 | [-3.2671, 5.5629] | 10 / 6 / 135 |
| Added151 | compact | +4.9890 | [0.2423, 9.8455] | 14 / 4 / 133 |
| Added151 | control | +4.8786 | [0.0221, 9.8455] | 15 / 5 / 131 |
| Pooled279 | generalist | +0.1135 | [-3.1662, 3.3692] | 17 / 14 / 248 |
| Pooled279 | compact | +2.7897 | [-0.9023, 6.4934] | 25 / 13 / 241 |
| Pooled279 | control | +2.3716 | [-1.4639, 6.2727] | 27 / 16 / 236 |

Intervals use the frozen seed0/4000 repetitions, are descriptive development intervals without a new multiple-comparison adjustment, and do not authorize promotion. The added candidate's interval versus Baseline includes zero. Added SLAKE improves +4.0104 pp versus Baseline; VQA-RAD declines −1.1494 pp. The pooled gain versus Baseline is only +0.1135 pp, below the original +0.5 pp screening requirement.

## Benefit, harm and stopping rule

- Added151: repairs **11/16** original compact harms, all 11 fully restored; retains **7/10** original gains and loses3. It damages **5/79** Baseline-full-score cases (6.33%). One new harm occurs where compact was not harmful.
- Pooled279: repairs **20/30** original harms, 19 fully restored; retains **13/24** original gains (54.17%) and loses11. It damages **11/149** Baseline-full-score cases (7.38%); three new harms occur where compact was not harmful.
- SLAKE added64: repairs5/7, retains6/8; VQA-RAD87: repairs6/9, retains1/2. The VQA original-gain count2 is below the screening minimum3, so its separate quality result is inconclusive, not a passing dataset claim.

For the actual added151 C decision, both +0.5 pp improvement thresholds, positive matched-control difference, at least one repair, at most10% Baseline damage, mean decoding at most10 seconds, coverage, intervention and answer-change gates pass. **Only original-gain retention fails: 70% <75%.** `summary.json` records every gate; `node-decisions.json` contains the controller's immutable decision. The inherited `blocked_node` field is historical; the authoritative current `node` is `EXPLORATORY_COMPLETE`.

The original A→C→STOP_NO_CANDIDATE result remains unchanged. The user's later request authorized this fixed-candidate exploratory expansion; its mode never advances to D/E or READY, even if a development point estimate passes. This completed negative screen does not justify adding a new candidate or resetting attempt limits.

## Intervention and generation

There are100 active projection cases and51 empty-evidence bypasses; those51 are the only zero-projection cases. Candidate tokens differ from compact on72/151 and Baseline on76/151. There are three distinct layouts on36 cases, two on64, one on51. Across4948 projection steps, rank1 occurs2555 times and rank2 occurs2393 times. Mean removed/retained weighted norms are0.245246/0.389745; maximum orthogonality residual is3.161e−13. Packet content was unchanged.

Candidate/control generate6773/4458 tokens, maxima316/342. All302 outputs end with EOS; zero1024-token limit hits. Changed tokens do not establish medical correction. No raw answer/reference text or medical images are published.

## Runtime and cost

Actual host GPU0 UUID: `GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023`. Python3.10.20, Torch2.0.1+cu117, Transformers4.37.2; existing HuatuoGPT-Vision-7B weights and shared offline cache. Generation source remains `9a554268250ef1ff07ca55de35947ac2f04091ee`; migration/controller execution HEAD is `101022ca05d75ad4e4dda8d1e39567c82d28cdfd`. Scientific code pins, weights, prompts, generation and scorer were verified unchanged. The migration utility preserves the GPU1 resource failure and changes only the authorized device policy/lineage.

| Phase | Seconds | Forwards |
|---|---:|---:|
| GPU1 failed resource attempt | 8.767 | 0 |
| GPU0 worker total | 1382.468 (23.04 min) | 34627 |
| Model load and runtime provenance, included above | 12.843 | Included |
| Case preparation, included above | 20.093 | 0 |
| Four canaries, included above | 22.247 | 653 |
| Candidate decode, included above | 805.136 | 21617 |
| Matched-control decode, included above | 496.626 | 12357 |
| Offline scoring | 0.176 | 0 |
| All inherited runs and probes | See prior cost report | 10164 |
| **Cumulative** | Full-task wall not instrumented | **44791 / 200000** |

Candidate/control mean decoding time is5.332/3.289 seconds per case. Each active step retains the same four-stream computation, but generated lengths differ, so realized total costs differ. Allocated-tensor peak for both arms is **22.480 GiB** (24137898496 bytes). A sampled device-residency reading was36406 MiB, including reservations and desktop, not the allocator peak or a measured device peak. Canary allocation peaks are not separately instrumented. Baseline cached deployment latency is unknown, not zero. Per-forward persistence and extra view preparation add experimental overhead; some phase time is included only in worker wall. New source-cache generation was unnecessary; initial expansion audit wall was not instrumented.

Both controller and worker exited successfully. GPU0 returned to428 MiB desktop use/48074 MiB free; GPU1's unrelated model job remained running. No task was killed. The narrowly frozen34 MiB Nautilus C+G exception is unchanged and ordinary competing compute is still rejected.

## Validation and artifacts

135 CPU tests passed, zero failures/skips. All four fixed canaries passed native/historical/off token equality and audit, with at least one real delivered-expert canary in each cohort. Rechecked frozen pins, all151 score/prediction bindings, recomputed decision equality, all279 unique pixel identities and the complete forward ledger. Both worker and CPU-only scorer exited0. No extra scorer overwrote results.

- `summary.json`, `group-results.json`: full metrics, paired intervals, benefit/harm, gates and behavior.
- `per-case-results.jsonl`: all279 hashed case identities and numeric paired outcomes; original128 clearly separated from added151.
- `node-decisions.json`, `failures.json`, `costs.json`, `validation.json`: full decision history, failures, actual cost and checks.
- `plan.redacted.json`, `provenance.json`: policy, generation, device lineage, source/model/code/scorer hashes; raw local input mappings redacted.
- [Executed commands](commands.md). Canonical local artifacts: `runs/chain-v1-expansion151-gpu0-v2/`, new outputs only under `C-attempt-3/`.

The earlier [GPU1 queued snapshot](../huatuo-decision-path-algorithm-chain-v1-expansion151/README.md) is historical and superseded by this completed report. Raw predictions, references, images, caches and complete immutable plans remain local. No main/PR merge.
