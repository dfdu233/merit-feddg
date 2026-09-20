# Completed conditional chain: A → C → STOP_NO_CANDIDATE

2026-09-20 UTC. **The repaired campaign reached its scientific stopping rule. A and C each processed and scored all 128 planned development cases without inference failures. Neither directional localization nor the C candidate passed its frozen gate. READY_FOR_SCALE was not reached.**

This is the end of the conditional protocol, not an interrupted run. A did not select B; C did not qualify for D; E was consequently not triggered. No new candidate, parameter search, forced confirmation, official TEST inference or training followed the stop. The controller and workers have exited and released their GPU memory.

## Results on the complete denominator

All figures use the same pinned offline scorer: `evaluate_rows` CLOSED and first-reference OPEN `answer_token_recall`. The scorer ran separately with CUDA disabled. Cached generalist/compact answers were reused and rescored; no full baseline or expert-cache regeneration occurred.

| Arm | Complete / planned | Mean score |
|---|---:|---:|
| Generalist / Baseline | 128/128 | 58.6068% |
| Compact | 128/128 | 57.4349% |
| C projection candidate | 128/128 | 57.6302% |
| Matched `layout_average` control | 128/128 | 58.2161% |

| Candidate comparison | Mean delta (percentage points) | Image-bootstrap 95% interval | Improved / harmed / unchanged |
|---|---:|---:|---:|
| vs Baseline | −0.9766 | [−5.6641, 3.9063] | 7 / 8 / 113 |
| vs compact | +0.1953 | [−6.0547, 6.4453] | 11 / 9 / 108 |
| vs matched control | −0.5859 | [−7.2266, 6.0547] | 12 / 11 / 105 |

C improves 9/14 previously harmed cases relative to compact; 8/14 fully recover the baseline score. However, it retains only **6/14 original gains (42.86%)**, losing 8. Six of 70 baseline-full-score cases are damaged (8.57%); 2 cases become worse than baseline where compact had not been harmful. These denominators describe different comparisons and should not be added together.

Automatic token-recall/closed-answer scores are not verified medical correctness. For example, the longest matched-control response has 602 tokens and score 1, while the paired candidate has 158 tokens and score 0; this score difference alone does not establish a medically better answer. Raw questions, outputs and references remain local. No clinical correction claim is made.

## Frozen decisions

A produced **66 first divergences, 62 same-answer cases, zero prefix-limit no-divergence cases and zero failures**. Both fixed canaries passed historical/native/off token equality and residual audit, and both exercised real delivered experts. All 128 historical prompt/transport comparisons passed. On every divergent case, full-prefix reconstruction and the incremental stream yielded exactly identical logits.

The 28 informative original gain/harm cases cover 28 distinct images. A uses the predeclared two-site adjustment, seed 0 and 4000 image-bootstrap repetitions:

| Boundary | Oriented restore-minus-roll effect | 97.5% interval |
|---|---:|---:|
| Layer 8 | 0.004977 | [−0.025751, 0.040287] |
| Layer 17 | 0.022618 | [−0.072478, 0.118828] |

Both intervals cross zero, so **A fails and automatically selects C**, without choosing a relay boundary. Descriptive gain/harm strata are retained in `A-results.json`: restoration tends to oppose original expert gains while helping original harms, especially at layer 17. These internal direction effects are not answer-accuracy improvements.

C's complete 128-image evidence passes coverage, intervention, output-change, harm-repair, point-estimate damage and mean-cost requirements. It fails all of the following:

- At least +0.5 percentage points over Baseline: actual −0.9766.
- At least +0.5 percentage points over compact: actual +0.1953.
- Positive improvement over matched control: actual −0.5859.
- At least 75% retention of original gains: actual 42.86%.

The frozen controller therefore writes **STOP_NO_CANDIDATE**. D/E are not missing work in this path. They also remain predeclared empty because the original data audit found no certified independent compatible cohorts; no exposed samples were relabeled as independent. The prior exposure/patient-independence limitations remain in the [original audit](../huatuo-decision-path-algorithm-chain-v1/data-audit.json).

## What C actually did

There were 96 active projection cases and 32 exact empty-evidence bypasses. Candidate tokens changed on 56/128 cases relative to compact and 49/128 relative to Baseline; output change is not the same as score improvement.

The central evidence plus two controls produced 3 distinct layouts on 50 cases, 2 on 46, and 1 on 32 bypass cases. No delivered packet was dropped or rewritten. Across 655 active projection steps, rank was 2 on 437 steps and 1 on 218. Mean removed/retained weighted residual norms were 0.29845/0.38742; maximum weighted orthogonality residual was 2.16e−14. The 32 bypass cases are the only cases with zero projection steps.

Candidate and matched control generated 784 and 1373 tokens respectively; maximum lengths were 158 and 602. All ended with EOS; no 1024-token limit hit occurred. Each active decoding step uses the same four-stream forward budget, though different output lengths make realized total costs unequal. There are zero failed candidate/control predictions and no excluded cases.

## Repair, authorization and immutable lineage

The original two attempts remain preserved: GPU0 resource-preflight failure, then GPU1's complete A run with one full-prefix/cached decision mismatch. The user explicitly authorized raising the per-node attempt ceiling from 2 to **3**, while retaining prior failures, forwards and scientific thresholds. Default policy remains 2; this amended plan alone sets 3.

Technical preflight reproduced the failing second-token decision. Projecting only the last hidden position did not fix it. Rebuilding the complete committed prefix using the native prefill followed by one-token cached forwards restored exact logit equality. The implementation now starts each diagnostic replay with a fresh private cache, reconstructs all prior tokens without intervention, and hooks only the final query. It retains the strict parity check, fixed layers, residual replacement, coordinate-roll definition, generation processors and offline orientation. Every replay forward is counted. The exact low-level source of BF16 execution-shape differences was not separately isolated.

The failed projection-only proposal was discarded. Real probes consumed 6 + 6 + 16 = 28 forwards, including failed comparisons, and these were inherited before the formal A attempt. A host input-read permission failure occurred before model loading, was corrected within this task's artifacts, and its log is retained.

`migrate_chain_repair.py` validates the predecessor and copies its artifacts, then creates a blocked successor retaining A=2 and **1046 + 28 = 1074 forwards**. Only the standard `retry` command grants A=3. The complete 128-case A was rerun under the newly frozen code; earlier partial diagnostics were not substituted into the new result.

- Branch: `experiments/huatuo-decision-path-algorithm-chain-v1`.
- Generation/repair source: `9a55426` (full SHA in `summary.json` and `provenance.json`).
- Requested implementation: `50668281a0ceadb70098660568655ce87b9f9cff`.
- Successor plan: `c5ce9086e8e2f68c5d7d241c14cec29b6266687ae6be191405a9613605bb6f13`.
- Predecessor plan: `3b7703020cbd480a68c40378fd728be64c01e32e77f392ed3b9e38a8cb0bc1f5`.
- HuatuoGPT-Vision-7B, Python 3.10.20, Torch 2.0.1+cu117, Transformers 4.37.2; unchanged existing weights, environment and shared offline cache.

## Device and costs

Host GPU1 had become occupied by another research job. The actual continuation used authorized **host GPU0**, UUID `GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023`, through the existing host connection. Its only pre-existing compute-query entry was a 34 MiB Nautilus C+G desktop context. A narrow resource-policy amendment explicitly lists that PID/name and requires driver XML to confirm C+G and at most 64 MiB; other compute processes remain rejected. No process was terminated, no dependency upgraded and no extra worker or multi-GPU sharding introduced.

| Phase | Time | Model forwards |
|---|---:|---:|
| Inherited attempts, total worker wall | 225.077 s | 1046 |
| Technical repair probes | 3.003 s, plus 28.484 s model loads | 28 |
| New A worker wall | 207.868 s | 1200 |
| A canary, included above | 3.010 s | 36 |
| A diagnostic, included above | 157.923 s | 1164 |
| C worker wall | 450.025 s | 7890 |
| C canary, included above | 3.041 s | 36 |
| C candidate decoding, included above | 161.904 s | 2749 |
| C control decoding, included above | 230.341 s | 5105 |
| A / C offline scoring | 0.087 / 0.102 s | 0 |
| **Cumulative total, including all inherited/probe costs** | Full-task wall not instrumented | **10164 / 200000** |

A/C model-load and runtime-provenance phases cost 14.913/13.336 s. Candidate mean decoding latency is 1.265 s/case; control mean is 1.800 s/case. The largest single control decode is about 72 s. A's diagnostic allocated-memory peak is 24.92 GiB; C's candidate/control peak is 23.09 GiB. Observed total device residency reached 38180 MiB, including allocator reservations and desktop context; this is distinct from allocated-tensor peaks. Preflight/canary peaks were not separately instrumented. The GPU returned to its original 428 MiB desktop use after exit.

Earlier data audit/initial hashing costs were 38.988/8.751 s. Some hash/controller and additional C view-preparation overhead is included only in worker wall. Persistent per-forward budget writes are experimental overhead. Cached Baseline latency is unknown, not zero. See `costs.json` for exact measured phases and limitations.

## Validation and artifacts

**128 CPU tests passed**, 0 failed/skipped, plus compilation, CLI help, shell syntax and diff checks. Each formal node passed both fixed native canaries. Runtime provenance was identical across A/C, and all source/model/generation/scorer pins were verified by the controller.

Local raw artifacts: `runs/chain-v1-repair/`; original failures remain in `runs/chain-v1/` and `runs/chain-v1-gpu1/`. The public package contains all 128 anonymized per-case records, decisions, A/C analyses, every gate, costs, failures, redacted plan, technical preflight, validation and content hashes. Raw answer/reference text, medical images, patient mappings and full expert payloads remain local. [Actual commands and inheritance procedure](commands.md).
