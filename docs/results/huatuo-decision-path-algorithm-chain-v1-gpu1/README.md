# GPU1 execution: complete A queue, technical stop

2026-09-20 UTC. **A processed all 128 planned cases and independently scored all 128 cached answer pairs. One diagnostic failed the full-prefix/cached decision parity check. The frozen controller stopped at BLOCKED_TECHNICAL; READY_FOR_SCALE was not reached.** This is not a scientific rejection or approval of B/C.

The user reported host GPU1 idle and authorized starting. Fresh device queries verified UUID `GPU-3846413a-4238-d307-b1f3-10c2dfbe002c`, PCI `00000000:05:00.0`, container index 0, 18 MiB used and no compute processes. The two-argument detached wrapper launched controller 3360098 and worker 3360541. Runtime provenance, model forwards, all case files, separate offline scoring and the final controller decision were verified. Both processes finished; GPU memory returned to 18 MiB.

## Frozen identity and inherited budget

- Branch: `experiments/huatuo-decision-path-algorithm-chain-v1`.
- Requested implementation: `50668281a0ceadb70098660568655ce87b9f9cff`.
- Unchanged generation source: `5d51ef2fddf9401ba12c498e7ba4b206745c05f3`.
- Successor plan: `3b7703020cbd480a68c40378fd728be64c01e32e77f392ed3b9e38a8cb0bc1f5`.
- Predecessor plan: `24b0c1e910978243905e6b750e176cd514b7cc39ca6fcdb549ab2528aea762a3`.
- Environment: Python 3.10.20, Torch 2.0.1+cu117, CUDA runtime 11.7, Transformers 4.37.2; existing HuatuoGPT-Vision-7B weights and shared offline cache. No dependency changes or downloads.

The original GPU0 resource failure remains intact in `runs/chain-v1` and its [public report](../huatuo-decision-path-algorithm-chain-v1/README.md). The new device-migration utility verifies this specific pre-model failure, pins and zero forwards, copies its immutable evidence, and creates a successor with the GPU UUID plus explicit lineage changed. The full scientific plan, model/code/scorer pins, generation, datasets, exposure registries, thresholds, attempt history and budget are inherited. The existing `retry` CLI then grants **A attempt 2**, rather than initializing fresh eligibility. No running inference source was edited. Migration utility content hash is recorded in the successor plan.

## Complete coverage and decision

| Node | Coverage | Disposition |
|---|---:|---|
| A attempt 1, GPU0 | 0/128 model cases | Resource guard rejected another compute context |
| A attempt 2, GPU1 | 128/128 processed; 127/128 valid diagnostics | 65 first divergences, 62 same-answer, 0 no-divergence-prefix-limit, 1 technical failure |
| Native canary | 2/2 | Historical/native/off tokens and residual audit match; both expert-exposed |
| Offline scoring | 128/128 | Separate CUDA-disabled process; pinned CLOSED and OPEN-first-reference metric |
| B / C | Not reached | No candidate or matched-control generation |
| D / E | Not reached; no cohorts declared | Independent data unavailable before freeze |

The error is `Full-prefix and cached decision parity failed`, raised inside `diagnose` before the affected case's two intervention sites. A complete-prefix recomputation changes at least one greedy decision relative to its private incremental KV stream. The stored error does not establish the numerical root cause; no kernel change or tolerance relaxation was made. Two native canaries do not prove every later full-prefix diagnostic will agree.

`localization()` checks diagnostic failures before scientific evidence. It therefore emits `technical_failure`, and the controller selects `BLOCKED_TECHNICAL`. Both A attempts are now consumed. No third attempt, budget reset, failed-case exclusion or manual C transition was performed. The failed diagnostic's effect is unavailable; its cached answer scores remain valid independently of that failure.

## Scores and descriptive direction evidence

The original cached answers were rescored together with the frozen scorer: generalist **58.6068%**, compact **57.4349%**, compact minus generalist **−1.1719 percentage points**. There are 14 original gains and 14 original harms. These are cached reference-arm scores, not new candidate performance. Candidate/control scores, repaired harms, retained gains and new candidate harms are **unavailable**, not zero.

For transparency, the complete diagnostic subset contains all 28 gain/harm cases. Descriptive effects below use the implemented score-oriented `(restore − roll)` probability-margin contrast, 4000 image-cluster bootstrap repetitions, seed 0 and 97.5% intervals. These are dimensionless diagnostic effects, not full-answer accuracy deltas. They do not override the technical stop or select a layer.

| Boundary | Overall effect [interval], n=28 | Original gains, n=14 | Original harms, n=14 |
|---|---|---|---|
| 8 | 0.00250 [−0.02603, 0.03319] | −0.03059 [−0.05657, −0.00774] | 0.03560 [−0.00952, 0.08605] |
| 17 | 0.02369 [−0.07143, 0.12007] | −0.12628 [−0.22068, −0.05183] | 0.17366 [0.05498, 0.29928] |

Both overall intervals cross zero. On the observed gain cases the restore direction tends to oppose the original expert benefit, while the harm cases favor it, especially at layer 17. This pattern is descriptive only; first-token differences and OPEN token recall are not verified medical corrections. Raw outputs stay local; public numeric records omit text, competing token identities and prefixes.

## Actual costs

| Component | Measured cost |
|---|---:|
| Prior failed resource attempt | 9.412 s; 0 forwards |
| GPU1 worker wall | 215.665 s |
| Model load including preflight/hash checks | 26.144 s |
| Context/image preparation across 128 cases | 19.964 s |
| Canary | 2.930 s; 36 forwards; 30 output tokens |
| A diagnostics | 152.203 s; 1010 forwards |
| Failed diagnostic, included above | 0.653 s; 6 forwards |
| Offline scoring process | 0.133 s |
| Total model forwards | **1046 / 200000** |
| Maximum instrumented diagnostic allocated GPU memory | **26,613,560,832 bytes (24.79 GiB)** |
| Candidate / matched-control decoding | Not reached |

The previous data audit cost 38.988 s, and initial runtime hashing 8.751 s. Repeated integrity checks and controller overhead are not all included in phase sums; canary peak memory was not separately instrumented. GPU memory above is the actual diagnostic allocator peak, not total device residency. Forward counts include failure, canary and shadow/control forwards, with persistent per-call budget overhead. Existing baseline/expert caches were reused; their deployment latency is unknown, not zero.

## Validation, independence and artifacts

Server checks: required chain tests 82 passed; device-migration tests 5 passed; combined chain/migration/pathway regression 120 passed in 4.14 s. Compile, wrapper syntax and diff checks passed. All 128 historical prompt/transport comparisons passed during execution. Actual MERIT and adapter imports match their pinned paths.

The original conservative exposure audit and its limitations remain unchanged: 2167 registered image fingerprints, 383 TEST-exclusion fingerprints, no certified independent compatible D/E cohorts, six unresolved historical image references and no usable patient mapping for these native development queues. The audit does not prove universal patient independence. [Original audit](../huatuo-decision-path-algorithm-chain-v1/data-audit.json) and provenance remain available. No official TEST inference/scoring, training, optimizer or backward pass occurred.

Local reproducibility: `runs/chain-v1-gpu1/plan.json`, `state.json`, `A-attempt-2/{runtime,predictions,scores,budget,decision}.json`, per-case files, durable inference/scoring logs and exit records. Original source/reference files and full expert payloads remain local. Public files contain summary, costs, failures, all 128 anonymized cases, descriptive localization, decisions, redacted plan, validation and provenance/artifact hashes. See [commands.md](commands.md) for actual invocation and lineage semantics. There is no running worker or queued retry.
