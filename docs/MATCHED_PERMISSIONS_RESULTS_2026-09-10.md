# Matched permissions evaluation, 2026-09-10

Verdict: the complete matched run provides a positive signal for the
`permissions` arm under one content-aware diagnostic, but it **does not validate
the proposed finite-relation mechanism and does not support scale-up**. All
finite relations were removed by the real context budget before generation.

## Frozen protocol

- Code: `bc530bd876b8440a7085e17199258acdd9c0df7b`.
- Run identity:
  `776bee263c4597ad6e11760792f5356a43f84d5b21bd284e7a094be34ae82154`.
- Dataset: the complete 451-question official VQA-RAD test manifest; no custom
  split, calibration or policy fitting.
- Arms: `generalist`, point evidence, uncertainty evidence and permissions
  evidence. All used the same image, question, prompt suffix, unconstrained
  output grammar and 64-token answer budget.
- The four arms shared persistent native-expert outputs. References and answer
  types were unavailable during generation and used only for offline scoring.
- The detached `nohup`/`setsid` job ran from 11:20:10 to 11:42:30 UTC (1,340 s,
  22 min 20 s), completed 451/451 in every arm, then released the GPU. There was
  no traceback, OOM, empty answer or recorded tool runtime error.

## Scores

The frozen ANCHOR v9 metric requires an explicit leading `yes` or `no` for
closed questions. Since every arm intentionally used the same unconstrained
free-generation protocol, its strict result is reported but is dominated by
format compliance:

| Arm | Strict mixed | Strict closed accuracy | Parseable closed | Open answer recall |
| --- | ---: | ---: | ---: | ---: |
| Generalist | 0.2945 | 0.2749 | 119/251 | 0.3190 |
| Point | 0.2158 | 0.1275 | 54/251 | 0.3266 |
| Uncertainty | 0.2152 | 0.1275 | 54/251 | 0.3253 |
| Permissions | 0.2274 | 0.1394 | 57/251 | 0.3378 |

A target-blind content-aware parser, applied identically to all arms as a
diagnostic, gives:

| Arm | Mixed diagnostic | Closed diagnostic | Delta from generalist |
| --- | ---: | ---: | ---: |
| Generalist | 0.4320 | 0.5219 | -- |
| Point | 0.4375 | 0.5259 | +0.0056 |
| Uncertainty | 0.3948 | 0.4502 | -0.0371 |
| Permissions | 0.4580 | 0.5538 | +0.0261 |

For permissions versus generalist this diagnostic marked 38 rows improved, 25
harmed and 388 tied. Those are evaluator outcomes, not clinician-adjudicated
medical changes. Generic Token-F1 moved in the opposite direction: generalist
0.0752, point 0.0708, uncertainty 0.0612 and permissions 0.0682. No lexical
metric is promoted selectively as the success criterion.

All 273 cases with no compatible tool produced byte-identical answer text in all
four arms. This establishes the intended null parity: observed answer changes
were confined to cases where an evidence intervention was possible.

## Actual evidence transport

Every evidence arm attempted 352 calls on 178 cases. Actual final presentation
was:

| Expert | Point | Uncertainty | Permissions |
| --- | ---: | ---: | ---: |
| XRV findings | 141 | 141 | 0 |
| CheXagent description | 172 | 31 | 172 |
| XRV anatomy | 31 | 31 | 31 |
| Biomed anatomy | 6 | 6 | 6 |

Two of 33 anatomy packets exceeded the token budget in every arm. In the
uncertainty arm, 141 later CheXagent packets did not fit after XRV. In the
permissions arm, all 141 XRV findings packets did not fit, after which the
shorter CheXagent packet was presented. Unlike the previous run, rejected
packets were not marked adopted and every rejection is recorded as
`token_budget` in the tool's packing preview.

This distinction changes the scientific interpretation. The permissions audit
constructed eight preview relations for each of 141 XRV cases (1,128 total),
but none of those XRV records fit the final prompt. Across the 209 permission
records actually presented, the number of `supported_measurement_relations` was
exactly zero; all had basis `single_unverified_observation`. Therefore the
finite-relation innovation was not exercised. The +0.0261 content diagnostic is
primarily associated with presenting CheXagent instead of the larger XRV packet,
not with relation-aware reasoning.

The perturbation audit executed 522 additional XRV forwards once in the shared
cache, taking 16.75 s. The permissions arm reused them without rerunning the
expert. Stability remains observed photometric variation, not correctness or a
domain certificate.

## Reference-aligned examples

These are engineering inspection examples, not blinded clinical adjudication.

Clear corrections in the permissions arm include:

- `0015`: reference no consolidation; generalist said consolidation, permissions
  said no consolidation.
- `0056`: reference cardiomegaly present; generalist denied it, permissions said
  cardiomegaly.
- `0091`: reference no pneumomediastinum; generalist asserted it, permissions
  denied it.
- `0175`: reference no pneumothorax; generalist asserted it, permissions denied it.
- `0268`: reference no mediastinal shift; generalist asserted it, permissions
  denied it.

Clear adverse changes include:

- `0003`: correct generalist right heart-border side changed to left.
- `0146`: reference cardiac-contour narrowing present changed to absent.
- `0226` and `0227`: correct enlarged-heart/cardiomegaly answers changed to no
  abnormality or no cardiomegaly.
- `0258`: reference organ system `chest`; correct generalist answer changed to
  `heart` after Biomed anatomy evidence.
- `0300`: reference ECG leads present; correct generalist answer changed to absent.

CheXagent-presented cases had a descriptive content-diagnostic delta of +0.0742
(38 improved, 24 harmed, 110 tied). XRV anatomy was -0.0232 and Biomed anatomy
was -0.1667 on only six cases. These subsets overlap, differ in difficulty and
are not causal per-channel medical-effect estimates.

## Decision and raw outputs

Do not scale from this run. The next permissions test must make at least one
finite relation fit while keeping expert output and prompt fixed, and it must use
source-side development data or a predeclared representation change rather than
tuning on these test answers. A matched ablation should separate packet length,
expert selection and permission relations. No threshold should be relaxed from
this result.

Raw local files are under
`runs/matched-permissions/776bee263c4597ad6e11760792f5356a43f84d5b21bd284e7a094be34ae82154/`:

- `generalist.json`, `point.json`, `uncertainty.json`, `permissions.json`: all
  per-case answers, raw evidence, traces, packing previews and timings;
- `protocol.json` and `routing.json`: frozen protocol and image-only routes;
- `evaluation-summary.json`: fixed aggregate, paired and transport audit.

SHA-256 values are `28afaf6a...0430a` (generalist),
`76159290...f8c2` (point), `3c91b742...8008` (uncertainty),
`b7754980...e9ae` (permissions), `a607c1f2...7b33` (protocol) and
`61eee66f...01a` (evaluation summary).
