# VQA-RAD evidence-revision pilot — 2026-09-11

## Result boundary

This is a fixed 24-case **selection pilot** sampled from the official VQA-RAD
train split: 12 closed and 12 open questions, with 24 unique image hashes. The
official test split was not used. All 17 arms share the frozen `anchor-ce-v1`
generation contract. The primary metric is the sample-weighted mean of strict
closed-question correctness and open-answer reference-token recall.

Both physical-GPU shards completed 12/12 cases and merged under identity
`f5cebc62ae47dba5fbe77c1f4fbc988e9b226a49eb3f5d661fa78a1097780e12`.
There were no runtime errors, OOMs or empty answers.

## Main results

| Arm | Mixed | CE accuracy | OE recall | vs. baseline | improved / harmed | changed text | mean engine time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Generalist | 55.90% | 66.67% | 45.14% | — | — | 0 | 0.648 s |
| `semantic_all` | **65.28%** | 75.00% | **55.56%** | +9.38 pp | 3 / 0 | 9 | 2.371 s |
| `compact_rows` | **65.28%** | 75.00% | **55.56%** | +9.38 pp | 3 / 0 | 9 | 1.245 s |
| `compact_all` | 64.58% | 75.00% | 54.17% | +8.68 pp | 3 / 0 | 10 | **1.199 s** |
| `compact_guarded` | 64.24% | 75.00% | 53.47% | +8.33 pp | 2 / 0 | 4 | 2.246 s |
| `compact_geometry` | 64.24% | 75.00% | 53.47% | +8.33 pp | 2 / 0 | 9 | 1.227 s |
| `compact_verified` | 60.07% | 66.67% | 53.47% | +4.17 pp | 1 / 0 | 2 | 2.246 s |
| `compact_spatial` | 55.90% | 58.33% | 53.47% | 0.00 pp | 1 / 1 | 10 | 1.460 s |
| `compact_all` + predictive entropy | 64.58% | 75.00% | 54.17% | +8.68 pp | 3 / 0 | 7 | 10.369 s |
| `compact_all` + semantic entropy | 64.58% | 75.00% | 54.17% | +8.68 pp | 3 / 0 | 9 | 10.369 s |
| `compact_all` + discrete semantic entropy | 64.58% | 75.00% | 54.17% | +8.68 pp | 3 / 0 | 9 | 10.369 s |
| `semantic_all` + predictive entropy | 61.11% | 66.67% | 55.56% | +5.21 pp | 2 / 0 | 5 | 11.483 s |
| `semantic_all` + semantic entropy | 61.11% | 66.67% | 55.56% | +5.21 pp | 2 / 0 | 6 | 11.483 s |
| `semantic_all` + discrete semantic entropy | 61.11% | 66.67% | 55.56% | +5.21 pp | 2 / 0 | 6 | 11.483 s |
| `compact_geometry` + predictive entropy | 60.07% | 66.67% | 53.47% | +4.17 pp | 1 / 0 | 6 | 10.455 s |
| `compact_geometry` + semantic entropy | 64.24% | 75.00% | 53.47% | +8.33 pp | 2 / 0 | 6 | 10.455 s |
| `compact_geometry` + discrete semantic entropy | 64.24% | 75.00% | 53.47% | +8.33 pp | 2 / 0 | 7 | 10.455 s |

Engine time is recorded inside each output and benefits from sequential shared
expert caches. It is not an independent cold-start benchmark. The actual
dual-GPU shard wall times were approximately 4:58 and 6:17, including model
initialization.

## What improved

- `vqarad-train-pilot-0755`: the Generalist gave a nonspecific opacity answer.
  CheXagent returned `Calcified pleural plaques`; `compact_rows` changed the
  answer to calcified pleural plaques and OE recall rose from 0 to 1.
- `vqarad-train-pilot-0832`: CheXagent returned `Volume loss and infiltrate`.
  The answer became more specific and OE recall rose from 0.25 to 0.50.
- `vqarad-train-pilot-1587`: the Generalist incorrectly answered that the heart
  was enlarged. CheXagent returned `No`; the final leading answer changed from
  Yes to No and CE correctness rose from 0 to 1.

These are real interventions rather than no-call parity. CheXagent evidence was
presented in nine cases and BiomedParse evidence in eight `compact_rows` cases.

## Safety and Gate findings

The primary metric misses an important bad case. For
`vqarad-train-pilot-1568`, whose reference is `4th ventricle`, the baseline
incorrectly said `right lateral ventricle`. BiomedParse routed the brain image
to its `MRI-Cardiac` group and the evidence arms changed this to `left
ventricle`. Token recall remains 0.5 for both wrong answers, so the aggregate
table reports no harm. This case must be explicitly audited in full scale.

Predictive entropy over `compact_all` rejected that bad rewrite and retained all
three score improvements, but it is not yet a validated Gate:

- most useful changes occurred when uncertainty was unavailable and the policy
  defaulted to the candidate (`abstain`), rather than from a positive high-
  confidence decision;
- it costs 10.369 s/case, 8.33 times `compact_rows` in recorded engine time;
- the 24-case pilot is too small for a safety or efficacy claim.

`compact_spatial` is not selected: its net primary gain is zero and it introduces
one measured harm. The external verifier is conservative but loses useful
corrections. Semantic-entropy variants add substantial cost without improving
the primary metric over their ungated source arm.

## Selection decision

`compact_rows` is the full-scale candidate because it ties the best primary
score, is substantially cheaper than `semantic_all`, and makes nonzero evidence
interventions. This is a method-selection decision, not a claim that it is
better than the Generalist. Full-scale VQA-RAD must report the frozen CE/OE
metric, paired improvements and harms, the ventricle-type semantic failure, CE
format failures, expert routing and end-to-end latency.

The complete compact machine-readable record is [result.json](result.json).
The uncompressed 4.2 GB masks and traces remain outside Git at
`runs/vqarad-train-evidence-revision-pilot/uncertainty-matrix-gpu/f5cebc62ae47dba5fbe77c1f4fbc988e9b226a49eb3f5d661fa78a1097780e12`.
Their input, evaluation and shard-log hashes are recorded in `result.json`.

