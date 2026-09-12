# MERIT evidence-agent-v1: complete VQA-RAD TRAIN acceptance

The frozen 1,793-case experiment completed on 2026-09-12. Engineering execution
and exact replay passed; candidate production and Agent efficacy did not.
All seven arms have identical answer text and token IDs. Every extension has
zero usable candidates, so preservation is not a successful medical gate.

[Machine-readable audit](summary.json) contains metrics, costs, controls,
artifact hashes and two compact case traces. Full outputs and unchanged ANCHOR
reports remain in the local run directory; no images, masks, weights or bulk
patient records are included in this result bundle.

| Acceptance item | Observed result |
|---|---|
| Completed shards | Host GPU0: 897/897; container GPU / host GPU1: 896/896 |
| Merge | Explicit `--shard-count 2 --shard-index 0 --merge-only`; protocol `shards_complete=true` |
| Coverage/order | All seven arms match all 1,793 manifest IDs in order; 313 pixel-image clusters |
| Incumbent reuse | Entire incumbent records equal the frozen `compact_rows` output |
| Replay parity | 1,793/1,793 exact text AND token IDs, including 0049 |
| Runtime integrity | No parity snapshots, runtime/OOM log failures, or recorded action failures |
| Crop integrity | All 550 saved crop instances rehashed successfully (160 per static/control arm, 70 agent) |
| Engineering tests | `tests/test_evidence_agent.py`: 36 passed in frozen environment |
| Real segmentation | 80/1,793 cases (4.46%), across 50 images; only `cxr_anatomy` |
| Candidate production | 0/1,793 in each of the four extension arms |
| Commit policy | `audit_only`; 1,793 unknown / no_usable_candidate / keep incumbent; 0 accepted |

The prior huatuo parity-failure snapshot is preserved outside this identity and
excluded. No baseline inference, model download, dependency change, threshold
change, strategy change or new experiment was performed during this completion.

## Seven-arm metrics relative to incumbent

The requested `agent_evaluate` ran only after completed merge. Its diagnostic is
closed leading-yes/no EM and open token recall using the first reference. The
unchanged ANCHOR `evaluate_rows` report uses decoded strict matching, including
short-answer exact matching. These are distinct lexical metrics, not clinical
accuracy or hallucination rates; neither is silently relabeled as the baseline's
mixed CE/OE paper metric.

| Arm | Diagnostic % | Δ pp | Improved / harmed | Changed texts | ANCHOR decoded strict % |
|---|---:|---:|---:|---:|---:|
| incumbent | 44.01 | 0.00 | 0 / 0 | 0 | 31.62 |
| graph_noop | 44.01 | 0.00 | 0 / 0 | 0 | 31.62 |
| graph_static | 44.01 | 0.00 | 0 / 0 | 0 | 31.62 |
| region_control | 44.01 | 0.00 | 0 / 0 | 0 | 31.62 |
| full_image_control | 44.01 | 0.00 | 0 / 0 | 0 | 31.62 |
| agent_candidate | 44.01 | 0.00 | 0 / 0 | 0 | 31.62 |
| agent_committed | 44.01 | 0.00 | 0 / 0 | 0 | 31.62 |

Diagnostic CLOSED: 51.60% on 940 cases; OPEN recall: 35.65% on 853.
ANCHOR decoded strict: 567/1,793; binary 567/940 (60.32%), short-answer
0/853 exact matches. Its binary parser accepts more explanatory answers than
the leading-yes/no diagnostic. The imported report's `primary_metric` string
points to a field that its standalone CLI adds; this report uses the actual
`decoded_strict` fields returned by the unchanged API. Evaluator version:
`medheval-decoded-eval-v11-explanatory-binary-source-audited`.

Paired bootstrap resamples 313 pixel-image clusters with replacement, retaining
all questions in each sampled cluster and computing sample-weighted deltas:
2,000 repetitions, seed 0. Every arm-versus-incumbent 95% interval is [0.00, 0.00]
pp for both reported metrics. These degenerate intervals follow from identical
outputs and provide no evidence about the quality of usable Agent candidates.

For attribution only, offline rescoring of the existing generalist output gives
42.13% on the same diagnostic versus incumbent 44.01%: +1.88 pp, 94 improvements,
56 harms and 550 text changes, image-cluster CI [+0.56, +3.30] pp. This is the
legacy generalist→compact_rows effect, entirely preceding the Agent. No part of
that gain belongs to evidence-agent-v1.

## Region, observation and candidate coverage

| Arm | Region cases | Cases with local observations | Crop / inspect operations | Synthesis attempts | Usable candidates |
|---|---:|---:|---:|---:|---:|
| graph_static | 80 | 80 (4.46%) | 160 / 160 | 80 | 0 |
| region_control | 80 | 80 (4.46%) | 160 / 160 | 80 | 0 |
| full_image_control | 80 | 80 (4.46%) | 160 / 160 | 80 | 0 |
| agent_candidate | 80 | 8 (0.45%) | 70 / 8 | 8 | 0 |

Each region-bearing case admits two `cxr_anatomy` regions. The additional
`region_budget` audit occurs in 80 cases per arm; it is the frozen two-region
cap, not a malformed mask or an action failure. BiomedParse remains explicitly
disabled by the frozen protocol. No substitute whole-image regions were invented
for the 1,713 cases lacking segmentation. Full-image windows occur only as the
explicit matched control on genuine region-bearing cases.

Static and both control arms each record 1,713
`no_usable_regions_or_observations` and 80 `evidence_budget_or_generation`.
Agent records 1,785 and 8 respectively: the first count includes 1,713 cases
without regions and 72 with regions but no acquired observation. All 80
region-bearing agent cases end at `planner_stop`, leaving pending actions;
none exhausts the six-action execution budget. There are 158 successful planner
call records. No recorded unavailable, failed or invalidated action exists.
RAG is `no_source_manifest` on every workflow; there are no retrieval calls.

All 248 synthesis callback attempts report zero output tokens and no candidate.
The frozen callback may refuse additions when it cannot preserve every incumbent
and added evidence item in context, before model generation. The persisted
`evidence_budget_or_generation` reason combines these conditions and does not
record a more specific rejection stage. Consequently, this run cannot establish
that these were 248 successful generation calls, empty model responses, or a
particular context-overflow subtype. No budget or threshold was relaxed to
manufacture coverage. This is a candidate-delivery limitation requiring separate
future work; the current frozen result remains unchanged.

The committed arm lacks `agent_workflow`, so the stock evaluator's zero
region/call counts for that arm describe missing workflow metadata, not zero
work. It inherits the candidate arm's execution and costs. Its 1,793 preserved
answers are audit-only abstentions, not successful verification decisions.

## Did the controls actually differ?

| Versus graph_static | Distinct crop boxes | Distinct pixel digests | Changed observation strings | Cases with changed observations |
|---|---:|---:|---:|---:|
| region_control | 160/160 | 160/160 | 25/160 | 18/80 |
| full_image_control | 160/160 | 160/160 | 32/160 | 20/80 |

No control is flagged identical to its expert crop. Pixel changes cover all 80
region-bearing cases. Observation changes demonstrate sensitivity to the
window, but final answers are unchanged because no candidate is delivered.
These data cannot establish that the anatomical windows improve answer quality.
Local descriptions remain dependent on the original segmentation and the same
frozen generalist, not independent expert votes.

## Recorded cost

| Arm | New seconds total | New seconds / all 1,793 cases | Incumbent + new seconds / case | Successful planner / local-observation records | Zero-output synthesis attempts |
|---|---:|---:|---:|---:|---:|
| graph_static | 142.264 | 0.07934 | 0.69175 | 0 / 160 | 80 |
| region_control | 137.005 | 0.07641 | 0.68882 | 0 / 160 | 80 |
| full_image_control | 155.058 | 0.08648 | 0.69889 | 0 / 160 | 80 |
| agent_candidate | 59.373 | 0.03311 | 0.64552 | 158 / 8 | 8 |

Inherited incumbent cost is 1,098.047 s total, 0.61241 s/case, with original
recorded counters of 1,270 expert calls and 1,270 controller calls, and 38,505
answer tokens. These are inherited costs, not newly run experts. Counters retain
the original baseline's semantics.

The required graph_noop replay regenerated 1,793 answers / 38,505 tokens. Its
shared replay audit took 1,319.934 s total, 0.73616 s/case; it is paid once for
the experiment, not once per extension. The stock evaluator's `model_calls=0`
for graph_noop reflects the absent workflow call list, not zero generation.

Recorded local-observation output tokens: 3,215 static, 3,176 region control,
3,170 full-image control, 131 agent. Agent planner output: 552 tokens. Each
static/control arm records 14,500 observation input tokens; agent records
716 observation and 66,353 planner input tokens. Synthesis input usage is
unrecorded (null), not zero. Full per-role timings are in the JSON audit.

agent_committed reuses agent_candidate, adds 0.154 s total for commit bookkeeping,
and reports 0.64561 s/case including incumbent and candidate work. It invokes
no external verifier. Do not count the candidate work twice. All timings use
shared warm assets; they exclude model loading and are not independent
cold-start, end-to-end wall-clock or GPU-normalized benchmarks. Means over all
cases are diluted by the 95.54% without regions.

## Case audit and limits of improvement/harm examples

There are no improvement or harm cases relative to incumbent under either
metric, and no changed final answers; inventing representative Agent rescues or
harms would be incorrect. Two useful failure/acceptance examples are retained:

- **0049 — parity recovery, retained wrong answer.** Reference locations are
  bilateral frontal lobes and body of corpus callosum. The frozen incumbent says
  bilateral basal ganglia and left thalamus. Every arm retains that exact text
  and token sequence (diagnostic recall 0.25). No regions or new observations
  exist. This verifies the formerly problematic left/right replay boundary,
  while illustrating that preservation does not correct the baseline error.
- **0198 — executed region chain, retained laterality error.** Reference is
  `left`; incumbent says the diaphragm is more depressed on the `right`.
  Real 512×512 anatomy RLE masks yield two lung windows. All three static/control
  arms execute both crop→inspect chains, but their observations repeat `right`
  despite the instruction against inferring patient laterality from a crop.
  Each synthesis attempt returns no candidate / zero output tokens. The agent
  crops once, then stops before inspect. Final score remains 0. These are
  dependent, fallible observations and a retained pre-existing error, not a new
  measured Agent harm or evidence of reliable medical verification.

This TRAIN-only, zero-candidate run establishes neither generalization nor
medical hallucination reduction. Structural graph validity, parity, real crops
and truthful fail-closed persistence pass engineering acceptance. Nonzero
candidate production, useful revision and medical gate efficacy remain unproven.

## Reproduction and local artifacts

Implementation commit: `ff7f69330ba3aeefbc71aef19073369bb46a45af` on
`implementation/evidence-agent-v1`. Result identity:
`21d7284ea2d895ac254b811e2ec554cce21e9de6d970ecd6ac23c7be49db05d0`.
Environment: `/home/dbw/merit-feddg/.venv/bin/python`, Python 3.14.6,
torch 2.14.0+cu130, transformers 4.57.6. GPU1 UUID was confirmed as
`GPU-3846413a-4238-d307-b1f3-10c2dfbe002c`; host physical GPU0 ran the other shard.

Run from `/home/dbw/merit-feddg-agent-a82f43c`. These commands merge and score
existing outputs; neither runs baseline or Agent inference:

```bash
PY=/home/dbw/merit-feddg/.venv/bin/python
BASE=/home/dbw/merit-feddg/runs/vqarad-official-train-v3/transport-fixed/0379506d53573b3a4ba8eb1c1e3648604b22fdf781949f780b6b1091d3f8dbd6
MANIFEST=/home/dbw/merit-feddg/runs/vqarad-official-protocol-v3/data/train/manifest.jsonl
REFERENCES=/home/dbw/merit-feddg/runs/vqarad-official-protocol-v3/data/train/references.json
ARTIFACTS=/home/dbw/merit-feddg/artifacts
OUT=/home/dbw/merit-feddg-agent-a82f43c/runs/evidence-agent-v1-vqarad-train-ff7f693
ROOT=$OUT/21d7284ea2d895ac254b811e2ec554cce21e9de6d970ecd6ac23c7be49db05d0
"$PY" -m merit_feddg.agent_run --base-run "$BASE" --incumbent compact_rows \
  --manifest "$MANIFEST" --artifacts "$ARTIFACTS" --output "$OUT" \
  --shard-count 2 --shard-index 0 --merge-only
# Only after ROOT/protocol.json confirms shards_complete=true:
"$PY" -m merit_feddg.agent_evaluate --run "$ROOT" --manifest "$MANIFEST" \
  --references "$REFERENCES" --anchor-root /home/dbw/ANCHOR \
  --output "$ROOT/evaluation.json"
```

`summary.json` binds local protocol, model provenance, evaluation and seven arm
files by SHA-256. The complete per-case evaluation remains `$ROOT/evaluation.json`.
The original worktree's uncommitted user changes were not modified.
