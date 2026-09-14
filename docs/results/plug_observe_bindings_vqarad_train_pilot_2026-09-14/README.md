# Plug-observe bindings: VQA-RAD TRAIN fixed-prefix pilot

## Decision

**Do not expand this configuration to the full 1,793-case TRAIN run yet.** The
engineering path works and shared encoding increases `plug_agent` candidate
coverage, but neither layout beats the matched incumbent on the frozen ANCHOR
diagnostic. Shared `plug_static` and `plug_agent` each score 50/154 versus the
incumbent's 52/154. The simpler token-recall diagnostic is worse still for the
evidence arms. No threshold, prompt, router, expert or budget was changed after
seeing these results.

This is a scheduling-stop pilot on the first 154 rows of the unchanged complete
TRAIN manifest. It is not a randomized subset, a completed protocol or a
publishable full-dataset score. The remaining cases were not evaluated.

## Identity and protected state

- Evaluated implementation: `e6f88744900a11b516279980124dc4775a5ee07b`
  from `implementation/plug-observe-bindings` (PR #3).
- Report branch: `experiment/plug-observe-bindings-vqarad-train-20260914`.
- Independent worktree:
  `/home/dbw/merit-feddg-plug-observe-e6f8874`; the existing dirty
  `/home/dbw/merit-feddg` worktree was not switched, reset or cleaned.
- Base identity:
  `0379506d53573b3a4ba8eb1c1e3648604b22fdf781949f780b6b1091d3f8dbd6`.
- Plain identity:
  `0b298c6ce398d9e7ee16dcedf0b02a370f8400dbd2fa0a3724c46f440e412bd8`.
- Shared identity:
  `3dbe0bd4510bdae3aba7d8612f3ccfa920ce957c793d21ff38b277900c724a06`.
- Complete source manifest:
  `/home/dbw/merit-feddg/runs/vqarad-official-protocol-v3/data/train/manifest.jsonl`,
  1,793 unique IDs. Its IDs and order exactly match the complete base run.
- Incumbent: preselected `compact_rows`; no result-driven incumbent selection.
- Generalist: `microsoft/llava-med-v1.5-mistral-7b`, existing local checkpoint,
  float16 CUDA, existing LLaVA-Med source. Existing BiomedCLIP, XRV DenseNet,
  XRV PSPNet, optional CheXagent and CONCH registrations were reused.
- Source-case RAG was disabled because no reviewed source manifest was supplied.
  BiomedParse was explicitly disabled. No model, data or dependency was downloaded.
- The run remained training-free: no bridge/router/adapter/gate training, threshold
  fitting, temperature calibration or cross-case adaptation.

The only layout difference was `content_encoding`: absent/plain in
`configs/plug_observe.yaml`, and `shared` in
`configs/plug_observe_shared.yaml`. Both used `max_calls=3`, `region_limit=2`,
64 answer tokens, 48 observation tokens, 16 planner tokens and the region view.

## Environment and engineering acceptance

- Python 3.14.6, torch 2.14.0+cu130, existing project virtual environment.
- Container CUDA device 0 maps to physical UUID
  `GPU-3846413a-4238-d307-b1f3-10c2dfbe002c`, an RTX 4090 reported with
  49,140 MiB. This was the only authorized visible GPU.
- Real execution used about 17.8 GiB and sustained approximately 92--100% GPU
  utilization after the unrelated evaluator stopped. No OOM or CUDA failure occurred.
- Focused suite: 124 passed:
  `tests/test_plug_bindings.py`, `tests/test_plug_observe.py`, and
  `tests/test_evidence_agent.py`.
- Full local pytest: 846 passed. CLI help and a CUDA tensor smoke test passed.
- GitHub's plug-observe CPU contract and regression jobs passed for `e6f8874`.
  The generic no-torch test job and inherited whole-repository ruff job remain
  failed as documented; this report does not call all GitHub jobs green.

Both preflights declared all 1,793 cases and the four arms
`incumbent/plug_read/plug_static/plug_agent`. Two-case real canaries produced
valid candidates. A separate answer-blind spatial engineering canary used the
earliest manifest row with cached native segmentation (`...-0177`): two real,
different lung crops retained original-image coordinates and parent references.
Shared encoding admitted both static local observations and enabled a dynamic
region candidate. Final synthesis still used the original image. This verifies
the mechanism, not medical correctness.

## Fixed 154-case result

Plain had already atomically completed rows 0--153 when the request changed to
small-sample confirmation. It was interrupted cleanly at that boundary and its
154 caches were retained. Shared then ran with `--canary-cases 154` against the
same full manifest. Both layouts contain exactly the same 154 IDs. Incumbent
text and token IDs match across layouts for all 154 cases.

### Repository diagnostic

This is the repository's **diagnostic** closed exact-match/open token-recall
against the first reference. It is not clinical accuracy.

| Layout / arm | Score | Delta vs incumbent | Improved | Harmed | Changed | Candidate coverage |
|---|---:|---:|---:|---:|---:|---:|
| incumbent | 46.83% | 0.00 pp | 0 | 0 | 0 | 0/154 |
| plain `plug_read` | 49.42% | +2.60 pp | 4 | 0 | 26 | 154/154 |
| plain `plug_static` | 40.90% | -5.93 pp | 6 | 15 | 109 | 148/154 |
| plain `plug_agent` | 43.50% | -3.33 pp | 6 | 11 | 99 | 136/154 |
| shared `plug_read` | 48.77% | +1.95 pp | 3 | 0 | 25 | 154/154 |
| shared `plug_static` | 39.82% | -7.01 pp | 5 | 16 | 111 | 148/154 |
| shared `plug_agent` | 39.82% | -7.01 pp | 5 | 16 | 111 | 148/154 |

Image-clustered exploratory 95% paired bootstrap intervals for the diagnostic
delta were plain read `[0.0000, 0.0658]`, static `[-0.1301, 0.0049]`, agent
`[-0.0863, 0.0187]`; shared read `[0.0000, 0.0511]`, static/agent
`[-0.1407, -0.0066]`. These intervals describe this non-random prefix only.

### Frozen ANCHOR v11 diagnostic

The existing ANCHOR decoder
`medheval-decoded-eval-v11-explanatory-binary-source-audited` was called only
after inference. References never entered generation. Its strict normalized
short-answer behavior gives zero credit to explanatory expansions in all 86
open rows here, so it is retained for protocol comparability rather than called
clinical accuracy.

| Layout / arm | Correct / 154 | Strict | Delta | Improved | Harmed |
|---|---:|---:|---:|---:|---:|
| incumbent | 52 | 33.77% | 0.00 pp | 0 | 0 |
| plain `plug_read` | 50 | 32.47% | -1.30 pp | 2 | 4 |
| plain `plug_static` | 46 | 29.87% | -3.90 pp | 5 | 11 |
| plain `plug_agent` | 49 | 31.82% | -1.95 pp | 5 | 8 |
| shared `plug_read` | 50 | 32.47% | -1.30 pp | 2 | 4 |
| shared `plug_static` | 50 | 32.47% | -1.30 pp | 6 | 8 |
| shared `plug_agent` | 50 | 32.47% | -1.30 pp | 6 | 8 |

Thus shared layout recovers four strict cases relative to plain static and one
relative to plain agent, but still has two net harms versus incumbent. Its
static and agent texts are identical in 154/154 cases; the dynamic controller
does not add answer value in this prefix.

## Delivery and cost audit

| Layout / arm | Candidate | Presented observation refs | Generative calls | Expert/region actions | Mean delivery input tokens | Mean new seconds |
|---|---:|---:|---:|---:|---:|---:|
| plain read | 154 | 39 | 154 | 0 | 689.8 | 1.179 |
| plain static | 148 | 180 | 149 | 215 | 1496.4 | 1.505 |
| plain agent | 136 | 168 | 350 | 169 | 1429.6 | 1.708 |
| shared read | 154 | 39 | 154 | 0 | 697.7 | 0.724 |
| shared static | 148 | 180 | 149 | 215 | 1522.3 | 0.925 |
| shared agent | 148 | 180 | 362 | 181 | 1522.3 | 1.091 |

`plug_static` omitted 66 observations for token budget in each layout, plus one
child because its parent was not presented. `plug_agent` recorded 33 token-budget
omissions in each layout. Shared framing increased, rather than reduced, mean
answer input tokens on this prefix. Runtime across layouts is not a controlled
speed comparison: most of plain overlapped an unrelated evaluator, whereas
shared ran alone with warm assets. Within each run the reported `new_seconds`
still shows the controller overhead. The inherited incumbent expert-time upper
bound is unchanged at mean 0.615 s/case (94.731 s summed).

## Mechanism findings and representative audits

- The path is real: packets were presented, expert actions ran, output text
  changed, and real region crops produced candidates. Fallbacks are excluded
  from candidate counts.
- Coverage is not correctness. Shared agent coverage rises from 88.31% to 96.10%,
  while its repository diagnostic falls and ANCHOR remains below incumbent.
- `biomed_anatomy` dominates new actions (plain agent 136; shared agent 148).
  It reports relative matches from a non-exhaustive major-organ catalog, not
  disease probabilities. The model nevertheless changes answers about infarction,
  enhancement, inflammation, signal intensity and other attributes outside that
  evidence's decision scope.
- Example improvement `...-0000`: incumbent negated brain infarction; both
  evidence arms changed to an affirmative answer matching the reference after
  a `biomed_anatomy` observation. The observation only established brain-like
  anatomy, not infarction, so this is a scored improvement but weak causal evidence.
- Example harm `...-0016`: a correct affirmative gyral-enhancement answer became
  a negative answer after a brain/anatomy catalog observation. The evidence was
  irrelevant to enhancement and should not have controlled the polarity.
- CXR harms `...-0010/0011` changed correct negative pulmonary answers into
  positive findings after an explicitly unverified inherited CXR observation;
  extra anatomy/classifier calls did not make that assertion reliable.
- No trained or calibrated clinical gate exists. `plug_agent` is action selection,
  not a correctness guarantee; the full-retention interpretation must not be used.

The immediate failure is therefore not packet reachability. It is a missing
mechanism for limiting the semantic authority of an observation to the question
attribute it can actually support, combined with an unreliable inherited CXR
description. Because changing scope rules after reading this TRAIN prefix would
be result-driven policy tuning, no such change was made in this run.

## Commands and retained artifacts

Preflight/canary/full commands used the following common arguments (plain and
shared differed only in config/output):

```bash
PYTHONPATH=/home/dbw/merit-feddg-plug-observe-e6f8874 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/dbw/merit-feddg/.venv/bin/python -m merit_feddg.plug_run \
  --base-run /home/dbw/merit-feddg/runs/vqarad-official-train-v3/transport-fixed/0379506d53573b3a4ba8eb1c1e3648604b22fdf781949f780b6b1091d3f8dbd6 \
  --incumbent compact_rows \
  --manifest /home/dbw/merit-feddg/runs/vqarad-official-protocol-v3/data/train/manifest.jsonl \
  --artifacts /home/dbw/merit-feddg/artifacts \
  --config configs/plug_observe_shared.yaml \
  --output runs/plug-observe-vqarad-train-shared-e6f8874 \
  --canary-cases 154
```

The server retains partial atomic caches and logs under
`runs/plug-observe-vqarad-train-{plain,shared}-e6f8874/`, plus spatial engineering
caches under `runs/plug-observe-spatial-engineering-{plain,shared}-e6f8874/`.
They intentionally have no complete `protocol.json`, no `shards_complete=true`
and no full `evaluation.json`. `agent_evaluate` was therefore not misused to
label them complete; the same frozen scoring functions were invoked read-only
for this explicit pilot. Raw images, weights, caches, logs and credentials are
not included in this Git commit.

## Next valid step

Do not resume simply to obtain a larger negative result. First freeze a
mechanism-level hypothesis without using test labels: for example, capability
contracts that prevent a major-anatomy observation from adjudicating disease
polarity, and an explicit unavailable state for unverified inherited prose.
Validate that contract on source/TRAIN development data and an answer-blind
mechanism suite, then run one prespecified holdout. If the present configuration
must still be characterized at full scale, resume the existing caches unchanged
and report it as a negative full TRAIN experiment, not as method selection.
