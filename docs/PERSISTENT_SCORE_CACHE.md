# Exact-prefix KV acceleration (2026-09-15)

The previous soft decoder invokes full production generation for each scoring
step, forcing every previously selected token again. For a T-token answer this
repeats multimodal prefill T times per branch and approximately T(T+1)/2 decoder
steps. Low allocated VRAM is not evidence of idle compute.

`PersistentScores` preserves each branch's own production multimodal prefill
and KV state. New selected tokens use the production input preparation and KV
update interfaces. This removes repeated vision/prompt computation within an
arm; it is not cross-case image caching or shared KV between different prompts.
The same selected prefix feeds both branches. A reset or diverging prefix
triggers a fresh prefill. Ordinary greedy `propose` remains unchanged.

Scope is semantic classification/generated-text evidence only. Tensor/spatial
interventions and unsupported generation processors fail explicitly. Model,
renderer, manifest, scoring, alpha=.5 and CAD strength=.5 remain unchanged.
Classification is still language-mediated; no native classifier-logit mapping
or new medical performance claim is made.

## Audit and continuation

`scripts/check_persistent_scores.py` uses the first available real case for each
dataset/channel cell on the existing full manifests. At every blend and CAD
step it compares the *whole vocabulary score vector* with production replay,
requiring exact equality, not just matching argmax. It also inherits both
zero-strength token checks. Failures stop the audit without replacing the old
full runner. This finite audit does not establish every future prefix's parity.

`scripts/run_class_text_cached.py` is a separate opt-in scheduling adapter. It
requires a successful four-cell audit and an admission file binding SHA256 of
the audit and scorer. The frozen original runner and old rows stay unchanged.
Every newly generated row records the execution backend and audit identity;
cache/implementation cost labels are corrected. Cached historical rows retain
their original costs. Analyses must distinguish the two cost regimes rather
than presenting their mixture as the optimized backend's average latency.

Local logs (contain no published dataset payloads):

- `runs/soft-full-checks/persistent-cpu-tests.log`
- `runs/soft-full-checks/persistent-full-pytest.log`
- `runs/soft-full-checks/persistent-audit-v1.log`
- `runs/soft-full-checks/persistent-audit-v1.json`

Only container GPU0 / physical host GPU1 is used. The first audit shares that
GPU with the pre-existing full run; its timing is not an isolated throughput
benchmark. No host GPU0 launch, dependency upgrade, model download or training.

## Validation results

- CPU suite: **876 passed** (`persistent-full-pytest-final.log`, 14.97 s).
  Final focused cache/CAD checks: **7 passed**. Compilation and diff whitespace
  checks passed; both audit and continuation help commands work.
- Shared-GPU real audit: all four cells passed, **338 complete score vectors
  exactly equal**, maximum tested prefix length 43. Both zero endpoints passed.
  These compare computational implementations, not medical quality.
- Scoring decoder forward work on those prefixes: **4506 replay forwards vs
  338 persistent forwards**; multimodal prefills **338 vs 16**. Ordinary control
  generation and inherited expert inference are excluded from these counts.
- Shared-GPU scoring timings were 343.82 s replay vs 21.47 s persistent. This
  approximately 16x ratio is exploratory, **not** isolated end-to-end speedup.
  A second, isolated four-cell audit is recorded separately before cutover.

The isolated audit also passed all **338 exact score comparisons**, with replay
**160.881452 s** versus persistent **10.316020 s**: **15.60x scoring-only speedup**.
Both zero-strength endpoints passed again. The full CPU suite was rerun after
the adapter changes: **876 passed in 13.72 s**. This does not measure end-to-end
case throughput, loading, ordinary controls or historical expert inference.
The original worker was stopped only after verifying its PID command and cwd;
completed atomic rows were preserved. `class-text-cached` continues the same root.

Admission identities:

- isolated audit SHA256: `0afe6266f9c6315d825085e6bed78ec084fcddaed3dd07f8811028a24d75205c`
- scorer SHA256: `fe3bc609a65e7d193b6a70d4b1a4457d31507e62d61fd1fb0988c2690da28ff7`

Logs: `runs/soft-full-checks/persistent-audit-isolated-v1.log`,
`persistent-audit-isolated-v1.json`, `persistent-full-pytest-cutover.log`,
and `class-text-cached.log` in that same directory. Raw case outputs stay local.

The audited cells were VQA-RAD classification `0067`, generation `0000`, SLAKE
classification `0037`, generation `0031`; these were fixed first-available
scheduling cases, not chosen by correctness. No answers or images are exported.

Continuation (after an explicit SHA-bound audit admission) uses the original
full output root, not a new sample or changed experiment identity:

```bash
scripts/with_gate_kb_env.sh scripts/run_class_text_cached.py \
  --cache-audit runs/soft-full-checks/persistent-audit-isolated-v1.json \
  --base-run runs/soft-guidance-full-v1/6ff365dc97261b29232acdab0a1060109bfd342c8dcb61d5621f7bfa07295440 \
  --output runs/class-text-guidance-v1
```
