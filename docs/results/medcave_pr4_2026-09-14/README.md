# PR #4 execution report — 2026-09-14

## Outcome

The free-answer baseline → native expert → real candidate → audited fallback
path was executed on the assigned GPU. ACCEPT and two-source accumulated-state
control have CPU integration coverage. **No medical efficacy or validated source
risk certificate is claimed.** The real adapter lacks required semantic,
qualification and OOD signals, so acceptance remains disabled.

The work continues PR #4 on `medcave-risk-agent`, starting from
`df7edce006f60702a58e69f1ead806f21a06dd72`, in the independent worktree
`/home/dbw/merit-feddg-medcave-risk`. Original worktree edits were untouched.
Server compatibility was committed separately as `c1b0e9c`.

## Verification

| Actual command/check | Result |
|---|---|
| Initial `python -m pytest tests/test_intervention_risk.py -q` | exit 0; 6 passed |
| Final `python -m pytest -ra` | exit 0; 740 passed, 0 failed, 0 skipped; 11.03 s |
| `python -m ruff check .` | exit 0; all checks passed |
| `git diff --check` | exit 0 |
| `bash -n run_open.sh run_capabilities.sh run_llava_med.sh bootstrap.sh` | exit 0 |
| `python -m merit_feddg.medcave_run --help` | exit 0; all six implemented stages listed |
| Real dry-run | exit 0; 1,793 input identities checked; ACCEPT disabled |
| Real two-case source-smoke | completed; valid baseline/candidate tokens and exact fallbacks |
| Exact own-cache replay | 2 available; remaining 1,791 explicitly unavailable |

There is no configured type-checker. CI initially failed on the PR. The ordinary
test job did not install torch; the workflow now installs CPU torch on GitHub
only. No package was installed/upgraded in the shared server environment.

An inherited test expected the loader to omit the now-normalized `task` field.
The regression now checks every input row plus the explicit `open_vqa` default;
no row/label assertions were removed. Fixture scoping, imports and one executable
mode were repaired for full regression/lint. Legacy methods and configs are kept.

The initial smoke failed at the old LLaVA `use_flash_attention_2` kwarg; the next
reached real CheXagent and failed at its legacy DynamicCache API. Both logs remain
local. Pinned-source compatibility was reused and scoped cache restoration has
regression tests. The final smoke completed successfully.

## Actual source smoke

Selection: first two rows of the unchanged complete VQA-RAD TRAIN manifest,
without consulting reference answers. `--limit 2` is a source-smoke scheduling
stop. It does not create a new test/calibration split or a complete experiment.
These TRAIN rows had appeared in development and are not claimed as unseen data.

- Python: `/home/dbw/merit-feddg/.venv/bin/python`, Python 3.14.6,
  torch 2.14.0+cu130; current GPU CUDA smoke succeeded under timeout.
- GPU: container index 0, UUID `GPU-3846413a-4238-d307-b1f3-10c2dfbe002c`,
  the assigned RTX 4090. No host permission, other GPU or unrelated process changed.
- Generalist: existing `microsoft/llava-med-v1.5-mistral-7b` checkpoint and pinned
  CLIP/LLaVA source under `/home/dbw/ANCHOR`.
- Specialist: existing local `StanfordAIMI/CheXagent-2-3b`, frozen bfloat16,
  maximum 64 generated tokens. No download. RAG absent.
- Config: `configs/medcave_source.yaml`; deterministic neutral prompt, 64 answer
  tokens, compact semantic packets, original image, two requests/two candidates,
  zero visual probes. Scheduling priority 1.0 is not a correctness probability.
- Binding: `b99d0717588d1466bc26e61bf40148b5f20a9f5dc7daac18f395eb91a50a1e91`.
- Run identity: `4d29695de9205095e3f4007965e8af1148331bd6936007cc38f98360b8b4c470`.
- Root: `runs/medcave-source-smoke/4d29695de9205095e3f4007965e8af1148331bd6936007cc38f98360b8b4c470/`.

| Measure | Actual result |
|---|---:|
| Scheduled / full manifest | 2 / 1,793 |
| Baseline calls | 2 |
| Image-type routing calls | 2 |
| Logical / actual expert calls | 1 / 1 |
| Native cache hits | 0 |
| Candidate generations | 1 |
| Candidate coverage | 1/2 |
| Candidate generated tokens | 21 |
| Visual probe calls | 0 |
| Accepted / coverage | 0 / 0% |
| Exact baseline text/token fallback | 2/2 |
| Harmful-accept rate | undefined (null; zero denominator) |
| Harm-and-accept frequency | 0/2 descriptive only |
| Closed normalized exact, baseline/final | 0/2 and 0/2 |
| Rescue / harm / net score change | 0 / 0 / 0 |
| Mean case-loop wall time | 6.255 s |
| Unknown verification fields | 7/10, 70% |

Case-loop latency excludes initial generalist loading and modality routing;
the specialist's first lazy load is included. It is not a steady-state benchmark.
Full explanatory answers scoring zero under normalized exact should not be
interpreted as two independently adjudicated clinical errors. Neither these
metrics nor the smoke are comparable to the old answer-type-conditioned paper
baseline. All final metrics are unchanged because both outputs fell back.

One MRI case had no applicable configured specialist. The CXR case actually
called CheXagent, retained its native result, delivered one observation and
generated a 21-token candidate. Its seven unknown fields were coverage,
source_reliability, pre_ood, post_ood, conflict, instability and specificity.
The reason was `missing_signals_or_risk_budget`. The other fallback reason was
`no_qualified_utility`. Neither is called gate success.

## Expanded real-data smoke: 32 TRAIN rows

After the two-case engineering check, a new background tmux run completed the
first 32 rows in the same original order, without answer-based selection. Same
config, binding and identity as above; new parent `runs/medcave-source32/`.
All 32 trajectories have `complete=true`. Full-manifest `shards_complete=false`
correctly remains set: this is 32/1,793, not a full experiment.

| Measure | Actual result |
|---|---:|
| Baseline / routing calls | 32 / 32 |
| Logical / actual specialist calls | 8 / 8 |
| Candidates / scheduled cases | 8/32 (25%) |
| Native evidence presented, no omission | 8/8 candidates |
| Candidate text and token changes | 7/8 candidates |
| Candidate generated tokens | 169 |
| Cache hits / visual probes | 0 / 0 |
| Candidate score improvements / declines | 0 / 0 |
| ACCEPT / exact baseline fallback | 0 / 32 |
| Closed normalized exact (14 cases), baseline/final | 0% / 0% |
| Open token F1 (18 cases), baseline/final | 6.901% / 6.901% |
| Conditional harmful acceptance | null (no accepts) |
| Case-loop total / mean | 45.130 s / 1.410 s |

Eight candidates comprise five CLOSED and three OPEN questions. Their individual
offline scores equal the corresponding baseline scores, despite seven changed
answers; this is neither proof of clinical equivalence nor candidate benefit.
The other 24 cases have no candidate and must not be counted as candidate tests.
Evidence delivery logs prove presentation, not causal use or medical usefulness.
All eight candidate trajectories lack seven of ten verification fields; 24 other
cases record `no_qualified_utility`. No runtime failures were recorded.

Timing includes baseline generation and lazy specialist loading within each
case loop, but excludes initial generalist loading and 32 modality-routing calls.
It is **not end-to-end latency or isolated additional agent overhead**. No
paper-baseline comparison is made: this smoke uses a neutral free-answer prompt
and the stated diagnostic scores, not the historical answer-type-conditioned
generation and frozen ANCHOR final scoring contract.

Reproduction (existing environment and authorized container GPU):

```bash
cd /home/dbw/merit-feddg-medcave-risk
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=. TOKENIZERS_PARALLELISM=false \
/home/dbw/merit-feddg/.venv/bin/python -m merit_feddg.medcave_run \
  --stage source-smoke --config configs/medcave_source.yaml \
  --manifest /home/dbw/merit-feddg/runs/vqarad-official-protocol-v3/data/train/manifest.jsonl \
  --artifacts /home/dbw/merit-feddg/artifacts \
  --output runs/medcave-source32 --limit 32 \
  --references /home/dbw/merit-feddg/runs/vqarad-official-protocol-v3/data/train/references.json
```

Existing exact-identity cases are reused on rerun. Use a new output parent for
a different smoke limit; finalized partial protocol files are not overwritten.
No raw patient images or answers are included in this public report.

## Remaining blocked work and limits

1. No independently audited, previously unexposed source-dev/source-cal pair
   with patient/study isolation was established. Existing TRAIN/image metadata
   alone cannot validate patient independence. No official test was repartitioned.
2. CheXagent prose lacks a reliable candidate-specific semantic bridge; missing
   coverage is kept unknown. A localized organ is not proof of normality.
3. No matching source QualificationArtifact/pre/post feature references were
   available for the configured real expert. Raw native confidence is not a
   calibrated correctness probability.
4. Candidate-bound visual consistency/sensitivity were not measured. The zero
   probe budget is explicit. No saliency or local-removal gain is called truth.
5. The real smoke used one applicable expert. Two distinct-model acquisitions,
   cumulative evidence and accepted joint candidates were tested with a CPU fake
   backend only; no real multi-expert clinical benefit is asserted.
6. Source-cal/evaluate stages are executable and covered by CPU integration
   tests, but no real source certificate or full target result was produced.

The statistical implementation performs fixed-policy, one-unit-per-group exact
binomial validation after generation, not threshold-search Wilson certification.
Insufficient accepted groups disable ACCEPT. Local certificate digests detect
accidental changes; trusted files/plugins are not an adversarial security model.

Commands and input requirements for dry-run, source-dev, source-cal, replay,
source-smoke and full evaluation are in
[`docs/MEDCAVE_RISK_AGENT.md`](../../MEDCAVE_RISK_AGENT.md).
Raw images, case outputs, logs, cache contents and model weights remain on the
server and were not staged for GitHub.
