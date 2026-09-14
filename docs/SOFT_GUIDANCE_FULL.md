# Full VQA-RAD / SLAKE matched soft-guidance evaluation

Started 2026-09-14 following the user's explicit request to scale after the
pilot. This document records a RUNNING experiment, not completed scores.

## Resolve the comparison, do not relabel historical parity

The previous pilot and its exact-parity failure remain untouched. Historical
0053 differed at a near-tied token although prompt/evidence hashes matched.
This full run does NOT declare that failure a pass or change tie-breaking.
Instead it constructs **fresh, same-environment generalist and compact-text
controls**. These are the primary comparisons. Historical text/token drift is
reported separately on every row; old results are retained for provenance and
the pre-existing correction/harm cohorts.

Prompt SHA, evidence SHA, presented/omitted records and context accounting must
still match the historical decode trace where recorded. A mismatch stops the
run. Current-control alpha-zero token parity also remains strict. Runtime/OOM,
missing arms and malformed cache records are engineering failures, not abstention.

## Frozen scope

- Original complete manifests: VQA-RAD 451 and SLAKE 2094, **2545 total**.
  No question selection by old or new outcomes and no new dataset split.
- Alpha remains 0.5, equal-weight original predicted-mask geometry, original
  image and original answer-type prompt protocol. No strength search/training.
- Same cached expert outputs; no model/data download or expert rerun.
- Segmentation channel replacement only. Previously presented classification,
  text and other non-spatial evidence remain the same compact text. This is
  not native classification-logit guidance or a new Gate/Agent experiment.
- No presented usable segmentation: predeclared identity policy returns the
  current compact control for intervention arms. Such rows remain in full
  denominators, marked not applicable / guidance_applied=false. No zero-call
  success or simulated soft candidate is claimed.
- Scope and source-dependency limits of the spatial operator remain those of
  the pilot. Mask geometry is not a verified disease finding.

Seven stored arms:

1. historical_generalist (reused).
2. historical_compact (reused compact_rows).
3. generalist_matched (fresh current-environment image-only control).
4. compact_matched (fresh identical-expert text control; primary comparator).
5. other_text_only (remove segmentation text on applicable rows).
6. text_soft (same text-conditioned distribution, alpha .5).
7. native_spatial_soft (native spatial branch, alpha .5).

No full earlier output is overwritten. This necessary new matched-control cost
is explicit, rather than passing historical scores off as current-control scores.

## Full offline evaluation and the requested correction cases

Scoring is chained after both complete datasets finish and exact ID/arm checks
pass. The existing ANCHOR decoded-CLOSED scorer and OPEN token-recall code are
SHA-pinned in `scorer-frozen.json` before the long run. No references enter the
generator. No non-yes SLAKE answer is automatically mapped to no.

For each arm report full mixed score, CLOSED accuracy, OPEN recall, paired
improvement/harm against fresh compact and fresh generalist, and image-cluster
bootstrap. The mixed metric is not uniform clinical accuracy.

Special cohorts are determined OFFLINE from unchanged historical predictions:

- **Old corrections:** historical compact score > historical generalist score.
  Report retained versus lost old gains, and new deltas against the fresh
  compact control on exactly these cases.
- **Old harms:** historical compact score < historical generalist score.
  Report recovery relative to old compact and old generalist, plus deltas
  against the fresh compact control on these cases.
- **New corrections and harms:** all full-manifest score gains/losses against
  the fresh compact control, not only the known bad-case subset.

OPEN score changes are token-recall diagnostics, not automatically factual
corrections. Old-cohort comparisons cannot hide environment drift: both old
and fresh controls are retained and scored by the same pinned evaluator.

Cost reporting separates cached historical engine costs, current arm decoding,
zero-check overhead, all-arm wall time, startup/restart records, score calls,
applicable coverage and explicit shared-control reuse. Prefix replay is still
the verified but slow implementation; no unverified KV optimization was added.

## Execution and initial validation

Existing environment, authorized container GPU 0 / physical GPU 1 only:
`GPU-3846413a-4238-d307-b1f3-10c2dfbe002c`. No host GPU was started.
The initial 864-test CPU suite passed in 16.49 s. After adding two evaluator
guards, the full 866-test suite passed (exit 0;
`runs/soft-full-checks/tests-final.log`); Ruff and compilation checks pass.
Two scheduling-canary rows completed with all arms and current zero parity,
42.463 s and 36.266 s all-arm wall time. These rows are reused on continuation.
They are not full performance estimates.

Output identity:
`6ff365dc97261b29232acdab0a1060109bfd342c8dcb61d5621f7bfa07295440`.

Root:
`/home/dbw/merit-feddg-soft-guidance/runs/soft-guidance-full-v1/6ff365dc97261b29232acdab0a1060109bfd342c8dcb61d5621f7bfa07295440`.

```bash
cd /home/dbw/merit-feddg-soft-guidance
scripts/with_gate_kb_env.sh scripts/run_soft_guidance_full.py --check-only
scripts/with_gate_kb_env.sh scripts/run_soft_guidance_full.py --max-cases 2
scripts/with_gate_kb_env.sh scripts/evaluate_soft_guidance_full.py \
  --run runs/soft-guidance-full-v1/6ff365dc97261b29232acdab0a1060109bfd342c8dcb61d5621f7bfa07295440 \
  --freeze-scorer
```

The full continuation and subsequent scoring run in tmux session
`soft-guidance-full`, surviving VS Code/disconnection:

```bash
scripts/with_gate_kb_env.sh scripts/run_soft_guidance_full.py
# Run only after the command above succeeds:
scripts/with_gate_kb_env.sh scripts/evaluate_soft_guidance_full.py \
  --run runs/soft-guidance-full-v1/6ff365dc97261b29232acdab0a1060109bfd342c8dcb61d5621f7bfa07295440
```

Log: `runs/soft-full-checks/full.log`. Resume refuses a conflicting identity,
locks out duplicate workers, and reuses complete same-run rows. Completion
markers require all IDs, including unsupported/fallback cases. No imputation
of failed/missing rows. Initial preflight-only roots are not inference results.
Full results will appear in `evaluation.json` and a patient-text-free
`evaluation-summary.json`; neither file exists until full evaluation finishes.
Raw per-case records, masks, images and credentials are not pushed to GitHub.
