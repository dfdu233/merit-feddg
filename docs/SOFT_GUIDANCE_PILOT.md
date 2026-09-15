# Segmentation soft-guidance probe: partial, not ready for full testing

Date: 2026-09-14. Independent branch `implementation/capability-soft-guidance-v1`,
based on `93c9c72b6e8a7b360e8d16bf1141265bcbfafc5f`. Original worktrees, expert
weights, semantic_all/compact_rows/compact_all and historical results preserved.

## Frozen intervention

The requested first stage changes only the segmentation interface, not the
experts, planner, Gate, classifier mappings or scoring thresholds. No training
or shared dependency upgrade. Classification-native guidance is NOT implemented
in this probe; classification and other non-spatial evidence remain text.

- Reuse the exact cached expert outputs from the complete VQA-RAD (451) and
  SLAKE (2094) test runs. No expert is rerun and no reference mask is used.
- Preserve the historical generalist, original image, answer-type prompt contract,
  decoder settings and 64-token limit. No new unified prompt is substituted.
- Recover which complete records the compact baseline actually presented. Keep
  only those non-segmentation records in the shared base context; do not admit
  newly available text just because removing masks freed the token budget.
- `compact_rows_replay`: exact same cached native values, compact text renderer.
- `other_text_only`: remove segmentation text, retain the other presented tools.
- `text_soft`: same full compact context supplies the conditional distribution.
- `native_spatial_soft`: segmentation text is replaced by predicted masks acting
  through the existing, parameter-free SpatialEvidenceBridge. It maps masks into
  the deterministic square-padded patch grid and mixes regional visual features.
  This is NOT a new attention-bias implementation or a reproduction of ARCD.
- Both soft arms use the single predeclared alpha=0.5:
  `score = (1-alpha)*base_logits + alpha*conditional_logits`, equivalent up to a
  token-independent constant to log-probability interpolation. No fitted trust,
  correctness confidence, class-logit/token-logit addition or disease blacklist.
- Both branches follow exactly the same selected token prefix. This initial
  implementation uses production prefix replay, not optimized paired KV caches.
  It is expensive; it cannot currently support a practical speedup claim.

Full compact text and its native mask contain the same underlying prediction
but not identical representations. Any gain could reflect geometry preservation
or removal of misleading label text. The deletion-only control is essential.
The old valid compact method remains the incumbent, not silently replaced.

## Engineering results and stop

CPU: 863 tests passed in 17.95 s; Ruff and diff checks passed. New tests cover
zero strength, endpoints, score-space checks, and identical branch prefixes.
Authorized container GPU 0 is physical GPU 1,
`GPU-3846413a-4238-d307-b1f3-10c2dfbe002c`. Existing environment and LLaVA-Med used.

First probe: first two cached-segmentation rows in each full manifest, without
consulting scores or labels: VQA-RAD 0000/0001 and SLAKE 0004/0005. All four
passed exact historical compact text/token parity and alpha-zero parity. Native
patch intervention was nonzero and real candidates were generated.

Historical-harm follow-up was selected only from the old paired score audit,
in manifest order. It was a diagnostic failure-set test, NOT a strategy-selection
set. Planned VQA-RAD 0035/0053 and SLAKE 0015, reusing already-completed SLAKE
0004 instead of rerunning it. Alpha and method were unchanged.

VQA-RAD 0035 passed. **VQA-RAD 0053 failed exact compact parity before soft
decoding.** The process stopped; no soft result exists for 0053 and SLAKE 0015
was not executed. Therefore 5 cases have valid soft outputs, 1 failed acceptance,
and 1 planned additional case is unexecuted. No full new score is published.

For 0053 the prompt SHA, evidence SHA, visible records and context counts match
the historical decode trace. The old answer ends at “chest X-ray”; the replay
adds “image” before the period. A fresh-process diagnostic reproduced the
difference at token index 17. Current production-prefix scores for token 3469
(image) and 28723 (period) both equal 16.75. This narrows the issue to a
numerically near-tied decoding fork, but does not establish the historical
numerical cause. No tie rule was changed and semantic equivalence was not used
to waive token parity. Original dirty-server compatibility edits were inspected,
not overwritten or blindly copied.

## Scoring: common frozen evaluator, not legacy numbers mixed together

Offline scorer: existing ANCHOR
`medheval-decoded-eval-v11-explanatory-binary-source-audited` for CLOSED answers,
plus unchanged ANCHOR answer-token-recall for OPEN answers. References are
loaded only by separate offline programs. Non-yes SLAKE references are NOT
silently converted to no. This mixed score is not uniform clinical accuracy.
The older legacy leading-yes/no “Strict” table is not directly interchangeable.

Both complete historical baselines were rescored with this same current metric:

| Dataset | N | Generalist | compact_rows | Paired improvements / harms |
|---|---:|---:|---:|---:|
| VQA-RAD | 451 | 48.47% | 53.96% | 49 / 20 |
| SLAKE | 2094 | 34.68% | 39.49% | 199 / 95 |

These are OLD predictions, not new full soft-guidance results.

Initial four-case probe only:

| Dataset / N | Generalist | compact text | Remove segmentation text | Text soft | Native spatial soft |
|---|---:|---:|---:|---:|---:|
| VQA-RAD / 2 | 50% | 50% | 50% | 50% | 50% |
| SLAKE / 2 | 100% | 50% | 100% | 50% | 100% |

Mean new decode time: VQA-RAD text-soft 18.93 s/native-soft 17.29 s, versus
other-text-only 0.594 s. SLAKE text-soft 8.70 s/native-soft 10.37 s, versus
other-text-only 0.543 s. Cached historical times are not zero-cost inference;
they were reused, not remeasured. Shared warm models and differing output lengths
also affect these small-sample timings. Prefix replay is currently too costly
for unqualified full-scale deployment.

## Two verified historical Bad cases

| Case | Question topic | Generalist | Compact text | Text soft | Native spatial soft | Remove segmentation text |
|---|---|---|---|---|---|---|
| SLAKE 0004 | Liver visibility | Correct | Wrong | Wrong | Correct | Correct |
| VQA-RAD 0035 | Small-bowel thickening | Correct | Wrong | Wrong | Correct | Correct |

For SLAKE 0004, BiomedParse provided eight region labels including liver. Only
segmentation evidence was in the presented context. The native branch changed
projected patch features (max absolute delta 0.2265625), but preserved the
generalist's correct absence answer; the text-soft branch kept the wrong
presence answer. This does not show that an organ name is a verified finding.

Both historical Bad cases were recovered by native guidance, but ALSO by simply
removing segmentation text. Thus the evidence supports possible avoidance of
text-induced interference, NOT an additional benefit from reading native masks.
The two cases were explicitly enriched for historical harm; their recovery rate
cannot estimate population rescue or harm. The preliminary image-cluster
bootstrap in the local probe evaluation is tiny-sample diagnostic only.

## Artifacts and commands

[Machine-readable sanitized summary](results/soft_guidance_2026-09-14/summary.json)
contains aggregate scores, costs, tiny-sample intervals and explicit failure/
unexecuted counts, without original images, masks or patient-level answer text.

Local raw results (not uploaded with patient-level outputs):

- `runs/soft-guidance-pilot-v1/`: four real outputs, frozen options, selections,
  original/replayed answers, per-token choices, transport and spatial audit.
- `runs/soft-guidance-harm-v1/`: one valid soft output and the parity failure;
  frozen planned IDs. No imputation for missing cases.
- `runs/soft-guidance-history-audit/`: aligned full historical scores and harm IDs.
- `runs/soft-guidance-checks/`: CPU/logs, four-case offline evaluation, fresh-process
  parity diagnostic. This report intentionally omits patient images and raw masks.

```bash
cd /home/dbw/merit-feddg-soft-guidance
scripts/with_gate_kb_env.sh -m pytest -ra
scripts/with_gate_kb_env.sh scripts/run_soft_guidance_pilot.py
scripts/with_gate_kb_env.sh scripts/audit_soft_guidance_history.py
scripts/with_gate_kb_env.sh scripts/run_soft_guidance_pilot.py \
  --output runs/soft-guidance-harm-v1 --ids configs/soft_guidance_harm_ids.json
scripts/with_gate_kb_env.sh scripts/evaluate_soft_guidance_probe.py \
  --run runs/soft-guidance-pilot-v1 --output runs/soft-guidance-checks/pilot-evaluation.json
```

Existing output roots refuse overwrite. GPU jobs ran in tmux with bounded
timeouts and have stopped. The wrapper now resolves its own worktree instead
of changing directory back into the older Gate branch. An initial runner import
typo was repaired before model execution; its failure log is retained locally.

Next decisions: recover the historical numerical execution path before scaling;
replace replay with verified per-branch KV caching without weakening parity;
only then extend fixed inference coverage. Scope-limited classification guidance
and any ICLR novelty claim remain untested, not implied by this spatial probe.
