# Frozen Huatuo anchored-editor confirmation

This follows the RARR-inspired development and 20-image confirmation described in
[the preceding report](HUATUO_PAPER_GROUNDED_ITERATIONS.md). The editor prompt and
1024-token answer budget are unchanged. The added blind-editor control has the
same draft, original image and editing instruction, but no expert context.
Neither arm is a trained gate, and no numeric admission threshold is used.

## Data and execution

- Worktree: `/home/dbw/merit-feddg-huatuo-anchored`;
  branch: `experiments/huatuo-anchored-confirm-v1`, parent `3185ec7`.
- VQA-RAD: full label-free TRAIN manifest has 1793 rows; schedule selects one
  question per each of 87 remaining eligible images, excluding test pixels and
  the 24 previously used TRAIN images.
- SLAKE: full label-free bilingual TRAIN manifest has 9835 rows; schedule selects
  64 of 444 eligible TRAIN-only images. Actual language counts are 33 English and
  31 Chinese; answer types are 32 OPEN and 32 CLOSED. These counts result from
  fixed hash ordering, not outcome balancing.
- Questions are selected by minimum SHA256(ID) within each image; images are
  ordered by pixel digest. References are separate and read only by the scorer.
  Pixel exclusion is not a guarantee of patient-level separation.
- Existing Huatuo environment, checkpoint, router and expanded native expert
  pool are retained. No new weights, training, dependency upgrades or test-based
  strategy selection. Native generalist and compact controls are computed once.
- VQA-RAD runs on container CUDA0 / host GPU1; SLAKE runs on host GPU0 via the
  existing merit-runner SSH connection. Persistent tmux protects both jobs.
- SLAKE expert prefetch initially stopped on UniMed Git ownership validation.
  After checking its exact clean revision, process-local exact safe-directory
  entries were supplied for UniMed and BiomedParse. No global trust wildcard,
  version-check bypass or experiment-output replacement was used.

## Complete SLAKE 64-image result

The frozen ANCHOR metric combines CLOSED correctness under its existing parser
and OPEN token recall. It is not clinical accuracy or a full test benchmark.

| Arm | Score | Improve/harm vs Baseline | Improve/harm vs compact |
|---|---:|---:|---:|
| Generalist Baseline | 52.8646% | — | 7 / 8 |
| Original compact MERIT | 52.9167% | 8 / 7 | — |
| Blind anchored editor | 48.6979% | 4 / 7 | 10 / 14 |
| Evidence anchored editor | 49.4792% | 3 / 6 | 8 / 12 |

Both editors generated all 64 candidates, with no empty answers. Evidence editor
minus Baseline is -3.3854 percentage points (image-cluster 95% bootstrap interval
[-11.9792, +5.2083] pp). Against compact the interval is [-15.0560, +7.9701] pp.
Evidence versus blind editing improves 6 cases and harms 5; +0.7813 pp is not
evidence of a reliable expert benefit. This does not reproduce the earlier
20-image positive signal and does not justify test-set scaling.

### Failure separation

- Fourteen cases have no delivered expert evidence. Both editing arms score
  50.00%, below unchanged native controls at 61.90%. These harms cannot be
  attributed to expert-context contamination: an extra rewrite alone can fail.
- Among 50 cases with evidence, Baseline/compact/blind/editor scores are
  50.33/50.40/48.33/49.33%. Expert context is not sufficient to repair rewriting.
- One Chinese location answer becomes a semantically similar English location;
  the frozen lexical scorer decreases. This is not proof of a clinical error.
- Other changes reverse categorical anatomy/visibility answers, or replace a
  direct answer with a cautious diagnostic qualification. They must not all be
  explained away as lexical changes. Native anatomy evidence is not a disease
  diagnosis, and a correct short answer does not validate its full rationale.
- The earlier all-AGREES RARR gate is not run here and is not called successful.

### Cost and completeness

128 new calls: 64 blind (24.6846 s cumulative call time) and 64 evidence editor
(45.1223 s). These are observed call times under shared server load, not isolated
throughput or full method cost. Native generalist calls add 16.8592 s and compact
outer calls 52.1040 s. Expert prefetch, routing and all failed-attempt wall time
are not completely instrumented and must not be treated as zero.

Native identity: `905c822687d3d52ce4826f9f04de3a378c55897424c92f6c6f00c116c1de64ce`.
Editor identity: `c6614721d2db98a000e24184cb93d9e0699ade85080375b710e706e29d36bb41`.
Raw server-only roots: `runs/native-slake64-v1`, `runs/editor-slake64-v1`.
Public reports contain IDs and numerical outcomes, never images or raw answers.

## Complete VQA-RAD 87-image result

| Arm | Score | Improve/harm vs Baseline | Improve/harm vs compact |
|---|---:|---:|---:|
| Generalist Baseline | 60.3448% | — | 9 / 2 |
| Original compact MERIT | 53.4483% | 2 / 9 | — |
| Blind anchored editor | 48.4674% | 11 / 22 | 15 / 18 |
| Evidence anchored editor | 56.5134% | 7 / 11 | 12 / 8 |

All 87 cases completed both real candidate calls, no empty outputs. The editor
recovers +3.0651 pp over compact, but remains -3.8314 pp below Baseline. Its
image-cluster 95% bootstrap intervals are [-6.8966,+12.2605] pp versus compact,
[-13.0268,+4.9808] pp versus Baseline and [-1.1494,+17.2414] pp versus blind
editing. It improves 12 and harms 5 relative to blind editing. This is not a
reliable improvement beyond Baseline and does not support full test scaling.

174 additional calls cost 406.9660 s cumulatively, separately from native
generalist194.3568 s and compact outer223.2506 s. Missing routing/prefetch and
failed-attempt cost limitations above apply. Actual candidate coverage is100%.
Both datasets passed their first-case exact compact replay check (2/2 total),
not an assertion that every case was generated twice for parity.

Native identity: `050cd806e43ac424183dcb7ffbc1b4775fe01c34ffe7dc4b0d05bb60a1ae1203`.
Editor identity: `2d6e522e00ff4391715096c38526c1c8e79f617f2a2bfc8d9af655dc5640b351`.
Public numerical report: `reports/anchored-vqa87-v1.json`.

The editor mechanism is now a negative larger confirmation, despite earlier20
positive results. Do not optimize prompts on these images and call their scores
an independent confirmation again. The next small development question is whether
pairwise selection can avoid unnecessary answer rewriting; it is a separate
experiment, not a retroactive repair of this result.
