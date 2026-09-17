# Historical channel ACD — current main-pipeline re-evaluation

## Scope and identity

Completed VQA-RAD 451 questions (251 CLOSED, 200 OPEN; 203 image clusters).
Generation identity: `3c609e9ab859e69e3c6634edeb6dd8821765e51b3831e811013f467e3e6cc2e4`.
Generation code: `818f492`, based on historical `350c313` / `d20c928`.
Both authorized host GPUs completed disjoint scheduling shards. Existing jobs
were not stopped. Four real canaries passed exact historical text/deletion/CAD
tokens and cached/replayed full-vocabulary scores. No new expert calls.

User explicitly authorized current-main-pipeline scoring after the original
v11 scorer guard failed. Original frozen identity and results remain unchanged.
All saved arms, including historical generalist and compact, were rescored with:

- `medheval-decoded-eval-v13-explanatory-uncertainty-review`;
- `mixed-medical-vqa-table-v4-source-typed-mmmu-official-primary`;
- direct `evaluate_mixed_vqa_table.score` invocation, checked against every
  arm's reconstructed per-case contributions to tolerance 1e-12;
- same 451 IDs and references, no inference rerun or parameter selection.

Main score is sample-weighted CLOSED correctness + OPEN answer-token recall,
not plain accuracy or clinical accuracy. CLOSED accuracy is reported separately.
Using the current scorer does not certify historical generation prompts/budgets
as equivalent to newer 1024-token benchmark runs; cross-method comparisons
still need generation-protocol alignment.

## Results (%)

| Channel / arm | Mixed VQA | CLOSED accuracy | OPEN recall | Improved / harmed vs text |
|---|---:|---:|---:|---:|
| Historical generalist | 48.4731 | 61.3546 | 32.3068 | — |
| Historical compact / text incumbent | 53.9566 | 68.1275 | 36.1722 | 0 / 0 |
| Classification deletion | 53.9566 | 68.1275 | 36.1722 | 0 / 0 |
| Classification fixed blend (0.5) | 53.9566 | 68.1275 | 36.1722 | 0 / 0 |
| Classification fixed CAD (0.5) | 53.2915 | 68.1275 | 34.6722 | 0 / 3 |
| Classification dynamic ACD | 53.9566 | 68.1275 | 36.1722 | 0 / 0 |
| Generated-text deletion | 48.5691 | 60.1594 | 34.0234 | 12 / 38 |
| Generated-text fixed blend (0.5) | 52.3016 | 65.3386 | 35.9401 | 6 / 14 |
| Generated-text fixed CAD (0.5) | 53.5797 | 67.7291 | 35.8222 | 2 / 5 |
| Generated-text dynamic ACD | 52.3016 | 65.3386 | 35.9401 | 6 / 14 |

Dynamic classification ACD is +0.6652 percentage points vs fixed CAD (3 gains,
0 harms), but has only 6/451 delivered candidates (1.33%); other rows inherit
incumbent and are not successful interventions. Image-cluster paired 95% CI
vs CAD: [0, +1.5353] points. No demonstrated benefit beyond text or fixed blend.

Dynamic generated-text ACD has 174/451 candidates (38.58%). Relative to text:
-1.6551 points, image-cluster paired 95% CI [-3.8302, +0.1859]. Relative to CAD:
-1.2781 points, 11 gains / 16 harms, CI [-3.5338, +0.6707]. Intervals include
zero; neither an improvement claim nor statistically established degradation
is supported. The observed point estimate is worse than text and CAD.

All 451 per-case scores equal fixed blend for each channel. Classification
candidate texts also match blend exactly; generated-text ACD changes five
texts vs blend without changing any per-case scores. Dynamic weights are
real (classification range 0.0809–0.8050, generation 0.0926–0.9989), but do not
produce measured incremental benefit here.

## Retention of old benefits

Under this same current scorer, historical compact improved 49 and harmed 20
cases relative to generalist, with net +5.4836 points. This gain belongs to
the old method, not ACD. Classification ACD preserves both cohorts' scores.
Generated-text ACD harms 10 of the original 49 improved cases (no gains in
that cohort), while improving 4 of the 20 original harmed cases (no harms in
that cohort). Repairing some old mistakes does not offset lost old benefits.

## Cost and limitations

- Classification: 14.25 seconds total ACD decoding, 2.37 seconds/active case,
  212 score calls; recorded row wall time including context/audits 33.73 seconds.
- Generation: 407.71 seconds total ACD decoding, 2.34 seconds/active case,
  6,860 score calls; recorded row wall time including context/audits 473.83 seconds.
- These are summed task times, not two-GPU elapsed wall time. Model loading,
  preserved failed preflight, and historical expert/control generation are
  separate costs. Concurrent device load and old scoring backends differ;
  old/new timings do not establish a controlled speedup.
- ACD uses same-prefix entropy-weighted convex interpolation, not fixed CAD's
  extrapolation. No extra verifier or new evidence was introduced.
- Source-only/source-ACD were not run: original delivered BiomedCLIP catalog
  and generated-text evidence do not supply matched native source uncertainty.
- This experiment does not rerun spatial/segmentation guidance, SLAKE, or MIMIC.

## Reproduction and artifacts

From the independent worktree, after complete generation:

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  /home/dbw/merit-feddg/.venv/bin/python scripts/evaluate_vqarad_dynamic.py \
  --run runs/vqarad-dynamic-v1/3c609e9ab859e69e3c6634edeb6dd8821765e51b3831e811013f467e3e6cc2e4 \
  --current-scorer \
  --output runs/vqarad-dynamic-v1/3c609e9ab859e69e3c6634edeb6dd8821765e51b3831e811013f467e3e6cc2e4/evaluation-current-v13.json
```

The evaluator refuses overwrites. The JSON records scorer hashes/dependencies,
manifest/reference hashes, per-case scores, paired intervals, costs and weight
traces. Published companion JSON contains aggregate scores and confidence intervals,
not patient images, questions, model outputs, weights, or credentials. Per-case
scores remain in the complete local evaluation file.

Validation: 14 uncertainty/class-text CPU tests passed; evaluator compilation
and git diff checks passed. Main scoring and per-case reconstruction agree for
all 14 channel/arm entries. These checks establish execution, not medical benefit.
