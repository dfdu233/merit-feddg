# CRES: time-limited diagnostic results

Both workers stopped at 2026-09-15 19:59:51 UTC, four hours after the user's
stop-budget request (earlier canary/run time is additional). No automatic resume.
This is an ordered, time-truncated prefix, **not a full-test result** or a new
random split. All eight arms use exactly the same completed cases within each
dataset. No parameters were selected from these scores.

Requested code: `c78986e2429053077e0b8e9db7a648d3ea345c30`; compatibility and
reporting commit: `f8958a64d93cbd72482c654584b6bef2f82b1faf`; explicit partial
evaluation/deadline changes: `4c86d0a`. See [server provenance and commands](../../CRES_SERVER_RESULTS.md).
Frozen run identities and scorer hashes are included in the JSON exports.

## Coverage and metric

- VQA-RAD: 291 / 451 completed cases, 142 image clusters; 137 spatially applicable.
- SLAKE: 350 / 2094 completed cases, 32 image clusters; 193 spatially applicable.
- All 330 applicable cases passed fresh zero-strength token parity and had
  available distinct geometry controls. Other cases retain explicit unavailable
  semantics. No new expert calls: original native expert results were cached.
- Metric: frozen ANCHOR decoded CLOSED correctness plus OPEN answer token recall,
  first reference; all fresh arms use the same uniform short-answer prompt.
  Mixed scores are not clinical accuracy and cannot be compared directly with
  historical full-test scores using different prompts or case sets.

| Arm | VQA-RAD mixed (%) | SLAKE mixed (%) |
| --- | ---: | ---: |
| generalist | 48.5379 | 46.3571 |
| compact | 53.1828 | 54.3571 |
| deletion | 53.6295 | 55.8571 |
| native_fixed | 53.9732 | 55.8571 |
| null_fixed | 53.6295 | 55.8571 |
| cres | 53.6295 | 55.8571 |
| kl_only | 53.9732 | 55.2857 |
| residual_unbounded | 53.6295 | 55.8571 |

The [VQA-RAD JSON](vqarad-summary.json) and [SLAKE JSON](slake-summary.json)
include CLOSED/OPEN scores, all pairwise comparisons, image-cluster bootstrap
intervals, revision coverage, conditional harm, strength/KL distributions and
recorded costs. Scores/deltas in JSON are fractions, not percentages.

## Interpretation

| CRES diagnostic | VQA-RAD | SLAKE |
| --- | ---: | ---: |
| Delta vs compact (percentage points) | +0.4467 | +1.5000 |
| Image-cluster 95% CI (percentage points) | [-1.9218, +2.7475] | [-2.0078, +5.3522] |
| Improvements / harms vs compact | 8 / 7 | 21 / 17 |
| Improvements / harms vs deletion | 1 / 1 | 0 / 0 |
| Changed token sequences vs deletion | 5 / 291 | 1 / 350 |
| Preserved compact gain cases | 24 / 30 | 35 / 51 |

The compact gain cases are fresh compact improvements over fresh generalist on
this same prefix; preservation means CRES retains at least the compact score.
Thus not all original expert gains survive. Improvement/harm refers to frozen
scorer changes, not independently verified clinical improvement/harm.

CRES does not beat deletion on either prefix. Its VQA-RAD mean ties deletion
through one improvement and one harm; SLAKE has identical per-case scores.
Its apparent gains over compact have intervals crossing zero. These conditional
bootstrap intervals do not correct time/order selection bias or extrapolate to
unprocessed images (particularly the 32-image SLAKE prefix).

Real geometry versus misplaced geometry: native_fixed gains just one scored
VQA-RAD case over null_fixed; SLAKE has no per-case score difference. CRES and
null_fixed have identical per-case scores on both prefixes. There is no convincing
geometry-specific benefit here.

KL does not explain a protective effect of CRES: its maximum token KL is
.00438105 / .01519002, below .05. All token sequences equal residual_unbounded.
Almost every CRES step uses strength 1 (only six zero-residual steps per dataset).
KL-only itself scores slightly higher on VQA-RAD and lower on SLAKE; these results
do not establish a consistent benefit from the control envelope either.

## Cost and decision

| Recorded mean seconds / completed case | VQA-RAD | SLAKE |
| --- | ---: | ---: |
| compact decode | 0.6075 | 0.5565 |
| CRES decode | 15.0311 | 12.4893 |
| all eight arms and recorded checks | 54.2118 | 42.7136 |

CRES decode is approximately 24.7x / 22.4x compact decode without demonstrated
incremental gain. These averages include unavailable cases; not all branches
run on every case. Score API calls are not transformer forward counts: this
implementation replays prefixes, unlike the semantic-only persistent KV path.
Completed-case wall totals are 15,775.63 / 14,949.76 seconds, including work before
the four-hour deadline request. Startup is separate; interrupted unfinished-case
work is not included, so these are not complete resource billing totals.
Inherited source timings/calls are reported separately and are not fresh
deployment measurements. Cached experts do not imply zero deployment cost.

Decision: keep this run stopped. The core claim of incremental control-referenced
spatial benefit is not supported by this diagnostic. Do not label it a success,
change epsilon using these results, merge PR #7, or resume unchanged full runs
without a new instruction. Raw patient questions, answers, images, masks,
weights and credentials remain outside GitHub.
