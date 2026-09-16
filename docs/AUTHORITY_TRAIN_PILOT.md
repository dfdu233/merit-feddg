# Capability authority projection: real TRAIN pilot

## Version and scope

2026-09-16. Remote implementation inspected at
`014d2eac52b9e077995060a3b4e4a247ecb91f3d`, branch
`implementation/capability-authority-projection-v1`. Independent worktree:
`/home/dbw/merit-feddg-authority`. No changes to prior worktrees, configurations,
results, environments, checkpoints or test splits. No training or calibration.

The upstream change supplies a finite free-text candidate-pool projection, not a
decoder or candidate generator. This validation reuses the previous frozen
12-case TRAIN pilot (10 image clusters, 1793-row parent manifest), its ten-arm
free-text candidates and cached native XRV outputs. This is **not a full test
evaluation**, nor a newly generated candidate experiment. No references enter
candidate construction, source mapping, or model scoring.

## Necessary implementation repairs

- Avoid division by zero when one semantic group's base softmax underflows;
  calculate within-group softmax and KL in log space.
- Conservative attribute mapping: uncertain/history/ambiguous clauses remain
  unknown; negation in another clause cannot reverse an explicit finding.
  Contradictory leading binary answers and finding statements remain unknown.
- Add bounded real-model runner and separate frozen ANCHOR offline evaluator.
- Ten targeted tests added; **900 tests passed in 14.78 s**. `git diff --check`
  passes. Ruff reports four import-order/modernization warnings; no claim of a
  clean lint run. Imports were left unchanged after experiment identity froze.

Initial run v1 stopped before inference because the new runner compared file
bytes to the existing manifest's pixel hash. v2 uses the existing `pixel_digest`
implementation. The failed run/log was retained, not overwritten.

## Fixed mechanism and runtime

Candidate pool: deduplicate all ten historical arm texts in their frozen order;
no inserted yes/no templates or reference-derived candidates. Score each text
with LLaVA-Med's sequence-mean log likelihood, canonical tokenization plus EOS,
using both incumbent evidence and incumbent-plus-focused-XRV text. These are
finite-pool preference scores, not normalized language probabilities.

XRV uses the official `densenet121-res224-all` metadata operating points:
Effusion 0.103246614, Cardiomegaly 0.050318155, Pneumothorax 0.0098118475.
The source coordinate is the published piecewise normalization of cached raw
sigmoid, **not calibrated correctness**. Unknown probability mass is fixed;
positive/negative within-group odds are preserved. Missing a semantic side
leaves the base pool distribution unchanged, not necessarily the incumbent text.

Reused actor: `/home/dbw/ANCHOR/hf_cache/llava-med-v1.5-mistral-7b` and existing
CLIP/source tree. Reused XRV checkpoint SHA256:
`56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899`.
No new source-model calls, verifier, RAG, downloads or dependency changes.
Runtime: existing `.venv`, torch 2.14.0+cu130, four CPU threads.
Authorized physical GPU1/container GPU0:
`GPU-3846413a-4238-d307-b1f3-10c2dfbe002c`; no second GPU needed for this pilot.
Background tmux execution with a 1800-second process timeout.

## Results

Frozen ANCHOR `medheval-decoded-eval-v11-explanatory-binary-source-audited`.
All selected questions are CLOSED; numbers below are this scorer's correctness,
not clinical accuracy. References were loaded only after all 12 cases completed.

| Arm | Correct | Score | Improvements / harms vs incumbent | Paired image-bootstrap delta CI |
|---|---:|---:|---:|---:|
| Historical incumbent, fresh token parity | 11/12 | 91.67% | 0 / 0 | [0, 0] pp |
| Base candidate-pool argmax | 11/12 | 91.67% | 0 / 0 | [0, 0] pp |
| Text-conditioned candidate-pool argmax | 8/12 | 66.67% | 0 / 3 | [-50, 0] pp |
| Authority projection | 11/12 | 91.67% | 0 / 0 | [0, 0] pp |

These resampling intervals describe this tiny pilot only; a zero interval with
identical observed predictions is not a generalization or noninferiority result.

- Fresh incumbent token parity: **12/12**. New focused text delivered with all
  previously delivered evidence retained: **12/12**.
- Nonempty cached candidate pools: **12/12**, 56 unique-per-case candidates.
- Feasible binary projection: **1/12 (8.33%)**; missing semantic side: 11/12.
- Authority versus base-pool selected text changes: **0/12**.
- In the one feasible case the source coordinate was 0.00827 versus base mapped
  positive mass 0.16101, but an unknown-group answer remained the argmax.
- Authority/base-pool text differs from historical incumbent in 12/12 cases,
  despite identical binary scores. Pool selection must not be described as an
  exact incumbent-retention gate.
- Transport direction diagnostic: 6 false, 6 undefined because no mapped mass;
  this is not six clinically harmful outcomes.

## Cost and interpretation

112 new full-sequence scoring calls: **20.46 s**. Twelve parity generation calls
and setup: **10.91 s**. Total case wall time: **31.38 s** (~2.62 s/case), plus two
successful model loads: **22.03 s**. The initial failed v1 load is extra engineering
overhead and is excluded from the successful-run timing. Historical candidate
generation cost: **171.52 s**, native source inference: **0.79 s**. Historical
incumbent cost: **7.41 s**, reported separately (do not double-count it with the
ten-arm candidate budget). Source/old model-loading costs remain in the original
run and are not zero. This is not a demonstrated low-cost online replacement.

**No demonstrated benefit from authority projection; do not scale up yet.** It
ties the base-pool comparator exactly. Its advantage over text-conditioned
ranking is not evidence of a working corrective mechanism. The immediate
bottleneck is safe semantic mapping and candidate support. The frozen mapper is
deliberately conservative and not a validated clinical entailment model.

Also, preserving within-group odds/unknown mass does not guarantee that changing
a selected whole answer preserves its other claims. Before any broader claim,
validate both sides of candidate support without target-label selection, and
audit out-of-scope assertions. Do not tune the mapper against these scores.

## Reproduction and private artifacts

From the worktree, using the existing environment wrapper:

```bash
OPENBLAS_NUM_THREADS=1 bash scripts/with_gate_kb_env.sh -m pytest -o addopts='' -q
OPENBLAS_NUM_THREADS=1 bash scripts/with_gate_kb_env.sh scripts/run_authority_pilot.py --output runs/authority-train-pilot-v2 --check-only
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 timeout 1800 bash scripts/with_gate_kb_env.sh scripts/run_authority_pilot.py --output runs/authority-train-pilot-v2 --max-cases 2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 timeout 1800 bash scripts/with_gate_kb_env.sh scripts/run_authority_pilot.py --output runs/authority-train-pilot-v2 --max-cases 12
OPENBLAS_NUM_THREADS=1 bash scripts/with_gate_kb_env.sh scripts/evaluate_authority_pilot.py --run runs/authority-train-pilot-v2
```

Use a fresh output for reruns; completed evaluation refuses overwrite. Existing
case outputs are resumed only with matching frozen identity. The pilot identity:
`3320d92dc31c611ea0aba6de28d16d7ebc239ec4d298b3593981a81ce586903e`.

Local output: `/home/dbw/merit-feddg-authority/runs/authority-train-pilot-v2/`:
`frozen.json`, `cases/*.json`, `evaluation.json`, `cpu-tests.log`, `canary.log`,
`pilot.log`, and model-load timing records. Actual prompts, answers, evidence and
patient-linked artifacts stay local. Only this aggregate report and source/tests
are published. Old full results and stopped CRES experiments remain untouched.
