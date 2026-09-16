# CS197 paired-support GPU1 diagnostic — completed 2026-09-16

## 1. Evidence update

**Verdict: guaranteed candidate support enables real interventions, but this pilot does not demonstrate a stable collaboration advantage over direct specialist replacement. Candidate phrasing is a major confound. Marginal locality does not guarantee locality of the selected answer.**

Independent branch `experiments/paired-support-gpu1-20260916`, worktree `/home/dbw/merit-feddg-paired-support`, based on `de056a8`; frozen GPU runner commit `18c33c2`. Experiment identity:

`6dc54d1eef5f9a3251ec7ee16277f1f5580881d2f91283bee7d111cfbe8fc724`.

Physical host GPU1 = container GPU0, UUID `GPU-3846413a-4238-d307-b1f3-10c2dfbe002c`. User authorized this GPU. The 12-case run and offline evaluation are complete; no GPU job from this experiment remains running.

Selection used the existing native-presence grammar, frozen TRAIN order, and input-modality metadata, without references. There were 45 eligible TRAIN questions before removing official-test RGB hashes; 24 questions on 12 unique images remained. The pilot takes the first question per distinct image: Effusion 4, Cardiomegaly 2, Pneumothorax 6. It has zero exact test-image hash overlap. Patient/study identity and pretraining overlap are not established. This is a narrow development diagnostic, not a random sample or full benchmark.

Unlike the old 12-case pilot, this is a newly selected set with newly computed XRV outputs and no inherited evidence in the generalist baseline. The old 11/12 result is **not** its baseline.

### Target-answer scores

Two candidate wordings and both normalizations were frozen before inference; all four cells are reported. Each four-state candidate expresses the target finding and an additional, unqueried, logically compatible finding. Target-only evaluation uses the first clause; the truth of the residual clause is not scored. The generalist still answers the original question. Residual clauses are diagnostic candidate carriers, not clinically validated report content.

| Method | Wording A, mean | Wording B, mean | Wording A, sum | Wording B, sum |
|---|---:|---:|---:|---:|
| Same-pool generalist | 4/12 | 8/12 | 4/12 | 9/12 |
| Same-pool focused evidence text | 4/12 | 4/12 | 4/12 | 4/12 |
| Fixed CAD-style sequence reranking | 4/12 | 3/12 | 4/12 | 4/12 |
| Direct expert, raw coordinate | 9/12 | 9/12 | 9/12 | 9/12 |
| Old authority, raw coordinate | 9/12 | 9/12 | 9/12 | 9/12 |
| Fixed residual marginal, raw coordinate | 9/12 | 9/12 | 9/12 | 9/12 |
| Direct expert, operating coordinate | 9/12 | 9/12 | 9/12 | 9/12 |
| Old authority, operating coordinate | 9/12 | 9/12 | 10/12 | 9/12 |
| Fixed residual marginal, operating coordinate | 9/12 | 9/12 | 9/12 | 9/12 |

The CAD arm is fixed `1.5 * conditioned_score - .5 * base_score` sequence reranking, not a new token-decoding experiment. Mean includes canonical EOS and sum multiplies by token count. Neither normalized finite-pool score is a calibrated full language distribution.

For both wordings under mean scoring, operating-coordinate authority and direct replacement make exactly the same 12 target decisions. The isolated 10/12 cell adds one correct target relative to direct replacement, but it does not survive the other wording/normalization cells. It is not selected as the winning configuration.

Under wording A/mean, authority and direct replacement each rescue 6 and harm 1 versus same-pool generalist. Under wording B/mean, both rescue 3 and harm 2. Thus a gain versus generalist alone is not evidence of a new collaboration mechanism. Direct raw and direct operating-coordinate scores happen to both be 9/12 but do not necessarily make the same predictions.

Ordinary free generation, evaluated separately on complete output by the frozen ANCHOR v11 scorer: generalist **4/12**, focused expert text **3/12** (0 rescues, 1 harm). The class balance, only inspected offline, is 8 No / 4 Yes; a constant-No response would obtain 8/12. This retrospective class-balance observation is not a newly selected method and limits interpretation of the specialist's 9/12.

### Locality and sensitivity

- Under mean scoring, changing between the two predefined wordings flips the generalist's target choice on **12/12** cases. The score changes from 4/12 to 8/12. Candidate wording is therefore a substantial unresolved confound.
- Old operating-coordinate authority has mean absolute residual-probability drift **0.02833** (A/mean) and **0.03055** (B/mean).
- Standard iterative proportional fitting constrains residual marginal drift to numerical precision, approximately `1e-17` in these cells. This is a mathematical execution property, not a learned ability.
- Despite that constraint, the selected answer changes its residual state on **7/12** cases (A/mean) and **9/12** cases (B/mean), relative to the same-pool base argmax. Direct replacement retains the base residual state by construction, with **0/12** such changes.
- Sum normalization substantially changes these properties; all cell-level metrics are in the aggregate JSON. Statistical correlation between target and residual can make some propagation appropriate, so none of these residual changes is labeled medical harm.
- Positive/negative pairs have complete authored support and reproduce direct-expert equivalence on **12/12** real-score cases. Artificially guaranteeing support does not demonstrate free-form candidate generation.
- Base versus focused free outputs have the same first eight tokens on **3/12** cases, and the same full tokens on those same **3/12**. There are **0** cases of a hidden later token change in this comparison. This gives no positive evidence for the short-window explanation here; it does not rerun or invalidate the old tensor-gate experiment.

### Execution and cost

All **12/12** target packets were admitted and actually presented using the shared enforce delivery path. Mutating nonrequested finding scores leaves the final conditioned prompt unchanged on **12/12**. Original native score semantics remain uncalibrated independent sigmoids; operating-point normalization is not target-domain probability calibration. Requests are explicit structures; automatic parsing was not improved.

| Cost/check | Result |
|---|---:|
| Fresh XRV inferences | 12 |
| Real LLaVA-Med free generations | 24 |
| Real full-sequence scoring calls | 288 |
| Case processing | 105.87 s |
| Actor loads, canary plus continuation | 22.92 s |
| Expert load and inference | 1.08 s |
| Sum of timed execution components | 129.87 s |
| Existing relevant CPU tests | 32 passed |

Timing excludes implementation, preparation and offline evaluation. It is not an end-to-end deployment latency estimate. After completion the GPU returned to its pre-run 18,052 MiB usage, with approximately 30,459 MiB free; these are snapshot values, not a reservation.

## 2. Affinity-map location

This result concerns attribute-controlled generation, finite-pool posterior constraints and expert deferral. Prior work already supplies those mathematical operations. The preceding literature review is in `/home/dbw/research-notes/heterogeneous-experts-20260916/RESEARCH.md`; a parallel repository literature audit also exists in commit `14fd616`. No new literature or novelty claim is implied by this GPU run.

## 3. Bit Flip status

The broad idea of restricting expert influence to its native semantic authority is still unproven. **The specific claim that old projection provides useful collaboration beyond direct replacement is weakened.** A standard marginal constraint does not resolve selected-text locality. Neither the old nor the constrained projection is presented as a new research algorithm.

## 4. Current Vector

Can a wording-stable semantic decision rule demonstrate complementary generalist information beyond direct expert replacement while preserving an explicitly defined residual property?

## 5. Core / Periphery

Core remains the existing models, real native outputs, semantic representation, strong direct-expert baseline and honest target/residual measurements. No new model was trained; no verifier, RAG, segmentation extension or new learned gate was added.

## 6. Minimum experiment and actual stopping point

The frozen experiment varied update rules on matched four-state support, with two fixed wordings and mean/sum sensitivity controls. It also compared fresh free outputs and verified shared admission. Reference files were opened only after all frozen case files existed and identity checks passed.

H1 required useful target intervention without unjustified residual propagation and with complementary value beyond replacement. H0 included wording effects and source-copy degeneration. The results favor those competing explanations strongly enough that extending this exact protocol to 48 cases or a full benchmark is not justified. There is no parameter tuning after looking at these scores.

## 7. Novelty collision check

The new evidence does not distinguish this method from standard constrained inference plus expert decisions. The 10/12 isolated cell is normalization-sensitive. It is not evidence of ICLR-level novelty. A subsequent research direction needs to address semantic scoring/decision stability and establish nontrivial complementarity, rather than rename IPF or add gate thresholds.

## 8. Velocity criterion

Resolved: matched support allows a real intervention, the two-candidate degeneration survives real model scoring, and distribution-level locality is insufficient for selected-output locality. Unresolved: natural answer support, wording-stable semantics, true residual correctness, broader generalization and meaningful complementary value.

## 9. Re-vector rules

- Positive: only proceed to a second existing capability type after a preregistered wording-stable test exceeds direct replacement with interpretable residual behavior.
- Negative: if decisions still reduce to expert copying, stop claiming collaboration gains and investigate source reliability or the task definition.
- Ambiguous: if wording/scoring changes dominate, improve the measurement substrate before comparing algorithms.
- Gate: this run does not support hidden changes beyond eight tokens; do not use its results to justify a longer-window gate as a new method.

## Reproduction and artifacts

```bash
OPENBLAS_NUM_THREADS=1 bash scripts/with_gate_kb_env.sh scripts/run_paired_support_probe.py --stage prepare --output runs/paired-support-gpu1-new
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 timeout 1800 bash scripts/with_gate_kb_env.sh scripts/run_paired_support_probe.py --stage run --output runs/paired-support-gpu1-new --max-cases 1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 timeout 1800 bash scripts/with_gate_kb_env.sh scripts/run_paired_support_probe.py --stage run --output runs/paired-support-gpu1-new
OPENBLAS_NUM_THREADS=1 bash scripts/with_gate_kb_env.sh scripts/evaluate_paired_support_probe.py --run runs/paired-support-gpu1-new
```

Local raw artifacts: `runs/paired-support-gpu1-v1/{frozen.json,sources.json,cases/,canary.log,pilot.log,evaluation.json}`. Image-linked IDs, packets, candidate answers and references stay local. The published aggregate is [results/paired-support-gpu1-v1.json](results/paired-support-gpu1-v1.json). Prepare refuses an existing directory; completed evaluation refuses overwrite. A fresh run on a later commit receives a new identity.
