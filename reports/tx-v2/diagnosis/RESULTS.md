# MERIT-Tx v2 source diagnosis

Diagnostic only: frozen qualification split, 224 cases/model (192 PathVQA TRAIN, 32 PathoROB development). No target labels, new generation, fitted thresholds, or independent-canary reuse. GPU0 was occupied by another job; GPU1 available. Cached outputs and native verifier observations suffice for this analysis, so neither GPU was used.

## Candidate opportunity under the existing token-F1 objective

| Model | Generalist % | Proposal % | Label-informed oracle % | Improved / harmed / tied |
|---|---:|---:|---:|---|
| LLaVA | 7.2543 | 6.9235 | 7.5120 | 16 / 15 / 193 |
| Huatuo | 13.5612 | 15.3226 | 20.8554 | 32 / 32 / 160 |

Oracle chooses the higher-scoring frozen answer using references; it is an unattainable diagnostic bound, not a deployable method. F1 improvement does not establish clinical correction. Huatuo PathVQA scores 6.4464/14.2305, while PathoROB scores56.25/21.875. Candidate PathoROB outputs collapse to25 tumor tissue and7 muscularis propria; this identifies concentration, not its causal expert source.

## Objective mismatch

On the81 source rows whose reference is yes/no, both Huatuo outputs have explicit leading yes/no: baseline51correct, proposal50correct,7 corrections and8 harmful flips. LLaVA has73 jointly explicit rows:48correct for both,0 polarity flips;8 ambiguous rows excluded, so this is not complete benchmark accuracy. This parser is a transparent diagnostic, not a replacement official evaluator.

Example PathVQA train00002-002311: incumbent starts No and explains; candidate No. Token-F1 gains0.987879 despite unchanged binary answer. Thus qualification's utility and specificity can reward brevity rather than correction. Preserve original metrics; do not silently refit cards with a new objective.

## Native verifier discrimination

AUC on improved versus harmed source group observations, excluding F1 ties; per-expert evaluable subsets differ and sample sizes are small. No generalization claim or automatic sign inversion justified.

| Model | BiomedCLIP D AUC | CONCH D AUC | PLIP D AUC |
|---|---:|---:|---:|
| LLaVA | .378 | .371 | .528 |
| Huatuo | .284 | .486 | .533 |

Requiring both positive raw margin and positive differential does not consistently improve conditional utility. Huatuo CONCH support mean gain changes from+.02237 to-.01540; BiomedCLIP from-.13281 to-.15699; PLIP+.04135 to+.04600 on only13 observations. These are unqualified retrospective diagnostics, not candidate algorithms. Larger GPUs or lowering qualification thresholds do not resolve this.

## Decision

Do not launch target expansion or present a new method as validated. First separate answer correctness from wording changes in source utility, using a fixed task-appropriate evaluation contract; then reassess proposal headroom and verifier discrimination. Existing80-case canary has been examined and is no longer an independent validation set for redesigned algorithms. Any changed protocol needs a versioned objective and fresh group-disjoint canary. No algorithm was modified during this diagnostic run.

Artifacts: source-diagnosis.json, source-changed-examples.json, explicit-binary-diagnostic.json. Reproduce from repository root with the two adjacent analysis scripts in the existing huatuo Python environment.
