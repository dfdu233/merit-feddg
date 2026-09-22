# Full6719 Huatuo PathVQA TEST expansion

Completed on hostGPU1, two disjoint workers, no other GPU used. Exact6719unique IDs and all256pilot case files byte-identical.256 reused,6463 newly evaluated. No missing rows or duplicate IDs. All hypotheses/rules retained. GPU workers finished.

Evaluation: medheval-decoded-eval-v14-explicit-binary-conclusion; sample-weighted closed correctness/open answer-token recall. Scores are mixed VQA scores, not plain accuracy. Raw frozen outputs and per-case native effects in runs/test-full. Historical proposal budgets differ, source qualification failed, and proposer independence is unverified; this is exploratory postprocessing, not formal MERIT-Tx v3 superiority.

| Arm | Full6719 score % | New6463 score % | Accepted | Improved / harmed vs Greedy |
|---|---:|---:|---:|---|
| Historical Greedy |38.6567|38.6646|0|0 /0|
| Historical BARD |35.8285|35.9324|4841|465 /668|
| BiomedCLIP filter |38.5612|38.6260|1603|163 /164|
| CONCH filter |38.1974|38.1857|1365|166 /191|
| PLIP filter |38.8949|38.8649|683|79 /55|
| Joint filter |38.8118|38.8175|877|91 /77|

Image-cluster bootstrap3000replicates, fixedseed20260922, gain95%interval percentage points versusGreedy: PLIP[-.05745,+.53188],joint[-.17198,+.48092],BiomedCLIP[-.52945,+.34238],CONCH[-.91571,-.01985],BARD[-3.61101,-2.05128]. Marginal exploratory intervals, not adjusted for multiple arms. PLIP/joint do not establish superiority.

Pathology-routed3821subset: Greedy39.3286,BARD34.1792,PLIP39.7474,joint39.2632. Thus joint's slight full-dataset advantage does not hold within pathology; other modalities can only use supported verifiers, often BiomedCLIP. Do not describe joint as three pathology experts agreeing on all6719cases.

Coverage: PLIP1600overlength unavailable observations and1327outside-scope, CONCH1327outside-scope, BiomedCLIP225insufficient-controls and180outside-scope. Other rows include available evidence OR unchanged answers (see case artifacts). PLIP's preservation includes substantial abstention; equality is not evidence of reliable contradiction.

Conclusion: filtering can recover much of historical BARD's loss. PLIP/joint show small positive point estimates on both full and new portions, but uncertainty still includes no benefit. No expert/threshold was promoted based on these TEST results; existing v3 qualified policy remains baseline-preserving. Full expanded MUSK pool still untested. Future confirmation must use a separately frozen evaluation, not this already-inspected TEST as fresh evidence.
