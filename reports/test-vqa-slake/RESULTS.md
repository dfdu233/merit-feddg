# Huatuo VQA-RAD and SLAKE full TEST verifier transfer

All451 VQA-RAD and2094 SLAKE completed on hostGPU1. Two disjoint shards, original frozen paired Generalist/BARD answers reused, no new generator inference, no target-trained qualification. These paired original generations share1024-budget/eager protocol (unlike historical PathVQA budget mismatch). No missing output IDs. Source controls, D threshold and6arms unchanged from PathVQA exploration. No expert selected using labels.

| Dataset | N | Paired Generalist | BARD proposal | BiomedCLIP/joint filter | CONCH/PLIP filters |
|---|---:|---:|---:|---:|---:|
| VQA-RAD |451|63.8910|63.0648|63.4919|63.8910|
| SLAKE all |2094|54.5504|57.1073|54.3912|54.5504|
| SLAKE English |1061|56.1603|55.5540|55.8462|56.1603|
| SLAKE Chinese |1033|52.8969|58.7028|52.8969|52.8969|

Scores percentages of existing CE correctness/OE recall mixture, not pure accuracy. Source VQA-RAD official references; SLAKE frozen historical references were verified equal to native per-image answers, and source CLOSED/OPEN plus language annotations restored.2087 exact question matches and7 sole spelling corrections exsit→exist for metadata alignment. No prediction edits. Current evaluator medheval-decoded-eval-v14-explicit-binary-conclusion.

Radiology modalities only:1053CT,578MRI,914CXR. CONCH/PLIP abstain by scope, so their equality to baseline is not demonstrated verification strength. Joint equals BiomedCLIP.617 changed cases have insufficient CXR source controls (3available<4required); no target images substituted as controls. Additional unchanged rows require no verification. Frozen source controls are image-routed mixed-source cases, not calibrated radiology-source qualification.

VQA-RAD filter accepts25:1score improvement,2harms; gain image-cluster95%CI[-1.07767,+.09009]pp. SLAKE filter accepts28:5improvements,8harms;CI[-.50884,+.13858]pp. All28acceptances are English. Original SLAKE candidate has strong Chinese contribution; filtering returns Chinese entirely to incumbent and loses that gain. Do not claim a generally beneficial filter or statistically confirmed degradation from these marginal intervals.

Decision: PathVQA's small exploratory positive point estimate does not transfer to these two complete TEST sets. Existing pathology experts lack scope here, source controls under-cover chest radiographs, and available native decisions do not deliver net observed benefit. No full-v3 or SOTA claim; no automatic threshold tuning or substitution of unsupported experts.
