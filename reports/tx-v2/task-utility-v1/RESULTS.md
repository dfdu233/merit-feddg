# Task-aware source utility experiment

Executed on host GPU1, both224-case source qualification jobs. Host GPU0 was busy with unrelated work and was not used. Frozen proposals, controls, verifiers, thresholds and original results retained. No target or previous canary labels entered this experiment. Recomputed native observations before worst-within-group collapse, rather than reweighting already collapsed observations.

Only source outcome scoring changed: existing local CE evaluator, OE reference-token recall, and unambiguous category match for the six PathoROB tissue classes. OE recall remains a lexical proxy, not factual accuracy. Metric source hashes retained. Five real evaluator invariants pass;55 targeted algorithm tests pass. This optional diagnostic metric requires the existing ANCHOR checkout on PYTHONPATH; default token-F1 behavior is unchanged.

| Model / source subset | N | Generalist % | Proposal % | Improved / harmed |
|---|---:|---:|---:|---|
| LLaVA PathVQA closed accuracy |81|65.43|61.73|1 /4|
| Huatuo PathVQA closed accuracy |81|62.96|61.73|7 /8|
| LLaVA PathVQA open recall |111|2.73|4.35|5 /0|
| Huatuo PathVQA open recall |111|4.04|3.27|1 /3|
| LLaVA PathoROB category accuracy |32|28.13|28.13|0 /0|
| Huatuo PathoROB category accuracy |32|53.13|21.88|0 /10|

Both receivers: zero commit-authorized and zero veto-authorized cards. No new canary or target expansion was launched because source qualification fails first.

A structural issue is now quantified: specificity counts all zero-utility transactions as failures. Even a perfect directional classifier on nonzero outcomes could achieve a Wilson lower bound of only .03654 for LLaVA pathology CONCH/BiomedCLIP (7nonzero/94) and .14654 for Huatuo (25/119), below the fixed .5 requirement. Thus failure cannot be attributed solely to verifier discrimination; this statistic conflates prevalence of useful changes and directional reliability. Yet utility lower bounds also remain nonpositive, so simply removing that criterion would not establish safe benefit.

Decision: stop automatic target expansion. Further redesign must explicitly separate useful-action prevalence, discrimination conditional on consequential changes, and harm among actual accepted actions. This experiment does not validate relaxing thresholds, inverting verifier signs, or changing candidate generation. A redesigned algorithm requires a fresh group-disjoint canary; observed80-case canary cannot be reused as independent confirmation.
