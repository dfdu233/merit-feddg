# Exploratory unqualified verifier ablation

Executed real native verifier scoring for both receivers on hostGPU1; other training/cache processes preserved.50 previously observed development pathology cases:34 PathVQA TRAIN-derived and16 PathoROB. Not fresh canary, source-P qualification, or official TEST. No MUSK. No formal qualification thresholds changed.

Frozen proposals reused. Candidate/control margins newly measured with four matched controls within this50-case development panel. Enumerated7 verifier subsets. Explicitly unqualified ablation: independent fault groups only; accept if any available member has D>0 and no available member has D<0; missing evidence abstains. Labels used only for post hoc evaluation, not thresholds or per-case acceptance. No subset promoted based on these results.

Task-aware mixed development score uses closed correctness, open token recall, and PathoROB unambiguous tissue-category match; NOT plain accuracy. Same50cases in every arm.

| Arm | LLaVA score % | Huatuo score % |
|---|---:|---:|
| Immutable Generalist |20.40|42.00|
| Frozen BARD candidate |20.40|32.00|
| BiomedCLIP only |20.40|38.00|
| CONCH only |20.40|38.00|
| PLIP only |20.40|42.00|
| BiomedCLIP+CONCH |20.40|38.00|
| BiomedCLIP+PLIP |20.40|38.00|
| CONCH+PLIP |20.40|38.00|
| All three |20.40|38.00|

Huatuo: candidate0 improved/5 harmed; BiomedCLIP andCONCH each block3 harmful changes but support2. PLIP has no available evidence on all5 harmed cases, so its equality to baseline is abstention rather than evidence of reliable negative judgments. All-three accepts6 modifications,0 score gains/2 harms. LLaVA:1 gain/1 harm, all3 experts positively support both; adding consensus fails to separate the harmful change.

Decision: no benefit justifying expansion or deployment from this bounded exploratory test. Distinguish proposal headroom (none on Huatuo under this panel's metric), verifier discrimination (shared false support), and missing-evidence fallback. Await MUSK authorization for added pathology capacity; do not infer full expanded-v3 failure from unavailable experts. Per-case effects and all7subsets retained in cases.json/results.json. Reproduce analyzer from repository root with existing huatuo environment and referenced local ANCHOR evaluator.
