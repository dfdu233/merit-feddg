# Huatuo OmniMedVQA verifier expansion

Completed all17,673 fixed snapshot IDs; not the full88,995 test. Existing raw Baseline/BARD reused. All256 pilot input records and output bytes unchanged;17,417 additional rows. Two disjoint native verifier workers ran on hostGPU1 and exited; no raw generation rerun. Frozen original route, controls, option mapping, acceptance rules and v14 evaluator unchanged. Unqualified exploratory filter, not source-qualified MERIT-Tx v3.

| Arm | Correct/17673 | Accuracy % | New17417 accuracy % | Accepted | Corrected/broken vs baseline |
|---|---:|---:|---:|---:|---:|
| Baseline |12036|68.1039|68.0485|0|0/0|
| Original BARD |12066|68.2736|68.2207|520|145/115|
| BiomedCLIP filter |12051|68.1888|68.1403|56|32/17|
| CONCH filter |12041|68.1322|68.0772|18|9/4|
| PLIP filter |12037|68.1095|68.0542|21|8/7|
| Joint filter |12054|68.2057|68.1575|42|27/9|

Joint improves on Baseline by18 answers but loses12 versus original BARD. It removes106 BARD-induced errors while rejecting118 beneficial corrections. Filtering reduces harmful changes, but also loses useful corrections; the net point estimate is worse than unfiltered BARD here. No stable superiority claim or threshold selection from TEST.

Paired image-cluster bootstrap (16,074 imageSHA groups,2,000 replicates,seed20260922): joint minus Baseline95% interval[+0.0394,+0.1697] percentage points; joint minus original BARD[-0.2382,+0.0957] points. The first interval excludes zero in this fixed pool; the second includes zero. These are exploratory marginal intervals, not multiplicity-adjusted and not evidence about all88,995 test samples or other datasets. Thus a small positive within-pool signal versus Baseline exists, but no demonstrated advantage over original BARD.

Coverage:520 textual changes;260 unambiguous-letter-only mapping skips (some merely equivalent letter-plus-text);119 cases with at least one available native effect. Out-of-scope verifiers abstain;7 BiomedCLIP observations lacked4 matched source controls. No other error category observed. All17,673 final answers parsed in all arms. Original routing mistakes remain to preserve this method version. Do not confuse missing applicability with successful clinical verification.

Machine-readable scores and changed-case effects: results.json. Frozen inputs and provenance: runs/test-omni17673. Native masks/other original benchmark outputs not modified. Results comparable only on these fixed IDs; do not mix with newer20k snapshots or full-test baseline percentages.
