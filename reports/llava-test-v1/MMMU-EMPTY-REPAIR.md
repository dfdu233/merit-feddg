# LLaVA-Med Adaptive BARD MMMU empty-output repair

Official full TEST: 10,500 unique IDs. Original predictions retained unchanged in the original run directories. Fixed official MMMU evaluator, unparseable/random fallback counted wrong.

| Protocol | Correct | Accuracy | Empty | Unparseable |
|---|---:|---:|---:|---:|
| Original | 2131 | 20.295238% | 659 | 3014 |
| Selected-EOS visible-continuation repair | 2246 | 21.390476% | 0 | 2605 |

All 659 initially empty records were selected without labels. When the original BARD decision selects EOS before any visible content, reuse the same scores and exclude EOS plus candidates that still decode to blank. Once visible content exists, resume ordinary decoding. Prompt, expert cache, BARD aggregation, and total 64-token budget remain unchanged. This is an explicitly revised decoding protocol, not a retroactive change to the original score. The other 9841 answer records are exactly unchanged. No baseline generation was rerun.

Native and cached replay previously reproduced whitespace→EOS exactly. EOS-only masking initially exposed a newline loop; the frozen visible-continuation rule was selected from token-level diagnostics without scoring against labels. Real three-empty/one-nonempty control pilot passed; the nonempty control retained identical tokens. All24 existing BARD tests passed, plus direct blank-loop/nonempty-parity checks. Every final repaired case records an actual guard intervention.

Two host retries did not reproduce source-environment empty termination. Both were rechecked on their original cloud environment: original tokens [28705,2] reproduced; repaired source-environment outputs replaced both host retries irrespective of their labels. All attempts retained. Neither source retry was correct, so the final accuracy is unchanged from the initial merged score.

115 of659 repaired outputs were correct (17.4507%). Remaining errors were not dropped.228 repaired outputs still reached the64-token cap withoutEOS, and2605 full-set outputs remain unparseable. Empty termination is resolved; semantic correctness, option adherence and incomplete answers are not thereby solved. Historical ICD20.8857/VCD17.7238/AVISC17.5333 used their archived decoding protocols, so any comparison with the repaired score is not an isolated algorithm ablation.

Machine-readable audit: mmmu-eos-repair-results.json (filename without the space). Full original/repaired answers, exact selection, retry traces, source controls and official per-ID judgments are retained under runs/llava-test-v1/mmmu-eos-repair/. Implementation commit cd75c55. No ground-truth labels were used for generation or retry selection.
