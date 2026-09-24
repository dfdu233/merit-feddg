# Mechanism ablation: manuscript-ready draft (results pending)

This file is a writing scaffold, **not** evidence that the experiments have finished. Replace every `TBD` only from the full-coverage, provenance-checked paired analysis. Do not fill cells from interim logs or historical main-table numbers.

## Experimental setup

We evaluate two frozen receivers, HuatuoGPT-Vision-7B and LLaVA-Med-7B, on the complete VQA-RAD TEST set (451 questions) and SLAKE TEST set (2,094 questions). For each receiver–dataset pair, all five arms use the same image, question, receiver checkpoint, native expert cache, answer-blind source selection and evidence-rendering protocol. The arms are: (i) the generalist without expert evidence; (ii) `joint_all`, which presents the selected evidence jointly to the receiver; (iii) `isolated_mean`, which decodes from source-isolated receiver branches and aggregates arithmetically; (iv) `isolated_geomedian`, which changes only the branch aggregation rule; and (v) `BARD (matched evidence)`, which adds anchored support and bounded commitment to the same isolated branches. The matched-evidence BARD arm is distinct from the previously reported full-native-schedule BARD: the latter may deliver more evidence and is not a causal comparator in this ablation.

The VQA score is the sample-weighted mean of CE strict correctness (0/1) and OE reference-token recall, expressed as a percentage. It is **not** pure accuracy. Each TEST question contributes once; nonempty text from a decode that reached `max_new_tokens` is scored as generated, while the unfinished flag is reported separately. Empty outputs remain in the denominator. Deterministic, answer-blind repairs, if any, are reported as a separate analysis and never substituted silently for raw generations. Because these TEST outcomes were previously inspected, statistical intervals quantify paired variability rather than a pristine confirmatory test of a newly tuned rule.

For arm `m` and question `i`, with score `s_mi` and generalist score `s_0i` in `[0,1]`, define `rescue_m = mean_i max(s_mi-s_0i,0)`, `harm_m = mean_i max(s_0i-s_mi,0)`, and `net_m = rescue_m-harm_m`. We use the same question/image clusters across all arms for paired bootstrap intervals (10,000 resamples, fixed seed). The prespecified contrasts are `joint_all - generalist` (descriptive evidence utility), `isolated_mean - joint_all`, `isolated_geomedian - isolated_mean`, and `BARD - isolated_geomedian`. The joint-to-isolated contrast identifies the **isolation-and-merge package**, not isolation alone. Holm adjustment applies to the latter three mechanistic contrasts; source, modality, language and CE/OE strata are descriptive.

## Table A: complete paired ablation

Create **one table per receiver and dataset** from the corresponding full-coverage analysis. Report raw outputs first; any length-extension or EOS repair belongs in a separate table with its own denominator and selection rule. Units: score/rescue/harm/net/change/parse/unfinished as %, time as seconds or GPU-hours (specify), calls as count per question.

| Receiver | Dataset | Arm | N | Score | Rescue | Harm | Net | Exact change | Decision change | Parse | Empty | Unfinished | Forward calls | Receiver decode time |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HuatuoGPT-Vision-7B | VQA-RAD | generalist / joint_all / isolated_mean / isolated_geomedian / BARD | 451 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| LLaVA-Med-7B | VQA-RAD | same five arms | 451 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| HuatuoGPT-Vision-7B | SLAKE | same five arms | 2,094 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| LLaVA-Med-7B | SLAKE | same five arms | 2,094 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

The compact rows above are placeholders, **not** four five-arm result tables. Expand each row into five rows after the coverage gate passes. Account for native-expert acquisition time separately from cached receiver decoding; otherwise a cached comparison understates deployment cost.

### Verified full-TEST result: LLaVA-Med-7B on VQA-RAD

The first completed paired analysis covers **all 451 official TEST questions** (251 CE, 200 OE; 203 image/patient clusters), with the frozen raw outputs and identical delivered evidence for all paired evidence arms. The score below is the mixed CE-strict/OE-reference-token-recall measure, not pure accuracy; rescue and harm are relative to the generalist. This is a descriptive analysis of an already inspected TEST set, not a new held-out tuning result. The authoritative files are `reports/researchstudio_ablation/llava_vqarad_raw_full/{summary.json,paired_rows.jsonl}` on the experimental runtime branch.

| Arm | Score % | Rescue % | Harm % | Net % | Exact changes / 451 | Parsed / 451 | Empty | Unfinished |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| generalist | 48.47 | 0.00 | 0.00 | 0.00 | 0 | 450 | 0 | 11 |
| joint_all | 45.44 | 11.91 | 14.94 | −3.03 | 379 | 421 | 0 | 26 |
| isolated_mean | 43.52 | 9.19 | 14.15 | −4.95 | 373 | 425 | 0 | 26 |
| isolated_geomedian | 43.74 | 8.75 | 13.48 | −4.73 | 373 | 425 | 0 | 26 |
| BARD (matched evidence) | 50.33 | 4.23 | 2.37 | +1.86 | 130 | 447 | 0 | 10 |

The anchored-commit contrast, BARD minus isolated_geomedian, is **+6.59 percentage points** (95% paired cluster-bootstrap CI +2.60 to +10.63; Holm-adjusted paired sign-flip p=0.0048). Relative to the generalist, BARD's +1.86-point net gain has a 95% CI of −0.72 to +4.54 points; do **not** claim a statistically resolved generalist win. Joint_all minus generalist is −3.03 points (CI −8.19 to +2.16); isolated_mean minus joint_all is −1.92 (CI −4.81 to +0.90); isolated_geomedian minus isolated_mean is +0.22 (CI −0.69 to +1.25). Thus this dataset supports a reduction of harm by anchored commitment relative to unbounded isolation, **not** a claim that isolation itself improves the receiver. CE and OE descriptive scores for BARD are 63.35% and 33.99%, respectively; the generalist scores are 61.35% and 32.31%.

No arm produced an empty answer or triggered the predeclared long-span repetition screen. Parser coverage is 450/451 generalist, 421/451 joint_all, 425/451 each isolated aggregation and 447/451 BARD. These quality counts are in the denominator, not filtered out.

Of 451 rows, 243 delivered one source group, 187 delivered two, and 21 delivered three or more. For 185 rows the answer-blind selection included evidence that could not fit the common joint context; all five evidence arms were nevertheless compared on the **same delivered** set. The raw LLaVA answer budget was 64 tokens, and nonempty unfinished answers were scored from their generated text. A separately frozen 128-token, answer-blind extension is **not** included in the primary table. Cached native access totaled 662.8 s; complete native expert acquisition cost is still unavailable and must not be inferred from this value.

The separate extension subsequently covered all 37 raw cases with at least one unfinished arm: 99 unfinished arm outputs fell to 2 while all original token prefixes and evidence matched. Re-scoring the complete 451-case denominator with only those answer-blind extensions changed **none of the five reported score percentages to four decimals**. Thus length limits did not explain the observed arm ranking; the raw table above remains primary.

**Mechanism limitation, not to hide in the aggregate:** 243/451 rows have only one delivered source group. On these rows BARD exactly equals the generalist (42.89%) while unbounded isolated_geomedian scores 32.43%; this conservative fallback explains most of the aggregate BARD–geomedian difference. On the 208 rows with at least two groups, geomedian is 56.96%, BARD 59.02%, a descriptive +2.07-point paired difference (99 image/patient clusters; 95% cluster-bootstrap CI −2.41 to +6.61). Relative to the generalist within this 208-row stratum, geomedian rescue/harm are 10.71%/8.75% and BARD's are 9.17%/5.14%: BARD reduces harm by 3.61 points but also loses 1.54 points of rescue. This subset is post-hoc descriptive, not a new confirmatory test. VQA-RAD alone does **not** establish that anchored commitment helps beyond conservative intervention; the full SLAKE result and strict consensus comparator are needed before writing that claim.

### Closest-prior strict consensus comparator: full LLaVA VQA-RAD

The independently free-decoded, base-relative **strict `q=m` probability consensus** also completed all 451 IDs. Its `joint_all` control reproduced the Stage-1 token IDs exactly on every case, and prompt, image, native cache and delivered evidence matched. This is an evidence-conditioned adaptation of the published consensus rule, not a reproduction of its trained-reference-model setup. The consensus score is **44.22%**, rescue/harm against the generalist **8.07%/12.33%**, parsed **433/451**, empty **0**, unfinished **21**, long-span repetition flags **0**. Against BARD's 50.33%, consensus is −6.11 points (95% paired image/patient-cluster CI −9.79 to −2.38); against isolated_geomedian's 43.74%, it is +0.48 (CI −1.76 to +2.80). These are descriptive TEST intervals, not a tuning license. The full result and per-ID rows are in `reports/researchstudio_ablation/llava_vqarad_consensus_strict_full/` on the experimental branch.

The same source-count caveat persists: for the 243 one-group cases, consensus and isolated_geomedian both score 32.43%, while BARD falls back to the generalist at 42.89%. On the 208 cases with at least two groups, consensus scores 57.99%, BARD 59.02%; the descriptive BARD–consensus difference is +1.03 points (99 clusters; 95% cluster-bootstrap CI −1.58 to +3.89). Therefore the full-set comparator establishes a **better risk-control policy overall**, but does not isolate a statistically resolved multi-source anchored-commit advantage. The latter remains an open claim for SLAKE and later failure-domain analysis.

**Compute-accounting gate:** the frozen cache's `progress.json` currently records 602.91 seconds summed over five local producer stages (`biomed_anatomy`, `cxr_findings`, `cxr_anatomy`, `chexagent_description`, `biomedparse_objects`) across the shared 2,545-case SLAKE/VQA-RAD manifest. This is **not** complete native-expert cost: `conch_tissue` and `medcpt_pubmed` timings are absent from that file, and the stage totals do not establish per-case latency or end-to-end wall time. The paired analyzer therefore leaves `native_expert_inference_seconds_total` null while reporting measured cached-access and receiver-decoding time separately. Do not turn the 602.91-second partial subtotal into a method latency claim; recover the missing producer logs or run a separately labelled acquisition-cost measurement before any full-cost comparison.

## Table B: paired effects and claim gate

| Receiver | Dataset | Prespecified contrast | Paired score effect (95% cluster CI) | Rescue change | Harm change | Adjusted p | Interpretation |
|---|---|---|---|---|---|---|---|
| TBD | TBD | joint_all − generalist | TBD | TBD | TBD | N/A (descriptive) | evidence utility, not isolation |
| TBD | TBD | isolated_mean − joint_all | TBD | TBD | TBD | TBD | isolation plus merge |
| TBD | TBD | isolated_geomedian − isolated_mean | TBD | TBD | TBD | TBD | center rule |
| TBD | TBD | BARD − isolated_geomedian | TBD | TBD | TBD | TBD | anchored commit |

The anchored-commit claim requires a favorable rescue–harm trade-off, not merely fewer changed answers. A lower harm accompanied by a larger loss of rescue is a negative or ambiguous result unless a tolerance was frozen on development data. No TEST-selected threshold, route, expert pool or repair rule may be relabeled as confirmatory. If the later base-relative/quorum consensus baseline matches or exceeds BARD, narrow the claimed contribution accordingly.

## Figure and diagnostic material

Plot each arm's rescue against harm (same axes and dataset-specific scale), annotate changed-decision coverage, and add the Stage-2 consensus baseline only after its branch-score protocol is frozen. Report the count of delivered source groups (`0/1/2/3+`), selected-versus-presented evidence discrepancy, and context-token budget for each arm. The case appendix should be selected by predeclared categories—rescued, harmed, unchanged despite useful evidence, missing source coverage, unfinished generation—and show the generalist answer, native evidence, joint and isolated outputs, BARD anchor/commit decision, reference answer and image identifier. Do not select anecdotes by desired outcome.

The first deterministic CE case selection for LLaVA VQA-RAD is recorded in [CASE_APPENDIX_VQARAD_LLAVA.md](CASE_APPENDIX_VQARAD_LLAVA.md); it deliberately includes both harm and missed rescue, and awaits exact-image visual review before any illustrated paper figure.

For Stage 2, start with strict base-relative unanimity (`q=m`), whose paper equation and released code agree. If a relaxed quorum is added, name whether its downward branch follows the paper's `q`-th ordered change or the released code's least-negative-supported change; the two differ when `q<m`. Freeze that choice before evaluating TEST and cite both the paper and code revision.

## Required checks before replacing `TBD`

1. Exact TEST ID count, no duplicates or missing IDs; source/manifest/image bytes match the run ledger.
2. Prompt, receiver, model weights, native cache and delivered evidence identity match for all paired evidence arms; ineligible mismatched rows are disclosed, not dropped silently.
3. Raw empty, unfinished and repeated-span counts, parse rate and scorer version are reported for every arm.
4. Cluster unit and number of clusters are reported; paired bootstrap uses identical clusters across arms.
5. Old full-native BARD and historical baselines remain separately labelled and are never pooled with the matched-evidence ablation.

The execution identities and commands are frozen in the experimental branch's `VQARAD_RUN_LEDGER.md` and `SLAKE_RUN_LEDGER.md`. The authoritative result source will be the full-coverage `paired_rows.jsonl` plus `summary.json` produced by `scripts/analyze_researchstudio_ablation.py` after each run is complete.
