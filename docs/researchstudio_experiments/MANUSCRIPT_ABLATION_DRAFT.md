# Mechanism ablation: manuscript-ready draft (results pending)

This file is a writing scaffold, **not** evidence that the experiments have finished. Replace every `TBD` only from the full-coverage, provenance-checked paired analysis. Do not fill cells from interim logs or historical main-table numbers.

## Reuse of existing headline results

For the paper's **main benchmark table**, reuse the already supplied full-test Greedy and MERIT rows; do not rerun those rows solely to fill the main table. The user-supplied values relevant to this draft are Huatuo VQA-RAD Greedy 62.82/MERIT 63.06 and SLAKE Greedy 54.69/MERIT 57.16; LLaVA VQA-RAD Greedy 48.47/MERIT 53.96 and SLAKE Greedy 34.68/MERIT 39.49 (percent, mixed CE/OE metric). Preserve their original provenance and quality caveats in the benchmark source. These headline values are **not** interchangeable with the matched-evidence raw BARD and generalist arms below: the MERIT/BARD evidence schedule differs, and the Huatuo SLAKE and VQA-RAD generalist numbers are not identical. Consequently, main-table reuse saves repeat inference but cannot supply per-question rescue/harm, paired uncertainty, or a causal BARD-versus-isolated contrast for this ablation. Complete already-running matched arms may be retained; do not launch new redundant full Greedy/MERIT jobs for this draft.

## Experimental setup

We evaluate two frozen receivers, HuatuoGPT-Vision-7B and LLaVA-Med-7B, on the complete VQA-RAD TEST set (451 questions) and SLAKE TEST set (2,094 questions). For each receiver–dataset pair, all five arms use the same image, question, receiver checkpoint, native expert cache, answer-blind source selection and evidence-rendering protocol. The arms are: (i) the generalist without expert evidence; (ii) `joint_all`, which presents the selected evidence jointly to the receiver; (iii) `isolated_mean`, which decodes from source-isolated receiver branches and aggregates arithmetically; (iv) `isolated_geomedian`, which changes only the branch aggregation rule; and (v) `BARD (matched evidence)`, which adds anchored support and bounded commitment to the same isolated branches. The matched-evidence BARD arm is distinct from the previously reported full-native-schedule BARD: the latter may deliver more evidence and is not a causal comparator in this ablation.

The VQA score is the sample-weighted mean of CE strict correctness (0/1) and OE reference-token recall, expressed as a percentage. It is **not** pure accuracy. Each TEST question contributes once; nonempty text from a decode that reached `max_new_tokens` is scored as generated, while the unfinished flag is reported separately. Empty outputs remain in the denominator. Deterministic, answer-blind repairs, if any, are reported as a separate analysis and never substituted silently for raw generations. Because these TEST outcomes were previously inspected, statistical intervals quantify paired variability rather than a pristine confirmatory test of a newly tuned rule.

For arm `m` and question `i`, with score `s_mi` and generalist score `s_0i` in `[0,1]`, define `rescue_m = mean_i max(s_mi-s_0i,0)`, `harm_m = mean_i max(s_0i-s_mi,0)`, and `net_m = rescue_m-harm_m`. We use the same question/image clusters across all arms for paired bootstrap intervals (10,000 resamples, fixed seed). The prespecified contrasts are `joint_all - generalist` (descriptive evidence utility), `isolated_mean - joint_all`, `isolated_geomedian - isolated_mean`, and `BARD - isolated_geomedian`. The joint-to-isolated contrast identifies the **isolation-and-merge package**, not isolation alone. Holm adjustment applies to the latter three mechanistic contrasts; source, modality, language and CE/OE strata are descriptive.

## Table A: complete paired ablation

Create **one table per receiver and dataset** from the corresponding full-coverage analysis. Report raw outputs first; any length-extension or EOS repair belongs in a separate table with its own denominator and selection rule. Units: score/rescue/harm/net/change/parse/unfinished as %, time as seconds or GPU-hours (specify), calls as count per question.

| Receiver | Dataset | Arm | N | Score | Rescue | Harm | Net | Exact change | Decision change | Parse | Empty | Unfinished | Forward calls | Receiver decode time |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HuatuoGPT-Vision-7B | VQA-RAD | five arms; detailed table below | 451 | complete | — | — | — | — | — | — | — | — | — | — |
| LLaVA-Med-7B | VQA-RAD | five arms; detailed table below | 451 | complete | — | — | — | — | — | — | — | — | — | — | — |
| HuatuoGPT-Vision-7B | SLAKE | five arms; detailed table below | 2,094 | complete | — | — | — | — | — | — | — | — | — | — |
| LLaVA-Med-7B | SLAKE | five arms; detailed table below | 2,094 | complete | — | — | — | — | — | — | — | — | — | — |

Both SLAKE and VQA-RAD receivers have passed the full paired audit and are expanded below. Report raw outputs first. Score/rescue/harm/net/change/parse/empty/unfinished are percentages; decoder time is summed per-arm receiver seconds, **not** end-to-end expert acquisition. The multi-arm runner recorded forward calls jointly per case, not attributable per arm, so no arm-specific call number is asserted. Native-expert acquisition time is incomplete for two producer stages and must not be folded into cached decode time.

**HuatuoGPT-Vision-7B, SLAKE, all 2,094 TEST IDs, raw matched evidence**

| Arm | Score % | Rescue % | Harm % | Net % | Parsed / 2,094 | Empty | Unfinished | Long-span repetition flags |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| generalist | 54.67 | 0.00 | 0.00 | 0.00 | 2,094 | 0 | 0 | 0 |
| joint_all | 49.31 | 12.48 | 17.83 | −5.35 | 2,090 | 0 | 0 | 0 |
| isolated_mean | 48.08 | 9.86 | 16.44 | −6.58 | 2,093 | 0 | 0 | 0 |
| isolated_geomedian | 48.00 | 9.59 | 16.26 | −6.67 | 2,092 | 0 | 0 | 0 |
| BARD (matched evidence) | 57.24 | 5.22 | 2.65 | +2.57 | 2,093 | 0 | 0 | 0 |

The fail-closed analyzer verified all 2,094 IDs (836 CE/1,258 OE), source images, frozen manifest/receiver/cache and equal delivered evidence. BARD minus isolated_geomedian is +9.24 points (95% paired image-cluster CI +7.17 to +11.45; Holm-adjusted p=0.0003); BARD minus generalist is +2.57 (CI +1.14 to +4.07). However, 1,182 single-source rows have BARD=generalist=58.72% against geomedian 42.95%. In the 912 multi-source rows, BARD is 55.32%, geomedian 54.53%, generalist 49.42%; an exploratory 109-image-cluster paired bootstrap gives BARD-minus-geomedian +0.78 points (95% CI −0.59 to +2.28). Thus a superior multi-source commit rule remains unresolved despite the positive overall score. [Raw summary](https://github.com/dfdu233/merit-feddg/blob/experiments/huatuo-adaptive-bard-validation-v1/reports/researchstudio_ablation/huatuo_slake_raw_full/summary.json) and paired rows are archived; do not conflate these matched-evidence values with the reused main-table Greedy 54.69/MERIT 57.16 values.

**LLaVA-Med-7B, SLAKE, all 2,094 TEST IDs, raw matched evidence**

| Arm | Score % | Rescue % | Harm % | Net % | Parsed / 2,094 | Empty | Unfinished | Long-span repetition flags |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| generalist | 34.68 | 0.00 | 0.00 | 0.00 | 2,069 | 0 | 52 | 1 |
| joint_all | 32.28 | 10.10 | 12.51 | −2.41 | 2,034 | 0 | 45 | 1 |
| isolated_mean | 32.93 | 10.50 | 12.25 | −1.75 | 2,037 | 0 | 47 | 1 |
| isolated_geomedian | 33.12 | 10.65 | 12.21 | −1.56 | 2,037 | 0 | 46 | 1 |
| BARD (matched evidence) | 36.11 | 2.70 | 1.28 | +1.42 | 2,064 | 0 | 54 | 1 |

The paired analyzer verified all 2,094 native TEST IDs (836 CE/1,258 OE), exact source images, receiver/cache identity, and equal *delivered* evidence across the four evidence-conditioned arms. BARD minus isolated_geomedian is +2.99 points (95% paired cluster-bootstrap CI +1.06 to +4.97; Holm-adjusted p=0.0084); BARD minus generalist is +1.42 points (CI +0.54 to +2.32). The one long-span repetition is the same genuine model loop at native qid `13817` in every arm; raw outputs remain in the denominator. Nonempty unfinished generations are likewise scored as generated, not silently replaced. Source files: [`summary.json`](https://github.com/dfdu233/merit-feddg/blob/experiments/huatuo-adaptive-bard-validation-v1/reports/researchstudio_ablation/llava_slake_raw_full/summary.json) and paired rows in the same directory. These TEST intervals are descriptive because outcomes were previously inspected; the older full-native MERIT/BARD SLAKE score follows a different evidence schedule and is not a matched comparator.

**Critical mechanism qualification:** On 1,277 single-source rows, BARD exactly retains the generalist's 35.26%, while isolated_geomedian falls to 26.81%. On the remaining 817 multi-source rows, the descriptive scores reverse: generalist 33.78%, joint_all 40.82%, isolated_mean 42.50%, isolated_geomedian **42.99%**, BARD **37.43%**. The two-source stratum alone (661 rows) is 41.11% geomedian versus 34.54% BARD; the 3+-source stratum (156 rows) is 50.96% versus 49.68%. An exploratory paired image-cluster bootstrap on the 817 rows (87 distinct image clusters, 10,000 resamples, seed 20260924) gives BARD minus geomedian **−5.56 points**, 95% percentile CI **−7.11 to −4.00**. This post-hoc interval describes the size of the negative finding; it is not a preregistered confirmatory claim. Thus the aggregate BARD win on SLAKE is driven by conservative single-source fallback and **does not establish a superior multi-source anchored-commit rule**. This negative result must be disclosed when interpreting the overall score.

**HuatuoGPT-Vision-7B, VQA-RAD, all 451 TEST IDs, raw matched evidence**

| Arm | Score | Rescue | Harm | Net | Exact change | Decision change | Parse | Empty | Unfinished | Decode s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| generalist | 62.84 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.00 | 0.00 | 0.00 | 701.4 |
| joint_all | 44.79 | 8.28 | 26.33 | -18.05 | 93.13 | 59.20 | 100.00 | 0.00 | 0.00 | 907.3 |
| isolated_mean | 47.23 | 5.31 | 20.92 | -15.61 | 91.57 | 51.44 | 100.00 | 0.00 | 0.00 | 4670.5 |
| isolated_geomedian | 47.95 | 5.26 | 20.15 | -14.89 | 91.13 | 50.33 | 100.00 | 0.00 | 0.00 | 5664.2 |
| BARD | 61.84 | 0.99 | 1.99 | -1.00 | 44.57 | 12.86 | 100.00 | 0.00 | 0.00 | 4548.3 |

**LLaVA-Med-7B, VQA-RAD, all 451 TEST IDs, raw matched evidence**

| Arm | Score | Rescue | Harm | Net | Exact change | Decision change | Parse | Empty | Unfinished | Decode s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| generalist | 48.47 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 99.78 | 0.00 | 2.44 | 556.4 |
| joint_all | 45.44 | 11.91 | 14.94 | -3.03 | 84.04 | 59.42 | 93.35 | 0.00 | 5.76 | 1159.5 |
| isolated_mean | 43.52 | 9.19 | 14.15 | -4.95 | 82.71 | 55.65 | 94.24 | 0.00 | 5.76 | 3930.1 |
| isolated_geomedian | 43.74 | 8.75 | 13.48 | -4.73 | 82.71 | 54.77 | 94.24 | 0.00 | 5.76 | 4086.9 |
| BARD | 50.33 | 4.23 | 2.37 | 1.86 | 28.82 | 14.86 | 99.11 | 0.00 | 2.22 | 2710.2 |

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

### Verified full-TEST result: HuatuoGPT-Vision-7B on VQA-RAD

The second receiver also passed exact 451-ID full coverage, native source/image/prompt/cache identity and equal delivered evidence across arms (251 CE, 200 OE; 203 image/patient clusters). All five arms have **zero** empty, unfinished and repeated-long-span outputs and **451/451** parsed decisions. The same frozen mixed scorer gives:

| Arm | Score % | Rescue % | Harm % | Net % |
|---|---:|---:|---:|---:|
| generalist | 62.84 | 0.00 | 0.00 | 0.00 |
| joint_all | 44.79 | 8.28 | 26.33 | −18.05 |
| isolated_mean | 47.23 | 5.31 | 20.92 | −15.61 |
| isolated_geomedian | 47.95 | 5.26 | 20.15 | −14.89 |
| BARD (matched evidence) | 61.84 | 0.99 | 1.99 | −1.00 |

BARD minus isolated_geomedian is **+13.89 points** (95% paired cluster-bootstrap CI +9.63 to +18.31; Holm-adjusted paired sign-flip p≈0.0003). Yet BARD minus the generalist is **−1.00 point** (net CI −2.66 to +0.54): this receiver does **not** show a resolved generalist improvement. Joint evidence and both unbounded isolated aggregators are substantially harmful. Huatuo CE scores are 76.49% for both generalist and BARD; OE scores are 45.70% and 43.45%, respectively. Selection exceeded the common joint presentation budget on 189 cases, but compared arms received the same *delivered* set.

The source-count decomposition again matters. For 218 one-group rows, BARD equals the generalist at 66.60% while geomedian scores 41.95%. For 233 rows with at least two groups, the scores are generalist 59.32%, geomedian 53.57%, BARD 57.39%; BARD–geomedian is +3.82 points (113 clusters, descriptive 95% CI +0.79 to +7.20), but BARD remains −1.93 points below the generalist. Thus BARD is a strong **harm limiter relative to naive evidence fusion** in this setting, not evidence of consistent improvement over the frozen receiver. The matched-evidence result must not be conflated with the historical full-native-schedule MERIT row.

### Closest-prior strict consensus comparator: full Huatuo VQA-RAD

The independently generated strict base-relative `q=m` consensus also completed all **451/451** official TEST IDs. The per-ID paired audit verified the original Stage-1 prompt, row, manifest, receiver configuration, cache, joint token IDs and delivered evidence. Its score is **50.69%**, rescue/harm against generalist **4.53%/16.68%**; parse 451/451, empty 0, unfinished 0 and long-span repetition flags 0. Consensus is **−11.15 points** versus BARD (95% paired 203-cluster CI −15.56 to −6.97), **−12.15** versus generalist (CI −16.55 to −7.87), and **+2.74** versus isolated_geomedian (CI +0.83 to +4.73). The full source is `reports/researchstudio_ablation/huatuo_vqarad_consensus_strict_full/` on the experimental branch. These are descriptive TEST comparisons, not grounds to tune the consensus quorum or BARD fallback.

On the 218 one-group rows, consensus scores 41.72% and geomedian 41.95%, while BARD/generalist score 66.60%. On the 233 rows with at least two groups, consensus scores 59.09%, BARD 57.39%, generalist 59.32% and geomedian 53.57%; consensus−BARD is +1.70 points with paired 113-cluster CI **−0.56 to +4.11**, so the multi-source difference is unresolved. The aggregate BARD advantage over consensus is therefore mainly conservative single-source anchoring, **not** an established superiority of its multi-source commit rule. This negative mechanism qualification belongs beside the main score.

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

The [VQA-RAD rescue–harm figure](https://github.com/dfdu233/merit-feddg/blob/experiments/huatuo-adaptive-bard-validation-v1/reports/researchstudio_ablation/figures/vqarad_rescue_harm.svg) uses both verified raw full-TEST five-arm summaries and both verified strict-consensus summaries. The dashed diagonal denotes zero net change relative to the corresponding generalist; above it is positive net score. The [plotting source](https://github.com/dfdu233/merit-feddg/blob/experiments/huatuo-adaptive-bard-validation-v1/scripts/plot_researchstudio_rescue_harm.py) rejects non-full, repaired or wrong-rule inputs; the figure is descriptive because TEST outcomes were previously inspected. Changed-decision coverage remains in Table A rather than being encoded as an unvalidated third visual dimension.

The first deterministic CE case selection for LLaVA VQA-RAD is recorded in [CASE_APPENDIX_VQARAD_LLAVA.md](CASE_APPENDIX_VQARAD_LLAVA.md); it deliberately includes both harm and missed rescue, and awaits exact-image visual review before any illustrated paper figure.

For Stage 2, start with strict base-relative unanimity (`q=m`), whose paper equation and released code agree. If a relaxed quorum is added, name whether its downward branch follows the paper's `q`-th ordered change or the released code's least-negative-supported change; the two differ when `q<m`. Freeze that choice before evaluating TEST and cite both the paper and code revision.

## Required checks before replacing `TBD`

### PathVQA failure-domain diagnostic (exploratory, not a benchmark row)

The LLaVA-Med PathVQA native cache now covers the full 6,719-case TEST set, but receiver inference has only been completed for a frozen, answer-blind **58-image diagnostic subset** (up to eight distinct images per routed modality). Its selected-case CE/OE composite percentages are generalist 25.06, joint_all 25.29, isolated_mean 23.28, isolated_geomedian 23.28, and BARD 28.79. The BARD-minus-geomedian paired image-cluster bootstrap difference is +5.52 percentage points, 95% CI [-1.72,+13.10]. The subset intentionally overrepresents rare modalities and cannot be presented as full PathVQA performance or an established gain.

The diagnostic points to an evidence-delivery bottleneck: 27/58 cases received only one expert group, and 23/58 had fewer presented evidence items than selected (127 selected versus 99 presented in total). In 53/58 cases, isolated geomedian did not improve on generalist, including ties. This motivates reporting *actual delivered* source count and evidence packing, not only the router's scheduled modalities. There were no empty outputs or long-span repetition flags; unfinished raw outputs were 1/9/9/9/2 for the five arms in the order above. These flags were not silently repaired. Source/manifest/cache hashes, per-case scores and modality strata are frozen in `reports/researchstudio_ablation/pathvqa_stage3_58_full/` on the experimental branch.

1. Exact TEST ID count, no duplicates or missing IDs; source/manifest/image bytes match the run ledger.
2. Prompt, receiver, model weights, native cache and delivered evidence identity match for all paired evidence arms; ineligible mismatched rows are disclosed, not dropped silently.
3. Raw empty, unfinished and repeated-span counts, parse rate and scorer version are reported for every arm.
4. Cluster unit and number of clusters are reported; paired bootstrap uses identical clusters across arms.
5. Old full-native BARD and historical baselines remain separately labelled and are never pooled with the matched-evidence ablation.

The execution identities and commands are frozen in the experimental branch's `VQARAD_RUN_LEDGER.md` and `SLAKE_RUN_LEDGER.md`. The authoritative result source will be the full-coverage `paired_rows.jsonl` plus `summary.json` produced by `scripts/analyze_researchstudio_ablation.py` after each run is complete.
