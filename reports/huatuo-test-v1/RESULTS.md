# Huatuo Adaptive BARD — complete official TEST

OmniMedVQA update (2026-09-23): one **complete shard**, 21,993 exact IDs, has been archived and matched-rescored at BARD 75.5332% versus Greedy 74.5965% (+0.9367 points); this is not the 88,995-ID full TEST result. See `omnimedvqa-shard2-complete-matched-v15-20260923.md`. Other Omni shards are still running or have unverified gaps; do not enter this subset score into a full-dataset/SOTA table.

An independently completed 2,048-ID interval [18144,20192) is nearly neutral: BARD 87.3535% versus Greedy 87.4023%, same IDs and evaluator. It overlaps shard2 and must not be added to that denominator. See `omnimedvqa-18144-20192-matched-v15-20260923.md`.

Current-score correction (2026-09-23): the existing PMC-VQA table below is a historical v14 snapshot. The current evaluator code (SHA-256 `497831651bfb9e178c7a79b9b5312ef578e6e36c3fa413c92cb12d3c63132163`) rescored the same frozen 33,430 answers at BARD **51.6991%** and Greedy **52.3602%**, without new generation. See `pmcvqa-current-evaluator-rescore-20260923.md`. Do not combine its numbers with the historical table or label BARD as SOTA.

OmniMedVQA ongoing quality spot-check (2026-09-22): inspected each cloud's50 most recently modified tail `bard.json.gz` records for each model (4nodes ×2models ×50 =400 outputs). Zero whitespace-empty answers and zero `finished=false` records in this sample. Tail file counts at observation: Huatuo40297/44450/42865/51493=4098/4096/4097/4096; LLaVA=4015/4007/3794/4096. These are file counts and a recent-window quality check, NOT a full-ID coverage audit or proof that earlier empty/unfinished outputs have all been repaired. No additional retries launched on this sample.

Persistent full PMC raw answer archive: `pmcvqa-full-raw-answers.json.gz`,33,430 unique frozen IDs, SHA256 `1a623466219058469bce6fffc93ff7933d81b41a8f6af19177d553d9b06cef3e`. Includes text/prompt/image/cache provenance; full native evidence remains at source. No longer necessary to depend on temporary `/tmp` exports for answer rescoring.

## MMMU parser correction audit (2026-09-22)

The previously reported 31.6286% is NOT a reliable estimate of all unambiguous answers: the pinned upstream parser fails on leading `C. option text` for short responses. Of 3,023 fallback cases, 2,972 have a valid leading label followed by the exact corresponding option text (whitespace and terminal periods normalized). Answer-blind deterministic rescue recovers 1,214 correct answers. Raw generations remain unchanged.

| Full 10,500 test rows | Exact-option-label repair accuracy (%) | Remaining fallbacks counted wrong |
|---|---:|---:|
| Adaptive BARD | 43.1905 | 51 |
| Archived Greedy, oldcloud_20260919 | 43.3333 | 51 |

Separate audit protocol, not unchanged upstream evaluation. Both methods use the identical opt-in `--exact-option-label-repair` evaluator. Do not compare corrected BARD to an uncorrected Greedy score. Artifacts: `mmmu-candidate-exact-label-audit.json`, `mmmu-greedy-exact-label-audit.json`. Remaining 51 cases per arm still require review; no random guesses enter these scores.

Additional archived full10,500-ID baselines rescored with the same repair (no generation): VISTA43.4667, DoLa43.2000, PAI42.9714, AGLA42.4000, MedRAG40.9333, AvisC40.0952, VCD39.4286, ICD38.8857 percent. Per-method original answer path/hash and rescued IDs are recorded in `mmmu-METHOD-exact-label-audit.json`. These are archive comparisons, not proof of identical runtime configurations. All input ID sets exactly match the full manifest.

Residual51 BARD fallback review shows both harmless typography differences and label/body contradictions (e.g.3692.50 vs3592.50). They remain unparsed in this conservative audit; no truth-dependent override is applied.

PMC independent source-label/image audit: all33,430 candidate IDs agree with authoritative source and archived GT. With authoritative source options and existing parser, BARD17,281 correct, Greedy17,502; BARD fixes602 baseline errors but introduces823 errors. This decline is not explained by the MMMU parsing defect. Article-normalized PMC audit remains a separate scoring variant.

## Historical pre-correction MMMU and PMC-VQA audit (2026-09-22)

The MMMU 31.6286% below is retained only as the old-parser diagnostic, NOT the current corrected result. Use 43.1905% and the identically rescored baseline table above for the exact-option-label repair protocol. PMC-VQA remains unchanged.

| Dataset | Full test n | Adaptive BARD historical audit score (%) |
|---|---:|---:|
| PMC-VQA | 33,430 | 51.6901 |
| MMMU, all 30 subjects | 10,500 | 31.6286 |

PMC-VQA uses locally rescored v14 multiple-choice accuracy (Greedy 52.3482%). MMMU uses the pinned official parser with unparseable/random-fallback cases counted wrong: 3,023 fallbacks, parse rate 71.2095%, zero empty answers and one unfinished answer retained. The upstream random-fallback diagnostic is 40.00% and is NOT the primary table value. Exact MMMU IDs, prompts and image SHA256 were verified during export. See `mmmu-full-official-20260922.json`, `mmmu-full-answers-20260922.jsonl`, and `../pmcvqa-full-two-model-20260922.json`. No baseline inference was rerun.

## Newly completed PathVQA (2026-09-21)

All6719 unique official test IDs aligned against the authoritative manifest;
original answers retained in runs/huatuo-test-v1/pathvqa-full-export.json.
Zero empty answers;3 EOS-unfinished outputs retained and scored by content.
Uniform local parser v14 for candidate and existing full historical answers.

| Method | PathVQA6719 mixed CE accuracy/OE recall (%) |
|---|---:|
| Adaptive BARD |35.83|
| Greedy |38.66|
| MedRAG |39.65|
| PAI |37.23|
| AGLA |36.08|
| AvisC |34.48|
| VCD |33.58|
| ICD |31.86|
| DoLa |29.35|

Source:pathvqa-full-current-scores.json. This is a full-test result, replacing
the earlier5673-case34.84% interim result; no new baseline inference.
BARD remains below Greedy and MedRAG. No blanket absence-of-content-errors claim.

Independent repeat audit: all6,719 exported answer IDs exactly match `data/pathvqa/official_test_v1.json`; realignment and full rescoring reproduce every stored score field exactly. None of the3,357 OE references becomes empty after normalization. Example `pathvqa-test-00-000001` answers “They are not charged” against “positively charged”; `...000002` describes eosinophilia/inflammation rather than the reference “early (reversible) ischemic injury”. These are content errors, not empty-output parsing failures. Token recall can still credit overlapping words in a contradictory answer; preserve the requested metric rather than interpreting it as semantic correctness.

## Previously completed SLAKE and VQA-RAD

All2545 candidate cases are complete: VQA-RAD451 and SLAKE2094 (English1061, Chinese1033). Per user instruction the comparison below uses historical baselines only; no further baseline inference.

| Method | VQA-RAD451 | SLAKE2094 |
|---|---:|---:|
| **Adaptive BARD** | **63.06%** | **57.16%** |
| greedy | 62.82% | 54.69% |
| cve | 62.79% | — |
| pai | 60.40% | 52.73% |
| opera | 60.27% | — |
| agla | 58.01% | 54.42% |
| vista | 57.84% | 54.56% |
| avisc | 57.63% | 49.98% |
| vcd | 55.83% | 52.83% |
| icd | 52.37% | 51.52% |
| dola | 49.23% | 48.48% |
| medrag | 48.88% | 49.88% |

The scores use the existing frozen benchmark evaluator: **both VQA-RAD and SLAKE are sample-weighted mixed CE/OE scores** (closed-answer strict 0/1 correctness plus open-answer reference-token recall). SLAKE has 836 CE and 1,258 OE TEST questions; the frozen `full-test-historical-comparison.json` records those counts and the exact metric definition. Do not label either column pure accuracy or all-answer recall. Missing complete historical runs are shown as —, not replaced with partial scores. Historical model runtime differs; this is an archive comparison, not an isolated causal estimate of algorithm improvement.

| SLAKE subset | n | Adaptive BARD | Historical Greedy |
|---|---:|---:|---:|
| SLAKE-English | 1061 | 55.55% | 55.98% |
| SLAKE-Chinese | 1033 | 58.80% | 53.37% |

BARD ranks highest among the complete historical methods verified in the main table. The SLAKE result is driven by the Chinese split; English remains below historical Greedy. This is not a verified external SOTA claim. VQA-RAD BARD CLOSED251 accuracy77.6892%, OPEN200 recall44.7112%.

Full precision and per-arm evaluator fields: full-test-historical-comparison.json. Earlier same-runtime audit and case-level failure analysis are retained separately in vqa-final.json, english-diagnostics.json and the changed-scores artifacts, but are not the requested primary comparison. No TEST-based parameter tuning was performed.

All completed raw candidate records and expert masks/CAMs remain under runs/huatuo-test-v1. Every counted case has provenance.json; total2545 unique candidate IDs checked against2545 generalist reference records before offline scoring. All6987 native requests passed exact-key/cache-identity validation on both machines. See PROTOCOL.md for frozen routes, generation and provenance corrections.
