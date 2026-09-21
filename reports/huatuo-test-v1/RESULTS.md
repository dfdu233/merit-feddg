# Huatuo Adaptive BARD: official TEST interim results

This is an ongoing full-TEST experiment, not a completed full-dataset claim. Frozen methods and prompt/model generation contract are documented in PROTOCOL.md. All6987 native expert requests passed exact identity/key validation on host and cloud. Baseline completed2545 cases. VQA has been reprioritized ahead of remaining SLAKE Chinese, without changing any predictions, labels or policy parameters.

## Complete SLAKE English TEST (1061 questions / 96 images)

| Method | Existing benchmark answer-token recall (%) |
|---|---:|
| Fresh Generalist, matched runtime | 56.1603 |
| Adaptive BARD | 55.5540 |
| Archived Greedy | 55.9789 |
| Archived PAI | 54.7401 |
| Archived AVISC | 53.0850 |
| Archived DoLa | 51.6121 |
| Archived VISTA | 51.1990 |
| Archived MedRAG | 46.1781 |

Archived methods are reference-only: identical cases and scoring, but numerical model runtime is not identical. Archived OPERA covers245/2094 SLAKE cases and is omitted rather than assigned a full-dataset score. These are not external SOTA comparisons. Recall is not plain accuracy.

Matched delta -0.6063 percentage points. Image-cluster bootstrap95% interval [-2.3231,+1.1683] points (10000 resamples, seed20260921), spanning zero.48 questions improve in score,46 worsen,967 tie. Native CLOSED416: Generalist74.7596%, BARD70.9135%; native OPEN645: reference-token recall44.1645%→45.6477%. CLOSED degradation outweighs OPEN improvement.

Structural fallback in528/1061 cases;226 cases contain at least one committed non-Generalist token. Expert branch counts: n1=528,n2=172,n3=18,n4=316,n5=27. Agreement or fallback cannot be interpreted as new clinical evidence. Among all scored harms,25 are exact No→Yes flips, versus10 such beneficial flips. Harms occur most often in Organ23 and Position15 categories. This suggests excess positive evidence can be a useful failure hypothesis, but the current observation does not identify a causal expert. Some measured gains are lexical: e.g. Lungs→Lung matches reference Lung. Therefore these metrics alone do not establish deeper visual reasoning improvements. Per-case score changes are retained in english-changed-scores.json.

## VQA-RAD — COMPLETE451/451

| Metric | Fresh Generalist | Adaptive BARD |
|---|---:|---:|
| Mixed CE-accuracy/OE-answer-recall | 63.8910% | 63.0648% |
| CLOSED251 accuracy | 77.2908% | 77.6892% |
| OPEN200 answer-token recall | 47.0743% | 44.7112% |

Matched delta -0.8262 percentage points.9 questions improve,13 worsen,429 tie. Paired bootstrap over203 image clusters (10000 resamples, seed20260921)95% interval [-2.4990,+0.7503] points. No demonstrated overall gain. CLOSED has one net extra correct answer; OPEN loses2.3631 points. Full per-method scores including11 archived baselines are in vqa-final.json; archive numerical runtime differs, so primary causal comparison remains the fresh Generalist. Archived Greedy62.8193%, CVE62.7864%, PAI60.4039%, OPERA60.2677%, AGLA58.0094%, VISTA57.8355%, AVISC57.6270%, VCD55.8293%, ICD52.3685%, DoLa49.2296%, MedRAG48.8819%. Being slightly above archived Greedy does not establish improvement over the matched current baseline. Per-case changes: vqa-changed-scores.json.

Raw native evidence and all masks/CAMs remain under runs/huatuo-test-v1/expert-cache. Cloud prediction files are retained by batch; count a case only when its provenance.json exists. No incomplete prediction is interpreted as fallback.
