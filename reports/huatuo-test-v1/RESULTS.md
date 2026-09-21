# Huatuo Adaptive BARD — complete official TEST

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

The scores use the existing frozen benchmark evaluator: VQA-RAD is sample-weighted closed correctness + open answer-token recall; SLAKE follows the existing all-short-answer contract and is answer-token recall. Do not label these two table columns uniformly as pure accuracy. Missing complete historical runs are shown as —, not replaced with partial scores. Historical model runtime differs; this is an archive comparison, not an isolated causal estimate of algorithm improvement.

| SLAKE subset | n | Adaptive BARD | Historical Greedy |
|---|---:|---:|---:|
| SLAKE-English | 1061 | 55.55% | 55.98% |
| SLAKE-Chinese | 1033 | 58.80% | 53.37% |

BARD ranks highest among the complete historical methods verified in the main table. The SLAKE result is driven by the Chinese split; English remains below historical Greedy. This is not a verified external SOTA claim. VQA-RAD BARD CLOSED251 accuracy77.6892%, OPEN200 recall44.7112%.

Full precision and per-arm evaluator fields: full-test-historical-comparison.json. Earlier same-runtime audit and case-level failure analysis are retained separately in vqa-final.json, english-diagnostics.json and the changed-scores artifacts, but are not the requested primary comparison. No TEST-based parameter tuning was performed.

All completed raw candidate records and expert masks/CAMs remain under runs/huatuo-test-v1. Every counted case has provenance.json; total2545 unique candidate IDs checked against2545 generalist reference records before offline scoring. All6987 native requests passed exact-key/cache-identity validation on both machines. See PROTOCOL.md for frozen routes, generation and provenance corrections.
