# LLaVA-Med candidate TEST results

2026-09-21T09:09:43.212653+00:00

Historical baselines only; no baseline reruns. SLAKE is complete. PathVQA is a completion-order subset and must not be treated as full TEST. Scores combine closed-answer correctness and open-answer reference-token recall.

## slake: 2094 samples

| Method | Score (%) |
|---|---:|
| Adaptive BARD | 35.8299 |
| greedy | 34.6813 |
| ICD | 33.2972 |
| opera | 35.9552 |
| MedRAG | 27.1863 |

Empty outputs: 0. Historical methods scored on exactly the same IDs.

## pathvqa: 789 samples

| Method | Score (%) |
|---|---:|
| Adaptive BARD | 31.2069 |
| greedy | 30.9273 |
| VCD | 29.9022 |
| ICD | 30.9798 |
| AGLA | 30.5189 |
| VISTA | 30.2776 |
| DoLa | 30.9273 |
| PAI | 30.6572 |
| avisc | 29.9097 |
| opera | 31.6490 |
| MedRAG | 31.2146 |

Empty outputs: 0. Historical methods scored on exactly the same IDs.

SLAKE candidate uses shared frozen Huatuo image-only native routing; historical runs have their archived configurations. This comparison does not isolate a causal algorithm effect. LLaVA-Med VQA-RAD is excluded at user request.
