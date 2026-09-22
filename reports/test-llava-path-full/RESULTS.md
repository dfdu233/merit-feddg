# LLaVA-Med PathVQA full TEST verifier experiment

All 6,719 official TEST IDs were processed on host GPU1 using the frozen exploratory postprocessing protocol. Historical LLaVA-Med Greedy and Adaptive BARD answers were reused; no answer generation was rerun. Current v15 source-typed mixed VQA scorer was applied identically to every arm: closed-ended exact correctness for 3,362 cases, open-ended answer-token recall for 3,357 cases. The table's mixed score is neither plain accuracy nor a qualified MERIT-Tx v3 result.

| Arm | Mixed score % | Accepted BARD changes | Improved cases vs Greedy | Harmed cases vs Greedy | Paired image-cluster 95% interval vs Greedy, pp |
|---|---:|---:|---:|---:|---:|
| Greedy Baseline | 31.6663 | 0 | 0 | 0 | — |
| Original BARD, rescored v15 | 30.8774 | 3,989 | 299 | 350 | [-1.3534, -0.2377] |
| BiomedCLIP filter | 31.3281 | 1,358 | 105 | 128 | [-0.7115, +0.0326] |
| CONCH filter | 31.1434 | 1,495 | 115 | 146 | [-0.9202, -0.1411] |
| PLIP filter | 32.1441 | 1,161 | 82 | 45 | [+0.2014, +0.7457] |
| Joint filter | 31.3472 | 506 | 40 | 62 | [-0.5844, -0.0466] |

PLIP's observed gain over Greedy is +0.4778 percentage points, concentrated in closed questions: closed accuracy 56.9601% versus Greedy 55.9786%; open recall 7.2910% versus Greedy 7.3178%. On the 6,463 cases outside the prior Huatuo pilot, PLIP scores 32.1713% versus Greedy 31.7046%. Image-cluster bootstrap uses 858 image SHA groups, 3,000 replicates, seed 20260922. The interval is exploratory and not corrected for looking at multiple arms or prior TEST results; it does not validate source qualification or a causal PLIP contribution.

The frozen rule accepts a BARD answer if a verifier's real-image candidate-vs-incumbent margin exceeds the median margin on four matched source controls (D>0). Joint accepts at least one D>0 and no available D<0. Unsupported modalities or missing/invalid evidence fall back to Greedy. PLIP is pathology-only; 679 PLIP observations were rejected for overlength instead of truncating clinical claims. BiomedCLIP lacked a fourth CXR control on 232 cases. Historical proposer independence is not established, so this is an unqualified TEST filter ablation.

The previous historical Adaptive BARD report gave 30.5946% under the earlier v13 evaluator. Rescoring the exact same 6,719 exported BARD texts with the current v15 evaluator yields 30.8774%; Greedy remains 31.6663%. The v13 number must not be compared directly with the v15 filtered arms. The current evaluator protocol and data are recorded in `results.json`.

Completeness: 6,719 unique input IDs, 6,719 atomic case files, 858 unique images with SHA-256 matching all manifest hashes, worker exit 0. The final score uses the pinned official TEST reference only after verifier inference. Source answers, raw effects and failure reasons remain under `runs/test-llava-path-full`; no thresholds or expert choices were selected using TEST labels.
