# PathVQA expert removal: completed progressive experiment

LLaVA-Med, official TEST, 2026-09-21. **Removing CONCH gives a small, uncertain improvement over the original method, but does not surpass matched Greedy. Stop expanding the blanket-removal policy; retain the original default.** No expert is established as consistently harmful.

## Protocol and coverage

Original full-test audit: 6,719 cases. The exact CONCH + BiomedParse + MedCPT admitted-group cohort contains 3,502 cases. Select a fixed SHA256 ordering without filtering by correctness. Screen three single-group removals on 64 cases, expand the best arm on 192 disjoint cases, then confirm the unchanged arm on 768 further cases: **1,024 unique cases complete**. This is TEST-guided exploratory ablation, not an independent held-out result or a new full-test score.

Freeze original descriptor selection, then omit the chosen group before acquisition; never refill the budget. Remaining prompts, requests, routing, model configuration, and 64-token limit are unchanged. Empty-EOS repair is OFF. Historical baseline predictions are rescored on identical IDs with the same local evaluator; no baselines regenerated. Host candidate replay control reproduced 8/8 original token sequences. Different existing Torch environments remain a comparison limitation.

Metric: `medheval-decoded-eval-v14-explicit-binary-conclusion`, sample-weighted closed-question accuracy plus open-question answer-token recall, expressed as a percentage. This is a mixed score, not pure accuracy.

## Progressive results

| Method | First 64 | New 192 | New 768 | Combined 1,024 |
|---|---:|---:|---:|---:|
| Original BARD | 35.72 | 32.19 | 32.57 | 32.70 |
| Remove CONCH | 38.32 | 32.43 | 33.28 | 33.43 |
| Historical Greedy | 33.32 | 32.52 | 34.54 | 34.09 |
| Historical OPERA | 41.13 | 35.87 | 37.34 | 37.30 |

The other screened removals were weaker: remove BiomedParse 33.63; remove MedCPT 33.44, versus original 35.72 on those same 64 cases. Neither advanced. This limited screen does not establish that these experts are universally beneficial.

All available matched historical comparisons on the same 1,024 IDs:

| Method | Mixed score (%) |
|---|---:|
| OPERA | 37.30 |
| AGLA | 34.67 |
| PAI | 34.47 |
| AVISC | 34.18 |
| DoLa | 34.14 |
| Greedy | 34.09 |
| ICD | 34.01 |
| VISTA | 33.90 |
| VCD | 33.59 |
| MedRAG | 33.48 |
| Remove CONCH | 33.43 |
| Original BARD | 32.70 |

## Paired evidence and uncertainty

Across 1,024 cases, removing CONCH raises per-case score on 55, lowers it on 48, and leaves 921 equal versus original BARD. Against Greedy: 45 higher, 51 lower, 928 equal. These are mixed-score comparisons, not all binary correct/incorrect flips. On the disjoint 768 confirmation cases: 40 higher/33 lower versus original, but 32 higher/41 lower versus Greedy.

Image-cluster bootstrap (406 images, 3,000 resamples, seed 20260921): the combined improvement over original is 0.73 percentage points, with a descriptive 95% interval of [-0.97, 2.39]. The difference from Greedy is -0.65 points, interval [-2.06, 0.80]. On new768 alone, the corresponding intervals are [-1.21, 2.62] and [-3.05, 0.46]. All include zero. Combined intervals include the arm-selection screen and are not post-selection-adjusted inferential guarantees.

Combined set: 583 closed questions and 441 open questions. Remove-CONCH closed accuracy 54.55%, open answer-token recall 5.52%; original 53.34%/5.41%; Greedy 55.57%/5.68%. Zero empty outputs; 94 outputs reached the unchanged token budget without finishing. All outputs remain in scoring.

## What the expert evidence suggests

The current CONCH adapter compares the whole image against ten fixed tissue categories. Native records explicitly say `query_used: false` and `score_semantics: relative_similarity`; they do not claim diagnostic probabilities. This evidence can be poorly matched to questions about specific diseases, structures, or gross images. The likely issue is relevance and granularity of this adapter's evidence, rather than proof that the CONCH model itself is inherently harmful.

Examples retained in `conch-evidence-examples.json`: for `pathvqa-test-00-000827`, a coronary/atherosclerosis/thrombus question receives top tissue similarities for adipose and muscle; removing CONCH changes a wrong No to a correct Yes. For `pathvqa-test-01-001390`, removal changes a refusal to the correct Yes. Conversely, on `pathvqa-test-01-002033`, removal changes a correct No to a wrong Yes. Examples are illustrative, selected after scoring, and do not establish the mechanism alone.

Removal also changes the current BARD decision rule from three-expert robust aggregation to two-expert unanimous commit. Consequently, the experiment measures the end-to-end effect of removing the branch; it does not isolate expert factual quality from changed consensus requirements.

## Decision and artifacts

Do not globally disable or replace CONCH based on this result. The gain is small and uncertain, and every listed historical baseline remains numerically higher on the combined subset. No replacement model has been tested or claimed beneficial. A future mechanism test would need to separate question–evidence relevance from expert count and consensus effects, with policy selection outside this TEST subset.

All ablation workers finished and their GPUs were released; other benchmark jobs were preserved. Original full-test outputs and all failed/partial attempts remain intact. Reports: `screen64-results.json`, `expansion256-results.json`, `confirmation1024-results.json`, `confirmation1024-uncertainty.json`, and `confirmation1024-provenance-audit.json`. Exact IDs, decoded outputs, token IDs, and per-case scores are under `runs/llava-test-v1/pathvqa-expert-ablation/`; native evidence remains in its existing cache. All 768 confirmation cases passed the provenance audit: identical receiver configuration, no labels loaded, EOS repair off, CONCH only removed, no refill, and validated cache identity.
