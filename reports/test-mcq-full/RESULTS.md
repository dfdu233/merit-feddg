# Huatuo full PMC-VQA and MMMU exploratory verification

Completed on host GPU1 only. PMC-VQA 33,430 + MMMU 10,500 = 43,930 exact unique cases; both workers exited 0. The 1,024 pilot outputs were reused byte-for-byte after exact input-row comparison. Existing Generalist and BARD answers were reused; this is native verifier postprocessing, not fresh answer generation.

| Arm | PMC-VQA accuracy % | MMMU accuracy %, medical subject scope |
|---|---:|---:|
| Historical Generalist | 52.3542 | 43.3333 |
| Historical BARD | 51.6931 | 43.1905 |
| BiomedCLIP filter | 52.3871 | 43.3048 |
| CONCH filter | 52.3422 | 43.3714 |
| PLIP filter | 52.3063 | 43.3333 |
| Joint filter | 52.4170 | 43.3048 |

Frozen arms were all reported without selecting a winner or tuning thresholds on TEST. Single verifiers accept D>0; joint accepts at least one positive and no available negative. Missing experts abstain. Four fixed matched source controls are required. CONCH/PLIP only apply to pathology; joint does not imply three available independent votes.

## pmcvqa_original_route

| Arm | Accepted changes | Corrected | Broken | New-only n | New-only accuracy % |
|---|---:|---:|---:|---:|---:|
| Historical Generalist | 0 | 0 | 0 | 32918 | 52.3543 |
| Historical BARD | 4390 | 602 | 823 | 32918 | 51.7042 |
| BiomedCLIP filter | 458 | 161 | 150 | 32918 | 52.3908 |
| CONCH filter | 344 | 114 | 118 | 32918 | 52.3483 |
| PLIP filter | 348 | 106 | 122 | 32918 | 52.3118 |
| Joint filter | 259 | 98 | 77 | 32918 | 52.4212 |
## mmmu_original_route

| Arm | Accepted changes | Corrected | Broken | New-only n | New-only accuracy % |
|---|---:|---:|---:|---:|---:|
| Historical Generalist | 0 | 0 | 0 | 9988 | 43.3520 |
| Historical BARD | 1222 | 155 | 170 | 9988 | 43.2119 |
| BiomedCLIP filter | 89 | 27 | 30 | 9988 | 43.3020 |
| CONCH filter | 11 | 6 | 1 | 9988 | 43.4021 |
| PLIP filter | 18 | 5 | 8 | 9988 | 43.3220 |
| Joint filter | 77 | 23 | 26 | 9988 | 43.3120 |
## mmmu_medical_subject_scope

| Arm | Accepted changes | Corrected | Broken | New-only n | New-only accuracy % |
|---|---:|---:|---:|---:|---:|
| Historical Generalist | 0 | 0 | 0 | 9988 | 43.3520 |
| Historical BARD | 1222 | 155 | 170 | 9988 | 43.2119 |
| BiomedCLIP filter | 18 | 5 | 8 | 9988 | 43.3120 |
| CONCH filter | 6 | 4 | 0 | 9988 | 43.3921 |
| PLIP filter | 11 | 4 | 4 | 9988 | 43.3420 |
| Joint filter | 10 | 2 | 5 | 9988 | 43.3220 |

PMC joint corrects 98 and breaks 77 cases: only 21 net additional correct answers. MMMU scoped joint corrects 2 and breaks 5. These small, inconsistent results do not establish superiority; no statistical significance claim is made. Pilot CONCH improvement did not transfer to full PMC.

Evaluation: PMC uses pinned existing multiple-choice correctness; MMMU uses pinned official evaluator with exact-option-label repair and strict no-random-fallback scoring. All arms use identical full denominators. MMMU medical subject scope permits edits only in Basic Medical Science, Clinical Medicine, Diagnostics and Laboratory Medicine, Pharmacy, Public Health; all other cases retain baseline. This label-free guard was introduced after pilot route audit and frozen before full expansion. Original-route results remain diagnostic because old routes misclassify many nonmedical images as MRI/CT. Subject membership alone does not validate image applicability.

Limitations: current source qualification failed, so these are unqualified exploratory TEST ablations, not formal qualified MERIT-Tx v3 results. Historical proposer independence is unverified. Baselines were not regenerated; MMMU uses the frozen oldcloud20260919 export, not an asserted latest repaired baseline. Strict option mapping abstains on 3,716 PMC and 4,554 MMMU pairs. No target labels entered GPU worker inputs; references were used only for offline scoring. No SOTA claim.

Coordination: shared STATUS.md notice in /home/dbw/merit-feddg-huatuo-spatial reserved these ranges and preserved the other session’s OmniMedVQA producer/cloud tasks. No direct cross-thread acknowledgment was available. At completion host cache PID 4100252 remained running; GPU1 memory was 5,090 MiB after our workers exited.

Reproduction: prepare.py, score.py, results.json and completion-audit.json in this directory; raw inputs/cases/logs under runs/test-mcq-full. Evaluation requires PYTHONPATH=/home/dbw/ANCHOR and the huatuo Python environment.
