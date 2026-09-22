# Huatuo PathVQA TEST exploratory postprocessing

Completed256fixed pathology-routed TEST cases (205images) on hostGPU1. Native verification ran without labels; original frozen historical answers reused. See PROTOCOL.md for scope, budget mismatch and missing proposal-provenance limitations. Not full6719TEST, not qualified v3, not a new independent benchmark confirmation.

| Arm | Mixed VQA score % | Accepted | Score improved / harmed vs historical Greedy |
|---|---:|---:|---|
| Historical Greedy |38.4589|0|0 /0|
| Historical BARD |33.2054|230|25 /36|
| BiomedCLIP filter |36.9241|64|8 /10|
| CONCH filter |38.4936|84|14 /11|
| PLIP filter |39.6508|39|7 /2|
| All-three filter |38.6689|26|6 /4|

Metric is existing sample-weighted CE correctness/OE answer-token recall, not plain accuracy. Open scores do not establish clinical correctness. Every row uses identical256IDs. No target labels in inference or threshold choices.

Image-cluster bootstrap3000 draws, seed20260922: PLIP gain95%interval[-.3021,+2.9338]percentage points; joint[-1.5749,+2.0456]. Both include zero; no superiority claim. PLIP is the largest observed score among predeclared arms, not a source-qualified selected policy. Differences in missing evidence and model coverage must not be confused with reliable contradiction.

No expert or threshold was selected for deployment from this TEST result. All original arms retained. Formal v3 with zero qualified authorities still preserves its incumbent. New MUSK-dependent v3 remains untested pending authorization.
