# Claim-unit result

Actual hostGPU1 run complete, matched full-answer and leading-polarity verification,81 Huatuo source-Q closed cases. Unqualified exploratory ablation, not official TEST/v3 deployment. No MUSK or candidate regeneration. See PROTOCOL.md for frozen decisions and controls.

| Rule | Full-answer correct /81 | Short-proposition correct /81 | Full corrected / broken | Short corrected / broken |
|---|---:|---:|---|---|
| Generalist |51|51|0/0|0/0|
| BARD proposal |50|50|7/8|7/8|
| BiomedCLIP |52|49|2/1|3/5|
| CONCH |49|52|4/6|4/3|
| PLIP |52|51|1/0|3/3|
| All three |52|50|2/1|0/1|

Generalist62.96%, raw candidate61.73%. Full all-three64.20%, short all-three61.73%. Tiny one-case gains are not evidence of reliable improvement or grounds for post hoc choosing an expert. No statistical superiority established. Missing expert evidence abstains; availability differs by text length. PLIP full accepts1flip, short accepts6; improved coverage does not mean safer edits.

Conclusion: the hypothesis that leading-polarity proposition verification consistently improves correction/harm separation is unsupported here. CONCH improves, BiomedCLIP worsens, PLIP expands coverage with equal3helps/3harms, and consensus in the short arm selects only1 harmful flip. This exposes sensitivity of contrastive verifier decisions to claim representation and correlated false support. No threshold tuning or deployment follows this experiment. Any next mechanism should test calibrated discrimination/negation sensitivity independently, rather than assume a shorter proposition is more trustworthy.
