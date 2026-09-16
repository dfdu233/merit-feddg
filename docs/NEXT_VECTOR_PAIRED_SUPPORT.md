# Next Vector after the authority TRAIN pilot: matched free-text support

## Evidence update

The completed 12-case TRAIN pilot does not demonstrate benefit from authority projection. Historical incumbent and base-pool reranking score 11/12; authority projection also scores 11/12 and changes the base-pool selected text in 0/12 cases. Text-conditioned reranking falls to 8/12. Most importantly, the finite-pool projection is feasible in only 1/12 cases because 11/12 historical candidate pools do not contain both semantic sides of the specialist-authorized variable.

This means the current experiment did not seriously test the central authority hypothesis. It mostly tested a projection over unsupported finite candidate sets. Scaling the same protocol would spend more compute without resolving that confound.

## Bit Flip status

The proposed bit remains:

> move from **how much an expert should be trusted globally** to **which semantic variable the expert is authorized to change**.

The latest result neither validates nor falsifies that bit. The projection rarely had support on which to act. Its lack of benefit therefore shifts the immediate risk from the projection formula to the representation/support substrate.

The next experiment must not present paired candidate generation itself as novel. Contrast Sets, CheckList, Polyjuice, FUDGE and GeDi already establish minimal counterfactual testing and attribute-controlled generation as existing ideas. We use those ideas as experimental machinery to test the authority hypothesis.

## Current Vector

> **When the free-text answer space contains a matched pair that differs only in the specialist-authorized claim, does capability-bounded projection produce an identifiable and useful intervention beyond base reranking and ordinary text conditioning?**

This is now the riskiest unanswered question because failure would make further work on the current authority projection unjustified, while success would isolate candidate support as the previous bottleneck.

## Core

- frozen LLaVA-Med actor;
- one frozen XRV classifier and the same three native findings;
- the incumbent free-text answer and its non-target residual text;
- one explicitly represented target claim state per case;
- matched present/absent free-text support;
- base sequence scoring and the existing minimum-change authority projection;
- TRAIN-only mechanism experiment with references loaded only by the offline evaluator.

## Periphery frozen

Do not add or modify:

- segmentation;
- retrieval/RAG;
- Agent planning;
- uncertainty weighting;
- visual or language verifiers;
- additional experts;
- full VQA-RAD/SLAKE runs;
- test-set parameter selection;
- new calibration models.

## Minimum discriminating experiment

### Case selection

Use TRAIN only. Select cases from the existing whole-image finding grammar without reading references. To make the mechanism experiment informative, form a frozen disagreement subset using only model-side quantities: the base mapped claim state and the XRV source decision coordinate. This subset is a stress test of intervention behavior, not a population accuracy estimate. Preserve a separate unselected eligible set for later validation if the mechanism survives.

### Matched support construction

For each selected case construct two complete natural-language candidates:

- `y_present`: expresses the target finding as present;
- `y_absent`: expresses the same target finding as absent.

They must share the same non-target residual text exactly. Only the target claim carrier may differ. Construction must not inspect the reference answer.

The first experiment should prefer deterministic, auditable claim carriers over a learned rewriter. A learned/free-form counterfactual generator would add a second unresolved problem: whether the generator changed other clinical content.

Existing historical free-text candidates may remain in the pool, but the matched pair supplies guaranteed support for the controlled variable.

### Arms

1. `incumbent`: unchanged historical free-text answer.
2. `balanced_base`: matched candidate pool, ranked only by the base LLaVA-Med sequence scores.
3. `balanced_text_conditioned`: same pool ranked after ordinary focused XRV text is added.
4. `balanced_authority_raw`: same base pool plus authority projection using the raw source coordinate.
5. `balanced_authority_op`: same base pool plus authority projection using the published XRV operating-point coordinate.

Do not add adaptive alpha, ACD, a verifier, or a new planner to this Vector.

### Primary mechanism metrics

The experiment is successful only if it establishes mechanism behavior, not merely a score increase.

Report:

- paired-support feasibility rate;
- fraction of source/base disagreement cases where authority changes the selected target claim;
- direction compliance: whether the authority posterior moves the target claim mass toward the declared source coordinate;
- exact non-target residual invariance for the matched pair;
- selection of historical `unknown` candidates versus the matched pair;
- base-rerank versus authority choice changes.

After inference is frozen, load references and report rescue/harm and the existing scorer as secondary outcome metrics.

### Strong controls

- `balanced_base` separates the effect of adding paired candidates from the effect of authority projection.
- `balanced_text_conditioned` tests whether ordinary language transport can use the same support.
- The present/absent candidates must be matched in wording and residual content so style does not become the hidden treatment.
- If desired for offline interpretation only, an oracle that chooses the reference-consistent side may be reported as an upper bound; it must never enter inference or parameter selection.

## Stop rules

### Stop authority projection

If balanced support is available but authority still almost never changes the base decision on source/base disagreement cases, the current projection is not a useful intervention substrate. Do not add more experts or tune projection strengths; re-vector away from this mechanism.

### Re-vector to specialist reliability

If authority reliably changes the target claim in the source direction but task rescue/harm is poor, the next risk is source expert correctness/domain transfer, not semantic authority.

### Re-vector to claim representation

If matched support cannot preserve the non-target residual or claim mapping remains frequently ambiguous, the next risk is claim representation/parsing. Do not blame the projection.

### Advance authority hypothesis

Only if authority changes the authorized claim when appropriate, preserves the non-target residual, and improves the rescue/harm tradeoff beyond `balanced_base` and `balanced_text_conditioned` should the next Vector ask whether the same mechanism transfers to a second capability type or unseen specialist.

## Novelty collision audit

- **Contrast Sets** and **CheckList**: minimal behavioral contrasts are established diagnostic methodology.
- **Polyjuice**: controlled counterfactual text generation is established; paired text generation is not the contribution.
- **FUDGE / GeDi**: attribute-controlled generation is established, though they use learned/generative discriminators rather than frozen medical specialist-native authority.
- **Posterior projection / constrained inference**: the mathematical projection itself is established.

Therefore a future novelty claim must remain at the level of a general, transferable **specialist-native capability boundary for free-text VLM collaboration**, not at the level of counterfactual pair construction or KL projection alone.

## Velocity criterion

This iteration has high velocity only if it resolves the support confound. At the end we must be able to say one of the following with evidence:

1. paired free-text support unlocks a distinct authority intervention;
2. paired support exists but authority still adds no value;
3. the dominant unresolved problem is source reliability;
4. the dominant unresolved problem is claim representation.

Producing more code, more candidate generators, or a larger benchmark without resolving one of these branches does not count as progress on the current Vector.
