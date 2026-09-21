# Frozen Adaptive BARD: mechanism audit

The tested algorithm is upstream `4d6416faab2d6191b55d9429dc3c014804fead0c`. Changes on the experiment branch preserve source split metadata and provide a cache-only receiver harness; `bard.py`, `bard_protocol.py`, the LLaVA adapter and the method configuration are unchanged.

## What changed mechanically

The receiver expresses heterogeneous evidence as a change in its own next-token distribution at an identical committed prefix. Segmentation branches retain their native spatial intervention, so evidence is not restricted to prose. Capabilities belonging to one declared fault group are grouped into a single receiver vote. Adaptive commit replaces the previous structural four-expert threshold: one delivered group anchors to the Generalist, two require unanimous candidate support, and three or more use geometric-median aggregation with bounded support.

These are meaningful changes relative to the old run. They do not establish that fault groups are statistically independent or that a non-corrupted medical specialist supplies correct patient-specific evidence. A literature retriever supplies background knowledge, not another independent reading of the patient image.

## Existing ideas and the possible research contribution

Geometric-median aggregation is established in [RFA](https://arxiv.org/abs/1912.13445). Isolate-then-aggregate inference is also established in [RobustRAG](https://arxiv.org/abs/2405.15556). Therefore neither component alone supports a novelty claim. The candidate contribution is their application to native heterogeneous medical evidence through branch-local receiver residuals, with an explicit incumbent and actual spatial intervention. A strong claim still needs controlled evidence that this construction preserves useful corrections while reducing harmful expert influence. This is a focused comparison, not an exhaustive novelty search.

## Attribution and threat-model limits

- In this frozen implementation, `joint_all` is semantic-only, while isolated branches can additionally receive native spatial evidence. Its comparison with `isolated_mean` changes both isolation and evidence channel; it is not a clean estimate of isolation alone. Joint token budgeting may also omit different evidence than isolated contexts.
- All experts condition the same Generalist. Different `fault_group` strings prevent duplicate votes but do not eliminate shared receiver errors or correlated specialist failures.
- `f < n/2` is a condition for the declared centralized aggregation setting, not a proof of clinical correctness or correct token selection. Actual expert support and the geometry of their residuals remain necessary empirical questions.
- Sign-reversal tests evaluate one specified residual corruption at fixed clean prefixes. They do not certify arbitrary malicious raw evidence, an adaptive attacker, or complete corrupted trajectories. The canary compares mean, geometric median and BARD on the same prefixes; it does not invent a comparable residual corruption for joint context.
- A higher clean-decision retention rate can be obtained simply by falling back to the Generalist, especially at n=1. Report n>=2 and n>=3 separately and inspect retained corrections, not only pooled retention.

## Direct early mechanism observation

On TRAIN case `vqarad-train-1088` (right-lung opacities/calcifications; reference “no”), the Generalist and isolated mean answered Yes, whereas geometric median and BARD answered No. At the first token, the delivered CheXagent and BiomedParse branches supported No with candidate-versus-baseline margins +0.6015625 and +0.7421875; MedCPT supported the opposing answer with margin -2.1875. Geometric median overcame that opposing influence. Removing retrieval in the same-prefix diagnostic also selects No. Thus this particular rescue is evidence for robustness against misleading retrieval, not evidence that retrieval supplied the missing fact or that bounded commit improves over geometric median.

When either visual residual was sign-reversed, this first-token rescue reverted to the baseline Yes. Reversing the retrieval residual retained No. This exposes the correction-retention tradeoff; it is not a failure of a clinical correctness guarantee, since no such guarantee exists and naturally mistaken specialists need not form a correct majority.

## Coverage observation before final scoring

The 32-row official TRAIN canary was selected by a fixed hash without selecting for model correctness. Planned coverage was 21 cases with two fault groups and 11 with at least three. However, BiomedParse returned `unknown_anatomy_or_sequence` on 20/32 requests. Final conclusions must use delivered receiver branches, not these planned counts. The imported official MedCPT chunk contains 851839 usable abstracts; no VQA answers enter the knowledge base or inference process.
