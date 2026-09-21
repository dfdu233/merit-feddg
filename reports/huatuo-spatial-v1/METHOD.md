# Huatuo Adaptive BARD: frozen method and runtime validation

Code base: implementation/huatuo-adaptive-bard-spatial-v1 at 5b15b16b2ad7426482e44a2584c51c5d7851bfd5. This experiment does not modify BARD, expert selection, the spatial operator, or its thresholds.

The spatial operator averages native receiver patch features within each expert mask, then blends region means back only at covered patches with unit original-token mass. It adds no learned parameters or independent diagnostic knowledge; it reallocates/smooths the receiver's existing visual features. Segmentation labels remain semantic evidence. Each fault-group gets an isolated receiver branch at the same committed prefix. Base-centered log-probability residuals are aggregated by mean/geometric median. Adaptive BARD uses exact baseline for one node, unanimous commit for two, and centralized bounded-fault consensus for three or more. Distinct models are fault groups, not necessarily statistically independent medical opinions.

Potential mechanism: spatially grounded expert masks can change visual feature use; robust aggregation can suppress a misleading branch. Limitations: semantic evidence and spatial input change together; joint_all is semantic-only whereas isolated arms can receive spatial tensors, so joint-versus-isolated is not a pure isolation ablation. Median robustness does not certify medical correctness or independence of honest experts. A strong receiver has less rescue headroom and more opportunity for harmful changes. One-node fallback must not be counted as demonstrated adversarial robustness.

## Real runtime corrections (no algorithm tuning)

- Historical generation is repetition_penalty=1.2, min_new_tokens=1; upstream config had 1.0/0. Answer/block budgets remain 1024.
- Explicit eager attention reproduces historical cloud PyTorch 2.8 outputs. Auto-SDPA diverges by one token in the 92-token source record. Host PyTorch 2.0.1 eager also diverges; re-running the original legacy adapter on that host gives the same divergence. Historical token arrays were never altered. Only cloud eager has passed all three frozen cases so far.
- Permit min_new_tokens == max_new_tokens for one-token score probes.
- Technical canary compares shared finite score support, excluding the intentionally masked EOS. It still rejects mismatched support or zero/nonfinite spatial changes.
- Disabled vector-gate probe setting corrected from invalid 1024 to valid 64. The gate remains off; this is not the answer budget.
- Preserve source domain/role during manifest loading instead of silently assigning official-test/target.

Both original host GPU1 and cloud 5090 exercised the real spatial projector hook. Cloud eager passes all three frozen historical source sequences (4, 13, 92 tokens) and the spatial canary. Host results are excluded from paired efficacy until runtime parity is established. 32 targeted CPU tests pass, including actual config construction and the observed single-step budget failure. No main merge.

## Evaluation contract

Exactly the same 16 SLAKE + 16 VQA-RAD official TRAIN/source cases as the completed LLaVA canary, reordered SLAKE-first without selecting by correctness. Reuse native expert outputs only: unchanged expert specifications, donor identity, exact request keys and image hashes. Frozen image-only LLaVA routes are deliberately shared across receivers; this is not a test of Huatuo routing. No LLaVA Generalist outputs reused. Every Huatuo arm runs afresh with the same prompt and 1024-token budget. Reference labels are read only offline.

The existing anchor-ce-v1 prompt for slake-train-4466 incorrectly requests Yes/No for a nonbinary question. It is preserved for exact queue/prompt comparison and must be separately reported in sensitivity analysis. Mixed CE accuracy/OE answer-token recall is not plain accuracy. These source canaries cannot rank SOTA on official TEST. Stress uses only the first eight clean BARD prefixes, with residual sign reversal; it is not full corrupted free generation.
