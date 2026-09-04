# Med-DEFER research design

## Why the old mechanism is now a baseline

The first 16-per-domain public run produced 6/32 correct for the medical generalist,
7/32 for GSCo-style context and routed score fusion, and 5/32 for MERIT, MERIT-FedDG,
wrong-route and shuffled-expert controls. Every method scored 0/16 on OCT. These results
do not validate expert-confirmed residual recovery: real, wrong and shuffled experts were
indistinguishable, and the OCT expert/task interface was not qualified.

Med-DEFER therefore does not assume that a generic hidden-layer residual is expert-specific.
It treats MERIT as an ablation and moves collaboration into the generation process.

## Research question

Can a medical generalist VLM reduce unsupported clinical claims by conditionally borrowing a
single capability from a qualified small model during decoding, while abstaining when that
expert is unreliable under unseen-domain shift?

This is narrower than a generic medical agent. The unit of routing is a clinical claim, the
action space is `NONE` plus registered capabilities, and the output remains authored by the
generalist VLM.

## Inference flow

1. The medical VLM begins autoregressive generation.
2. At the beginning of generation or after a clinical-claim delimiter, the controller measures
   normalized token uncertainty.
3. If the generalist is confident, choose `NONE` and continue without loading an expert.
4. Otherwise, filter expert cards by image modality and required capability.
5. Rank compatible experts by expected correction utility minus latency cost.
6. Load and call only the selected expert. Cache its result for that sample and claim.
7. Preserve its native mask, box, retrieved item or generated text, and map only supported
   concepts into a centered, norm-bounded evidence direction.
8. Recompute domain trust with the expert's own OOD and quality signals.
9. If trust remains sufficient, bias matching concept-phrase tokens until the claim ends.
10. Record a trace containing selection, utility, trust, gate and exact logit delta.

The current implementation is synchronous. The trace/cache boundary is intentionally compatible
with a later asynchronous path that speculates with the generalist and rolls back only the current
claim if expert evidence arrives before the claim is committed.

## Domain-generalization gate

For expert `e` on target study `x`, the source-only trust is

```text
T(e, x) = LCB_source(e)
          * CVaR_lower(source-domain performance)
          * exp(-temperature * OOD(e, x))
          * image_quality(x)
```

`LCB_source` is computed from federated aggregate correctness and calibration error. The lower
tail prevents a high mean on easy hospitals from hiding one failed source domain. OOD and quality
are label-free target-study signals. No target outcome is used to fit the gate.

The pre-call controller utility is

```text
U(e) = uncertainty * route_confidence * capability_match
       * T_pre(e, x) * expected_gain(e) - cost_weight * latency(e)
```

The controller selects `argmax({0, U(e)})`. After the call, the intervention gate also multiplies
expert confidence and the recomputed trust. The centered evidence vector is normalized and capped,
so multiplying raw expert logits cannot arbitrarily increase its influence.

## Expert contract

`NativeEvidence` supports five initial capability types:

| Capability | Native result retained | Initial language bridge |
|---|---|---|
| Classification | class/concept scores | signed concept phrase bias |
| Retrieval | retrieved cases or reports in provenance | grounded concept scores |
| Segmentation | masks | region-derived concept scores |
| Detection | boxes | object/finding concept scores |
| Generation | specialist text | extracted supported/contradicted concepts |

The bridge is explicit rather than pretending that tensors from unrelated architectures share a
logit space. New experts register an `ExpertCard` and one lazy provider; the controller itself has
no hard-coded model name. A spatial-to-token projector can be learned later, but it must be trained
and evaluated separately instead of being silently assumed.

## Training protocol

- Freeze the generalist and all experts for the first study.
- On source clients, calibrate per-expert correctness, calibration error and per-domain lower-tail
  performance. Share aggregate statistics only.
- Fit controller thresholds and expected gain on source validation clients.
- Construct synthetic/augmented pseudo-domains only from source data to stress the trust gate.
- Freeze all parameters before revealing the held-out target labels.
- Qualify each expert on its native task before allowing it into the pool. An expert such as the
  current OCT adapter with zero source accuracy is quarantined, not merely assigned a small weight.

## Required comparisons

1. Medical generalist alone.
2. Post-hoc specialist verification of a completed draft.
3. Dense all-expert precomputation.
4. GSCo-style specialist context and direct routed fusion.
5. Legacy MERIT and MERIT-FedDG.
6. Med-DEFER without domain trust.
7. Med-DEFER with mean-domain rather than lower-tail trust.
8. Full Med-DEFER.
9. Wrong-capability, wrong-route and shuffled-evidence controls.
10. Oracle modality/capability routing, reported separately.

Report task accuracy plus claim-level factuality, ECE/selective risk, worst-domain performance,
expert call rate, latency, memory, rescued/harmed claims and expert-specific counterfactual effects.
The existing `hallucination_rate = 1 - accuracy` is only a closed-set error proxy and must not be
presented as an open-ended hallucination metric.

## Falsification criteria

Stop or redesign the mechanism if correct experts do not beat shuffled/wrong evidence, the benefit
vanishes after controlling for output length, domain trust only copies the generalist without
improving selective risk, or expert qualification fails on its native source task. Real hospital
or scanner domains are required before making a cross-hospital FedDG claim.

## Implemented versus next

Implemented now: typed heterogeneous evidence, `NONE`-or-one controller, source-only lower-tail
trust, lazy loading, per-claim cache, bounded concept guidance, Transformers generation hook,
deterministic end-to-end study and legacy baselines.

Next empirical work: repair/replace the OCT expert-task mapping; add native segmentation and
detection adapters; define claim extraction and factuality annotations; run real MedVL generation;
then add asynchronous execution and claim-local rollback only if synchronous latency is limiting.
