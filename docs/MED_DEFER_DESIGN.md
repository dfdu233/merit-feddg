# Med-DEFER v0.5 research design

## Motivation from the failed experiment

The earlier 16-per-domain proxy run produced 6/32 correct for the medical
generalist, 7/32 for direct specialist fusion and 5/32 for MERIT/MERIT-FedDG.
MERIT tied wrong-route and shuffled-expert controls, every method scored 0/16
on OCT, and the apparent cached Med-DEFER gain (6/32 to 8/32) did not transfer
to two live pathology canaries. One live error was highly confident and occurred
before CONCH was called.

The redesign therefore makes four falsifiable changes:

1. yes/no proxy VQA is replaced by a real six-class tissue task;
2. hash partitions are replaced by named medical-center domains;
3. a qualified expert is considered before the first answer even for a
   high-confidence generalist;
4. OOD is computed from frozen source features rather than `1-route_confidence`.

No learned binary `P(generalist error)` estimator is used.

## Research hypothesis

A medical generalist can make a more domain-robust first clinical decision by
allowing one task-qualified specialist to provide bounded semantic evidence
before that decision is committed, while a two-stage source-only domain gate
rejects specialists outside their validated operating region.

This is not a generic agent system: the action space is `NONE` plus registered
expert capabilities, only one compatible expert is invoked, and the medical
VLM candidate likelihood remains the base decision score.

## Stage 1: real multiclass clinical decision

The initial mechanism study uses `bifold-pathomics/PathoROB-tolkach_esca` with a custom LOCO
protocol (not the official APD leaderboard protocol):

- 16,300 real 256×256 histopathology patches;
- six biological tissue classes;
- four explicit medical-center domains;
- `slide_id` grouping for leakage checks;
- four leave-one-medical-center-out evaluations.

The pilot samples a fixed number per medical center by hashing center, slide and
patch identity. It never reads target class labels. The six answer classes are
the frozen official PathoROB task taxonomy, not a vocabulary inferred from a
held-out fold.

For candidate `c`, the generalist base score is its true teacher-forced,
length-normalized sequence log-likelihood:

```text
g(c) = mean_t log p_generalist(c_t | image, question, c_<t)
```

The specialist produces semantic class evidence `s(c)`. Its calibrated,
centered direction is normalized and bounded:

```text
d = normalize(softmax(s / temperature_source) - uniform)
g_guided = g / temperature_generalist_source + lambda * gate * d
||lambda * gate * d||_2 <= max_delta_norm
```

Both temperatures use only real source-center labels. Target labels are not an
input to the decision function and enter only after predictions are frozen.

## Source-only qualification and domain generalization

Every expert has a versioned qualification artifact bound to exact
model/adapter/task/modality/capability fingerprints. It contains:

- per-source-center accuracy, balanced accuracy, macro-F1, NLL, multiclass
  Brier score, ECE and predictive entropy;
- a lower confidence bound over source-center performance;
- worst-source-center lower-tail CVaR;
- latency distribution;
- shrinkage-diagonal class-conditional feature references.

OOD calibration distances are source-only cross-fitted: leave one source center
out when possible, otherwise leave one source sample out. The deployed mean and
diagonal variance are subsequently refit on all source observations. This keeps
calibration rows out of their own reference fit, especially in the `n << d`
embedding regime. Missing artifacts, insufficient aggregate support for any
frozen class, legacy self-included calibration, weak native-task results,
fingerprint mismatch or invalid features all fail closed.

Two OOD checks separate computational cost from specialist-specific safety:

```text
pre_ood  = empirical Mahalanobis percentile in frozen BiomedCLIP features
post_ood = empirical Mahalanobis percentile in frozen expert-native features

source_robustness = sqrt(LCB_source * CVaR_lower)
trust = source_robustness
        * exp(-temperature * max(pre_ood, post_ood))
        * image_quality
```

LCB and lower-tail CVaR summarize overlapping source-validation evidence. The
geometric mean avoids counting that evidence twice. Hard qualification
thresholds and OOD rejection provide the conservative veto; this directly
addresses the earlier FedDG result in which multiplicative attenuation greatly
reduced the intervention but rescued no additional case.

Before the expert is loaded, pre-call OOD uses the nearest source-class cheap
reference (with global fallback) and never conditions a veto on the
generalist's argmax. This is essential for high-confidence wrong answers: a
wrong class prediction must not be mistaken for domain shift. After the sparse
call, the expert-predicted class selects the native reference. Neither stage
uses the target ground-truth class.

## First-claim versus uncertainty policy

The main `qualified_first_claim` policy checks one qualified expert before the
first diagnostic output. It is designed specifically to make high-confidence
generalist errors observable. `uncertainty_only` remains a required ablation.

For later free-text claims, the original candidate set cannot be reused. A new
`ClaimSpec` must describe the current question, generated prefix, capability and
semantic propositions; otherwise the decision is `NONE`. This prevents a later
sentence from being guided by stale yes/no evidence.

## Heterogeneous expert contract

`NativeEvidence` preserves classification scores, retrieved references,
segmentation masks, detection boxes and specialist text. An `EvidenceBridge`
converts only explicitly linked semantic evidence into bounded claim support:

| Capability | Native payload | Required semantic link |
|---|---|---|
| Classification | class/claim scores | proposition score |
| Retrieval | reports/cases | grounded proposition score + provenance |
| Segmentation | masks | candidate/claim support + mask provenance |
| Detection | boxes | candidate/claim support + box provenance |
| Generation | specialist text | supported/contradicted claim score + text |

An unrecognized capability or spatial/text payload with no semantic link
abstains. External adapters can be loaded through a `package.module:factory`
configuration entry, so adding a model does not require editing the controller.

## Required matched comparisons

1. Generalist.
2. Specialist alone.
3. Routed direct fusion.
4. Uncertainty-only Med-DEFER.
5. Med-DEFER without DG.
6. Mean-source-domain trust.
7. Full LCB + CVaR + two-stage native-OOD Med-DEFER.
8. Equal-budget shuffled evidence.
9. Wrong capability (must fail closed).

Every fold reports accuracy, fixed-taxonomy macro-F1 (including zero-support
frozen classes), target class support and structural/sample absence, ECE,
worst-center accuracy, call rate, rescue/harm and same-as-generalist rate. It
also reports a direct full-vs-shuffled slide-cluster bootstrap interval and
paired sign test as descriptive mechanism falsification. `1-accuracy` is not
reported as an open-ended hallucination rate.

## Post-hoc falsification diagnostics

After predictions are frozen, flag the mechanism as unsupported when any of
these holds:

- full Med-DEFER does not outperform shuffled evidence;
- rescued cases do not exceed harmed cases;
- a called expert leaves at least 95% of predictions unchanged;
- an expert lacks valid real-source qualification;
- source/target slides overlap.

These checks are descriptive outcomes, not model-selection or sample-size
rules. The pilot size is fixed before target labels are inspected. Any
confirmatory study must pre-register its scale and hyperparameters and use a
disjoint final cohort (or a genuinely external center); it cannot expand the
same nested target sample after seeing these diagnostics.

## Stage 2: open-report hallucination

The real multiclass experiment validates only the first clinical claim. The
next stage will generate several natural-language claim beams from the medical
VLM, score those claims with the selected expert, commit one claim, and then
continue decoding. It requires held-out claim-level factuality annotations,
unsupported-claim rate and blinded review. The current semantic `ClaimSpec` and
evidence bridges are foundations for that stage, not evidence that open-ended
hallucination has already been reduced.
