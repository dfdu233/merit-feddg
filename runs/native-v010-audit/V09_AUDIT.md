# v0.9 read-only audit for the v0.10 source experiment

Date: 2026-09-07. This audit treats all 32 old target cases as a
development/diagnostic cohort. Token-F1 changes below are lexical, not clinical
factuality or hallucination measurements.

## Frozen inputs inspected

- Source interventions: `runs/native-v09-value-full-gpu1/llava/b8013cc570604684/source-interventions.json`
- Policy: `runs/native-v09-value-full-gpu1/llava/b8013cc570604684/value-policy.json`
- Old development result: `runs/native-v09-value-full-gpu1/llava/b8013cc570604684/evaluations/b0a71687490f275a/result.json`
- Old development predictions: the sibling `predictions.json`
- Actual v0.9 generated config: `runs/native-v09-value-full-gpu1/config-llava-ccc4803d0175.yaml`
- Real manifests: `runs/native-v09-value-full-gpu1/data/both-source16-target16-seed17/{source.jsonl,target.jsonl,references.json}`

The 293 source intervention rows come from 64 independent image groups. Extra
states/prefixes are not counted as additional support.

## Independent source support and group-balanced lexical gain

| Tool | Source proxy domain | Independent image groups | Mean of per-group gains | Positive groups | Negative groups |
| --- | --- | ---: | ---: | ---: | ---: |
| BiomedCLIP anatomy | VQA-RAD proxy 0 | 1 | +0.00000 | 0 | 0 |
| BiomedCLIP anatomy | VQA-RAD proxy 1 | 1 | +0.00000 | 0 | 0 |
| CheXagent description | PathVQA proxy 0 | 1 | +0.02899 | 1 | 0 |
| CheXagent description | VQA-RAD proxy 0 | 6 | +0.03994 | 3 | 0 |
| CheXagent description | VQA-RAD proxy 1 | 4 | -0.00743 | 0 | 2 |
| CONCH tissue | PathVQA proxy 0 | 8 | +0.01110 | 1 | 0 |
| CONCH tissue | PathVQA proxy 1 | 9 | -0.00478 | 2 | 2 |
| XRV anatomy | VQA-RAD proxy 0 | 1 | +0.00000 | 0 | 0 |
| XRV findings | PathVQA proxy 0 | 1 | +0.05062 | 1 | 0 |
| XRV findings | VQA-RAD proxy 0 | 5 | -0.04087 | 0 | 2 |
| XRV findings | VQA-RAD proxy 1 | 4 | -0.00011 | 1 | 1 |
| Source retrieval | PathVQA proxy 0 | 16 | -0.00094 | 2 | 1 |
| Source retrieval | PathVQA proxy 1 | 16 | -0.01331 | 3 | 6 |
| Source retrieval | VQA-RAD proxy 0 | 16 | -0.05699 | 1 | 4 |
| Source retrieval | VQA-RAD proxy 1 | 16 | -0.02379 | 3 | 3 |

Only CONCH initial/selected-history conditions met the old exact support rule.
CheXagent had 37 intervention rows but only 11 independent images and no domain
with the required eight groups. Its positive signal was concentrated in generic
or modality/plane questions: among nine `unknown`-typed groups, four had positive
mean gain and two negative; the single anatomy and explanation groups were both
unchanged. This is insufficient for a deployable CXR policy.

## Routing audit

- The source PathVQA case `pathvqa-train-14925` is visually a chest radiograph;
  its CXR routing is plausible.
- The old target/development case `pathvqa-test-338` is visibly a newborn clinical
  photograph, not a chest radiograph, but was routed as CXR. CheXagent then emitted
  `Cradle cardiomegaly` and LLaVA-Med answered with cardiomegaly. Its apparent
  +0.04440 lexical gain against a photograph-caption reference is not evidence of
  valid expert use.
- Routing is model-inferred applicability metadata, not a validated OOD detector.
  The v0.10 source routing audit must be completed blind to questions and answers.

## Development-cohort failure examples

The v0.9 mean-value policy made six tool calls on four target/development images:

| Case | Calls | Paired Token-F1 gain | Observation |
| --- | --- | ---: | --- |
| `pathvqa-test-1335` | CONCH then retrieval | -0.01831 | Replaced a parasite answer with fixed-catalog adipose tissue. |
| `pathvqa-test-3021` | CONCH | +0.00000 | Changed wording but still missed `many macrophages source`. |
| `pathvqa-test-3675` | CONCH then retrieval | +0.00000 | Produced generic tool-observation prose; missed `organisms`. |
| `pathvqa-test-5871` | retrieval | -0.11111 | Replaced a lung answer with a disclaimer about a different patient/image. |

Thus there were no lexical rescues and two harms. These examples show task and
communication mismatch, not merely a threshold problem.

## XRV anatomy failure inspection

The predicted lung/heart masks for the two applicable development radiographs are
geometrically plausible after center-crop coordinate remapping. They are thoracic
anatomy masks, not mass or pneumothorax masks. Under the native two-image prompt,
both answers collapsed to only EOS (empty text):

| Case | Question | Baseline -> tool Token-F1 change | Baseline/tool output tokens |
| --- | --- | ---: | ---: |
| `vqarad-c8e8636957b860192115-c2114db66127` | thoracic mass side | -0.14286 | 18 / 2 |
| `vqarad-d83984c6991935029b0dd3d548cbcc77fed09878501b381aba6b805fd725ff51` | pneumothorax side | -0.33333 | 17 / 2 |

Rendered overlays are in `runs/native-v010-audit/overlays/`. The original image
is first, the second image is explicitly predicted evidence, and the mask metadata
uses normalized coordinates in the original-image frame. The text budget exposed
only the first structure record in the inspected prompt while the overlay contained
left lung, right lung and heart. The v0.10 duplicate-original and text-only controls
are therefore necessary before attributing harm to mask quality.

## Stage-1 diagnosis

The three hypotheses carried into v0.10 are:

1. **Expert/task and routing validity:** some observations are irrelevant because
   the image type or requested capability is wrong.
2. **Evidence communication:** long native catalogs, retrieved source answers and
   multi-image formatting can dominate or destabilize the LLaVA-Med answer.
3. **Conditional utility support:** useful-looking CheXagent changes lack enough
   independent per-domain source support; a richer value model cannot repair this.

Every “single-tool” result means LLaVA-Med plus that tool's evidence. It is not the
standalone accuracy of the specialist.
