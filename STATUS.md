# Current status

## Hypothesis

Med-DEFER lets a medical generalist VLM conditionally borrow one qualified
capability from a small specialist while source-only lower-tail trust suppresses
experts that may fail under unseen-domain shift.

## Working set

- Current upstream commit: `9d1ef28` (`Add domain-robust expert deferral during decoding`).
- Restored real-runtime compatibility for OpenMed, CheXagent, CONCH and LO-VLM.
- Experts without any source-domain validation score now fail closed and cannot
  be selected.
- Claim-boundary decisions are now always recorded, including `NONE`; claim
  indices no longer shift when the first claim is confident.
- Concept-token guidance now covers sentence-initial/non-initial and casing
  variants instead of only a leading-space lowercase token.
- 33 tests and Ruff pass.

## Latest evidence

- Deterministic mechanism smoke: generalist 0.6563, Med-DEFER 0.8542, but
  `abstentions=0`; this proves top-1 sparsity, not useful `NONE` selection.
- Frozen real-model cache, 32 target samples: generalist 6/32, Med-DEFER 8/32;
  16 pathology calls, 16 OCT abstentions, 2 rescued and 0 harmed.
- The same cache comparison at 4/8/16/32 target samples changes accuracy by
  0/0/+1/+2 correct examples. This is an early signal, not statistical evidence.
- Cache comparison is counterfactual. It does not prove live lazy loading or
  open-ended hallucination reduction.
- Pathology live canary: the route selected pathology at 0.9848 probability and
  lazily loaded only CONCH. The first claim was confidently wrong (`No`) and
  hallucinated a 3D brain reconstruction for a gross heart specimen, so the
  controller correctly recorded `NONE`; it called CONCH only after the first
  sentence boundary. Default and 10x guidance did not change the text.
- A cache-selected pathology rescue diagnostic produced the same correct `Yes`
  answer with and without live CONCH guidance. This confirms that cached
  candidate-score gains do not directly transfer to autoregressive generation.
- OCT live canary loaded no expert. With the uncertainty threshold forced to
  zero, its source lower-tail trust remained zero and utility was -0.005, so the
  live domain gate still rejected LO-VLM.

## Design gaps before a formal claim

- Live requests currently ask only for classification; segmentation, detection,
  retrieval and generation are typed envelopes but lack end-to-end adapters.
- Configured expert construction still uses a central hard-coded adapter factory.
- Target OOD uses `1 - modality route confidence`, not an expert-specific
  distance to its source training/validation domains.
- The public hash partitions are not hospitals or scanner domains.
- The required no-DG, mean-trust, post-hoc and dense performance ablations are
  not yet implemented; dense execution is only a call-count counterfactual.

## Active job / next action

No project GPU job is running and the GPU is available. Do not scale the current
live configuration yet: an uncertainty-only trigger missed a high-confidence
hallucination, while the later expert call could not repair the already emitted
claim. The next coherent mechanism change is a source-calibrated defer-risk
trigger that can flag overconfident/OOD generalist claims before generation;
then repeat the matched pathology and OCT live canaries before a larger batch.
