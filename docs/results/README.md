# Experiment result archive

Formal experiment results are published here as compact, reviewable bundles.
Each bundle should contain the frozen protocol, aggregate metrics, audit output,
and per-case predictions with the intermediate routing, expert, evidence, gate,
and timing records needed for analysis.

Large generated tensors, base64 image/mask payloads, model weights, caches, and
background logs remain under the ignored `runs/` tree. A published bundle must
state every omission, preserve the metadata needed to identify the omitted
payload, include checksums, and link to the corresponding raw local run.

Publishing an archive does not change the evaluation decision rules: target
labels must not enter generation or threshold selection, a zero-call/null
intervention is not success, and lexical changes are not medical benefit.

Published bundles:

- [`vqarad_native_claims_stopped_2026-09-10`](vqarad_native_claims_stopped_2026-09-10/README.md):
  complete compact snapshot of the stopped 362-case-common, six-arm run, with
  all available per-case outputs and Gate/expert traces. It is diagnostic, not
  a completed 451-case result.
- [`vqarad_spatial_full_2026-09-10`](vqarad_spatial_full_2026-09-10/README.md):
  completed earlier four-arm spatial experiment.
