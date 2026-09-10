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
