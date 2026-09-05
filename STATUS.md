# Current status

## v0.6 open-generation upgrade (2026-09-05)

- Actual short-block beam proposals, native expert evidence, bounded reranking and exact-token
  commitment now form a separate open-generation path (`run_open.sh`). This is not the old
  frozen-score comparison and not a token-level logit processor.
- The first block can receive expert evidence despite high generalist confidence. Every block
  constructs new prefix-dependent propositions. No original yes/no candidates are reused.
- A single source-only qualification rule uses paired *complete generated answers* and continuous
  token-F1 gain. Worst-source mean-minus-SE is a heuristic margin, not an OOD safety guarantee.
- Default actual models: medical OpenMed 3B generalist, CONCH and BiomedCLIP evidence providers.
  Heterogeneous masks/boxes/retrieval/text are supported by the native plugin contract, not claimed
  to be clinically validated new model integrations.
- Real PathVQA non-yes/no train/test preparation, separate references, pixel/group leakage audit,
  per-case resumability, beam-only and ungated controls, corrupted-support control, blind annotation
  templates, latency and memory reporting.
- PathVQA train partitions are explicitly proxy domains. Retain the v0.5 PathoROB experiment for
  independent-center closed-set checks. No new medical GPU performance result is claimed here.
- Research overlap and limitations are documented in `docs/OPEN_DECODING_RESEARCH.md`:
  especially CCD, GSCo, FUDGE, GeDi, VGS and FedDG. No first-expert-decoding novelty claim.
- Local verification: 121 tests passed with PyTorch 2.6.0 CPU, Transformers 4.57.1 and
  PyArrow 23.0.1; Ruff and Bash syntax checks passed. Tests include real tiny GPT-2 and
  image-bearing Qwen2.5-VL generation, not downloaded medical checkpoint inference.

## What changed in v0.5

The primary validation path no longer treats yes/no VQA or a learned binary
error predictor as the scientific endpoint. It now uses the real public
PathoROB Tolkach ESCA task: six tissue classes from four named medical centers,
evaluated leave-one-medical-center-out.

- The medical generalist is scored with its real length-normalized answer
  sequence log-likelihoods. The previous all-zero live placeholder is gone.
- A qualified compatible specialist is considered before the first diagnostic
  claim even when the generalist is highly confident. Entropy-only triggering
  remains an ablation because it missed the observed high-confidence error.
- The implemented live closed-set path locks the bounded candidate-space argmax
  before asking the generalist for a separate explanation. The retained token
  logits processor is an unvalidated open-generation prototype, not the v0.5
  empirical result.
- Specialist qualification uses real source-center multiclass probabilities,
  macro-F1/balanced accuracy, lower confidence bounds and worst-domain CVaR.
  There is no binary `P(error)` model.
- Qualified experts use geometric-mean LCB/CVaR reliability rather than
  multiplying two correlated source metrics. Hard qualification and OOD remain
  the vetoes; this avoids repeating the over-suppression seen in the old FedDG gate.
- Pre-call OOD is measured in frozen BiomedCLIP features. Post-call OOD is
  measured in the selected expert's frozen native features (CONCH in the first
  benchmark), both calibrated only on source centers in the PathoROB evaluator.
- Classification, retrieval, segmentation, detection and generation have
  explicit semantic evidence bridges. Missing bridges, capabilities,
  qualification artifacts or model fingerprints fail closed in the qualified
  PathoROB path. The older generic live command does not yet claim this strict
  qualification contract for every heterogeneous adapter.
- The bridge API accepts the current question, generated prefix and semantic
  claims. Static yes/no strings are not sent to a specialist. Dynamic later
  claims are not yet validated; without a new `ClaimSpec`, reuse is forbidden.
- External expert adapters can be registered by an importable factory in the
  model configuration without editing the central adapter switch.

## New real experiment

`./run_pathorob.sh` installs or reuses OpenMed/Qwen2.5-3B-MedVL,
BiomedCLIP and CONCH, downloads the 317 MB PathoROB subset, prepares six-class
manifests from label-blind per-center samples, audits slide leakage, extracts
evidence once and evaluates all four leave-one-center-out folds. Frozen caches are reused only when data,
configuration, model-snapshot and extraction-contract fingerprints match.

The required comparison includes Generalist, Specialist, routed fusion,
uncertainty-only, Med-DEFER without DG, mean-domain trust, full LCB+CVaR+
native-OOD Med-DEFER, shuffled evidence and wrong capability. It reports
accuracy, macro-F1, ECE, worst-center accuracy, call rate, rescue/harm, paired
slide-cluster bootstrap intervals and a slide-cluster paired sign test. It does
not label `1-accuracy` as an open-ended hallucination rate.

The comparison reuses frozen outputs from real models. Its accuracy and
shuffled-evidence controls are real, but selected-call rates remain
counterfactual until a matched live batch is run; they are not latency claims.

No new medical performance number is claimed in this commit: the real models
must be run on the GPU server after pulling. The result JSON includes post-hoc
falsification diagnostics. They describe whether full Med-DEFER beats shuffled
evidence, whether rescues exceed harms, and whether calls change predictions;
they are explicitly forbidden as target-label-based tuning or sample-size rules.

## Why the previous result is insufficient

The old frozen cache improved from 6/32 to 8/32, but it was a candidate-score
counterfactual. In live generation, one pathology example was confidently wrong
before CONCH was invoked, while another correct answer remained unchanged.
Therefore the cache result cannot establish live hallucination reduction.

The old OCT source expert was unreliable and remained correctly rejected. The
new fail-closed qualification rule preserves that behavior; no expert without
real source-task evidence is silently assigned a default trust score.

## Remaining boundary

The new PathoROB study validates the first, closed multiclass clinical claim.
It is a real domain-generalization experiment, but not yet an open-report
hallucination experiment. Open generation requires a dynamic claim-candidate
provider (claim-beam decoding) plus claim-level factuality annotations. The
semantic `ClaimSpec` and heterogeneous evidence bridge are implemented as the
foundation for that second phase; no open-ended benefit is claimed yet.
