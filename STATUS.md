# Current status

## v0.9 source-conditioned native capability value (2026-09-06)

- Additive `run_value.sh` / `llava_run --study value`; legacy v0.8 remains available.
- Frozen LLaVA-Med remains the final free-text generator. Native evidence is used
  before/between answer blocks, never as a post-answer replacement. Registered
  classification/retrieval/generation and predicted mask/bbox views are supported;
  original images remain intact. Default real tools remain the existing v0.8 set.
- Continuous state x tool ridge predicts marginal quality, not binary model error.
  Source branches compare exact prefixes and identical no-further-tools continuations,
  including declared A->B histories and actual empty/negative outcomes. Independent
  image groups, not extra prefixes, determine support. Unknown histories fail closed.
- Mean and empirical worst-held-source overprediction ablations share support checks.
  This is source-balanced regression and LODO pessimism, NOT a proved DG guarantee,
  minimax GroupDRO implementation or federated training protocol. Shared source-bank
  retrieval means LODO is policy-level, not an entirely fold-isolated pipeline.
- Added deterministic image padding for the value profile only: the inspected
  official LLaVA-Med padding helper can jitter a nonsquare image by one pixel.
  Source block-NONE must match single-pass token IDs before fitting; old preprocessing
  remains the v0.8 default for historical reproduction.
- Source/evaluate stages, fingerprint-bound branch caches, target-label isolation,
  native visual provenance, source support diagnostics and blind factuality templates.
  Default quality is explicitly lexical F1; custom continuous scorers need version identity.
- No new v0.9 remote medical-GPU result has been run locally. Previous negative
  results below are retained; engineering tests cannot establish clinical benefit.
- Local verification for this revision: 422 tests passed (including deterministic
  padding, two-image protocol, source/target cache separation and paired-history
  collection); Ruff, both modified Bash entry syntax checks and git diff checks passed.

## v0.8 LLaVA-Med native-tool revision (2026-09-06)

- Default remote entry `run_llava_med.sh` reuses the supplied LLaVA-Med Mistral 7B,
  official source and huatuo Python. OpenMed remains a same-protocol comparison.
- Compact finite tool IDs replace free JSON action generation in the new config.
  Medical answers remain unconstrained free text; exact committed tokens survive
  evidence-context reconstruction. Image-only routing removes the all-PathVQA-is-
  histology assumption. Native masks and generation adapters use actual models.
- Default native capabilities: CONCH tissue appearance; BiomedCLIP major anatomy
  and image/question source retrieval; small XRV CXR classification and anatomy
  segmentation. CheXagent generation is optional and local-only. No new generalist
  or 3B generation model is downloaded by this entry.
- Source utility is measured using forced single-tool real generation, not calls
  selected by the same potentially broken controller. Empty evidence and runtime
  failures are distinguished. No relaxed source threshold or target-label fitting.
- Real non-yes/no PathVQA + image-disjoint re-split VQA-RAD; proxy groups remain
  explicitly non-hospital. New single-tool baselines, shorter evidence prompts,
  routing cost accounting and external LLaVA/CLIP provenance.
- The remote pretrained medical GPU validation recorded below completed. It validates
  execution and fail-closed DG gating, but does not establish medical hallucination
  reduction, DG improvement or ICLR novelty. Local verification: 300 tests collected,
  298 passed and 2 environment-specific tests skipped in both supported environments;
  Ruff and Linux entry syntax checks passed.
- Read `docs/V08_LLAVA.md` for exact paths, dependencies, coverage and limitations.
- `run_llava_med_full_gpu0.sh` is the host handoff for the complete offline method
  matrix. It checks actual free memory, defaults to the local CheXagent with a
  28 GiB free-memory floor, and resumes from the same per-case cache/output root.
  The host environment is `/home/dbw/.runtime/miniconda3/envs/huatuo`; launchers
  auto-detect it before the older `/opt/miniconda3` container path.

## GPU validation after v0.8 (2026-09-06)

### Full 48 GiB run with CheXagent

- The completed full run is
  `runs/native-v08-full-gpu0/llava/faa0513e4fd6b8ba` (exit code 0). It used
  the local LLaVA-Med v1.5 Mistral 7B generalist plus CONCH, BiomedCLIP,
  TorchXRayVision classification/segmentation, source retrieval and the local
  CheXagent-2-3b generation specialist. The run evaluated 64 source and 32
  target questions (16 PathVQA and 16 image-disjoint VQA-RAD targets) and wrote
  457 code-fingerprinted case-cache records. No model download was required.
- Generalist token-F1 was 0.07338. Forced bounded all-evidence reached 0.06342,
  paired gain -0.00995 with bootstrap interval [-0.03930, 0.02197] (5 improved,
  9 harmed). Ungated adaptive control reached 0.07075, gain -0.00262 with
  interval [-0.02624, 0.02151] (4 improved, 3 harmed). Neither establishes an
  improvement over the generalist on this 32-question target sample.
- All 11 source qualification cards failed: two sufficiently supported cards
  had observed negative robust gain (retrieval/pathology -0.03152 and
  CONCH/pathology -0.03805), while the remaining nine had insufficient
  per-domain support. Consequently adaptive DG made no target tool call, exactly
  matched the generalist token-F1, and logged 44 insufficient-support, 12
  observed-negative-gain and one missing-source-scope veto. This validates the
  intended fail-closed gate, not the efficacy of specialist assistance.
- Among routed single-tool target arms, CheXagent was the strongest numerical
  result: token-F1 0.07924, paired gain +0.00587, interval
  [-0.00593, 0.01957], with 10/32 calls and zero runtime errors. CONCH reached
  0.07437 (+0.00099; interval [0.00000, 0.00269]); CXR anatomy reached 0.07329
  (-0.00009); XRV findings reached 0.05946 (-0.01391; interval
  [-0.03497, -0.00062]); and source retrieval reached 0.05964 (-0.01373;
  interval [-0.05124, 0.02298]). These are routed whole-set means and the
  intervals remain sample-limited.
- Forced all-evidence made 57 target specialist calls (32 retrieval, 13
  classification, 10 generation and 2 segmentation) and adopted 1.281 pieces
  of evidence per question on average. Ungated adaptive control made 14 target
  calls on 14/32 questions, restricted itself to 11 retrieval and 3 CONCH calls,
  and produced no invalid compact action. It did not dynamically compose two
  experts on a target case in this run.
- Across the complete fixed-cache trace there were zero tool runtime errors.
  CheXagent's BF16 visual-input fix was exercised in both source qualification
  and target evaluation; the full trace contains 41 CheXagent calls and 31
  adopted results across all cached experimental arms. Peak method-level
  PyTorch allocation was 23.91 GiB; observed total device use peaked near
  46.3/48.5 GiB because an unrelated baseline allocation remained on the GPU.
- Automatic EM/token-F1 are lexical answer-overlap measures, not medical
  hallucination or clinical factuality metrics. The modality labels are
  image-inferred routing predictions, PathVQA source groups are hash proxies,
  and the 32-question target set is too small for an efficacy claim. The result
  supports execution correctness and conservative DG rejection; it does not yet
  support the claim that the framework reduces medical-VLM hallucination.

- The final offline run is `runs/native-v08-scale8-final/llava/8fffb22c963d85dd`.
  It reused `/opt/miniconda3/envs/huatuo/bin/python`, the existing 15 GiB local
  LLaVA-Med v1.5 Mistral 7B checkpoint, the clean Med-LVLMs source tree and a
  complete local CLIP ViT-L/14-336 vision tower. No generalist download occurred.
- The real-data pilot used 32 source questions and 8 target questions: four
  PathVQA targets plus four targets from a custom RGB-image-disjoint VQA-RAD
  resplit. Image-only routing predicted three pathology, one CT and four CXR
  target images. These predictions are routing labels, not clinical metadata.
- Generalist token-F1 was 0.1290. Forced all-evidence fell to 0.0804, a paired
  gain of -0.0486 with bootstrap interval [-0.1319, 0.0139] (one improved, two
  harmed). Ungated adaptive control reached 0.1151, gain -0.0139 with interval
  [-0.0417, 0.0000] (zero improved, one harmed). The sample is too small for an
  efficacy claim and lexical token-F1 is not a hallucination metric.
- Every final target tool result that fit the applicable single-tool arm was
  adopted by the language bridge: CONCH 2/2, XRV anatomy 2/2, XRV findings 2/2,
  and retrieval 8/8. CONCH and XRV anatomy matched baseline token-F1; XRV findings
  fell by 0.0399 and retrieval fell by 0.0486. Thus the earlier zero-adoption XRV
  result was an evidence-prompt budgeting bug, now fixed, while specialist utility
  remains unproven and can be harmful.
- All source qualification cards remained `insufficient_support` under the fixed
  minimum of eight interventions per domain and two-domain rule. The DG-gated arm
  therefore made zero target tool calls and preserved baseline F1 exactly. This is
  the intended fail-closed behavior, not evidence that DG assistance improves answers.
- No tool runtime error, invalid controller action, data split audit failure or
  2,048-token context overflow occurred. Peak PyTorch allocation was 17.85 GiB.
  Optional CheXagent was excluded from this run because another process occupied
  roughly 23 GiB of the 48 GiB GPU; CONCH, BiomedCLIP, XRV and source retrieval
  were the active native capability pool.

## v0.7 native capability collaboration (2026-09-05)

- Primary entry: `run_capabilities.sh`. The frozen medical VLM requests registered
  capabilities before and between answer blocks, incorporates native evidence into
  memory, and continues with exact committed token IDs. No candidate score fusion.
- Default actual adapters: CONCH fixed tissue appearance catalog and BiomedCLIP
  source-image retrieval. Optional MedSAM prompted segmentation and Qwen-compatible
  generative specialist adapter; other native factories can supply detection etc.
- Source qualification is task/capability/scope-specific; missing source support is
  distinct from observed negative gain. No new binary risk or two-stage OOD model.
- Four real-generation comparisons, source-only retrieval with leave-domain/group/
  pixel exclusion, cache/provenance binding, complete native traces and clinical
  annotation templates. Default PathVQA remains a proxy-domain mechanism pilot.
- Controller generation cost is measured, including invalid JSON; no-tools paths
  skip the controller. This is evidence-conditioned re-prefill, not KV reuse or
  token-logit guidance. Single-scope qualification does not certify composition.
- At release time no medical-checkpoint GPU result was reported; the subsequent
  server validation is recorded below. See `docs/CAPABILITY_COLLABORATION.md` for
  exact algorithm, integration boundaries and outstanding academic requirements.
- Local verification: 191 tests passed, Ruff passed, Bash syntax/entry-point
  checks passed. Includes real tiny image-bearing Qwen controller/memory/continuation
  and tiny SAM forward checks on CPU; no downloaded medical-weight performance claim.

## GPU validation after v0.7 (2026-09-05)

- The real default pilot used 32 PathVQA source questions (16 per proxy group) and
  16 target questions. The OpenMed 3B generalist, CONCH catalog tool and BiomedCLIP
  source-image retrieval all ran with native traces; peak PyTorch allocation was
  9.58 GiB. Source/target pixel and group audits passed.
- Generalist token-F1 was 0.2431. Adaptive no-DG reached 0.2826, a paired gain of
  0.0396 with bootstrap 95% interval [0.0000, 0.1021], two lexical improvements
  and no lexical harms. It invoked retrieval four times over three target cases,
  adopted three results, and never selected CONCH. No target case composed more
  than one capability.
- The apparent automatic gain is not evidence of medical factuality. One changed
  answer only moved from `kidney` to the lexically closer `renal`; another changed
  from a mandible radiograph to a femur implant while the reference was a bone
  marrow defect. Independent clinical claim annotation is required.
- Static all-evidence was strongly harmful: token-F1 fell to 0.0159, paired gain
  -0.2271 with bootstrap 95% interval [-0.4388, -0.0555]. It expanded answers from
  4.0 to 64.6 mean tokens and often copied or over-interpreted irrelevant catalog
  and retrieved-source context.
- Neither scope qualified. CONCH was actually invoked in only 6 cases per source
  proxy domain (below the configured 8) and its observed paired gains were negative.
  Retrieval was never requested in its isolated source qualification runs, so it
  had no intervention support. Adaptive DG therefore correctly made zero target
  tool calls and matched the generalist.
- Adaptive no-DG issued 19 target controller decisions; 13 were invalid. Most mixed
  the retrieval expert with the classification scope, supplied `[]` as a region,
  or exceeded the controller output budget and produced truncated JSON. Whitelist
  and ROI validation failed closed, but the unconstrained 3B JSON controller is not
  reliable enough for a larger efficacy experiment.
- The implementation and safety plumbing are validated; usefulness, dynamic
  multi-capability composition, real-domain generalization and hallucination
  reduction are not established. The next experiment should follow controller and
  evidence-policy repairs rather than merely increasing target sample count.

## GPU validation after v0.6 (2026-09-05)

- Environment and entry points passed 121 unit tests, Ruff and Bash syntax checks. Pinned
  OpenMed/Qwen2.5-3B-MedVL, BiomedCLIP and CONCH snapshots passed payload fingerprint
  verification; the PathoROB and PathVQA split audits reported no source/target leakage.
- The real PathoROB 12-patch-per-center pilot evaluated 48 frozen real-model records over four
  leave-one-center-out folds. Aggregate accuracy was 18.75% for the medical generalist, 54.17%
  for the always-on CONCH specialist, 27.08% for shuffled evidence and 33.33% for full Med-DEFER.
  Full Med-DEFER therefore improved over the generalist by 14.58 points and over shuffled
  evidence by 6.25 points, with three patches rescued and none harmed. The slide-cluster
  bootstrap interval for full minus shuffled was [0.00, 13.64] points and the paired sign-test
  p-value was 0.25, so this is a mechanism signal, not a statistically established gain.
- Full Med-DEFER called the expert on 70.83% of target patches. It failed closed for the UKK fold
  because regressed-tumor tissue had no source support outside UKK; the other three folds showed
  a positive full-versus-shuffled diagnostic. Selected-call counts are counterfactual over frozen
  model evidence and are not live latency measurements.
- In the open PathVQA pilot, 16 source examples (8 per proxy group) and 8 target examples were
  generated with real short-block proposals. Both expert cards had sufficient source support but
  failed source-only gain qualification: conservative token-F1 gain was -0.3597 for CONCH and
  -0.0403 for BiomedCLIP. Robust decoding consequently made zero expert calls and matched the
  beam-only token-F1 of 0.25. Forced ungated experts changed 12.5% and 37.5% of target answers,
  respectively, without improving target token-F1.
- These pilots validate execution, fail-closed qualification and evidence-sensitive controls.
  They do not establish hallucination reduction: PathoROB is closed-set classification, PathVQA
  uses lexical EM/F1 and proxy rather than hospital domains, and both target samples are small.

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
  independent-center closed-set checks. Subsequent GPU pilot results are recorded above.
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

At code-landing time no medical performance number was claimed; the subsequent
GPU pilot is recorded at the top of this file. The result JSON includes post-hoc
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

The PathoROB study validates the first, closed multiclass clinical claim and is
a real domain-generalization experiment. The separate open-generation path now
provides dynamic block candidates, but it is not yet an open-report hallucination
result: source-qualified experts did not pass the current PathVQA pilot, and
claim-level factuality annotations are still required. The semantic `ClaimSpec`
and heterogeneous evidence bridge remain the foundation for that next phase;
no open-ended benefit is claimed yet.
