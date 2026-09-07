# Current status

## v0.11 opt-in bounded evidence bridge (2026-09-07)

- Built on `8bdd256`, retaining the remote allocator-cache and compatibility fixes.
  Historical reports and default v0.8/v0.9/v0.10 entry points are preserved.
- Added single-original-image, same-prefix base/evidence token distributions, clipped
  evidence residuals, per-token/case KL budgets, exact-payload deduplication and
  token-distance expiry. NONE/zero uses production generation; no post-hoc rewrite.
- Added `--study evidence --evidence-stage source|evaluate`, default source-only,
  offline, existing local weights/environment. No model training or new downloads.
  Source uses the same actual tool output for direct, bounded and format controls.
- Source calibration selects strength on independent groups and confirms the fixed
  choice on other groups. Proxy domains cannot qualify; matching source/code/model/
  runtime fingerprints are required for the separate target stage. This is empirical
  source confirmation, NOT LODO, a coverage guarantee, or a complete DG solution.
- New profile reuses classification, generation and segmentation observations from
  the existing registry; retrieval is excluded until independent bank partitioning.
  Current experiment is initial single-tool bridge validation, not demonstrated
  multistep capability composition. All medical conclusions remain unverified.
- Correctness-first backend replays each committed prefix through the production
  KV-cache path for base and evidence; there is no cross-branch KV sharing. Default
  guidance lasts 16 tokens, not a semantically detected claim.
- Local tests include numerical KL/mask tests, source/target isolation and cache
  checks, fake-backend integration, and real random CPU Mistral score replay.
  Verified: 488 tests passed (Python 3.10.9, Torch 2.6.0 CPU, Transformers 4.57.1),
  Ruff passed, three Bash entry-point syntax checks and git diff checks passed.
  A subsequent real-GPU canary and full source diagnosis are recorded below. See
  `docs/BOUNDED_EVIDENCE_V011.md` for the server handoff.

## GPU validation after v0.11 (2026-09-07)

- Reused the existing huatuo Python, local LLaVA-Med v1.5 Mistral 7B, CheXagent,
  CONCH, BiomedCLIP and XRV assets in forced offline mode. No model/data download,
  target generation, target scoring or target tuning occurred. Exact commands and
  paths are in `runs/bounded-evidence-v011-audit/HANDOFF_RESULT.md`.
- The first real canary exposed two hard integration failures before any result was
  accepted. CheXagent `device_map: auto` placed the remote-code model on CPU beside
  LLaVA, so the profile now explicitly maps it to CUDA. More importantly, one-step
  full-prefix FP16 prefill disagreed with production KV-cache generation on a near
  tie (`right` versus `superior`, logit gap 0.0078125). `next_scores` now forces the
  exact prefix through the production KV path; the failing real case then matched
  16/16 tokens. The trace names this backend `production_kv_replay`.
- The final 8-case canary and frozen 64-source run both exited successfully. The full
  run is `runs/bounded-evidence-source-v011-kvreplay-full/llava/e634250537642f06`:
  64 cases, 82 strength-specific records from 41 real expert executions, 35 cases
  without a compatible expert, and zero tool/runtime failures. All 992 checked
  production baseline tokens matched; target generations were exactly zero.
- Across all real and format-control guidance traces, per-token KL never exceeded
  0.02, maximum cumulative case KL was 0.25635 versus the 0.32 budget, evidence was
  active only at token positions 0--15, and all branches retained a single original
  image. Peak PyTorch allocation/reservation was 23.91/46.84 GiB; observed device
  use briefly reached about 48.5 GiB during expert loading and returned after cases.
- On the 41 applicable source expert/case interventions, direct context averaged
  +0.00738 Token-F1 (8 improved, 3 harmed, 30 unchanged). Bounded strength 0.25
  averaged -0.00658 while format-only averaged -0.00007; the entire net real-versus-
  format loss (-0.00650) came from one CXR case changing correct `both sides` to
  incorrect `right side`. At strength 0.5, bounded and format-only both averaged
  -0.01308, so the aggregate net content effect was zero.
- The content-sensitive signal is therefore sparse and not robust: guided and
  format outputs differed in only 3/41 interventions at strength 0.25 and 4/41 at
  0.5, with no positive real-versus-format Token-F1 case. Direct CheXagent context
  averaged +0.01771 over 11 applicable cases, but bounded CheXagent was identical
  to format control in quality; this does not establish medical factuality.
- All six calibration cards correctly failed closed with
  `proxy_or_unverified_domains` and strength 0.0. The four source domains are
  dataset/hash proxies rather than hospitals, and several tools have only one
  represented domain/case. No new strategy was fitted or unlocked, and target
  evaluation remains intentionally unrun. The next valid step is real source-domain
  coverage plus blinded evidence/answer review, not weaker gates or old-target tuning.

## v0.10 source communication diagnostics and scoped evidence (2026-09-07)

- Synced the complete negative v0.9 report at `c99d488` before these changes.
  No historical result was overwritten or reinterpreted as clinical improvement.
- Added deterministic capability-specific EvidenceNeed subrequests and opt-in
  scoped native presentation. Raw observations are preserved, scores stay native,
  no ROI/negative diagnosis is fabricated, and retrieval-answer context is an
  explicit ablation. The deployed generator remains LLaVA-Med / the chosen VLM.
- `run_diagnostics.sh` uses existing real models/data, defaults to an offline
  source-only canary, and never fits a policy or generates targets. It compares
  identical-prefix text/scoped/overlay/duplicate-original views, plus genuine
  generative-expert query variants and predeclared joint evidence pairs.
- Presentations reuse one native tool result. Joint gain, conditional second-tool
  gain and interaction are measured separately; this is NOT ROI handoff or a
  trained macro-action policy. Runtime errors and block-NONE mismatch stop diagnosis.
- Added per-source-domain/state independent-group summaries, raw branch evidence,
  separate routing audit and correctness/relevance/factuality annotation templates.
- `run_value.sh --evidence-profile scoped --value-stage source` can fit the existing
  continuous value/LODO policy with the new communication configuration. Legacy
  defaults remain available. No DG threshold was weakened to force calls.
- Local verification: 449 tests passed; Ruff, Bash syntax and diff checks passed.
  These include offline model doubles, NOT new real-model medical evidence.
- The remote source diagnosis and bounded source-only fits were subsequently completed
  as recorded below. No new clinical/target performance result was produced. Frozen
  native features, retrieval bank LODO isolation, real hospital coverage, learned
  evidence alignment and ROI/sequence-policy extensions remain unsolved. Read
  `docs/SERVER_CODEX_V010.md` before any later target execution.

## GPU validation after v0.10 (2026-09-07)

- Completed handoff stages 0--4 with the existing huatuo environment, local
  LLaVA-Med v1.5 Mistral 7B, CheXagent, CONCH, XRV and the exact 64-source v0.9
  manifests. No model/data download, target generation, target scoring or target
  hyperparameter search was performed. Detailed paths and commands are recorded in
  `runs/native-v010-audit/HANDOFF_RESULT.md`.
- A first full diagnostic attempt exposed a per-case CUDA allocator-cache lifecycle
  issue: recorded live allocation stayed near 17.4 GiB while total device use grew
  to 48.5 GiB after 11 cases. The value/diagnostic cache lifecycle now resets case
  state, runs GC and releases unused CUDA cache after success or failure without
  unloading expert weights. A focused regression test and a real 8-source canary
  verified the repair. Full checks passed with 447 tests and two environment skips;
  Ruff, Bash syntax and diff checks passed.
- The 64-source diagnostic completed at
  `runs/native-v010-source-diagnostics/llava/33f84d09e0bdb3a1`: all 64 block-NONE
  token sequences matched baseline, all 233 tool events executed without a runtime
  error, and peak per-case PyTorch allocation was 24.03 GiB. The run fitted no policy
  and generated zero target answers.
- Duplicate-original input alone had mean initial Token-F1 gain -0.05095 and harmed
  17/64 groups, exposing a substantial multi-image-format confound. Scoped XRV
  findings changed the 10-CXR mean from -0.04175 to +0.00646, primarily by reducing
  verbose evidence harm. CONCH scoped continuation changed 16-group mean from
  -0.00108 to +0.00994. These are lexical communication signals, not medical
  correctness or hallucination reduction.
- Source retrieval remained harmful at initial state (about -0.031 mean over 64
  groups), and removing attached source answers did not solve it. The predeclared
  CONCH-to-retrieval pair averaged -0.00612 over 17 initial groups and beat the best
  single action only once; the CXR anatomy-to-CheXagent pair had only one supported
  case and zero gain. No stable complementarity claim is supported.
- The diagnostic justified a bounded source-only legacy/scoped fit, not target use.
  Legacy fitted 292 and scoped 294 real source interventions, with zero runtime
  failures and 64 independent groups each. Only five exact conditions in each policy
  met support requirements, all involving pathology CONCH/retrieval. Both robust
  policies selected NONE at all 200 observed source states after their source-LODO
  overestimation penalties. This is conservative rejection, not a performance win.
- The four source domains are dataset/hash proxies, not hospitals. CXR conditions
  lack two domains with eight independent groups, routing and evidence factuality
  annotation sheets remain unfilled, and pretraining-source exposure is unresolved.
  The next step is blinded routing/medical evidence review plus source-only CXR
  coverage, not lowering thresholds or tuning on the old 32-case development target.

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
- The first full v0.9 medical-GPU result is recorded below. Previous negative
  results remain retained; engineering execution cannot establish clinical benefit.
- Local verification for this revision: 422 tests passed (including deterministic
  padding, two-image protocol, source/target cache separation and paired-history
  collection); Ruff, both modified Bash entry syntax checks and git diff checks passed.

## GPU validation after v0.9 (2026-09-06)

- The completed source-conditioned value run is
  `runs/native-v09-value-full-gpu1/llava/b8013cc570604684`; its frozen target
  evaluation is `evaluations/b0a71687490f275a`. Both source and evaluate stages
  exited with code 0, using the existing LLaVA-Med v1.5 Mistral 7B, CONCH,
  BiomedCLIP, XRV and local CheXagent weights. It evaluated 64 source and 32
  target questions and wrote 448 fingerprint-bound case caches.
- Source collection produced 293 real same-prefix state/action interventions:
  105 initial, 127 continuation and 61 after-tool records. All 293 tool
  executions completed, five produced legal empty/unusable evidence, and no
  duplicate or runtime-failure record was fitted. The 133-dimensional frozen
  state/action encoder and four policy-level LODO heads contained no target
  labels or reference fields.
- Six of 35 exact action/state/history conditions met both independent-group
  support and held-source residual requirements. They covered pathology CONCH
  and source retrieval at initial, continuation and selected ordered-history
  states. CheXagent had 37 source interventions with mean lexical gain +0.0210,
  but its CXR conditions lacked eight independent groups in every represented
  source domain and therefore remained unsupported under the predeclared rule.
- The v0.9 generalist token-F1 was 0.07358. Deterministic padding and the new
  protocol mean this is a newly measured baseline, not a value that should be
  silently copied from v0.8. Block-NONE exactly matched every baseline token.
  Bounded all-evidence fell to 0.04248 (paired gain -0.03111, bootstrap interval
  [-0.07016, 0.00872]; 4 improved, 12 harmed). The finite-action agent reached
  0.06573 (-0.00785, [-0.02581, 0.00353]; 3 improved, 3 harmed).
- The source mean-value policy reached token-F1 0.06954, paired gain -0.00404
  with interval [-0.01100, 0.00000]. It called CONCH three times and retrieval
  three times across four target cases, including two multi-capability cases;
  there were no lexical improvements and two harms. Thus the implementation
  demonstrates state-dependent and history-conditioned selection, but the
  learned mean utility did not transfer beneficially to this target sample.
- The robust-value policy subtracted source-LODO overestimation penalties from
  the same supported mean predictions. It made zero target tool calls and
  exactly matched the generalist (token-F1 0.07358). For example, predicted
  positive CONCH initial utilities of +0.0105 to +0.0963 were below its 0.1239
  penalty, while source-retrieval predictions up to +0.1579 were below penalties
  of 0.2725--0.3206. This is evidence of conservative rejection, not evidence
  that robust specialist collaboration improves answers.
- Single-tool CheXagent was again the strongest numerical arm: token-F1 0.08044,
  gain +0.00686 with interval [-0.00448, 0.02022] (4 improved, 2 harmed; 9
  routed calls). CONCH reached 0.07198 (-0.00160); source retrieval 0.05865
  (-0.01494); XRV findings 0.06095 (-0.01263, interval entirely below zero);
  and XRV anatomy 0.05870 (-0.01488). One CheXagent call followed an image-only
  CXR routing error on a gross specimen, showing that modality routing remains
  a material failure mode.
- No target method logged a tool runtime error or invalid controller action.
  Original images were always preserved, predicted segmentation was supplied
  only as a second labelled overlay, and deterministic padding was enabled.
  Peak method-level PyTorch allocation was 23.90 GiB for CheXagent; observed
  total device use stayed within the shared 48 GiB GPU.
- These results validate the new paired-intervention collection, continuous
  value head, exact-history support, multi-capability execution and pessimistic
  fallback. They do not show target lexical improvement for either learned
  value policy, and token-F1 is not medical factuality or hallucination. All
  recorded domains are still proxy groups rather than hospitals.

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
