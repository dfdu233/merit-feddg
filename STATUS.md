# Current status

## Complete packets and external answer verification (2026-09-10; full GPU run active)

- Added `verified_packets`: complete native packets, lossless scalar/shared-field
  compression, an identical-content layout comparison, and frozen external answer
  arbitration. See [method, code lineage and experiments](docs/VERIFIED_PACKETS.md).
  The optimization directly addresses the diagnosed information-loss and Gate
  problems, but is not yet an efficacy result. The verifier margin is not a
  calibrated correctness probability; unsupported modality, same-model checking,
  text truncation and inconsistent preferences abstain to the exact baseline.
- The evaluation is now aligned to conversation
  `01a05d29-9d0b-7121-8a53-488b8cd1a125`: the official full 451-case VQA-RAD
  test manifest, `anchor-ce-v1` CE/OE prompt contract, deterministic 64-token
  generation and the frozen ANCHOR scorer. `answer_type` selects only the same
  prompt applied to every arm and is removed before routing, experts and
  arbitration. Scheduling-only strided sharding does not divide the dataset;
  all 451 cases must merge before scoring.
- Completed work is reused without reading labels at runtime. Generalist imports
  a sanitized 451-answer package from the dedicated ANCHOR Greedy run and is not
  regenerated; its text and token IDs were independently reproduced exactly on
  all 364 overlapping stopped-run cases. Expert inference first uses the complete
  451-case semantic-spatial run's request-keyed cache after exact expert-weight,
  exclusion, route-ID and manifest-size checks. Old routes predate per-route
  `group_id`, which is disclosed in provenance; every expert-cache lookup still
  includes the current image path and image-hash group and therefore cannot hit
  on a mismatched image. Unavailable requests run the real frozen expert.
- A real one-case GPU preflight produced all five arms. Baseline was imported;
  `semantic_all`, `compact_rows` and `compact_all` were newly generated; and
  `compact_verified` reused the exact `compact_all` candidate. In that case the
  verifier accepted only a semantically equivalent wording change, which will
  not be counted as a medical success. It recorded two image-text renderings,
  paired scores, margins, decision and timing.
- Full regression at commit `9703c57` collected 704 tests and completed with no
  failures (two optional skips); the verified-packet suite passed 22/22 and
  `git diff --check` passed. No dependency upgrade, new download, threshold
  change, target run or result-driven protocol change was made.
- Container-visible GPU 0 (host GPU 1) is running even indices as detached PID
  `2607735`, identity
  `72f36cd3907456c90166699cd0a7fe007de3eff937f6696e04f41a0dae5e5082`,
  under `runs/matched-verified-packets-anchor`. At the first health check it had
  completed 10/226 cases, used 18.6/49.1 GiB and reached 98% GPU utilization,
  with no traceback or OOM. `shard-0.failed-reuse-validation.log` preserves the
  harmless pre-model validation failure that led to the explicit legacy-route
  audit fix. Final results do not exist until host shard 1 also completes and
  the atomic merger writes the root protocol/results.

## ANCHOR-aligned native-entry evaluation (2026-09-10; stopped at 362 common cases)

- Added an independent, strictly training-free `native_claims` protocol; see [implementation and runbook](docs/NATIVE_CLAIMS.md). Six arms isolate native-entry packing, cheap attribute checks, spatial delivery and paired local-removal gating.
- The new protocol now uses the exact paper-baseline generation boundary from
  Codex conversation `01a05d29-9d0b-7121-8a53-488b8cd1a125`: the complete 451
  VQA-RAD manifest, frozen `anchor-ce-v1` closed/open prompts, deterministic
  64-token generation, the same images and no target subset. `answer_type` only
  selects the two fixed prompt suffixes and is removed before routing, expert
  selection, Gate and runtime generation. Final scoring uses ANCHOR
  `mixed-medical-vqa-table-v2-source-typed-primary` and decoder v11, with
  regenerated Generalist required to match the dedicated Greedy answers on
  451/451 cases.
- Local checks are bounded per case. Spatial evidence beyond the check budget is
  downgraded to an explicitly unverified semantic fallback, while evidence that
  is semantic-only is currently accepted as `semantic_only_unverified`; this is
  a diagnosed Gate weakness, not a correctness claim. No fitted confidence
  weights, calibration cards or learned bridge are introduced.
- The run was stopped by request and both workers are no longer running. The
  frozen six-arm common snapshot contains 362/451 cases (206 closed, 156 open);
  available per-arm caches are 364/364/364/364/363/362 in arm order. This is
  completion-order selected and diagnostic only. Unified/CE/OE scores are:
  Generalist 51.06/62.62/35.79, `semantic_all` 52.38/62.62/38.85,
  `entry_all` 47.97/54.85/38.88, `entry_filtered` 47.97/54.85/38.88,
  `hybrid_all` 47.69/54.85/38.24 and `hybrid_gate` 48.45/56.80/37.44 percent.
  Gate recovers +0.76 points versus `hybrid_all` but remains -2.61 points below
  Generalist. The 364 available Generalist answers match the dedicated ANCHOR
  Greedy baseline exactly 364/364, excluding baseline drift. Mean engine time
  is 0.676 s Generalist versus 3.140 s Gate (4.65x).
- Gate audited 408 native entries: 246 accepted and 162 rejected. Of the
  accepted entries, 152 are `semantic_only_unverified`, 93 have positive local
  removal gain and one is an unverified budget fallback. There is no calibrated
  Gate probability: expert confidence is null, XRV scores are explicitly
  uncalibrated, and local-removal gain measures reliance rather than medical
  correctness. `attribute_check` passed 2,784/2,785 entries, so filtering is
  functionally inactive. Concrete good/bad cases show both lucky CheXagent
  corrections and wrong semantic/spatial evidence overriding a correct
  Generalist. See the [complete stopped result bundle](docs/results/vqarad_native_claims_stopped_2026-09-10/README.md)
  and [Gate/literature diagnosis](docs/results/vqarad_native_claims_stopped_2026-09-10/GATE_FAILURE_AND_CONFIDENCE_RESEARCH_2026-09-10.md).
- Validation: 680 tests passed and two optional tests skipped; the focused
  native/semantic suite passed 31 tests and `git diff --check` passed. A real
  closed/open two-case smoke reproduced both paper Greedy answers exactly in
  the regenerated Generalist. It also exposed a genuine bad case: on
  `vqarad-official-test-0003`, Generalist and semantic arms retained the correct
  `right` heart border, while `hybrid_all` and `hybrid_gate` changed it to
  `left`. Gate execution is therefore not being treated as medical success.
- Host physical GPU 0 remained unavailable from this container. To use it
  without duplicating the full run, commit `5346bb7` enables deterministic
  scheduling-only sharding for `native_claims`; the two shard unions must still
  cover the complete 451 cases before finalization, and `dataset_partitioned`
  remains false. The initial unsharded PID `2495870` was stopped after 12/451
  complete cases; its 77 atomic per-arm cache files are retained as an
  incomplete historical run but are not reused because the cache identity
  intentionally binds implementation bytes. Container GPU 0 (previously
  mapped to host GPU 1) ran shard 0/2 as detached PID `2501208`, while host GPU
  0 ran shard 1/2. Both shards use identity
  `edcf48420dd394d764533ec9cc24be2e555887d916adc7fa29a9d74181ac603d`
  and the existing offline weights. Both jobs have now stopped; their partial
  caches and logs are retained. Because container `/root` is not visible
  on the host, the already-cached XraySigLIP, CLIP tokenizer, BiomedBERT and
  CheXagent dynamic-module dependencies were copied without downloading to the
  3.3 GB shared `/home/dbw/hf-shared` cache. All required tokenizer/config
  lookups resolve in strict offline mode. The first host attempt passed
  XraySigLIP but failed before a complete case because the shared cache did not
  yet contain `openai/clip-vit-base-patch32`; that log is retained as
  `shard-1-host.failed-missing-clip.log`. After completing the shared cache,
  the restarted host shard advanced from 6/225 to 8/225 in 15 seconds with no
  traceback, OOM, runtime error or empty sampled output. At the same checkpoint
  the container shard was at 73/226. Logs are
  `runs/matched-native-claims-anchor/shard-0.log` and
  `runs/matched-native-claims-anchor/shard-1-host.log`. No dependency upgrade,
  download, threshold change, target run or result-based protocol change was made.

## ANCHOR-aligned semantic-spatial rerun (2026-09-10; shard 0 complete, shard 1 recovery required)

- The matched runner now renders the frozen `anchor-ce-v1` contract used by the
  dedicated paper baseline: closed questions append `Please answer Yes or No.`
  and open questions append `Give only the short answer. Do not explain.` All
  five arms receive the identical per-case prompt and 64-token budget. Only the
  answer-blind `closed`/`open` task type selects the template; the runtime still
  rejects labels and references, and the type is removed before routing, expert
  selection and generation.
- A real two-case GPU preflight (one closed and one open VQA-RAD case) reproduced
  the dedicated LLaVA-Med baseline exactly, character for character: `No, there
  is no evidence of an aortic aneurysm in the chest X-ray image.` and `The right
  side of the heart border is obscured in the chest X-ray.` This establishes
  prompt/backend alignment before the full rerun; complete 451-case equality
  remains a post-run acceptance check.
- The runner supports deterministic strided case sharding. Two workers share one
  scientific identity but write disjoint shard artifacts and case caches; an
  advisory lock merges all five arms and routing records in original manifest
  order only after every shard is complete. This changes scheduling, not the
  experiment definition. Full regression passed 659 tests with two optional
  skips; the focused semantic-spatial suite passed 10 tests.
- The current container exposes one device as container GPU 0; this maps to host
  GPU 1, while host GPU 0 is not visible inside the container. Shard 0 completed
  all 226 even-indexed cases under identity `e63b52e0...`; its regenerated
  generalist is exactly identical to the dedicated paper baseline on 226/226
  answers. Relative to it, `semantic_all`, `hybrid_all`, `hybrid_contrast` and
  `hybrid_gate` changed 73, 77, 18 and 0 answer strings respectively. These are
  intervention counts, not accuracy results; scoring waits for all 451 cases.
  The completed artifacts and log are under
  `runs/matched-semantic-spatial-anchor/e63b52e0ade6b740854572a2f5305678873a564b0edc3bed9f959f67abe48fa4/shards/0000-of-0002`
  and `runs/matched-semantic-spatial-anchor/shard-0.log`.
- Host shard 1 is not running. Its two launch attempts reached the first
  CheXagent-required case and exited because the host process did not resolve
  the nested XraySigLIP dependency from the existing `/root/.cache/huggingface`
  cache while offline. The checkpoint is present and complete locally, so the
  recovery is to relaunch host GPU 0 with explicit `HF_HOME`, `HF_HUB_CACHE`
  and `HF_MODULES_CACHE`; no download or dependency change is needed. The
  per-case cache makes this resumable. Because later commit `dcfed50` adds
  default runtime fields and therefore changes the cache identity even though
  its new branches are inactive for this protocol, the recovery worktree is
  pinned at the shard-0 commit `8010604` in
  `/home/dbw/merit-feddg-eval-8010604`; its output and artifact arguments point
  back to the shared main workspace. The failed log is
  `runs/matched-semantic-spatial-anchor/shard-1.log`.
- For the final apples-to-apples table, the frozen evaluator is
  `mixed-medical-vqa-table-v2-source-typed-primary` with current answer decoder
  `medheval-decoded-eval-v11-explanatory-binary-source-audited`. A read-only
  rescore of the immutable 451-answer VQA-RAD baselines gives unified/CE/OE:
  Greedy 48.47/61.35/32.31, ICD 47.16/59.36/31.84, VCD 44.89/53.39/34.23,
  DoLa 46.55/57.77/32.47, MMedPO 49.91/60.16/37.04 and MedRAG
  39.48/46.61/30.53 percent. Unified is the paper-facing primary score; CE is
  strict accuracy and OE is answer-token recall. The current ANCHOR-aligned
  run will be evaluated by this exact code after both shards merge.

## Detailed diagnosis and experimental successor (2026-09-10)

- Final reporting is frozen to the paper-baseline evaluation provenance in
  Codex conversation `01a05d29-9d0b-7121-8a53-488b8cd1a125`. In addition to
  the within-run matched analysis, every arm will be scored by ANCHOR
  `mixed-medical-vqa-table-v2-source-typed-primary`: sample-weighted CE strict
  0/1 plus OE answer-token recall, with CE accuracy, OE recall, parse rate,
  repetition and unresolved cap hits reported separately. The MERIT and ANCHOR
  VQA-RAD manifests match on 451/451 ordered questions, references, mapped task
  types and decoded RGB images; their IDs and JPEG byte hashes differ and will
  be mapped explicitly. The dedicated paper Greedy baseline and this aligned
  rerun both use the `anchor-ce-v1` prompt contract and deterministic 64-token
  generation. The regenerated in-run generalist is therefore required to
  reproduce that baseline on 451/451 answers; the completed even shard
  currently passes this acceptance check on 226/226. Historical pre-alignment
  unrestricted-prompt runs remain contextual only and will not be mixed into
  the final table.
- Real-model validation at commit `e6ec0ed` passed 54 focused tests and a
  one-case five-arm GPU smoke test. The final semantic context was bounded to
  1,834 input tokens plus 64 reserved tokens under the 2,048-token limit;
  CheXagent semantics and BiomedParse evidence were presented, and the hybrid
  arm also built a spatial packet. The complete 451-case run is active as
  detached PID `2270821`, output identity
  `46662ead4eb2ebeb14ace380ea76a5179a7157e8e7f959a0f771422a09d477c6`
  under `runs/matched-semantic-spatial`, with log
  `runs/matched-semantic-spatial.background.log`. It uses the existing huatuo
  environment and offline local weights; no dependency or threshold changed.
- Read [results, research and method review](docs/RESULTS_AND_METHOD_REVIEW_2026-09-10.md)
  before launching another run. The 451-case bundle checksums pass. Nine of the
  twelve accepted gate interventions lower original-image likelihood; the old
  contrast gain can increase merely because the neutral-image score falls more.
  CT receives evidence in 51/172 cases and MRI in 4/105, by predicted modality.
- Added an opt-in `semantic_spatial` matched protocol and
  `configs/matched_semantic_spatial.yaml`: semantic-only, hybrid ungated,
  hybrid contrast and multidimensional hybrid gate, plus a regenerated baseline.
  Scalar evidence uses existing frozen token embeddings; dense geometry uses
  the existing parameter-free spatial operator. This is not arbitrary latent
  alignment, and dense tensors are not transmitted losslessly.
- Gate dimensions cover delivery, relevance, semantic redundancy, original-image
  gain and visual-contrast gain. Uncalibrated reliability remains unknown;
  perturbation stability is audited, not promoted to a correctness certificate.
  Neither target answers nor CE/OE labels enter generation or gate decisions.
- Local suite: 642 passed, 17 optional skips. No new real-model GPU outcomes yet.
  Existing negative results remain the empirical evidence; the new configuration
  is a hypothesis to test, not a performance claim.

## Full VQA-RAD training-free spatial run (2026-09-10; complete, do not scale)

- The first detached run was stopped at 61/451 because the optional BiomedParse
  assets were absent; its partial run `bf41177c...` is retained but is not a
  result. After explicit download authorization, the official gated
  `microsoft/BiomedParse` v1 checkpoint was installed locally (1,803,167,371
  bytes, SHA-256 `66716517a59e5b8060dc87732a4d65fea1f699ecec3f6a8440749f3ea2b917dc`)
  with clean source pinned to `db5c10782dab2377db4f68bbc03f71c54572e51b`.
  Required BiomedBERT and CLIP tokenizer assets are cached for offline use.
- The existing environment was retained. Missing v1 inference packages were
  added without dependency upgrades; the required detectron2 fork was built
  from `42121d75e10d9f858f3a91b6a39f5722c02868f0` as a CPU extension because
  system NVCC 13.2 is incompatible with the existing PyTorch CUDA 11.7 build.
  A local Pillow compatibility alias was necessary for the pinned detectron2
  code; Torch, Transformers, Pillow and the LLaVA-Med weights were not changed.
- A real VQA-RAD chest-radiograph probe loaded the frozen v1 checkpoint and
  returned 1024x1024 left/right lung soft masks with nonzero foreground. The
  focused regression suite passed 100 tests with one optional skip. Formal run
  `f42e591e4c92265c1c3d1827ac0378167285a353b143079d2657e8714d794b4b`
  completed all four arms on all 451 official test questions without runtime
  error or empty output. Its regenerated baseline is exactly identical on
  451/451 texts and token sequences to the preceding matched vector run.
- Fixed-reference scores are negative versus the generalist. Strict mixed is
  0.2945 generalist, 0.2885 spatial_equal, 0.2885 spatial_weighted and 0.2923
  spatial_gate. Target-blind content-aware mixed is respectively 0.4364,
  0.4193, 0.4193 and 0.4320. Ungated strict pairing has zero improvements and
  four harms; gated pairing has zero improvements and one harm. The content
  diagnostic counts 1/10 improve/harm for ungated and 1/3 for gate, but its
  apparent gate improvement is the medically false statement that chest X-ray
  is safe in pregnancy because it does not use ionizing radiation, so it is not
  evidence of clinical gain.
- Spatial evidence genuinely entered generation in 229 cases and changed 46--47
  texts. The channel diagnostics are all negative: XRV findings n=141,
  mean content delta -0.0440 (1 improve, 8 harm); XRV anatomy n=33, -0.0455
  (0/2); adopted BiomedParse n=229, -0.0336 (1/10). Equal and lexical-weighted
  arms have identical aggregate scores and differ in only three texts, so the
  relevance weighting shows no measurable benefit.
- Gate accepted 12/403 usable-evidence events (2.98%), rejected 51 negative
  visual-contrast gains and saw 340 unchanged candidate pairs. It reduced harm
  but accepted concrete reversals including right-to-left heart border,
  no-pneumothorax to pneumothorax and no-free-air to free air. Gate cost was
  437.3 s, 17,618 candidate tokens and 252 verifier forwards. Mean engine time
  rose from 0.528 s generalist to 0.843 s spatial_equal and 1.661 s gate; the
  latter is 3.14x baseline even with shared expert caches. Do not scale this
  method without a stronger task-relevance/reliability mechanism.
- The compact Git-tracked result bundle is
  [`docs/results/vqarad_spatial_full_2026-09-10`](docs/results/vqarad_spatial_full_2026-09-10/README.md).
  It includes aggregate metrics, protocol, spatial audit, routing and all 451
  per-case four-arm outputs with expert/gate/timing traces. The approximately
  18 GB base64 soft-mask values remain in the raw run directory; mask metadata
  and explicit omission markers are retained in the bundle. The evaluator was
  repaired so per-expert channel subsets include only adopted evidence, not
  applicability-rejected calls; generation outputs and global scores were
  unchanged.

## Strict training-free replacement (implementation summary; evaluated above)

- Current method: [training-free spatial collaboration](docs/TRAINING_FREE_SPATIAL.md).
  Supersedes the source-training recommendation in the historical null report below.
- The default vector config uses a zero-parameter spatial operator over existing
  VLM tokens; the factory refuses trained bridge checkpoints. Training and source
  preparation CLIs are retired. No new dataset split or CE/OE generation branch.
- Adds source-balanced lexical importance, XRV positive CAM and optional pinned
  BiomedParse v1 soft masks with explicit anatomy/sequence applicability.
  Score-only and generation/retrieval experts are excluded from the spatial protocol.
- New four-arm full-manifest runner: generalist / spatial_equal /
  spatial_weighted / spatial_gate. Gate probes the full answer budget and uses
  evidence-free sequence teacher forcing (four verifier forwards for differing pairs).
- Pre-run validation: 634 tests passed, 17 optional-dependency tests skipped;
  Ruff and launcher syntax checks passed. Official checkpoint execution and
  clinical evaluation are reported above. Nonzero token changes and nonzero
  gate acceptance were not counted as medical success.

## Matched vector-gate full evaluation (2026-09-10; complete, null result, do not scale)

- Fast-forwarded to `e7c4093` and ran the complete 451-question official
  VQA-RAD test manifest with matched `generalist`, `tensor_all` and
  `tensor_gate` arms. All arms used the same original image, prompt,
  unconstrained 64-token budget and deterministic preprocessing. Generation
  never loaded references or answer types; no target subset or tuned threshold
  was created. The detached run completed 1,353/1,353 outputs without runtime
  error or empty answer.
- The official QA split is not image-disjoint: 202/203 test images occur in the
  official train split, covering 1,059/1,793 train questions. Those rows were
  excluded by exact RGB identity before bridge training. Of the remaining 111
  source images and 734 questions, one answer-blind deterministic question per
  image was considered and 39 images had a fully registered applicable expert
  result. The resulting source cache has zero image overlap with the full test,
  contains 32 XRV findings, three XRV segmentation and five Biomed anatomy
  items, and uses only test-manifest image identities for exclusion: no test
  question field or answer participates in source preparation.
- A one-epoch, 39-step source-only bridge was trained against frozen LLaVA-Med.
  Its learned fusion gate is nonzero but very small (`0.003635`). This satisfies
  the non-null checkpoint guard mechanically, but the complete evaluation shows
  that the intervention is functionally negligible; it must not be presented as
  effective bridge training.
- All fixed scores are identical across the three arms: strict mixed 0.2945,
  strict closed 0.2749, open answer-token recall 0.3190, target-blind content-
  aware mixed diagnostic 0.4320 and generic Token-F1 0.0752. `tensor_all`
  changed only 1/451 strings (`suggests` to `shows`) with no strict, content or
  lexical score change. `tensor_gate` was byte-identical to generalist on
  451/451. The matched generalist itself is byte-identical on 451/451 to the
  preceding matched-permissions baseline, so baseline drift is not the cause.
- Real native evidence did enter `tensor_all`: 141 XRV finding packets, 33 XRV
  anatomy packets (99 actual mask structures) and six Biomed anatomy packets
  were adopted and presented over 178 cases. Nevertheless the spatial channel
  changed 0/33 answers, Biomed changed 0/6 and findings changed only wording in
  1/141, with zero measured medical gain or harm. This is evidence of ineffective
  fusion, not evidence that the expert channels are clinically useless.
- The training-free gate executed after all 180 expert calls and accepted 0/180.
  In 179 calls the baseline/evidence candidates were identical within the fixed
  eight-token probe; the sole changed candidate reached the image-versus-mean-
  color verifier but had negative gain (`-0.01510`) and was rejected. Thus zero
  adoption is a null intervention, not safety or success. It generated 2,880
  probe tokens, made 26 verifier score queries and spent 91.18 s (0.507 s per
  eligible case) without changing a final answer.
- Mean recorded engine time was 0.576 s/case for generalist, 0.607 s for
  `tensor_all` (+5.4%) and 0.843 s for `tensor_gate` (+46.3%). These sequential
  shared-cache measurements are not independent cold-start latency, but they
  establish that the current gate overhead is material and brings no observed
  benefit. The full evaluation stage, including model load and 77.8 s image
  routing, finished in about 16.1 minutes.
- Do not scale and do not relax the `1e-6` admission threshold. The next valid
  step is a predeclared source-side bridge-training study with held-out source
  validation that demonstrates a material tensor effect before another target
  run. It should separate bridge undertraining/fusion amplitude from gate
  horizon and verifier behavior; the completed target answers must not be used
  to choose epochs, fusion strength or probe length.
- Necessary repair: experiment YAML inheritance now merges partial `generalist`
  overrides with the base model identity. Previously the new vector config
  discarded `id`, checkpoint, LLaVA source and vision-tower paths and failed
  before loading. A regression test covers this behavior. The source builder,
  complete 36-binding VQA-RAD contract, detached pipeline and offline evaluator
  are now reproducible scripts. Detailed results and raw paths are in
  `docs/VECTOR_GATE_RESULTS_2026-09-10.md`.

## Matched permissions full evaluation (2026-09-10; complete, do not scale)

- Ran the complete 451-question official VQA-RAD test manifest under the new
  matched four-arm protocol at commit `bc530bd`. Generalist, point, uncertainty
  and permissions used the same prompt and unconstrained 64-token generation;
  no split, calibration, policy fit or label access occurred during generation.
  The detached job completed 451/451 per arm without OOM, empty output or tool
  runtime error in 1,340 s, then released the GPU.
- Frozen strict mixed scores were generalist 0.2945, point 0.2158, uncertainty
  0.2152 and permissions 0.2274. Strict closed parsing is a major confound: only
  119/251 generalist and 57/251 permissions answers began with explicit yes/no.
- A target-blind content-aware diagnostic scored generalist 0.4320, point 0.4375,
  uncertainty 0.3948 and permissions 0.4580. Permissions was +0.0261 versus the
  matched generalist (38 improved, 25 harmed, 388 tied by that evaluator), while
  generic Token-F1 decreased from 0.0752 to 0.0682. These are not clinician-
  adjudicated medical-benefit counts.
- Null integrity passed: all 273 no-tool cases had exactly identical text in all
  four arms. Every evidence arm attempted 352 calls on 178 cases, and rejected
  packets were no longer silently marked adopted.
- The intended relation mechanism was not exercised. Permissions presented
  CheXagent on 172 cases, XRV anatomy on 31 and Biomed anatomy on 6, but XRV
  findings on 0/141 because all permission packets exceeded the real token budget.
  Although pre-packing audits constructed 1,128 finite relations, exactly zero
  relations entered a final prompt. The apparent gain therefore reflects packet/
  channel selection, principally CheXagent replacing XRV, not validated finite-
  relation reasoning.
- Clear reference-aligned corrections coexist with clear harms: cardiomegaly,
  pneumothorax and mediastinal-shift errors were sometimes corrected, while heart-
  border laterality, cardiac-contour narrowing, cardiomegaly, organ-system and ECG-
  lead answers were also made wrong. No lexical or zero-call improvement is counted
  as medical success.
- The shared perturbation cache executed 522 extra XRV forwards in 16.75 s. These
  are response-variation audits, not correctness or domain guarantees. Sequential
  per-arm times are warm/cache-order dependent and not a latency comparison.
- Do not scale or relax thresholds. A future source-designed ablation must make
  finite relations actually fit and isolate packet length, expert selection and
  relation content without tuning on these official-test answers. Full results,
  examples and raw artifact hashes are in
  `docs/MATCHED_PERMISSIONS_RESULTS_2026-09-10.md`.

## Generic evidence patch integration (2026-09-10; GPU not run)

- Integrated supplied 4ce0538 patch after fast-forwarding to 065a03d, preserving
  the server's latest reports and uncertainty transport fixes.
- Added optional measurement permissions, native precision API, actual-context
  evidence packing and a four-arm full-manifest generation entry point. No new
  clinical benefit or arbitrary-domain guarantee is claimed.
- Integration fix: matched-evaluation caches now include expert provenance and
  recursively hashed adapter code; JSON manifest/cache reads use UTF-8.
- Full patch regression: 614 passed in 96.07 s. After cache hardening, all 21
  permissions/transport targeted tests passed, including the new cache regression.
  Ruff, CLI --help and diff checks passed. No GPU experiment or calibration ran.
- Input patch SHA256: 5471e0bace39e694977079fd96a60ee7c3295e0603d010a37e6eae16236bcd4c.
- See docs/GENERIC_EVIDENCE_RUNBOOK.md for server execution and evaluation limits.

## Full official VQA-RAD new-method evaluation (2026-09-10; negative, do not scale)

- Completed all 451 official test questions (251 closed, 200 open; no custom
  split) in a detached `nohup`/`setsid` job using the existing huatuo environment,
  LLaVA-Med and local expert weights. Only the new method was generated; the
  existing same-row baseline was reused offline. No dependency/checkpoint was
  downloaded or upgraded, and no train/test reference entered generation.
- This was the training-free uncertainty-preserving `all_evidence` component,
  not the full source-fitted value/DG policy: no source cohort means no fitted
  gate or source retrieval. It cannot establish a domain-generalization claim.
- Generic Token-F1 was 0.0694 overall (closed 0.0386, open 0.1080). Frozen ANCHOR
  v9 mixed evaluation was 0.2163 versus the historical LLaVA-Med baseline 0.4825;
  strict closed accuracy was 0.1355 versus 0.6096, heavily affected by 195/251
  new answers violating the leading yes/no contract. A same-parser content-aware
  diagnostic still decreased 0.0600 overall and 10.36 percentage points on closed
  questions; it marked 48 rows improved, 73 harmed and 330 tied. Generation
  protocol differences make this diagnostic rather than a clean causal ablation.
- Evidence transport is a blocking failure. Of 352 executed/adopted expert calls,
  only 178 case prompts contained compiled evidence: XRV findings 141/141, XRV
  anatomy 31/33 and Biomed anatomy 6/6, but CheXagent 0/172. Whole CheXagent
  packets were silently excluded by the 2,600-character packing budget after an
  earlier packet, even though traces said adopted. Therefore no CheXagent benefit
  can be claimed and 172 calls were wasted.
- Real 512x512 XRV anatomy masks ran, but only structure names, boxes and foreground
  fractions entered 31 prompts; `visual_views=0`, so mask pixels/overlays/crops did
  not. This is serialized spatial evidence, not validation of the visual bridge.
  Clear laterality/measurement harms include right-to-left heart-border and gastric-
  bubble flips, plus a correct cardiac-silhouette relation changed to its negation.
- Findings sensitivity over 141 audits was min/mean/max
  0.00427/0.02131/0.08362; anatomy over 33 was
  0.00728/0.01877/0.05243. The 522 extra probe forwards took 20.47 s. These are
  photometric variations, not clinical correctness or domain certificates.
- Total wall time was 408.01 s (6 min 48 s), mean 0.924 s/question, recorded expert
  work 76.77 s and peak allocation 22.46 GiB. Cost is manageable at this size but
  scientifically wasteful until presentation-aware calling is fixed.
- Do not scale or run target on this result. Required next fixes are explicit
  adopted-versus-presented tracing, budget-aware calls/packing, closed-answer format
  enforcement and a matched same-generation baseline/evidence ablation without
  tuning on test results. See `docs/VQARAD_OFFICIAL_NEW_METHOD_RESULTS_2026-09-10.md`
  for examples, scorer caveats, hashes and raw local artifact paths.

## Independent uncertainty source confirmation (2026-09-10; do not scale)

- Froze and ran a new 20-case VQA-RAD source-only cohort (seed 29, 10 cases per
  proxy group) with `configs/uncertainty_source_pilot.yaml`. It has zero sample,
  group and pixel-hash overlap with the prior six-case canary. The final run is
  `runs/uncertainty-source-confirm-seed29/llava/e48b658050cd0748`; source cases
  were 20, target generations exactly 0, policy fitting false, Block-NONE parity
  20/20 and all observed domains remained proxies. The source manifest SHA-256 is
  `48ddc71ef063030eeebfa79c96cc29a8dd1bcc6163b0a202af6c5958361d8ef5`.
- This confirms execution, not safety or domain generalization. A conservative
  reference-aligned semantic review of the 20 `source_cases` uncertainty answers
  found four clear corrections, two new wrong answers and fourteen without a
  clear medical correction. Token-F1 instead marked eight improved, one harmed
  and eleven unchanged, demonstrating that the lexical diagnostic misses harm
  and overstates useful change. This review is not an independent clinician audit.
- The four clear retrieval-assisted corrections were `vascular`, `axial`,
  `T2-weighted MRI` and `left MCA`. The two clear harms were `psoas` changed to
  `erector spinae`, and a correct `calcified` mass rim changed into a left-frontal
  location answer. Related source answers are explicitly scoped to their original
  images, but the generalist can still follow an irrelevant association.
- Among the four CXR cases, CheXagent clearly corrected one right-lower-lobe
  answer; the other three provided no clear new medical benefit. XRV findings
  produced three nonempty 18-label outputs but 0/3 clear medical corrections.
  The one real XRV anatomy segmentation produced nonempty 512x512 Left Lung,
  Right Lung and Heart masks, but changed an already correct `right` answer to
  the wrong `left lung`. Thus nonempty expert calls and visible serialized
  evidence are not successes.
- The segmentation branch transported structure names, mask-derived boxes and
  foreground fractions into the text memory, but used zero visual views. It did
  not pass a pixel mask/overlay/crop to LLaVA in this configuration. Consequently
  this run tests serialized spatial evidence, not the visual spatial bridge.
- Photometric sensitivity was measured for three XRV findings calls and one XRV
  anatomy call: 12 additional forwards took 0.643 s; maximum changes were
  0.0138248, 0.0145904 and 0.00316687 for findings and IoU distance 0.0201379 for
  anatomy. These are response variations, not correctness or domain certificates.
  Point and uncertainty-text answers were textually identical in 31/32 matched
  arms; the remaining pair differed only by the word `image`, with no score or
  medical change. There was therefore no range-specific medical benefit.
- Diagnostic generation took 171.42 s and recorded expert events took 23.83 s
  (retrieval 8.31 s, XRV findings 1.73 s, CheXagent 11.87 s, anatomy 1.92 s), for
  195.25 s of summed measured work or 9.76 s/case across the full multi-arm
  diagnostic. This is not production latency: 180 answer branches replayed shared
  tool outputs, and cold starts dominate several expert totals. Peak PyTorch
  allocation was 22.68 GiB.
- No threshold was relaxed, no target case was generated, and no dependency or
  checkpoint was downloaded or upgraded. No code repair was required. The result
  does **not** support scale-up: proxy-only data, sparse per-expert counts, two
  demonstrated harms, no range-specific effect and no pixel-level spatial arm
  remain blocking scientific gaps. See
  `docs/UNCERTAINTY_CONFIRM_RESULTS_2026-09-10.md` for the case-level interpretation
  and raw artifact hashes.

## Native tensor bridge patch integration (2026-09-09; no medical GPU result)

- Applied the supplied non-text bridge patch while preserving the pending
  uncertainty pilot. Tensor mode is a separately trained, opt-in experiment, NOT
  a replacement for the training-free framework or a learned DG certificate.
- Added base-model provenance checks on production checkpoint loading; rejected
  silent tensor-to-text diagnostics and unsupported uncertainty/tensor mixing.
- Patch-author benchmark/test statements are not new local results. See
  docs/NATIVE_TENSOR_BRIDGE.md for scope, training requirements and review limits.
- Integration verification: 591 passed in 66.88 s; full Ruff and diff checks passed.
  tensor_train --help also passed. These are CPU/protocol checks, not real medical inference.

## Uncertainty-preserving interface source canary (2026-09-09; negative medical result)

- Optional finite native-observation envelopes now reach NativeSession; the original
  observation always participates, numeric ranges widen, and unsupported semantic
  intersections are omitted. No disease threshold, alias repair or training added.
- Ran the fixed previous 6-case source-only compatibility cohort with existing huatuo,
  LLaVA-Med and expert weights, fully offline. The final run is
  `runs/uncertainty-source-pilot/llava/a726dadab20bfb9a`: source cases 6,
  target generations 0, policy fitted false, all domains proxy, and baseline/Block-NONE
  parity held. This reused cohort is a plumbing canary, not the separately frozen
  confirmation set required for a performance claim.
- The first attempt exposed that private `native_uncertainty` alternatives leaked into
  legacy text controls and exceeded the 2048-token LLaVA context. Controls now receive
  only the original point observation. A second attempt showed that repeated typed
  fields made a complete 18-label packet exceed the character/context budgets. The
  repaired schema factors shared semantics and uses a label-keyed value/range table;
  all labels and exact native float values remain present. Full verification collected
  593 tests: 591 passed and 2 skipped; Ruff lint and diff checks passed.
- Real XRV classification audit outputs populated 2/2 empirical packets with all 18
  labels, four observations each (the original tool output plus three probe forwards).
  Other capabilities remained explicitly `unknown`; empty retrievals stayed empty.
  This is NOT native-task calibration, a truth certificate or a completed DG algorithm.
- Medical evidence was negative. On the left-retrocardiac-opacity case, both the point
  and range presentations answered a `right lower lobe ... mass`; Token-F1 rose from
  0.1333 to 0.1818 only through lexical overlap, while the finding and laterality were
  wrong. On the cardiomegaly case, the range changed a generic point response into an
  unsupported right-lower-lobe mass; both scored zero. Thus 0/2 cases showed a
  range-specific medical benefit and 1/2 showed a medically adverse semantic change.
- The nine real XRV probe forwards took 0.450 s total; sensitivities were 0.0118836 for
  anatomy (reported only, no spatial envelope), 0.0086983 and 0.0180077 for findings.
  Six-case replay generation time was 85.81 s, recorded tool-event time 19.21 s, and
  peak PyTorch allocation 23.71 GiB. Sensitivity is photometric response variation,
  not correctness or domain applicability.
- Existing domain gate and default generation behavior remain unchanged. New pilot
  uses one original image and no keyword request contracts. See
  docs/UNCERTAINTY_PILOT_RESULTS_2026-09-09.md for examples and artifact paths.
- Previous lexical request-scope repair remains a diagnostic baseline, not the proposed
  general method. No threshold was relaxed, no target was run and no zero/lexical-only
  change is counted as success.

## Request-scope repair after negative canary (implementation; GPU pending)

- Reviewed the synchronized `4d05a4a` negative source report and new shared discussion.
  No new clinical performance claim: panel transport, finite task coverage and
  domain risk are separate issues. Existing remote LLaVA compatibility fixes stay.
- Added opt-in question-only request contracts and focused native evidence with no
  unrelated top-score fallback. Source replay compares the same raw expert output.
- New pilot keeps the original single image, disables panel diagnostics/probes and
  leaves existing domain-neighborhood gating unchanged. No KV or parameter edits.
- Read docs/RESEARCH_REASSESSMENT_20260908.md for checked literature, limitations
  and the fixed-cohort server task. configs/request_scoped_pilot.yaml is the new pilot.
- Finite lexical coverage is explicit; this does not fix image routing or establish
  correctness. Domain-risk calibration still needs actual source support.

## Evidence-operator source canary (2026-09-08; negative medical result)

- A compact case-by-case view is in
  `docs/EVIDENCE_OPERATORS_CANARY_RESULTS_2026-09-08.md`.
- Safely fast-forwarded `main` from `11061d4` to requested commit `e2c99c4` while
  preserving the two pre-existing untracked weekly-report files. Reused the existing
  huatuo environment (Torch 2.0.1, Transformers 4.37.2, XRV 1.5.4), LLaVA-Med,
  CLIP and expert weights with all Hugging Face/dataset offline flags set. No package
  was installed/upgraded and no model/data was downloaded.
- Ran the fixed 6-case source-only pilot from `configs/evidence_operators_pilot.yaml`.
  The final repaired run is
  `runs/evidence-operators-source-final2/llava/12a7597d747e9f7a`; the earlier native
  two-token and intermediate compatibility failures remain under
  `runs/evidence-operators-source`, `runs/evidence-operators-source-fixed`,
  `runs/evidence-operators-source-panel`, `runs/evidence-operators-source-panel-padfix`
  and `runs/evidence-operators-source-final`. Every final case had exact
  baseline/Block-NONE token parity. Source cases were 6, target generations exactly
  0, policy fitting false, and every observed domain kind `proxy`.
- The released LLaVA-Med path accepted two image tokens structurally but generated
  blank/whitespace plus EOS for all initial two-image branches. The necessary repair
  makes a pixel-preserving left/right panel before deterministic padding, uses one
  image token, and places the concise panel-source statement before the medical
  question. The final run produced nonempty output for 6/6 duplicate controls and
  all five anatomy spatial arms. This proves transport into generation, not benefit
  or native multi-image support. Duplicate formatting itself changed answers: the
  already-correct MRI `axial` answer became `coronal`, the CT artery changed, and one
  CXR diagnosis changed from effusion to pneumothorax.
- Real XRV anatomy segmentation produced nonempty 512x512 masks for Left Lung,
  Right Lung and Heart, with foreground fractions 0.22076, 0.23244 and 0.06831.
  Their original-image normalized boxes were respectively
  `[0.5488,0.0892,0.9336,0.7038]`, `[0.0898,0.0892,0.5137,0.7199]` and
  `[0.4238,0.3957,0.7070,0.6444]`. The selected left-lung crop was
  `[541,107,921,846]`; its equal-size control `[0,461,380,1200]` had zero pixel-box
  overlap. Crop, control and overlays had distinct recorded pixel digests. These are
  predicted thoracic anatomy masks, not lesion masks or anatomical ground truth.
- Spatial evidence did not provide medical benefit. For the anatomy question whose
  reference is `breasts`, baseline/duplicate said `liver`, crop and matched control
  both said `heart`, while overlays mostly described the left lung or overlaid regions
  instead of answering the organ. Thus neither the zero Token-F1 gains nor nonempty
  calls are successes; crop equaling control gives no localization-specific gain.
- Other evidence channels were also negative on this canary. For the left
  retrocardiac-opacity case, XRV led to `atelectasis` and CheXagent to
  `consolidation`; both were directionally related but failed left-retrocardiac
  localization and all reported Token-F1 gains were negative. For the cardiomegaly
  case, XRV emphasized infiltration/lung opacity and CheXagent answered enlarged
  pulmonary artery, so neither established the reference. MRI and CT retrievals were
  empty/unusable. CONCH ranked connective tissue/smooth muscle on a gross specimen
  and did not answer `prostate`; this is a scope mismatch, not a pathology success.
- Photometric probes executed 9 additional real XRV forwards. Anatomy sensitivity
  was 0.0118836 (mean probability change 0.0009385; 0 empty channels; 0.191 s for
  three forwards). Findings sensitivities were 0.0086983 and 0.0180077 (0.063 s and
  0.073 s for three forwards each). They measure gamma-0.95/1.05 response changes,
  not correctness, applicability or verified clinical invariance; the audit threshold
  was not changed and the audit did not alter adoption.
- Final source diagnostic replay totaled 50.42 s (2.47--20.13 s/case) with maximum
  PyTorch allocation 23.47 GiB. Recorded tool-event time was 12.29 s: Biomed anatomy
  2.01 s, retrieval 0.070 s, XRV anatomy 0.750 s, XRV findings 0.285 s, CheXagent
  6.49 s (including a 5.61 s first-load call), and CONCH 2.68 s. Presentation replays
  reused tool output and are not online end-to-end latency; individual spatial answer
  generations were 0.56 s crop/control and 0.96--1.00 s overlays versus 0.385 s for
  that case's baseline replay.
- Engineering transport now works, but the scientific result is negative. No target
  run, threshold relaxation, sample expansion, clinical-benefit claim or DG claim was
  made. The fixed cohort remains too small and proxy-only for statistical conclusions.
  Full verification passed 541 tests with 2 skips (543 collected); Ruff checks and
  `git diff --check` passed.

## Typed behavior probes and evidence operators (implementation)

- Added opt-in source diagnostics sharing native tool outputs across text, overlay,
  region crop and equal-area location control. Existing dynamic framework stays intact.
- XRV classification/segmentation photometric probes report sensitivity and extra
  forwards, not correctness. Default off; pilot audit does not alter adoption.
- Rejection requires an explicit source-selected threshold; unsupported/empty probes
  cannot certify applicability. Existing source-neighborhood admission is unchanged.
- See docs/DOMAIN_AWARE_EVIDENCE_OPERATORS.md and configs/evidence_operators_pilot.yaml.
  No new GPU, clinical benefit or domain-generalization result is claimed.

## Input applicability collaboration implementation (remote GPU pending)

- Core remains NativeSession/CapabilityRuntime with heterogeneous native evidence.
  Optional input-aware admission filters tool descriptors before controller choice.
- Added source intervention collection, expert-native neighborhoods, four gate
  ablations, whole-domain residual cross-fitting, cache fingerprints and live traces.
- Default risk measures paired answer harm, not native expert correctness. Native
  task losses can be provided explicitly; neither is automatically hallucination rate.
- Native features: CONCH, BiomedCLIP/retrieval, XRV findings. Other plugins need
  a validated domain_embedding adapter; no hidden shared-encoder substitution.
- Read docs/INPUT_APPLICABILITY.md before GPU runs. Three real source domains are
  normally needed for robust residual calibration, four for nested source-only
  evaluation with the default two-domain support. Existing proxy data is diagnostic.
- No new clinical result or domain-generalization improvement is claimed.

## ROVER source pilot implementation (2026-09-07; remote GPU pending)

- Added `python -m merit_feddg.rover_run`: five matched-input region arms,
  per-token position/stability traces, production baseline parity and case cache.
- Existing XRV anatomical segmentation can prepare proposals in a separate
  process; generic predicted region JSON supports later lesion-model integration.
- 11 new CPU tests cover gating, geometry, prefix sharing, provenance rejection,
  and end-to-end mocked cache recovery. No local medical-model inference performed.
- Start with 2 source images / 16 tokens using `docs/ROVER_QUICKSTART.md`.
  The replay implementation repeats forward work; no speed or DG guarantee.
  Real anatomy proposals are not lesion localization or clinical verification.
- The v0.12 negative scientific result below remains unchanged.

## v0.12 source-only GPU canary (2026-09-07)

- Ran the prescribed check-only and source-only canary on pulled commit `e417662`
  with `/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python`, Torch 2.0.1,
  Transformers 4.37.2 and GPU 0 (NVIDIA GeForce RTX 4090, 48 GiB). All hub and
  dataset offline flags were set. Existing local LLaVA-Med, CLIP vision tower,
  CONCH, BiomedCLIP, CheXagent and XRV assets were reused; no model was downloaded
  and no dependency was installed or upgraded. The run is
  `runs/claim-grounded-v012-canary/llava/4fa43d6d663f956b`.
- The canary completed 8 source cases and 15 calibration records from 5 actual
  expert/case executions: CONCH 3, XRV findings 1 and CheXagent 1. Four other
  cases had no compatible expert. Every executed observation was adopted, tool
  and runtime errors were zero, and target generations were exactly zero. The
  successful hard assertion established baseline/Block-NONE/zero-strength token
  identity for all 8 cases; 127 production next-token positions were also checked.
- The real path emitted `typed-clinical-evidence-v1`. CONCH scores remained
  `visual_match` with `relative_similarity`, unknown polarity, whole-image scope,
  a non-exhaustive-catalog limitation and no diagnosis interpretation. XRV scores
  remained whole-image `finding_score` values with
  `uncalibrated_independent_sigmoid`; low scores did not become negatives.
  CheXagent's `Consolidation` remained an unverified specialist statement with
  unspecified spatial scope. Format-null packets retained tool identity/scope but
  contained only `withheld control`; no real observation, score or summary survived.
  Spatial and retrieval semantics were not exercised by these routed cases, but the
  full 497-pass/2-skip suite covered their fail-closed contracts and the unconnected
  `ClaimCommitVerifier` contract. Ruff, three Bash syntax checks and diff checks passed.
- Across the 5 expert/case interventions, direct real versus baseline had mean
  Token-F1 gain +0.02373 (2 improved, 1 harmed, 2 unchanged); direct-format had
  -0.01823 (1/2/2). Direct content gain was +0.04196 (3 positive, 0 negative,
  2 zero). All 5 direct real outputs differed from direct-format, so expert content
  affected surface text, while the nonzero direct-format gains also expose a material
  format-envelope effect. At both bounded strengths 0.25 and 0.5, real and format
  each had mean output/control gain 0, content gain 0, and 0/0/5
  improved/harmed/unchanged. Bounded real differed from format-null only for
  `pathvqa-train-4584` at each strength, without any Token-F1 change.
- Per expert, CONCH direct content/output/control gains were
  +0.01480/+0.02899/+0.01418 over 3 cases; XRV findings were
  +0.16226/+0.06667/-0.09560 over 1 case; CheXagent was
  +0.00312/-0.03497/-0.03810 over 1 case. For every expert, bounded content,
  output and control gains were all exactly zero at both strengths. Every one of
  the 15 records satisfied `content_gain = output_gain - control_gain`.
- Manual review does not support a clinical improvement claim. In
  `pathvqa-train-10960`, every branch failed to answer the expected "alternate
  areas" and the CONCH matches were irrelevant to the alveolar question. In
  `pathvqa-train-15641`, the Token-F1 increase came from generic tissue wording;
  the image is a gross specimen routed as microscopic pathology, so CONCH's catalog
  use was scope-inappropriate. `pathvqa-train-4584` had the same gross-specimen
  routing problem and no branch answered "prostate". In the CXR case, XRV
  atelectasis/infiltration and CheXagent consolidation were directionally closer to
  the reference left retrocardiac opacity than the baseline right pleural effusion,
  but neither localized the finding to the left retrocardiac region and neither is
  sufficient to establish correctness. Token-F1 is not a medical hallucination rate.
- All three qualification cards selected no action (`NONE`, strength 0); source
  selection and independent confirmation were not attempted because every observed
  domain kind was `proxy`. Each card correctly rejected with
  `proxy_or_unverified_domains`. These hash/dataset proxy domains are not hospitals
  and cannot prove domain generalization.
- Source replay totaled 258.08 s (32.26 s/case; 5.75--87.45 s); actual tool calls
  totaled 13.42 s. Peak PyTorch allocation/reservation was 22.95/23.66 GiB. Maximum
  per-token KL was 0.02, maximum case spend was 0.24136 versus the 0.32 budget, and
  evidence stopped after token 15. Per-case cleanup returned observed device use to
  18 MiB. All 8 fingerprinted case caches were saved; a same-config recovery rerun
  exited in 15.59 s without loading checkpoint shards or executing any new case/tool,
  and reproduced 8 source cases, 15 records and zero target generations. The original
  timing is source replay, not online end-to-end latency.
- Engineering canary acceptance passed, but the scientific stop condition fired:
  bounded evidence did not change quality, direct gains were sparse/confounded, and
  two CONCH routes were medically mismatched. No sample expansion or target
  generation/evaluation was started. `ClaimCommitVerifier` remains a tested semantic
  contract and has not entered live decoding.

## v0.12 typed evidence and content-controlled qualification (2026-09-07)

- Added a real-path `typed-clinical-evidence-v1` presentation for classification,
  spatial, retrieval and generative specialists. It retains native score semantics,
  source-only provenance and spatial scope without converting them into diagnoses.
- Added a direct format-null arm. Source records now separate content gain
  (`real - null`), output gain (`real - generalist`) and control gain. Qualification
  requires positive content and output gains in every real source domain on both the
  selection and independent confirmation groups, and can select direct or bounded use.
- Added a fail-closed `ClaimCommitVerifier` contract: spatial claims need spatial
  evidence, negative claims need explicit exhaustive coverage, proxy domains cannot
  produce qualified certificates, and conflicts cause REVISE rather than voting.
- The verifier is not yet wired into live claim-boundary decoding. The immediate real
  experiment is source-only typed-bridge diagnosis; no target benefit or clinical
  hallucination reduction is claimed. See `docs/CLAIM_GROUNDED_V012.md`.
- Local verification: 499 tests passed under the repository Python 3.10 environment;
  Ruff, three Bash syntax checks and `git diff --check` passed. These are protocol and
  orchestration checks, not a real-weight clinical efficacy result.

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
