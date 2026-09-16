# Native capability contract: real-model boundary validation

## Evidence update

Tested upstream **d7d37cf25589b332a5f41a2a29150537951f1cd5**, branch
`implementation/native-capability-contract-v1`, on 2026-09-16. Independent
worktree: `/home/dbw/merit-feddg-contract-validation`. No changes were made to the
contract implementation, parser, thresholds, expert configuration or historical
results. The validation runner and this report are published separately on
`experiments/native-contract-canary-20260916`.

**Result: descriptor filtering works for recognized forbidden requests, but the
end-to-end semantic authority boundary is not enforced. Do not scale up yet.**

This branch introduces native-task declarations and a descriptor-entry filter,
not a new reliability estimator or a candidate-support solution. The previous
authority-projection accuracy numbers must not be attributed to this branch.

## Question, controls and scientific boundary

Vector: does the declared native capability prevent out-of-scope evidence from
influencing actual generation while preserving in-scope access?

H1: forbidden requests do not deliver evidence via either normal routing or
inherited packets; accepted requests receive the declared native output.
Counter-explanation: this is only a front-door tool filter, and entity recognition
or downstream reuse still permits out-of-scope influence.

This is an applicability/interface-enforcement diagnostic. It does not establish
a novel Bit Flip, expert correctness, calibration, medical benefit or superiority
over prior constrained tool-use methods. The minimum-KL/projection hypothesis is
unchanged by these results; projection is not exercised here.

Core: frozen LLaVA-Med, existing DenseNet classifier and PSPNet anatomy segmenter,
the new contracts, real native packets, and actual text transport. Periphery
frozen: no RAG, verifier, new expert, learned gate, training, calibration,
paired-candidate construction, or full benchmark generation.

### Frozen design

1. Audit normal descriptor routing on **all 1793 unchanged TRAIN questions**, with
   their existing incumbent input modalities. Compare the current configuration
   against the same configuration with only `authority_contract` removed.
2. Select the first image in the prior frozen TRAIN pilot, without reading any
   reference answer. Pin its pixel identity and both checkpoint hashes.
3. Run the real classifier and anatomy segmenter once each on that image.
4. Run eight predeclared controlled English questions on the same real image:
   expert-native presence/location, forbidden laterality/disease/measurement,
   mixed known/unknown entities, and declared-but-not-returned anatomy.
5. For each question generate up to 64 tokens in three paths: no evidence;
   evidence admitted by the actual `tool_descriptors`; the same real evidence
   supplied directly as inherited `NativeState.items` to the production session.

These eight questions are **controlled probes, not eight original dataset QA
examples**. The inherited-packet arm deliberately tests an API bypass; it is not
a claim that a full controller naturally chose that path. There are no references,
task scores, good/bad medical labels or threshold selection in this experiment.
Token changes demonstrate influence, not clinically meaningful harm by themselves.

## Engineering results

- Relevant upstream tests: **24 passed**.
- Complete suite: **914 passed**, recorded rerun **14.27 s**.
- Focused Ruff checks on `capability_contracts.py`, `capabilities.py` and
  `test_capability_contracts.py`: passed. `git diff --check`: passed.
- Authorized GPU: physical GPU1 / container GPU0,
  `GPU-3846413a-4238-d307-b1f3-10c2dfbe002c`; mapping and CUDA allocation checked.
- Existing Python environment and model sources reused. No downloads or upgrades.
- Offline-only environment; background tmux with a 900-second timeout. Run completed.
- Real native classifier output and nonempty predicted masks obtained. Masks were
  Left Lung (foreground crop fraction 0.1959), Right Lung (0.1886), Heart (0.1322).
  All three had bounding boxes; these are predicted anatomy, not lesion masks or
  ground-truth segmentation. Segmentation quality was not scored.

### Full TRAIN routing audit (no generation or labels)

| Expert | Legacy eligible | Contract retained | Blocked | Unknown | Denied | Partial |
|---|---:|---:|---:|---:|---:|---:|
| CXR classifier | 528 | 53 (10.04%) | 475 | 292 | 173 | 10 |
| CXR anatomy | 88 | 13 (14.77%) | 75 | 11 | 63 | 1 |

The denominator is each expert's legacy-eligible subset, not 1793. This is not
expert invocation frequency or false-rejection rate. Some old eligibility was
itself broad; manually labeling applicability is required before interpreting
the loss of coverage. In particular, `unknown` means the regex parser failed to
establish dimensions, not that the specialist is medically wrong.

### Controlled GPU boundary results

| Controlled question | Expert | Contract status | Normal route | Actual issue |
|---|---|---|---|---|
| Is there a pleural effusion? | Classifier | exact | allowed | In-scope packet delivered; selected output unchanged |
| Is there a left pleural effusion? | Classifier | partial | blocked | Direct inherited packet still delivered and changed tokens |
| Is there tuberculosis? | Classifier | denied | blocked | Direct inherited packet still delivered and changed tokens |
| Is there pneumothorax or tuberculosis? | Classifier | exact | allowed | Matching one native entity admits an unsupported second entity |
| Where is the left lung located? | Anatomy | exact | allowed | In-scope mask packet delivered; output changed |
| Where is the aorta located? | Anatomy | exact | allowed | Contract names aorta, but returned packet has no aorta mask |
| Where is the tumor relative to the heart? | Anatomy | exact | allowed | Heart match incorrectly admits a tumor-localization request |
| How large is the heart in centimeters? | Anatomy | denied | blocked | Direct inherited packet still delivered and changed tokens |

- Recognized blocked routes: **3/3** preserve exact no-evidence output token IDs.
- Recognized forbidden/partial packets supplied through inheritance: **3/3**
  are presented in the prompt and **3/3** change output token IDs.
- Unexpected route admission: **2/8** controlled probes (mixed entities and
  tumor-relative-to-heart). This is a purposive diagnostic, not a prevalence estimate.
- The aorta probe is a separate **declaration/output mismatch**, not counted in
  those two parser failures: declaration permits it, actual default packet lacks it.

## Root causes and belief update

1. `assess_authority` executes only in `tool_descriptors`. `CapabilityPool.infer`
   and `NativeSession.context` have no equivalent contract enforcement, so direct
   native packets and inherited evidence bypass the new check. Reusing an expert
   payload cannot be equated with retaining its original applicability decision.
2. `matched_entities` returns any catalog match. The subsequent decision tests
   whether that tuple is nonempty, not whether all requested entities and their
   relationships are covered. A known entity can therefore admit an unknown one.
3. The anatomy contract lists the model's full native catalog, while `_xrv_segment`
   returns only `Left Lung`, `Right Lung`, `Heart` by default. Model capability,
   configured output and actually delivered evidence are different layers.
4. The transparent regex parser has limited linguistic coverage. Large numbers
   of `unknown` responses require an answer-blind applicability audit before any
   claim that in-scope coverage is preserved.

H1 is not established; the front-door-only explanation is supported. Tests
passing do not constitute end-to-end acceptance. The novelty claim remains
unestablished; repairing these boundaries would be engineering, not automatically
a research contribution.

Stop rule applied: no full VQA-RAD/SLAKE generation, no aggregate accuracy claims,
no rule relaxation and no automatic retries with modified semantics. A future
repair should first cover actual transport of inherited/new packets and align
entity coverage with returned payloads, then rerun the **same** controlled cases.
Do not expand reliability or additional specialists to compensate for these gaps.

## Compute, identity and reproduction

Two expert calls, including loading: classification **1.504 s**, anatomy **0.572 s**.
LLaVA-Med load: **10.429 s**. Twenty-four actual generation calls: **16.482 s**.
Total of these timed components: **28.987 s**. This excludes Python import,
checkpoint hashing, CPU tests and routing audit. All three arms were generated;
zero-call blocked routing was not counted as a model-generated improvement.

Actor: existing `/home/dbw/ANCHOR/hf_cache/llava-med-v1.5-mistral-7b`, original
LLaVA-Med source and CLIP weights. Exact generalist configuration inherited from
the complete historical TRAIN incumbent protocol.

Checkpoint SHA256:

- DenseNet: `56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899`.
- PSPNet: `019b167eac6b729fc1bb92bbbc185fc1730aaa65819f4e3fe718186cadc044fc`.

Experiment identity:
`9d3ae2e1633e5f2b29bbe0802d3a06253b0bdc0a50a923245e1a2ecbba25ecfa`.

```bash
cd /home/dbw/merit-feddg-contract-validation
OPENBLAS_NUM_THREADS=1 bash scripts/with_gate_kb_env.sh -m pytest -o addopts='' -q
OPENBLAS_NUM_THREADS=1 bash scripts/with_gate_kb_env.sh scripts/run_native_contract_canary.py --output runs/native-contract-canary-v1 --check-only
tmux new-session -d -s native-contract-canary 'cd /home/dbw/merit-feddg-contract-validation && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 timeout 900 bash scripts/with_gate_kb_env.sh scripts/run_native_contract_canary.py --output runs/native-contract-canary-v1 > runs/native-contract-canary-v1/gpu.log 2>&1'
```

Use a fresh output path for reruns; the runner refuses to overwrite completed
results. Private server outputs under `runs/native-contract-canary-v1/`:
`frozen.json`, `routing.json`, `native-evidence.json`, `cases/0.json` through
`cases/7.json`, `result.json`, `gpu.log`, `cpu-tests.log`. Full masks and model
responses stay on the server. No patient images, responses, credentials or
checkpoints are committed. The controlled questions above are authored probes.
