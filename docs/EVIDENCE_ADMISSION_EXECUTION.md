# Evidence admission execution repair (2026-09-16)

## Frozen scope / CS197 evidence update

Base: `08f5d998637036b2527b0e434f0990845a573f20`. Independent worktree
`/home/dbw/merit-feddg-admission`, branch `experiments/evidence-admission-20260916`.
Historical worktrees, outputs and the original eight-question runner are unchanged.

The preceding canary established that descriptor-entry filtering did not protect
inherited evidence, and that matching one entity could admit an unsupported entity.
Anatomy's declared catalog also exceeded the default returned structures.

This iteration is **engineering enforcement, not a research algorithm or novelty
claim**. Following `skills/cs197-research/SKILL.md`, the single question is whether
the declared boundary is enforced at actual delivery without destroying both
positive controls. The broader authority hypothesis/Bit Flip remains unproven.
Its location is the existing capability/applicability interface, not uncertainty,
calibration, clinical reliability or a new constrained-decoding technique.

Core: one delivery-view function, explicit question-bound requests, the existing
XRV native definitions/configuration/returned packets, and both text/spatial
session paths. Periphery frozen: automatic parser, experts, weights, prompts,
scorer, full benchmarks, uncertainty, verifiers, RAG and training.

## Implementation

`merit_feddg/evidence_admission.py` introduces `EvidenceRequest` and `delivery_view`.
The latter returns a separate view and audit; it never mutates the raw item.

Admission checks the current question binding, explicitly complete entity and
dimension sets, modality/task, configured expert/capability/scope, native contract,
adapter-native semantics, configured structure selection, and actual returned
entries. All requested entities must be supported and returned; a partial request
is not permission to modify the entire answer.

- Classification: retain only matching native `finding`/`score` pairs and their
  original independent-sigmoid semantics. No normalization, new threshold or
  inference from absence. CAM is not admitted as finding-presence evidence.
- Anatomy: retain only matching configured, returned structures with their native
  masks and transforms. No disease inference, physical measurement, fabricated
  full-image ROI or fallback to another returned organ.
- Raw summary/provenance may describe excluded entries. The enforce view removes
  those summaries and retains only the native adapter identity; new admission
  audit remains outside prompt/payload. Packets marked as using target annotations
  are rejected **before** provenance reduction, preserving the existing safeguard.
- Missing explicit semantics, unknown entities, undeclared contracts or unsupported
  adapters fail closed in enforce. This is intentional limited deployment, not a
  universal parser or a replacement for all historical experts.

`NativeSession.context` filters before any text renderer, overlay/crop or spatial
packet compilation. `_tensor_session` uses the same filter; temporary evidence
sessions carry the same request/configuration. Session reuse keys include current
question, request, configured specs and admission mode, preventing stale admission
from surviving a changed request. `CapabilityRuntime` supplies current specs.

Modes (legacy remains default):

| Mode | Delivered input | Admission audit |
|---|---|---|
| legacy | Original items and original behavior | No new admission decision |
| audit | Original items, byte-compatible prompts | Would-be decision, stored separately |
| enforce | Filtered delivery views only | Actual decision, stored separately |

Example opt-in (caller supplies an explicit, complete request; not a model parser):

```python
request = EvidenceRequest(question, ('Effusion',),
                          frozenset({'finding_presence'}), 'cxr', 'open_vqa')
session = NativeSession(probe, image, prompt, question,
                        replace(config, admission_mode='enforce'),
                        authority_specs=specs, evidence_request=request)
# New, JSON-cached and inherited raw packets all enter the same boundary:
session.propose(NativeState(items=raw_items), length=64)
```

## Frozen validation design

`scripts/run_evidence_admission_canary.py` imports the **unchanged eight questions**
from the previous runner. It verifies the previous image's pixel identity, expert
configuration, checkpoint hashes, old packet file and old outputs. It uses the
same generation configuration, prompt construction, model and TRAIN image.

The explicit structures list requested entities and dimensions before inference.
They are authored execution fixtures, **not outputs of the automatic parser**.
`assess_authority` still runs separately and is saved per probe, including its old
incorrect `exact` decisions. No disease blacklist, eight-question regex patch or
new ontology parser was added.

Two fresh expert calls produce current native packets. The normal route uses the
existing descriptor gate; the cached path JSON-roundtrips the fresh packet; the
inherited path uses the actual previous run's saved raw packet. Model outputs are
query-independent XRV native outputs on the same image and are reused across the
eight probes. This is not 24 independent expert calls.

Each probe generates a no-evidence base and three modes × three entry paths:
**80 actual LLaVA-Med generation calls**. Positive controls are exactly the
effusion-presence and left-lung-location questions. Six other probes must deliver
no expert evidence, including aorta (declared but not configured/returned).

H1: forbidden text/spatial packets are absent for every path, while both positives
retain only their requested entries. H0: bypass remains or success is achieved by
rejecting everything. Stop on parity loss, leaked forbidden content or lost
positive coverage. No reference answers or medical scoring are used.

## Final measured results

Final identity:
`66c258394f21a1b4978509256c26b85169b035caddb77f16d8c3a3eeaa180183`.
All eight probes completed. **936 CPU tests passed in 14.97 s**, including 22 new
admission tests. New-module/runner/test Ruff checks and `git diff --check` pass.
One additional overlay test initially had incomplete synthetic coordinate
metadata; its fixture was corrected to provide the renderer's required original
and model sizes. Production geometry checks were not relaxed.

| Delivery path | legacy: probes with evidence | audit: probes with evidence | enforce: probes with evidence |
|---|---:|---:|---:|
| Normal descriptor route | 5/8 | 5/8 | 2/8 |
| JSON-cached packet | 8/8 | 8/8 | 2/8 |
| Historical inherited packet | 8/8 | 8/8 | 2/8 |

The two delivered probes are exactly the legal positives, not arbitrary survivors.

| Unchanged probe | Old automatic status | Explicit enforce result |
|---|---|---|
| Is there a pleural effusion? | exact | Only Effusion, original raw score |
| Is there a left pleural effusion? | partial | Block: unsupported laterality |
| Is there tuberculosis? | denied | Block: entity outside native catalog |
| Is there pneumothorax or tuberculosis? | exact | Block: incomplete native entity coverage |
| Where is the left lung located? | exact | Only actual Left Lung mask/transform |
| Where is the aorta located? | exact | Block: structure not configured/returned |
| Where is the tumor relative to the heart? | exact | Block: tumor outside native catalog |
| How large is the heart in centimeters? | denied | Block: unsupported physical measurement |

- Legal positive deliveries: **6/6** (2 questions × 3 paths); every delivered
  native list contains exactly the requested entry, with original score/mask.
- Forbidden deliveries: **0/18** (6 questions × 3 paths). No spatial records were
  compiled for them. Their token equality to no-evidence base is a regression
  check, **not an accuracy or medical safety claim**.
- Final enforce normal/inherited token parity: **8/8**.
- Cached/inherited token parity: **24/24** across three modes.
- Historical base/normal/inherited token parity: **24/24** against the old run.
- Audit/legacy token parity: **24/24**. New audit is outside the answer prompt.
- Actual spatial compiler: Left Lung positive yields one record; classifier
  presence yields zero spatial records as intended (classification is not a mask).
- Raw EvidenceItems remain unchanged after every generation/view check.

This rejects the "everything was denied" explanation and supports the narrowly
scoped **execution repair**. The old parser still misclassifies mixed entities;
these experiments do not validate a general request parser. No medical scorer,
good/bad-case accuracy, bootstrap clinical comparison, or full benchmark was run.

### Measured cost

Two fresh expert calls including loading: **3.036 s**; actor loading **11.469 s**;
80 generations plus delivery/spatial compilation **105.678 s**; sum of timed
components **120.183 s**. CPU tests/imports/preflight are outside that sum.

| Mode | Normal path seconds | Cached path seconds | Inherited path seconds |
|---|---:|---:|---:|
| legacy | 10.746 | 11.290 | 10.466 |
| audit | 10.746 | 11.126 | 10.767 |
| enforce | 10.394 | 10.157 | 10.653 |

Each cell contains eight generations; the eight no-evidence generations are extra
and included in the 80-call total. These single-run wall times do not establish
a speed advantage. The initial v1 replay consumed another 80 actor calls, two
expert calls and **63.898 s** of timed components before the provenance safeguard
was added. Its results/logs remain intact; its execution cost is not hidden.
The substantial v1/v2 timing variation is not attributed to the admission rule.

Device: authorized physical GPU1/container GPU0,
`GPU-3846413a-4238-d307-b1f3-10c2dfbe002c`. No host GPU0 or new environment used.

## Residual limitations

- An explicit request is a trusted caller input. The system cannot establish that
  a human/caller omitted no semantic dimension. General automatic parsing is
  unchanged and remains unsolved; no wider in-scope coverage claim is made.
- This is one TRAIN image with controlled prompts, not a benchmark sample or
  estimate of medical harm/rescue. Admission, token parity and tests do not prove
  correctness or improved accuracy.
- Only the two reviewed XRV adapter formats have enforce delivery views. Historical
  entrypoints still default to legacy; they must explicitly opt in with configured
  specs and complete requests. No global rollout or silent policy change occurred.
- The boundary is `NativeSession`; code directly calling a low-level generalist
  without this session is outside the enforced interface and needs its own review.
- Native definitions/configuration and cached packet provenance must be trusted.
  The canary verifies weight/image identities externally; this patch is not a
  cryptographic proof of arbitrary packet origin or a validated clinical ontology.
- The production spatial-packet compiler is exercised with real masks, and CPU
  tests cover native overlays, tensor and combined semantic/spatial sessions.
  GPU generation preserves the prior **text-semantic** configuration; no new
  spatial-guidance efficacy arm or geometric accuracy experiment is claimed.

## Reproduction and private outputs

```bash
cd /home/dbw/merit-feddg-admission
OPENBLAS_NUM_THREADS=1 bash scripts/with_gate_kb_env.sh -m pytest -o addopts='' -q
OPENBLAS_NUM_THREADS=1 bash scripts/with_gate_kb_env.sh scripts/run_evidence_admission_canary.py --output runs/admission-eight-probes-v2 --check-only
tmux new-session -d -s admission-eight-probes 'cd /home/dbw/merit-feddg-admission && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 timeout 900 bash scripts/with_gate_kb_env.sh scripts/run_evidence_admission_canary.py --output runs/admission-eight-probes-v2 > runs/admission-eight-probes-v2/gpu.log 2>&1'
```

Use a fresh output directory for reruns. The runner refuses to overwrite completed
results. Runtime environment, model source and weights are reused; no upgrade,
download, training, reference access or full benchmark generation is required.

Local artifacts: `runs/admission-eight-probes-v2/{frozen.json,fresh-packets.json,
cases/*.json,result.json,gpu.log,cpu-tests.log}`. They retain raw evidence, delivery
views, separate automatic/explicit audits, transport, outputs and timing locally.
No raw images, patient-linked answers, masks, credentials or weights are published.
The initial successful v1 replay is retained separately; after adding a target-
annotation safeguard, the final source was retested and replayed in v2 rather
than rewriting its identity or outputs.
