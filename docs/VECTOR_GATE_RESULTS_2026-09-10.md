# Matched vector-gate VQA-RAD results, 2026-09-10

Verdict: the unified tensor path executed with real classification and spatial
outputs, but the trained bridge was functionally inert and the dynamic gate
accepted nothing. This is a complete negative/null result and does not support
scale-up.

## Frozen experiment and leakage boundary

- Starting code: `e7c4093`; run identity:
  `306777618498b319de2466b413f45cd08e3e608f878ed587e0e0494eca1988c3`.
- Complete official VQA-RAD test: 451 questions, 203 unique images. The test was
  not split, filtered or used to fit a policy.
- Arms: `generalist`, `tensor_all`, `tensor_gate`; same image, question, prompt,
  64-token free-generation budget and deterministic padding.
- References and `answer_type` were absent from generation and loaded only by
  the offline evaluator.
- Existing local LLaVA-Med and expert weights were reused with offline mode. No
  dependency upgrade or model download occurred.

The official VQA-RAD train/test QA split shares 202 of 203 test images. Exact RGB
identity filtering removed 1,059 official-train questions on test images. The
remaining pool had 734 questions on 111 independent images. Deterministic
question selection did not inspect answer content; 39 unique images had an
applicable, fully registered native result and formed the source bridge set.
Their exact image overlap with the complete test is zero.

The source cache contains 32 XRV finding items, three XRV anatomy segmentations
and five Biomed anatomy items. One epoch produced 39 bridge updates. The bridge
checkpoint is source-only and its learned fusion gate is `0.003635`; nonzero is
not interpreted as a meaningful intervention.

## Results

| Arm | Strict mixed | Strict closed | Open recall | Content diagnostic | Token-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generalist | 0.2945 | 0.2749 | 0.3190 | 0.4320 | 0.0752 |
| Tensor all | 0.2945 | 0.2749 | 0.3190 | 0.4320 | 0.0752 |
| Tensor gate | 0.2945 | 0.2749 | 0.3190 | 0.4320 | 0.0752 |

There were zero improvements and zero harms under both the frozen strict score
and the target-blind content-aware diagnostic. `tensor_all` was text-identical
to generalist on 450/451 cases. The only difference was case `0198`:

- Reference: `right vs left sided pathology`.
- Generalist: “The image suggests that the right-sided pleural effusion is
  better identified on a PA ... compared to a lateral CXR.”
- Tensor all: the same answer with `shows` replacing `suggests`.
- Tensor gate: exactly the generalist answer.

The wording change has no metric or clear medical-content effect. The generalist
arm is byte-identical on all 451 rows to the earlier matched-permissions run,
confirming that baseline drift did not hide an effect.

## Evidence and gate audit

`tensor_all` executed, adopted and actually presented:

| Expert channel | Cases | Final text changed | Content gain/harm |
| --- | ---: | ---: | ---: |
| XRV findings | 141 | 1 | 0 / 0 |
| XRV anatomy | 33 | 0 | 0 / 0 |
| Biomed anatomy | 6 | 0 | 0 / 0 |

The 33 XRV anatomy items contained 99 real predicted mask structures. They were
compiled as spatial tensor evidence and passed through the bridge; this differs
from earlier text-only box/fraction transport. Their failure to change an answer
shows that the current bridge did not make spatial evidence effective. It does
not establish that segmentation itself has no medical value.

The gate saw the same 180 acquired expert results but accepted zero:

- 179: baseline and evidence candidates were identical through the predeclared
  eight-token horizon, so the verifier was skipped and the call failed closed.
- 1: candidates differed, but image-relative support gain was `-0.01510`, below
  the unchanged fixed tolerance `1e-6`, so it was rejected.
- Total: 2,880 candidate tokens, 26 verifier score queries, 91.18 seconds.

The gate arm therefore presented zero evidence and matched the generalist on
451/451. This is a null intervention and is not counted as a successful safety
result. No threshold was changed after observing the target outputs.

## Cost and decision

| Arm | Mean engine seconds/case | Relative to generalist |
| --- | ---: | ---: |
| Generalist | 0.576 | -- |
| Tensor all | 0.607 | +5.4% |
| Tensor gate | 0.843 | +46.3% |

These are sequential shared-expert-cache timings, not independent cold-start
benchmarks. Image routing added 77.8 seconds. The full evaluation stage ran from
the bridge report at 13:02:01 UTC to 13:18:07 UTC, about 16.1 minutes. The gate
overhead is already material despite producing no final intervention.

Do not scale this configuration and do not relax the gate threshold. A new
target run would only be justified after a source-only, predeclared bridge study
shows a material effect on held-out source images. That study must separately
test bridge training sufficiency/fusion magnitude, probe horizon and visual
verifier sensitivity, without using these target answers for model or budget
selection.

## Necessary implementation repair and raw artifacts

The supplied vector YAML used a partial `generalist` override, while the loader
previously replaced the complete base mapping. This discarded the model ID and
all local paths, causing an immediate `KeyError` before experimentation. The
loader now merges only this nested override, with a regression test.

Raw directory:
`runs/matched-vector-gate/306777618498b319de2466b413f45cd08e3e608f878ed587e0e0494eca1988c3/`.

- `generalist.json`, `tensor_all.json`, `tensor_gate.json`: all 451 answers and
  complete native evidence/gate traces.
- `protocol.json`, `routing.json`: frozen protocol and image-only routing.
- `evaluation-summary.json`: fixed metrics, paired outcomes, channel and gate
  audits.
- `runs/matched-vector-gate-source/audit.json`: train/test RGB-overlap and source
  evidence audit.
- `runs/matched-vector-gate-source/bridge-seed0-epoch1.training.json`: training
  losses and checkpoint identity.

SHA-256: `89c8ac55...61495` (generalist), `baf2a3d7...fd419`
(tensor all), `60c56391...a0697` (tensor gate), `46708969...0b5e`
(protocol), `998b3780...e2719` (evaluation summary), and
`83d056a7...2ad8a` (source audit).
