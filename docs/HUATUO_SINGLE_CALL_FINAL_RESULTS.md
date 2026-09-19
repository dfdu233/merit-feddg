# Single-call critic: complete dual-GPU TRAIN results

2026-09-20. HostGPU1 finished fresh SLAKE128 confirmation while hostGPU0 ran
the existing VQA-RAD87 development schedule. No TEST benchmark, training,
threshold tuning or shared-environment upgrade. Original candidates reused.

| Method | SLAKE128 fresh TRAIN images | VQA-RAD87 development |
|---|---:|---:|
| Generalist Baseline |58.6068%|60.3448%|
| Original compact MERIT |57.4349%|53.4483%|
| Independent single-call critic |59.7786%|58.0460%|
| Delta vs Baseline |+1.1719pp|-2.2989pp|
| Improve / harm vs Baseline |10 / 7|0 / 2|
| Delta vs compact |+2.3438pp|+4.5977pp|
| Improve / harm vs compact |7 / 4|7 / 2|

Frozen ANCHOR CLOSED parser + OPEN token recall, not clinical accuracy. Exact
full ID sets and complete markers verified by evaluator before scoring. SLAKE
images are pixel-disjoint from previous64 and TEST pixels; patient isolation
unknown. VQA87 has been used for development and is not a fresh holdout.

Image-paired bootstrap95% intervals vs Baseline: SLAKE[-3.9063,+6.4453]pp;
RAD[-5.7471,0]pp. Versus compact: SLAKE[-2.3438,+7.0313]pp;
RAD[-1.1494,+10.7375]pp. No statistically established general gain or
noninferiority. The earlier SLAKE64 double-order result must not be pooled
with this changed single-call confirmation protocol.

## Execution and format repair

Single judge per differing pair, deterministic SHA256(ID)-parity ordering,
reasoned prompt, BF16 SDPA,512-token ceiling. Exact model weights/source,
images, native generation configurations and scorer remain audited in protocols.
Original experts/routing are unchanged; no new patient answer generation.

SLAKEv1 stopped after45 cases on [A] instead of [[A]]. v3 replayed saved calls
without new inference, then stopped after66 cases on an explanation containing
[B] followed by final `Verdict: [[B]]`. v4 accepts repeated same-label mentions
only with a matching explicit terminal verdict. Conflicting decisions, repeated
labels without a terminal verdict and missing labels still fail. Conditional
option-list echoes are not treated as decisions. Tests cover these distinctions.
These are output-serialization repairs, not medical decision-rule changes.

Previous files were preserved. Reuse requires identical model/input/order/
generation protocol, stores hashes of source calls, reparses decisions and
retains their original measured costs. SLAKE complete identity:
f084b8d233821a549dd64c040a772de559bb9dd30cb97006b0cf1b923048ec0f.
VQA identity:b352404380f450aeb390634b4b7ba15ce3d8a73f64204994e75aed587b30b9ed.
VQA ran with the preceding parser; offline replay of all47 actual judgments
under the final parser produced identical labels. Its historical identity and
files are not relabelled as a new run.

## Coverage and cost

SLAKE:128/128 complete,66 real judge calls,62 same-text skips; compact33,
generalist95;16 explicit ties. Calls223.4649s, including inherited saved calls
exactly once (1.7458s per scheduled case). Three actual model loads across
attempts3.8826+12.6455+11.7856=28.3137s, additional to calls.

RAD:87/87 complete,47 real calls,40 same-text skips; compact5, generalist82;
12 explicit ties. Calls134.7163s (1.5485s per scheduled case), model load4.2910s.
Original candidate generation/expert costs remain in evaluator reports and
are not zero; preflight hashing, formatting/recovery, preparation, downloads
and all prior experiments are not included in an end-to-end speedup claim.

The two jobs ran concurrently on different physical GPUs. They are different
data schedules, not a misleading merged benchmark or duplicated judging of
the same answers. Reports contain no patient text, images, weights or tokens:

- reports/single-critic-slake128-confirm.json
- reports/single-critic-vqa87-dev.json

## Decision

SLAKE confirms a small positive point estimate but uncertainty remains large.
RAD recovers much of compact's loss yet still harms Baseline and captures none
of its candidate improvements. Keep original methods; do not claim universal
Gate safety, medical improvement or SOTA. Further work should diagnose missed
useful evidence and false acceptance separately, rather than interpret format
success or Baseline reuse as validation. Both jobs finished and released GPUs.
