# MedCAVE flow audit and bounded optimization

## What actually runs

Original image → frozen image-type inference → modality/intent tool filtering →
baseline answer → specialist observation → original-image candidate synthesis
with accumulated observations → candidate-bound checks → certified ACCEPT,
another eligible model source, or exact baseline FALLBACK.

The current configuration contains only CheXagent for CXR. It does not contain
a general CT/MRI expert or a second distinct specialist. In the first 32 TRAIN
rows, model-inferred types are 8 CXR, 17 MRI and 7 CT. Therefore the 25% candidate
coverage is primarily a configured-tool coverage limit, not evidence that the
risk threshold is too conservative. These predicted modalities are not oracle
labels or validated OOD detection.

## Why this cannot yet demonstrate correction

1. **No measured candidate-specific verification.** Native prose does not supply
   semantic support bound to a newly generated candidate. A post-generation
   verifier would need to construct that support explicitly. Relabeling raw
   expert confidence or copying an upstream score is not a fix.
2. **No validated risk certificate.** Qualification, OOD references and independent
   source calibration are absent. Acquisition cannot make those resources appear;
   adding a second unqualified expert would not make acceptance valid.
3. **Question relevance is not the same as modality relevance.** CXR description
   eligibility does not establish suitability for projection, modality or every
   anatomical question. Capability contracts should eventually narrow this using
   prespecified semantics and source validation, not rules selected from target
   errors. No new exclusion rule was fitted in this change.
4. **Score ties hide semantic changes.** The prior run changed 7/8 candidate
   answers, including an assertion reversal and changed imaging-plane wording.
   These are not adjudicated clinical good/bad cases. Long explanatory CLOSED
   answers can both fail normalized exact despite different meanings. The zero
   score-improvement/decline counts do not establish medical equivalence.
5. **Missing cost accounting and redundant routing.** The 32 questions have only
   16 image hashes, yet the original run made 32 fixed-prompt image-type calls.
   Its 1.410 s mean excluded model startup and routing.

## Implemented now

- Cache image-type outputs by validated image hash within one run/model binding.
  This fixed-prompt inference never uses the question. Do not cache answers,
  evidence, verifier decisions or reflections across questions. Original routing
  provenance is retained; separate cache-hit and actual-call counters distinguish
  inherited inference time from current lookup cost.
- Record model-load, routing, baseline, expert, candidate and verification time.
  Case wall time includes routing and the inference loop, excludes initial model
  load and file serialization; model load is reported separately. These are
  synchronous wall-clock measurements, not GPU-kernel profiling.
- Report candidate coverage, text changes and candidate-only per-task score
  improvement/decline counts. Questions without candidates are not candidate
  evidence. Existing final scores and scorer definitions remain unchanged.
- Regression checks exercise repeated-image routing reuse and empty candidate
  diagnostics. Missing legacy timing remains null rather than invented zero.

No medical threshold, model, prompt, evidence ordering, decoding, calibration
policy or original configuration changed. A new 32-row output parent preserves
the prior trajectories. This is an efficiency/parity check on development data,
not new evidence of generalization.

## Real validation result

New root: `runs/medcave-source32-routing-cache/f1c188c610bb9559d04c1189851909fc4d0c9fec265795cf29d4e957ab5e007f/`.
Same first 32 TRAIN rows, same configuration and authorized GPU. All 32 cases
completed; full-manifest completeness remains false.

| Check | Original | Optimized |
|---|---:|---:|
| Actual image-type calls | 32 | 16 |
| Image-type cache hits | 0 | 16 |
| Image-type wall time | 5.836 s | 3.192 s |
| Actual expert / candidate calls | 8 / 8 | 8 / 8 |
| Candidate coverage | 25% | 25% |
| Candidate text changes versus own baseline | 7 | 7 |
| Accepted answers | 0 | 0 |

Cross-run parity: baseline 32/32, generated candidates 8/8, final 32/32 have
identical text, token IDs and generation config. The remaining 24 candidates
are absent in both runs and are not counted as generated-candidate parity.
Thus observed answers and their existing diagnostic scores are unchanged.

Optimized measured costs: generalist/pool initialization 13.085 s, baseline
generation 24.217 s, specialist calls including lazy load 13.171 s, candidate
generation 6.125 s, verification 0.001 s. Routing plus case loops total 46.710 s;
adding initialization gives 59.795 s, excluding preflight, output serialization
and offline evaluation. Cheap verification reflects missing actual verification
models, not an efficient proven clinical verifier.

Routing calls decreased exactly 50%; measured routing time decreased about 45%
in this single pair of runs. Do not extrapolate that to 45% end-to-end speedup:
generation dominates, old startup timing is unavailable, and no repeated
steady-state benchmark was performed. All 740 tests and Ruff passed.

## Next scientific decision

Before a medical-effectiveness expansion, align the neutral generation/scoring
contract with the intended comparison and separately validate a real
candidate-specific evidence verifier on independently audited source data.
Then qualify modality/task-specific experts and measure improvement **and harm**
before enabling acceptance. These are substantive next steps, not completed by
the routing cache or by 740 passing CPU tests.
