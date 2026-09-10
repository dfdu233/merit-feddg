# VQA-RAD official-test new-method evaluation, 2026-09-10

Verdict: **do not scale**. The complete 451-question official test run finished,
but it does not support a benefit claim. Under the frozen strict evaluator the
mixed score fell from the existing LLaVA-Med baseline's 0.4825 to 0.2163. Much
of the closed-question collapse is an answer-format failure, but a same-parser
content-aware diagnostic still fell by 0.0600 overall and 10.36 percentage
points on closed questions. More importantly, the evidence audit found that all
172 CheXagent outputs were called and marked adopted but none fit into the final
generation context.

## Protocol and scope

- Dataset: all 451 questions in the official `flaviagiammarino/vqa-rad` test
  split (251 closed, 200 open; 203 unique images). There is no custom split.
- Only the new method was generated. The baseline was not rerun; the frozen
  historical LLaVA-Med greedy answers were reused offline on the same 451 rows.
- No official-train answers, test answers or reference-derived decisions entered
  generation. Source retrieval was disabled because this protocol has no source
  cohort.
- The evaluated branch is training-free uncertainty-preserving `all_evidence`,
  not the full source-fitted value/DG policy. It therefore cannot establish a
  domain-generalization result or evaluate the source-fitted gate.
- The successful job used the existing huatuo environment, local LLaVA-Med and
  expert checkpoints, with no dependency upgrade or download. It ran under
  `nohup` plus `setsid`, was reparented to PID 1 and completed independently of
  VS Code.
- The initial 4,000- and 3,000-character evidence budgets exceeded LLaVA-Med's
  2,048-token context on the first case. The completed run used a 2,600-character
  transport cap. No medical score, admission threshold or routing threshold was
  relaxed.

## Scores

The repository's generic lexical summary is:

| Subset | n | Exact match | Token-F1 | Mean end-to-end seconds |
| --- | ---: | ---: | ---: | ---: |
| All | 451 | 0.0000 | 0.0694 | 0.924 |
| Closed | 251 | 0.0000 | 0.0386 | 0.942 |
| Open | 200 | 0.0000 | 0.1080 | 0.903 |

Exact match is zero because the model produced sentence-form answers while the
references are usually short labels. It is not an official accuracy measure and
must not be interpreted as zero clinical accuracy.

For comparison with the existing same-row LLaVA-Med greedy file, the committed
ANCHOR v9 strict binary parser plus answer-token recall gives:

| Metric | Existing baseline | New method | Delta |
| --- | ---: | ---: | ---: |
| Mixed CE accuracy + OE recall, 451 | 0.4825 | 0.2163 | -0.2662 |
| Closed strict accuracy, 251 | 0.6096 | 0.1355 | -0.4741 |
| Open answer-token recall, 200 | 0.3231 | 0.3178 | -0.0052 |

Only 56/251 new closed answers were parseable under the frozen strict protocol;
195 began with prose rather than an explicit `yes` or `no`. This is a real
evaluation-contract defect, not evidence that all 195 answers were medically
wrong. To separate formatting from content, a target-blind content-aware parser
was applied identically to the old and new files as a diagnostic. It scored
0.4847 versus 0.4248 overall (delta -0.0600) and 0.6135 versus 0.5100 on closed
questions (delta -0.1036). Across all rows it marked 48 improved, 73 harmed and
330 tied. The historical and new generation budgets differ (64 versus 32 tokens),
so this paired comparison is diagnostic rather than a clean causal ablation.

The result therefore does not support the claim that the method is at least as
good as the generalist. Nor can the drop be assigned solely to gating: this run
used forced compatible-tool order, not the source-fitted value gate, and the
zero-call subset also changed because its answer protocol differs from the
historical baseline.

## Did evidence actually enter generation?

There were 352 real expert events: 174 cases had two calls, four had one and 273
had none. Every event executed without a recorded runtime error and was marked
adopted. Reconstructing the exact final `evidence_memory` under the 2,600-character
budget produced a materially different result:

| Expert | Called | Present in final prompt | Audit interpretation |
| --- | ---: | ---: | --- |
| XRV findings | 141 | 141 | Full uncertainty tables reached generation |
| XRV anatomy | 33 | 31 | Two whole packets did not fit |
| Biomed anatomy | 6 | 6 | Text evidence reached generation |
| CheXagent description | 172 | 0 | All calls were wasted by packet ordering/budget |

Thus only 178/451 final prompts contained any expert memory. An `adopted=true`
trace is not sufficient evidence of transport. This is a blocking framework bug:
the runtime records adoption before bounded serialization, and `all_evidence`
calls the next expert even when its packet cannot be presented. No score from
this run can be attributed to CheXagent.

The 31 transported XRV anatomy packets contained structure names, normalized
boxes and foreground fractions derived from real 512x512 masks. `visual_views=0`,
so neither mask pixels, overlay nor crop entered LLaVA-Med. This tests serialized
spatial facts only, not the pixel-level spatial bridge. Anatomy is not lesion
localization.

## Perturbation and cost

- XRV findings: 141 audits, sensitivity min/mean/max
  0.00427/0.02131/0.08362.
- XRV anatomy: 33 audits, sensitivity min/mean/max
  0.00728/0.01877/0.05243.
- The gamma 0.95/1.05 audits added 522 native-model forwards and 20.47 s.
  These values measure output variation, not correctness, calibration or
  clinical invariance; audit mode did not reject evidence.
- The complete job took 408.01 s (6 min 48 s), averaging 0.924 s per question
  including routing. Recorded expert work summed to 76.77 s, of which 20.47 s
  was perturbation probing. Peak PyTorch allocation was 22.46 GiB.

The wall-clock cost is not by itself prohibitive for this small benchmark, but
the present 352 tool calls are inefficient because 172 CheXagent results and two
anatomy packets never reached the answer model. Scaling before fixing transport
would spend more compute without testing the intended method.

## Direct examples

All image paths below refer to local official-test copies; original images are
not committed to Git.

1. `vqarad-official-test-0003`, image
   `runs/vqarad-official-full-test/data/images/75b1c97f4218632681600a58b55389ee9c2040b01ff9667ee169d7a433994c93.jpg`.
   Question: which heart-border side is obscured? Reference and baseline: right.
   New answer: left. XRV anatomy reached the prompt; CheXagent was called but
   omitted. This is a clear laterality harm.
2. `vqarad-official-test-0124`, image
   `runs/vqarad-official-full-test/data/images/9ee8f41c0c2fdb822a90be4e69a1d5c07227dfed5ad82e70ed01051f62f5798b.jpg`.
   The reference and baseline say the cardiac silhouette is less than half the
   diaphragm diameter. The new answer says it is not. Anatomy text entered;
   CheXagent did not.
3. `vqarad-official-test-0332`, image
   `runs/vqarad-official-full-test/data/images/d3757359983115a7ed42154aff1e090d6bb5c36e4855c795248407a4aceba721.jpg`.
   Reference and baseline: gastric bubble on the right. New answer: left. XRV
   findings entered despite not being a laterality expert; CheXagent was omitted.
4. `vqarad-official-test-0000`, image
   `runs/vqarad-official-full-test/data/images/b9c14f55bccd14627fe3f7318c3c48040278ba9d2c8706450d61373fcc3b9cf4.jpg`.
   Reference: aortic aneurysm present. Both baseline and new method deny it.
   This illustrates that a nonempty uncertainty packet does not guarantee a
   correction.

These examples are reference-aligned engineering review, not blinded clinician
adjudication. Lexical improvements and nonempty calls are not counted as medical
benefit.

## Raw local artifacts

- `runs/vqarad-official-full-test/new-method-uncertainty/predictions.json`:
  complete per-case answer, token IDs, raw native evidence, masks, expert
  requests, perturbation confidence/sensitivity, timing and traces.
- `runs/vqarad-official-full-test/new-method-uncertainty/evidence-transport-audit.json`:
  exact final-prompt packet reconstruction and per-expert transport counts.
- `runs/vqarad-official-full-test/new-method-uncertainty/result.json`: generic
  lexical and runtime summary.
- `runs/vqarad-official-full-test/new-method-uncertainty/evaluation_mixed_vqa_table.json`:
  content-aware diagnostic table for the new outputs.
- `runs/vqarad-official-full-test/new-method-uncertainty/comparison-to-historical-baseline-v9.json`:
  frozen strict paired comparison.
- `runs/vqarad-official-full-test/new-method-uncertainty/routing.json`: image-only
  route decisions (174 CXR, 172 CT, 105 MRI).
- `runs/vqarad-official-full-test/new-method-uncertainty.background.log`: detached
  job log.

Key SHA-256 values are `35421a6a...271f9` for `predictions.json`,
`e3674cb5...2796b9ec` for the transport audit, `9542d303...530ccd` for
the frozen comparison, and `c6dc0325...beb5` for the 451-row manifest.

Before another full run, the framework must report presented versus merely
adopted evidence, avoid calls whose packets cannot fit, enforce the benchmark's
closed-answer contract, and run a same-generation matched generalist/evidence
ablation. None of those criteria may be repaired by tuning on test answers or
relaxing a medical threshold.
