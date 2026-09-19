# Huatuo context admission pilot

This experiment targets the Huatuo degradation, not the previously completed
LLaVA-Med TRAIN run. Existing full-test artifacts show VQA-RAD generalist62.82%
vs MERIT57.16%, and SLAKE54.69% vs54.48%. These historical aggregate scores are
motivation, not selection criteria for individual TRAIN examples.

Hypothesis: a separate same-model question-conditioned usefulness decision can
exclude out-of-capability specialist packets without contaminating the final
answer context. The prior LLaVA experiment accepted everything; that result is
not assumed to generalize to Huatuo. Reuse the already frozen relevance/scope
prompts, grounded in Self-RAG's relevance annotation, S2A context separation and
the tool-review precedent in MedAgent-Pro. See CONTEXT_ADMISSION_RESEARCH.md.
No new novelty claim, training, numerical admission threshold or yes/no-specific
gate rule. Original formal answer instructions are retained for every arm.

Fixed initial sample: first4 IDs of the existing24 TRAIN-only-image probe,
selected before this experiment and without score-based filtering. Pixel identity
and test-image exclusion are checked. The generation input contains no reference
answers. Huatuo recomputes its own original image-only routing; no LLaVA route or
answer is reused. Expanded expert registry and native all_evidence acquisition
remain unchanged. Original formal1024-token answer budget, native4096 input
policy and Huatuo preprocessing/repetition settings are retained.

Arms: generalist, original compact MERIT, relevance, scope. Gates see the original
image/question, specialist definition and unchanged packet; no generalist answer
or reference. Only admitted packets enter a fresh answer context. All/none views
reuse exact controls and are explicitly logged, not counted as new candidates.
First compact answer must reproduce exactly in text and token IDs. Runtime,
budget and delivery failures stop the affected experiment, not medical evidence.

Independent worktree `/home/dbw/merit-feddg-huatuo-admission`, branch
`experiments/huatuo-context-admission-v1`, based on `8d1742a`. Existing dirty
expert-coverage worktree is read-only: pinned imports reuse its verified native
Huatuo adapter and runner plus the unchanged formal MERIT core. Every stage
checks hashes of the dependency files/config and image identities. Checkpoints,
environment and data are reused; no upgrades/downloads or other-job termination.

GPU0 host UUID `GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023`; GPU1 had insufficient
conservative headroom for another Huatuo worker. Background host tmux runs
route -> experts -> admission. Completed root `runs/huatuo-train-admission-v5`; v1/v2 are
preflight-only snapshots. Initial expert stage stopped on a missing relative
artifacts link; non-overwriting links to existing artifacts/upstream were added,
the attempted resume correctly rejected the changed registry. In v3 the missing
links excluded CheXagent/BiomedParse and prevented XRV loading. V4 is a separate
complete-registry run, recomputing routing. No v3 answers/caches are mixed in.
Original failure logs are preserved.

V4 then completed12 real expert calls but exposed an API compatibility mismatch:
the old formal compact_records function lacks the newer geometry keyword. Removed
the explicit default-false argument; independently verified exact serialization
equality on all12 real native records. No formal-core change. V5 reran the same4
inputs with the compatible call and saved controls before admission. The v4
partial answers were not persisted before that exception; their call-time costs
are not fully known and are not represented as zero. Host BiomedParse ownership
was handled by a process-local exact safe.directory after clean-revision checking,
not a wildcard trust or shared-environment modification.

Run with the existing `/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python`:

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/dbw/ANCHOR:/home/dbw/merit-feddg-expert-coverage/scripts
export HF_HOME=/home/dbw/ANCHOR/hf_cache HF_HUB_CACHE=/home/dbw/ANCHOR/hf_cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/run_huatuo_admission_probe.py --stage check --dataset vqa_rad --gpu-uuid GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023 --output runs/huatuo-train-admission-v5
# Same arguments with --stage route, experts, admission in sequence.
```

Only aggregate outcomes may be committed. Do not copy raw questions, answers,
images, expert caches or tokens into public reports. This four-case plumbing
probe cannot establish efficacy, SOTA or general safety; no score-conditioned
prompt tuning is permitted. Expand TRAIN only after inspecting actual decisions,
delivery and candidates. SLAKE remains the next target, not already evaluated.

## Completed outcome: do not scale this gate yet

All4 TRAIN-only images completed, all routed CXR by Huatuo. Each received XRV
classification, CheXagent description and BiomedParse segmentation:12 native
expert outputs. First original compact replay matched text and token IDs exactly.
No empty final answers. Identity:
`2f0d08b805d426560ace36bd5c8efd527b14609743296ffa45dca126176b4621`.

| Arm | Four-case mixed task score | Improvements/harms vs compact | New candidate calls |
|---|---:|---:|---:|
| generalist | 80.56% | 1 / 0 | control |
| compact MERIT | 55.56% | 0 / 0 | control |
| relevance | 30.56% | 0 / 1 | 3 |
| scope | 55.56% | 0 / 0 | 1 |

These are4-example diagnostic scores, not dataset accuracy estimates. No
statistical significance, holdout gain or SOTA claim. Current unchanged ANCHOR
CLOSED parser + OPEN token recall, source hashes and reference hash recorded in
the [sanitized result](../reports/huatuo-admission-train-probe-v1-final.json).

CPU regression:1032 passed in18.08s; targeted admission tests2 passed; Ruff and
diff whitespace checks passed. Real old/new renderer equality verified on12
native packets. The complete CPU regression is not a Huatuo medical validation.

Relevance labels9 A /3 C, scope7 A /5 C;24 real judgments total. No threshold,
training or answer-conditioned rule. Relevance produced fresh candidates on3/4
cases, scope1/4; scope reused compact on2 and generalist on1. Unlike the previous
LLaVA all-admit result, execution really changed contexts, but did not improve
answers. Filtering failure is not merely a failure to execute.

Two TRAIN failure mechanisms (paraphrased, no patient text/images exported):

- A cardiac-size case was correct for baseline/compact. Relevance removed the
  classifier while retaining generated description and segmentation; the new
  answer became wrong. Scope kept all packets. Topic relevance judgments can
  remove a helpful packet; an observed correlated change is not a general causal
  attribution to that expert.
- A lesion-presence case was correct for baseline but wrong for compact. Scope
  retained the classifier and removed description/segmentation, yet the answer
  remained wrong and cited the same raw class score as evidence of presence.
  The source score is not itself a calibrated diagnostic conclusion. This shows
  a distinction between admissible task relevance and faithful use of score
  semantics; it does not establish any score threshold or prove that classifiers
  are generally harmful.

Measured successful-v5 costs: relevance judgments4.86s, scope5.25s, four new
candidate generations9.43s, baseline generations10.57s, compact outer time10.75s,
actor model load10.49s. These are concurrent-call timings, not a complete pipeline
speed benchmark. Earlier failures, routing loads, expert-prefetch phase wall time
and parity timing are not fully instrumented; report missing rather than zero.

Decision: preserve controls, do not scale these two gates to full test or tune
them on these answers. The next mechanism to investigate is whether the answer
model respects an expert's native claim strength (measurement/score versus
diagnostic fact), separately from whether its output is related to the question.
This is a follow-up hypothesis, not an implemented or proven solution.
