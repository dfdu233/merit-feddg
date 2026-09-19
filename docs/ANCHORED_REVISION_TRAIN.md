# Anchored revision: frozen TRAIN engineering pilot

## Evidence update / Vector

The completed Huatuo RAD and SLAKE expanded MERIT runs used all_evidence with
vector_gate off. Their deficits do not measure an enabled gate's discrimination.
The earlier 12-case uncertainty pilot's pairwise judge accepted zero changes;
repeating that mechanism is not evidence of a useful gate.

Vector: can preservation-oriented revision retain specialist benefit while
reducing damage to the original answer, without a numerical acceptance threshold?

## Affinity map and collision check

* [RARR, ACL 2023](https://aclanthology.org/2023.acl-long.910/) separates evidence
  collection from preservation-oriented editing. Official implementation reviewed:
  [anthonywchen/RARR](https://github.com/anthonywchen/RARR), revision
  `51a1a10fe5bada837a368f98cb55288ac5168c9e`;
  `utils/editor.py`, `utils/agreement_gate.py`, `prompts/rarr_prompts.py`, and
  `run_editor_sequential.py`. Its edit stage conditions on original claim and
  evidence. The original pipeline also has retrieval, agreement checks and an
  edit-ratio cutoff. We do NOT reproduce those components or import that cutoff.
* [CRITIC, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/fef126561bbf9d4467dbb8d27334b8fe-Abstract-Conference.html)
  uses tool feedback to revise initial answers. The shared assumption is that
  feedback supplies information beyond the original generation; unreliable
  specialists can violate it. This is a literature neighbor, not a reproduced method.
* [MT-Bench judge, NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/file/91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf)
  studies preference judging and position bias. Official FastChat
  `fastchat/llm_judge/common.py`, `run_judge_pair` / `play_a_match_pair` were inspected.
  The categorical pairwise path and swapped order are distinct from its numerical
  score/TIE_DELTA path. The repository already tried swapped categorical judging;
  no new judge is introduced here.

Bit Flip status: **unproven; engineering ablation, not a novel scientific method**.
An anchored prompt is not itself an ICLR contribution. Negative results must end
this iteration rather than trigger prompt tuning against the observed examples.

## Minimum experiment (frozen before GPU execution)

* Historical source: complete 1,793-row VQA-RAD TRAIN transport-fixed run.
* Select 24 distinct TRAIN images with existing expert evidence by a fixed SHA256
  ordering of case IDs, excluding every image hash in the official test manifest.
  This is a development probe, not a replacement benchmark split. No outcome,
  reference, disease list, or yes/no special rule selects cases or controls inference.
* Official TRAIN/test share 202 image hashes (313 TRAIN images, 203 test images).
  Exclusion leaves 111 TRAIN-only images before the evidence-availability filter.
  Patient-level independence and near-duplicate independence remain unverified.
* Arms: original generalist; original compact MERIT; anchored revision without
  expert evidence; anchored revision with exactly the existing delivered evidence.
* H1: evidence-bearing anchored revision reduces harm without merely retaining
  everything. H0: preservation loses the useful changes, does not beat no-evidence
  revision, or changes only parser/style behavior.
* Constants: original model/weights, image, question, native evidence, renderer,
  greedy decoding and historical 64-token cap. No new expert calls. The changed
  variable is whether the original answer anchors the final generation. A separate
  no-evidence revision diagnoses generic extra inference versus specialist value.
* Each case first requires exact text AND token parity for both historical controls.
  Same evidence IDs/order must remain delivered after adding the draft. Any mismatch,
  empty candidate, image identity failure or resource error stops, with failures saved.
* Full 24 cases required for the pilot score. The first two are engineering canaries;
  their scores do not choose a prompt or strategy. An all-unchanged pilot is not success.
* Frozen ANCHOR strict closed / open answer-token-recall, paired improvements/harms,
  preservation of original MERIT gains, text/candidate coverage, image bootstrap,
  actual calls and inherited/new costs. Not a clinical accuracy claim.

## Core / periphery, velocity and re-vector rules

Core is a single-pass editing instruction around the unchanged evidence context.
Periphery frozen: routing, expert pool, acquisition, budgets, scoring, trained gates,
uncertainty, retrieval, verifier models and learned parameters.
Positive signal warrants a separate larger TRAIN confirmation and a cross-model
Huatuo check before a frozen test run; this small LLaVA pilot cannot establish repair
of the Huatuo regressions. Ambiguous CI, loss of gains or no useful changes means no
automatic benchmark expansion. A clear negative result rejects this prompt substrate;
do not call a new prompt a research breakthrough.

## Execution

Worktree `/home/dbw/merit-feddg-anchored-revision`, base commit `5889907`;
branch `experiments/anchored-revision-train-v1`. Existing source/dirty worktrees untouched.
Use the existing research Python and local model cache; no packages installed/upgraded.
`scripts/run_anchored_revision_train.py --output runs/pilot-v4 --check-only` freezes
inputs and code without loading reference labels. `--max-cases 2` is a scheduling
stop, not a new selection. Remove it to continue the identical pilot. Shard flags
only schedule disjoint cases, not change the identity.
GPU0 is occupied by PMC; do not terminate it without explicit direction. GPU1 has
room for the pilot, checked at launch. Memory headroom is an engineering resource
check, never a medical decision threshold. Raw patient data and outputs stay local.

## Actual outcome: stopped before full pilot completion

Execution commit `e09cc66109537628a2b96db3bee8f64c65a8001a`.
Experiment identity
`4f655115ddcdbc25c3743a5cf085c2a7cb1e042417b43c012d202f4b4cb8b009`.
All 1,029 CPU tests pass (16.48s), Ruff/diff check and CLI pass. Repeated preflight
matches. The incomplete-scoring negative check correctly rejects missing cases.

Commands used from the independent worktree:

```bash
PYTHONPATH=. /home/dbw/merit-feddg/.venv/bin/python \
  scripts/run_anchored_revision_train.py --output runs/pilot-v4 --check-only
# Executed inside detached tmux; no dependency changes or downloads:
env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  HF_HOME=/home/dbw/ANCHOR/hf_cache HF_HUB_CACHE=/home/dbw/ANCHOR/hf_cache/hub \
  TRANSFORMERS_CACHE=/home/dbw/ANCHOR/hf_cache/hub \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  /home/dbw/merit-feddg/.venv/bin/python -u scripts/run_anchored_revision_train.py \
  --output runs/pilot-v4 --gpu-uuid GPU-3846413a-4238-d307-b1f3-10c2dfbe002c \
  --max-cases 2
# After inspecting both real canaries, same command without --max-cases 2.
```

The first two canaries and a third pilot example completed all four arms, including
exact original control text/token replay. All three evidence SHA256 values match
between compact and anchored evidence revision: no evidence displaced by the draft.
The fourth example stopped at `revision_no_evidence`: tokens `[28705, 2]`, empty
decoded answer. Actual input751/reserved64/context2048 excludes a context-overflow
explanation. No crash/OOM was hidden as negative medical evidence. The failing
case's evidence-bearing revision was not reached.

| Arm | Complete records | Timed generation seconds, sum | Output tokens, sum | Non-EOS completions |
|---|---:|---:|---:|---:|
| Original generalist | 3 | 4.578 | 67 | 0 |
| Original compact MERIT | 3 | 5.477 | 106 | 1 |
| Anchored revision, no evidence | 3 | 4.230 | 88 | 0 |
| Anchored revision, same evidence | 3 | 4.819 | 91 | 0 |

These are completed-record timing diagnostics, NOT a partial accuracy table.
Two loads cost24.173s, complete-record wall time19.827s, failing call0.257s.
There were15 generation calls including both controls on the fourth case; only13
have individual persisted timings (12 complete-record calls + failure). The old
compact/generalist inherited times for completed cases are3.054/1.600s, not matched
fresh end-to-end latency. New expert calls0 because exact native outputs were
reused, not because experts were absent. No cost advantage is established.

Manual behavior inspection without reference-based selection: the first two
examples mostly repeated the draft with an unwanted `Final answer (JSON string)`
heading; the third similarly copied broad anatomical content. Text differences
are not evidence of useful corrections. The one cap hit belongs to the original
compact control, not the revision arm. The selected cohort is 22CXR/1CT/1MRI,
15open/9closed; the executed prefix cannot establish modality-general efficacy.

**Result / belief update:** the editing instruction is executable and evidence
delivery remains intact, but the frozen medical generator did not reliably follow
the edit-output contract. H1 is unestablished, not confirmed by unchanged answers.
The full24-case accuracy, gain retention, harm recovery and image CI are unavailable.
No references were loaded to score this incomplete pilot. Stop, do not scale.

**Residual uncertainty / next Vector:** a generic preservation prompt does not
establish source reliability or a reliable Gate. Before choosing a stronger editor
or a different evidence-admission mechanism, separate editing ability from expert
correctness on independent development examples. No new prompt was selected using
these outcomes; no larger experiment queued. This pilot does not test Huatuo.

Failure history retained locally: v1 preparation; v2 tuple-vs-list serialized
provenance resume mismatch before GPU inference; v3 token-exact but trailing-space
text mismatch. v4 uses the original engine's exact stripped text serialization,
with raw block text additionally logged, not a tolerant parity comparison.
Raw logs and generated patient-related text remain in `runs/pilot-v4`; only this
aggregate report and the sanitized summary are tracked. All old methods untouched.

## Follow-up: serialization is not a sufficient explanation

**Evidence update:** user authorized continuation. The prior failure did not
distinguish JSON-container behavior from editing behavior. A new diagnostic used
the first8 already-frozen TRAIN-only images, with no reference access. It changed
only the final draft container from JSON string to plain text. Same question,
instruction wording, model,64-token answer budget, original image and evidence.
Original generalist/compact were also replayed. Empty outputs were explicitly
recorded as diagnostic failures, never returned as successful clinical predictions.

**Affinity map / Bit Flip:** this remains a RARR-inspired engineering substrate,
not a research innovation. The nearest-neighbor assumption being examined is that
the chosen generator can execute a preservation edit. JSON-only causation is
weakened by the measured results; no broader impossibility claim follows.

**Vector / core versus periphery:** does removing JSON serialization fix empty
revision output? Core is one serialization contrast; all evidence acquisition,
routing, scoring, weights and thresholds remain untouched. No new expert or judge.

**Minimum experiment:** run6 arms on8 fixed images, exact historical parity and
identical evidence required. Runtime/parity/evidence mismatch stops. The two
representations were frozen together before GPU execution; there was no third
variant or answer-score-driven prompt search. Execute commit `5e52261`, identity
`59ef87eeec744dcdab555ce4115e12d29ea92655cc0d708ca60d2f4f6b170156`.

The process stopped on case7 at an evidence-displacement check, after6 complete
records. Counts below describe ONLY those same6 records, not a completed8-case
result and not accuracy:

| Diagnostic arm | Nonempty outputs / same6 completed cases |
|---|---:|
| Original generalist | 6/6 |
| Original compact | 6/6 |
| JSON draft, no evidence | 5/6 |
| JSON draft, same evidence | 6/6 |
| Plain draft, no evidence | 2/6 |
| Plain draft, same evidence | 3/6 |

All14 original control calls across7 started cases reproduced exact text and
tokens. The earlier JSON empty answer reappeared unchanged. Plain serialization
introduces more empty answers in the matched prefix, rather than fixing the issue.
The8 empty calls produce immediateEOS (7) or whitespace+EOS (1). Nonempty output
is not evidence of a useful edit, and no clinical labels were loaded to score it.

Case7 reveals a different problem: original compact input1880 + answer reserve64
fits the2048 limit. Full JSON revision input2014 +64 exceeds it by30; plain input
2011 +64 exceeds it by27. Packing drops the existing CheXagent description while
retaining the anatomy packet. The equality check stopped BEFORE generating this
unmatched evidence arm. The8th image received no model call. No context enlargement
or evidence deletion was accepted to turn this into a successful comparison.

**Engineering improvement:** added whole-diagnostic CPU token-only preflight to
`scripts/probe_revision_format.py`. It uses the same official Mistral conversation
template/tokenizer and declared fixed patch expansion; it rejects unsupported
visual formats. Compared against all39 persisted real GPU calls: context accounting,
presented evidence and evidence hashes match exactly. The new preflight, in a fresh
output `runs/format-preflight-v3`, correctly identifies the2 displaced-evidence arms
before loading model weights or making any model call. This does not claim CUDA
validation or a medical decision rule. Earlier raw results remain unchanged.

Commands (same existing offline environment as above):

```bash
# Actual completed/blocked GPU diagnostic, before adding whole-run preflight:
PYTHONPATH=. /home/dbw/merit-feddg/.venv/bin/python scripts/probe_revision_format.py \
  --output runs/format-probe-v2 --gpu-uuid GPU-3846413a-4238-d307-b1f3-10c2dfbe002c
# New guard: expected nonzero exit; zero inference calls, detailed local audit saved:
PYTHONPATH=. /home/dbw/merit-feddg/.venv/bin/python scripts/probe_revision_format.py \
  --output runs/format-preflight-v3 --gpu-uuid GPU-3846413a-4238-d307-b1f3-10c2dfbe002c \
  --check-only
```

Actual inference39calls,42.876s; weight load11.740s. Per-arm persistence now retains
timings even for a partially completed case. Real cached expert outputs were reused;
zero new expert calls does not mean zero specialist involvement. Both memory checks
and context-capacity checks are engineering guards, not confidence thresholds.
GPU1 used, model released after stop; occupied GPU0 PMC process not interrupted.

**Novelty collision / velocity / re-vector:** no novelty claim; no full-run score.
New knowledge is that simply changing the draft container does not fix this
checkpoint's editing behavior, and extra instructions can change evidence delivery
even with unchanged raw packets. Stop this prompt substrate; do not keep adding
instructions or suppressEOS to claim benefit. The next uncertainty is whether a
capability-bound evidence intervention can be effective without relying on this
fragile edit instruction. That requires a separately frozen experiment, not a
continuation of these failed results or a claim that Gate is now reliable.
