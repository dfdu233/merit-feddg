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
`scripts/run_anchored_revision_train.py --output runs/pilot-v2 --check-only` freezes
inputs and code without loading reference labels. `--max-cases 2` is a scheduling
stop, not a new selection. Remove it to continue the identical pilot. Shard flags
only schedule disjoint cases, not change the identity.
GPU0 is occupied by PMC; do not terminate it without explicit direction. GPU1 has
room for the pilot, checked at launch. Memory headroom is an engineering resource
check, never a medical decision threshold. Raw patient data and outputs stay local.
