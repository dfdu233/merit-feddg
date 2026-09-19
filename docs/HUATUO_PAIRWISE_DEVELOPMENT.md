# Pairwise selection development, not rewriting

## Evidence and scope

- [Judging LLM-as-a-Judge, NeurIPS2023 Datasets and Benchmarks](https://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)
  studies position/verbosity/self-enhancement bias. Inspected official
  [FastChat common.py at587d5cfa1609a43d192cedb8441cac3c17db105d](https://github.com/lm-sys/FastChat/blob/587d5cfa1609a43d192cedb8441cac3c17db105d/fastchat/llm_judge/common.py):
  `play_a_match_pair` calls `run_judge_pair` in both response orders and remaps
  local A/B labels back to original model identities. We use categorical
  pairwise judgments, NOT its separate scalar scoring/TIE_DELTA path.
- [LLaVA-Critic, CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Xiong_LLaVA-Critic_Learning_to_Evaluate_Multimodal_Models_CVPR_2025_paper.html)
  and its [official model-card inference code](https://huggingface.co/lmms-lab/llava-critic-7b)
  provide an image/question/two-answer interface. This experiment adapts that
  interface to the existing Huatuo model. It does NOT load LLaVA-Critic weights,
  reproduce its trained evaluator, or assume general-domain judging transfers
  to medicine. The latter remains a resource option, not a completed experiment.

## Frozen development experiment

Keep original generalist and compact outputs unchanged. An image-grounded judge
sees the question and two anonymous answers, not the raw expert context or gold
references. Judge both orders. Choose compact only when both decisions identify
compact as better; otherwise keep generalist and record tie/inconsistency or a
consistent generalist preference. This conservative categorical aggregation is
our explicit adaptation, not a claimed theorem or full official reproduction.
Identical answer texts have no selection opportunity and incur no judge call.
Malformed verdicts fail explicitly; no silent KEEP fallback. There is no numeric
admission threshold, answer-type rule, disease blacklist or training.

Scripts: `scripts/run_huatuo_pairwise.py`;7 focused CPU tests cover label parsing,
swapped-order mapping and explicit malformed-output rejection. Existing1032 CPU
tests passed before adding this isolated experiment; Ruff also passes.

The first8 existing scheduled cases per dataset are a canary stopping point,
not a new data split. These TRAIN images have already been scored during the
editor experiment: any result here is DEVELOPMENT, never an untouched holdout.
No full-test launch follows merely from valid labels or all-keep behavior.

Server outputs: `runs/pairwise-vqa87-dev-v1`, `runs/pairwise-slake64-dev-v1`.
Two persistent tmux sessions use the same authorized GPUs and unchanged Huatuo
environment. A full-schedule completion marker is written only when every
scheduled ID has a complete case with the same identity. Selected answer reuse
is recorded separately from judge calls and is NOT new candidate generation.

The v1 canary stopped on the first case of both datasets: Huatuo emitted a
unique explicit bracketed verdict before its explanation, contrary to the
requested final position. No score was produced. v2 accepts exactly one such
explicit label irrespective of position; missing, unknown or multiple labels
still fail. It never infers a label from explanatory prose. Failed v1 outputs
are preserved, and v2 has separate roots and a different source identity.

## Actual v2 outcome: failed canary, no efficacy claim

- VQA-RAD:6 cases complete, then the7th emits both B and C.8 recorded judge
  calls including the failed case,23.5524 s cumulative. Three completed cases
  have identical answer texts; the others keep generalist after judgments.
- SLAKE:1 case complete, then the2nd emits both A and B.3 recorded judge calls,
  8.3351 s cumulative. The completed case keeps generalist on a tie.
- Neither scheduled8-case canary completes; no complete marker or full-score
  evaluation is produced. No candidate is selected over Baseline in completed
  cases. This is NOT successful protection or a reliable gate. Multiple labels
  are not repaired by extracting the first one or guessing from the explanation.
- v1 recorded5.0974 s and3.5042 s of failed judge calls, also not zero cost.
  Model loading and native inherited costs are additional.

## Independent critic readiness, not an implemented result

Official LLaVA-NeXT source cloned at
`bce12e479bc4dfee2b9c50c88137b01ff51bd483` in the existing untracked artifacts
area. CPU import with current torch2.14.0 / transformers4.57.6 fails because
upstream imports `apply_chunking_to_forward` from an unavailable old
`transformers.modeling_utils` location. No shared dependency was upgraded and
no weights were downloaded in this checkpoint. Final CPU regression:1039 tests
passed; all affected scripts and the added test pass Ruff and diff whitespace
checks. A dedicated compatibility path
and real inference validation are still required before claiming the critic
is available. It must not be counted among evaluated methods here.
