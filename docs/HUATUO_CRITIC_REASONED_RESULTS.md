# Reasoned independent critic: complete SLAKE64 TRAIN diagnosis

2026-09-19; parent1c65e3a, same independent critic worktree/branch. This is
development evidence, not a full-test result, fresh holdout or SOTA claim.

## Change and literature basis

The preceding finite-label critic returned AA for three of four differing pairs
in the first8 fixed rows. The [official LLaVA-Critic model card](https://huggingface.co/lmms-lab/llava-critic-7b)
uses an explanation-first pairwise layout, not single-token classification.
Added `--decision-channel critic_reasoned`, using that layout and an explicit
terminal A/B/C verdict convention adapted from the existing FastChat-inspired
two-order interface. This is not an exact reproduction: deterministic512-token
explanations, explicit verdict, BF16 and compatibility changes are recorded.
The old free-text and finite-choice interfaces remain available.

Original image, original Huatuo generalist/compact answers, question, frozen
critic weights and both candidate orders remain fixed. Both orders must prefer
compact to select it. No numeric admission threshold, training, recalibration,
new experts, reference-at-inference or TEST optimization. Invalid verdicts
raise rather than count as a successful keep. More explanation is additional
compute, not an isolated parser-only ablation.

## Real execution

HostGPU1/containerCUDA0, UUID GPU-3846413a-4238-d307-b1f3-10c2dfbe002c.
Existing modern environment; no new download or installation this round.
Checkpoint revision498f2d719b83e50e48787c6958afe7100503c23f, strict hash/load
checks unchanged. Existing full64 TRAIN development identity
905c822687d3d52ce4826f9f04de3a378c55897424c92f6c6f00c116c1de64ce.
Run identity4c40f2bfa7ca6f3a838b21911f7cf11c413a823dfa69da00ed79adcb5ab470db.

```bash
cd /home/dbw/merit-feddg-huatuo-critic
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/dbw/merit-feddg/.venv/bin/python scripts/run_huatuo_pairwise.py \
  --base /home/dbw/merit-feddg-huatuo-anchored/runs/native-slake64-v1 \
  --output runs/critic-reasoned-slake64-v1 --judge llava_critic \
  --decision-channel critic_reasoned \
  --gpu-uuid GPU-3846413a-4238-d307-b1f3-10c2dfbe002c
```

First the same command used `--canary-cases 8`, then resumed without the stop.
All64 cases finished with exact identity/ID checks; frozen ANCHOR scoring ran
only after complete.json. Original Baseline/compact were reused, not regenerated.
Two allocator OOM retry warnings occurred, but the process recovered and completed;
no cases skipped, no runtime/config change, no fatal OOM treated as a medical error.

```bash
/home/dbw/merit-feddg/.venv/bin/python scripts/evaluate_huatuo_admission_probe.py \
  --run /home/dbw/merit-feddg-huatuo-anchored/runs/native-slake64-v1 \
  --candidate-run runs/critic-reasoned-slake64-v1 \
  --references /home/dbw/merit-feddg-huatuo-anchored/runs/inputs-slake64/references.json \
  --output reports/critic-reasoned-slake64-v1.json
```

## Complete frozen-score comparison

Metric: existing ANCHOR CLOSED parser + OPEN token recall, not clinical accuracy.

| Method | SLAKE64 score | Delta vs Baseline | Improve / harm vs Baseline |
|---|---:|---:|---:|
| Original generalist |52.8646%|0|0 / 0|
| Original compact MERIT |52.9167%|+0.0521pp|8 / 7|
| Previous Huatuo finite-choice selector |50.8333%|-2.0313pp|3 / 4|
| Independent explanation-first selector |53.9583%|+1.0938pp|3 / 2|

Image-cluster paired bootstrap95% interval of new minus Baseline:
[-4.3750,+6.8789]pp; new minus compact: [-7.0378,+9.3750]pp.
Against compact, five improvements and five harms. Intervals span zero.
The previous independent finite-label run is only8 cases and must NOT be shown
as a full64 comparator; model and interface differences are not causally separated.

34/64 pairs differed in text;68 actual judge calls, all parsed.30 pairs skipped
because texts match. Selected compact10/64 (29.41% of differing pairs), original
generalist54/64. Ten order-consistent compact, three consistent generalist,
21 tie/order-inconsistent. Labels A24/B19/C25. There are no newly generated
patient answers: this is selection among existing native candidates.

## Gain/harm interpretation, without patient content export

Five score-changing cases need different interpretations, not a single claim:

- Gain1: question asks an organ's function; candidate addresses function while
  original only names an organ. This is task relevance improvement under the
  frozen scorer, not independent clinical verification.
- Gain2: candidate describes paired anatomy and uses the question language;
  token recall rises0.2. We did not establish pixel-grounded superiority.
- Gain3: English and Chinese negative responses have the same broad meaning,
  yet the frozen parser assigns different scores. Do not call it medical gain.
- Harm1: short negative response becomes a Chinese negative plus a modality
  assertion; frozen score drops1. The judge prefers informativeness. This mixes
  parser sensitivity with potentially unsupported extra content, not proven
  polarity reversal. The scorer is unchanged; no yes/no-specific rule is added.
- Harm2: candidate denies disease in an image, while original names diseases;
  the judge endorses the denial because it treats lack of apparent evidence as
  evidence of absence. Frozen open score drops0.5. The generic critic is not a
  validated medical verifier; medical truth is not inferred from its explanation.

Another unchanged case confuses an organ name with the requested organ system.
Explanation-first removes the most obvious small-canary position pathology but
does not ensure correct task semantics or visual verification. Linguistic
completeness and medical utility remain conflated. Rationale is an observable
output, not proof of the causal mechanism of a model decision.

## Costs, checks, limitations and next decision

68 judge calls255.5981s (3.9937s per scheduled case;7.5176s per differing pair).
Two loads12.6143s +12.1375s, additional24.7518s. These exclude preflight hashing,
previous failed attempts/downloads and inherited candidate-generation costs.
Reused native logged generalist calls16.8592s, compact outer52.1040s; expert
prefetch and other components remain incompletely instrumented. Different
runtime/load conditions prevent an isolated speedup claim. The additional judge
cost is substantial relative to the small, statistically uncertain score delta.

Full CPU suite1043 passed, focused11 passed, Ruff passed. No new patient text,
image, weight or credentials are committed. Raw local outputs stay under runs/;
sanitized complete score report is reports/critic-reasoned-slake64-v1.json.

Do not scale to TEST on this signal. Next algorithmic question should isolate
factual incremental utility from stylistic preference using a task-grounded
criterion and controlled same-candidate comparison; avoid simply lengthening
reasoning, adding thresholds or patching Chinese/yes-no cases. Any such new
criterion must be a separately identified experiment, with fresh TRAIN-only
confirmation before generalization claims. This turn establishes a modest
development score gain and a clearer failure mechanism, not reliable Gate success.
