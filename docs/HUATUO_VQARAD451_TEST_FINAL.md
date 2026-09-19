# Huatuo-MERIT + independent critic: completed official VQA-RAD test

2026-09-20. Full451 official test questions,251 CLOSED and200 OPEN,203 unique
image clusters. User explicitly requested TEST evaluation after TRAIN exploration.
No test-score tuning, training, new data partition or shared dependency upgrades.

## Complete results

| Method | Official mixed score | CLOSED accuracy | OPEN token recall |
|---|---:|---:|---:|
| Huatuo Baseline |62.8193%|76.4940%|45.6576%|
| Original Huatuo-MERIT compact |57.1558%|68.5259%|42.8864%|
| MERIT + single comparison critic, format completion where required |60.3434%|72.9084%|44.5743%|

Mixed score is the sample-weighted existing CLOSED parser and OPEN token recall,
not clinical accuracy. Recomputed Baseline and original MERIT match their
independent official stored scores within1e-12. No prompt or scorer was changed
to improve results. Frozen ANCHOR version:
medheval-decoded-eval-v13-explanatory-uncertainty-review.

Compared with Baseline: -2.4760pp,9 improved/18 harmed questions. Image-cluster
paired bootstrap95% delta interval[-4.7464,-0.3720]pp. Compared with original
MERIT: +3.1876pp,35 improved/14 harmed; interval[+0.4334,+5.9626]pp.
The method recovers some MERIT degradation but does NOT preserve Baseline.
No SOTA, clinical safety or general Gate-success claim is supported.

## What was actually run

- Reused original formal451 Huatuo greedy and original native MERIT candidates;
  no rerouting, expert reruns or candidate answer generation. Exact IDs, image
  bytes, canonical questions/prompts,1024-token budgets and original source
  hashes checked before inference. Reference answers are only in offline scoring.
- Original image/question and two original answers go to frozen LLaVA-Critic-7B,
  revision498f2d719b83e50e48787c6958afe7100503c23f. Greedy BF16 SDPA,512-token
  critic limit, original anyres image preprocessing. Same question-ID SHA parity
  orders candidates once; no answer swapping or second independent comparison.
- Both host GPUs process disjoint even/odd indices226/225 of the SAME full
  manifest. Merge refuses incomplete/extra/mismatched IDs. All451 files have the
  same final identity, are complete, and were scored only after successful merge.
-349 differing candidate pairs judged;102 identical-text cases skipped.
  Final output reuses Baseline380 times, MERIT71 times.120 explicit ties.
  No patient answer was rewritten. Choosing Baseline is not medical validation.

## Transparent runtime recovery and protocol amendment

Original strict-parser runv1 stopped after373 completed cases when a response
echoed the three conditional options instead of producing one legal verdict.
We did NOT silently interpret the prose, fill a row, or default to Baseline.

Runv2 adds an explicitly identified, general output-completion policy: only on
invalid verdict, continue the same model's original generated token prefix
(removing terminal EOS only), append a final-verdict cue and constrain the next
output to A/B/C. Original image, question, candidate order and explanation are
retained. The explanation, cue and finite continuation must fit the original
512-token allowance; insufficient budget still fails rather than truncates.
This is an extra short model invocation, NOT zero cost and NOT a second
independent answer comparison. It is a runtime protocol amendment after a format
failure on TEST; do not present it as the exact original pre-registered strict
parser run. No references or scores select this recovery.

Three cases required completion, each producing label+EOS (two output tokens).
Total extra completion3.3852s. All three have zero score delta versus Baseline;
the negative final conclusion is not caused by the completion decisions.
373 already completed v1 cases were checked: selected answers and full reused
answer payloads remain identical. Old raw calls/files and failure logs preserved.

## Cost and engineering checks

349 original comparison invocations +3 format completions =352 model calls.
Summed call time1020.2948s across both GPUs and resumed attempts, including
inherited saved calls once. This is NOT elapsed wall time or a total project
cost. Original two model loads3.5016+11.8842s and recovery loads4.4972+11.9474s
add31.8305s. Preparation, hashing, monitoring gaps and original candidate costs
are additional; candidate timing missing from the formal exports is null, not0.

CPU1053 tests and focused21 tests passed; Ruff passed. Completion-prefix tests
verify exact token-prefix preservation and rejection when budget is exhausted.
The real blocked case passed with187 prefix tokens+2 generated tokens within512.
Other model/weight/image/merge checks remained strict. Both jobs finished;
no inference left running for this evaluation.

## Artifacts and identities

- Original MERIT identity741a7d2ebfd54b18eec6a31bb61c3eaf06e58a20af47652a3b566bef855bbfa3.
- Reuse-controls identityc9b7b988101470f7c8a59f168e99b0c0369635b61d5f499d4034bf42d55d860a.
- Final critic identity3df65e80985bda1de6b7f4391102a0c128086662c091821291152559983a65d1.
- Local full outputs: runs/critic-single-vqarad451-test-v2/{cases,progress,protocol.json,complete.json,evaluation.json}.
- Preserved earlier run: runs/critic-single-vqarad451-test-v1.
- Public sanitized scores, all451 per-case numerical scores, image-cluster
  intervals and costs: reports/huatuo-single-critic-vqarad451-test.json.

No raw patient text, images, response token IDs, model weights or credentials
are committed. Primary source context and prior TRAIN results remain in
docs/HUATUO_CRITIC_REASONED_RESULTS.md and docs/HUATUO_SINGLE_CALL_FINAL_RESULTS.md.

## Reproduction of resumed execution

Use existing environment and per-device CUDA_VISIBLE_DEVICES=0/1 on host;
pass matching physical UUID. Both workers share the same output root and differ
only in --shard-index0/1. Common options:

```bash
python scripts/run_huatuo_pairwise.py \
  --base runs/formal-vqarad451-controls-v1 \
  --output runs/critic-single-vqarad451-test-v2 \
  --judge llava_critic --decision-channel critic_reasoned \
  --comparison-orders single --critic-attention sdpa --shard-count 2 \
  --reuse-judgments runs/critic-single-vqarad451-test-v1 --complete-verdict \
  --shard-index 0 --gpu-uuid GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023
```

After both workers complete, the same common options with --merge-only replace
the device/shard-index arguments. Score using the unchanged evaluator and
controls/references.json only after successful complete451 merge. For a fresh
reproduction omit --reuse-judgments and choose a new output directory; never
overwrite published runs. Do not use TEST outcomes to choose a revised Gate.
