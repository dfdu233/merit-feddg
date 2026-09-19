# Independent frozen multimodal critic: preparation, not efficacy evidence

2026-09-19. Branch `experiments/huatuo-independent-critic-v1`, based on
`c080b826dff6a19075b222e9c566e56f1c20ec99`. Prior methods/results preserved.

## Hypothesis and fixed experiment

The previous Huatuo finite-choice selector scored 59.1954% on RAD87 and
50.8333% on SLAKE64, below Baseline 60.3448% and 52.8646%. These are TRAIN
development images, not full tests or fresh holdouts. On SLAKE, the offline
two-candidate oracle is 62.2917%; candidate availability and selection ability
are different bottlenecks. No oracle information is passed to inference.

Test a frozen independently trained visual judge using the same original
image/question, original Huatuo generalist/compact candidates, A/B/C prompt,
two answer orders and conservative selection policy. Do not rewrite answers,
change experts/routing, add numeric admission thresholds, train, or tune on TEST.
Only two order-consistent compact votes select compact; otherwise preserve the
generalist. This policy is NOT a guarantee against harm or calibrated confidence.

Reference: [LLaVA-Critic, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Xiong_LLaVA-Critic_Learning_to_Evaluate_Multimodal_Models_CVPR_2025_paper.html),
[official model card/code example](https://huggingface.co/lmms-lab/llava-critic-7b).
The official example uses free-form reasoning. Our finite-choice interface is
an explicit adaptation, not a paper reproduction. Different architecture and
preprocessing prevent attributing changes solely to critic training.

## Reproducible resources and compatibility

- Official LLaVA-NeXT source: `bce12e479bc4dfee2b9c50c88137b01ff51bd483`.
- Model: `lmms-lab/llava-critic-7b`, revision
  `498f2d719b83e50e48787c6958afe7100503c23f`; four official LFS SHA256 values
  enforced in `scripts/llava_critic_backend.py`. Total weights 16,060,803,168 bytes.
- Reuse `/home/dbw/merit-feddg/.venv`: Python3.14, torch2.14.0+cu130,
  transformers4.57.6. No dependency install/upgrade.
- Independent upstream worktree `artifacts/upstream/LLaVA-NeXT-critic-compat`;
  compatibility diff archived in `patches/llava-critic-existing-environment.patch`.
  Moves existing Transformers utility imports; initializes the exact native
  vision architecture for loading embedded critic weights. CPU meta inspection
  matched all421 visual parameter keys to the checkpoint index (missing0/extra0).
  This is structural evidence, NOT proof of loaded weights or inference.
- Native model load rejects any missing/unexpected/mismatched/error keys. The
  padded vocabulary is shrunk as in the official builder, never enlarged with
  random rows. Original-image anyres tensors, upper-bound context budget,
  greedy finite decoding, source identities and load diagnostics are recorded.
- Existing partial downloads preserved. HTTP was slow; mirror redirected to
  Hugging Face. A redundant base-SigLIP download was stopped once embedded visual
  coverage was verified; its attempted preparation cost is not zero. Aria2
  resumes main critic shards and verifies their checksums. Partial final filenames
  must not be mistaken for complete models; backend identity independently hashes
  every shard before inference. Download script logs remain local.

## Execution and present status

Full existing CPU suite passed1040 tests; then two additional incomplete-weight
guards and eight existing pairwise tests passed10/10. Ruff, CLI and whitespace
checks passed. GPU model loading and medical performance are **not yet validated**.
Weights are downloading in persistent tmux `merit-critic-aria2`.
Persistent tmux `merit-critic-canary` waits for that exact download process to
exit, then executes the two-case command below with a1800-second timeout. Failed
download checksums or device/load checks abort inference; no full run is queued.

Container CUDA0 maps to hostGPU1 UUID
`GPU-3846413a-4238-d307-b1f3-10c2dfbe002c`;44,695MiB free at preparation.
HostGPU0 had another active workload; do not interfere. Use hostGPU1 for the
first real two-case scheduling stop on the existing full SLAKE64 manifest:

```bash
cd /home/dbw/merit-feddg-huatuo-critic
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/dbw/merit-feddg/.venv/bin/python scripts/run_huatuo_pairwise.py \
  --base /home/dbw/merit-feddg-huatuo-anchored/runs/native-slake64-v1 \
  --output runs/critic-slake64-v1 --judge llava_critic \
  --decision-channel finite_choice --canary-cases 2 \
  --gpu-uuid GPU-3846413a-4238-d307-b1f3-10c2dfbe002c
```

The first two scheduled cases both have differing candidate texts, so this
cannot pass solely by skipping identical answers. After manual inspection of
actual loading/calls, extend to existing eight-case stop, then decide whether
full64 development diagnosis is warranted. No automatic full-test expansion.
Do not publish a full-manifest score for an incomplete canary. Reference answers
remain offline in the original inputs-slake64 references file; use the existing
frozen ANCHOR evaluator only after a valid complete marker.

No new effectiveness score, improvement/harm count, or real GPU inference cost
is available yet. Format validity, all-keep, checksum success and CPU tests must
not be presented as medical improvement. Subsequent reporting must include
selection coverage, improvement/harm, paired image confidence intervals, judge
calls, all model-load attempts and inherited candidate-generation costs. Answers
are reused, not newly generated; downloads and failed attempts are additional
preparation costs, not hidden within the per-call timing.
