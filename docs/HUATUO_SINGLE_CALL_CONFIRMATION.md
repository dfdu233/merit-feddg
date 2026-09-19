# Single-call confirmation: explicit user-requested protocol change

2026-09-19, parent551f5ef. User authorized both host GPUs and removed the second
answer-order comparison. Original128 fresh TRAIN image schedule remains frozen;
no labels, score selection, question-type special cases or numeric admission
thresholds were introduced.

## Selection and numerical implementation

`--comparison-orders single` chooses the sole pair order by SHA256(case ID)
parity, without labels, then maps A/B to the actual displayed candidate. C keeps
generalist. Same-text candidates skip the judge. The reasoned prompt and512
output-token budget are unchanged. This is NOT the old order-consistency gate;
single-order position bias remains possible. `double` is still the default.

`--critic-attention sdpa` is optional. Current cards had roughly29/30GB free,
while earlier eager long-image inference approached46GB. SDPA was tested with
the same pinned checkpoint and strict zero-missing-key load. No shared dependency
changed, image/candidate truncation or quantization. Numerical token parity with
eager was not established; separate output identity records the new backend.

On hostGPU0, fixed first8 previous development cases completed with four actual
calls,13.9866s, maximum one call/case; four identical-text skips. This is engineering
validation, not medical improvement. Outputs: runs/critic-single-sdpa-canary.
Full1044 CPU tests and focused12 tests passed; Ruff passed.

## Corrected native controls and execution

The new worktree lacked its artifacts symlink. An initial native route completed,
but expert loading failed; the same absence also excluded two optional experts
in preflight. Restored a non-overwriting symlink to existing artifacts. Do not
reuse the incomplete-configuration routes: preserved failed nativev1, created
nativev2 and reran128 routes. Corrected expert registry matches previous SLAKE64
apart from output paths and authorized GPU UUID; generation configuration is
identical. Nativev2 identity:
9602d9844e9ff3fcd464ceced136d77b3108b1046ebda1fad2c5cdf6b95665ff.

HostGPU1/containerCUDA0 runs route, experts, admission using original native
runner/environment and HUATUO_NATIVE_ONLY=1;128 candidate pairs must fully complete
before the hostGPU0 critic starts. These are sequential dependent stages, not
two independent64-case data partitions. No other workload is killed. Native
logs: runs/critic-preparation/confirm128-v2-{route,experts,admission}.log.

HostGPU0 tmux merit-critic-slake128-single waits for native complete.json, then:

```bash
cd /home/dbw/merit-feddg-huatuo-critic
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/dbw/ANCHOR:$PWD \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory \
GIT_CONFIG_VALUE_0=/home/dbw/merit-feddg/artifacts/upstream/LLaVA-NeXT-critic-compat \
  /home/dbw/merit-feddg/.venv/bin/python scripts/run_huatuo_pairwise.py \
  --base runs/native-slake128-confirm-v2 \
  --output runs/critic-single-slake128-confirm-v1 \
  --judge llava_critic --decision-channel critic_reasoned \
  --comparison-orders single --critic-attention sdpa \
  --gpu-uuid GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023
```

The existing evaluator uses inputs-slake128-confirm/references.json only after
full completion and writes runs/critic-single-slake128-confirm-v1/evaluation.json.
No128-case score exists at this checkpoint. Preparation failures, duplicate
route work, shared-GPU contention and canary cost are additional overheads;
halving the maximum judge-call count is not a measured2x end-to-end speedup.
