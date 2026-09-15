# CRES server execution — 2026-09-15

Status: real spatial scheduling accepted; full experiments launched/queued.
No complete CRES medical score yet.

## Version and preserved work

PR #7 requested and fetched commit:
`c78986e2429053077e0b8e9db7a648d3ea345c30`, branch
`implementation/control-referenced-steering-v1`. Independent worktree:
`/home/dbw/merit-feddg-cres`. Original repository has 26 dirty/untracked entries;
none were changed. Other existing worktrees and old results remain intact.
No merge, training, dependency upgrade, model download, or parameter selection.

Both prior full runs have `complete.json` and evaluation outputs: segmentation
root `6ff365dc97261b29232acdab0a1060109bfd342c8dcb61d5621f7bfa07295440`,
classification/text root
`13a3e59cbcb99caa714fea919b3b50fb99c80ae6b9aa90e3c8b8879872e35f35`,
under `/home/dbw/merit-feddg-soft-guidance/runs/`. They cover VQA-RAD 451 and
SLAKE 2094. Their historical prompts are not the fresh CRES uniform contract;
their scores are not substituted for CRES comparators.

## Environment and devices

Existing `/home/dbw/merit-feddg/.venv/bin/python`, LLaVA-Med Mistral 7B and
existing CLIP tower referenced by the original protocols. Offline environment,
4 torch threads, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4`. No weights changed.
Nonoverwriting `artifacts` and `upstream` symlinks point to the original repository.

Before launch both host GPUs were idle and neither host nor container had tmux
sessions. Time-limited CUDA allocation checks passed on both GPUs:

- Host GPU1/container GPU0: `GPU-3846413a-4238-d307-b1f3-10c2dfbe002c`.
- Host GPU0, existing `merit-runner` SSH account:
  `GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023`.

Two independent full-manifest dataset jobs, not new data splits: VQA-RAD on GPU1,
SLAKE on GPU0. Initial tmux sessions `cres-vqarad-canary` and
`cres-slake-canary`; logs `runs/cres-checks/{vqarad,slake}-canary.log`.

## Observed compatibility failure and correction

Initial preflight correctly stopped before GPU inference: legacy `routing.json`
lacks `group_id`. VQA-RAD manifest image identity is not a file digest. Inspected
the actual original `_route_records` implementation and image preparation code:
RGB identity is SHA256 of `str(image.size).encode() + image.convert('RGB').tobytes()`.
SLAKE uses file SHA256.

The adapter verifies either exact file identity or the exact legacy RGB formula,
and additionally freezes actual file bytes for every image and checks them again
before inference. It does not rewrite manifests or source hashes. Missing group
IDs require exact reconstruction of the original routing cache key. Reused route
protocols must match the stored donor SHA and identity before following the chain.
VQA-RAD routing origin is `46662ead4eb2ebeb14ace380ea76a5179a7157e8e7f959a0f771422a09d477c6`;
SLAKE routing origin is its source run itself. All 451 / 2094 rows passed checks.
Prompt/evidence delivery binding remains separately enforced by the original runner.

## Frozen setup and commands

`configs/control_evidence.json` unchanged: uniform short answers, 64 tokens,
KL budget .05, maximum strength 1, fixed strength .5, all eight arms.
No references are passed to generation. Scorer is pinned before scoring.

Run from this worktree with the existing environment wrapper:

```bash
CRES_SOURCE_RUN=/home/dbw/merit-feddg/runs/matched-verified-packets-anchor/72f36cd3907456c90166699cd0a7fe007de3eff937f6696e04f41a0dae5e5082
CRES_MANIFEST=/home/dbw/merit-feddg/runs/vqarad-official-full-test/manifest.jsonl
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 bash scripts/with_gate_kb_env.sh \
  scripts/run_control_evidence.py --source-run "$CRES_SOURCE_RUN" \
  --manifest "$CRES_MANIFEST" --output runs/cres-vqarad --max-cases 2
```

SLAKE uses source
`/home/dbw/merit-feddg/runs/slake-matched-verified-packets-anchor/a57a640a1eb9e395da0523c9e999f3b28e67ece1edfc4c5861e9883f0855e465`,
manifest `/home/dbw/merit-feddg/runs/slake-official-full-test/manifest.jsonl`,
output `runs/cres-slake`, identical flags. `--check-only` passed for both.
Removing `--max-cases` resumes the same frozen complete manifest after acceptance.

The six requested regression groups passed: 65 tests. Full CPU suite passed:
902 tests in 12.92 seconds; log `runs/cres-checks/pytest.log`.
New negative tests reject altered RGB content
and mismatched legacy route identities. Ruff and compile checks cover changed
runner and CRES implementation. CPU success is not medical validation.

## Real scheduling acceptance and pending results

VQA-RAD first two scheduled records passed exact zero-strength token parity,
source delivery binding, distinct controls and all eight nonempty candidate
outputs. Maximum spatial feature deltas were 3.390625 and 3.046875; CRES maximum
token KL was .0000407445 and .0003655015. Eight-arm walls were 35.980 and 65.373
seconds, of which CRES alone was 8.970 and 18.049 seconds. This is a replay
implementation, not the previous semantic-only persistent-cache backend.

SLAKE first four records had no presented spatial operator and were correctly
retained in the denominator as unavailable, not successes. Sequential record
0004 had eight native regions, feature delta .2265625, two different controls
(each changed 4608 cells), exact zero parity, and maximum CRES KL .0000150755.
Its eight-arm wall was 34.999 seconds. No labels or scorer results were inspected.

Frozen output roots:

- `runs/cres-vqarad/44d017cb7ba4b311f6631736355347edc3fcc3ca3f750ba7eecf0e056ea91584`
- `runs/cres-slake/8bd4a1a40e986cec46975492407bb74e96e6aa31d62b2cf5e765f92665dc5890`

Scorers are pinned in both roots. Container tmux `cres-vqarad-full` resumes
451 records; host tmux `cres-slake-full` waits for the existing canary lock and
requires its 12 completed scheduling records before resuming all 2094. Both
jobs run the complete-only evaluator only after successful generation. Logs:
`runs/cres-checks/{vqarad,slake}-{full,evaluate}.log`. VSCode exit does not stop
these tmux jobs. No data partition or new strategy was introduced.

The evaluator additionally reports all pairwise arm contrasts, text revision,
preservation of fresh compact gains, strength/KL distributions, zero residual
counts, control-unavailability reasons and recorded inherited source cost.
Inherited cached timings are explicitly not fresh deployment cost. No actual
forward counter exists here; score API calls must not be called forward counts.

Complete-only scorer must
then report all eight arms, paired image-cluster intervals and costs. No claims
about CRES exceeding compact/deletion, preserving benefit, geometry specificity,
KL-only explanations or acceptable cost are established at this checkpoint.
Patient images, raw answers, masks, weights and credentials are not committed.
