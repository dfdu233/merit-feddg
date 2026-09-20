# Actual amended continuation

Worktree: `/home/dbw/merit-feddg-chain-5066828`.
Python: `/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python`.

The user explicitly approved a maximum of 3 attempts per node. The complete
policy otherwise remains unchanged, including 200000 total forwards. The default
configuration still uses 2 attempts; only this successor freezes 3.

A label-blind technical preflight on the previous parity-failing input compared
full-sequence replay, last-query projection, and native incremental prefix
rebuilding. The first two reproduced the failure (6 forwards each); rebuilding
passed with exactly matching logits (16 forwards). All 28 preflight forwards are
charged to the campaign. A host read-permission failure before model loading is
also retained locally; ownership of this task's input artifacts was corrected.

Raw preflight scripts/logs: `runs/preparation/probe_chain_parity*.py` and
`runs/chain-v1-repair-preflight/`. These are technical probes, not a newly selected
scientific subset. The formal queue remains all original 128 cases.

Executed successor migration and normal retry:

```bash
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python scripts/migrate_chain_repair.py \
  --source runs/chain-v1-gpu1 --output runs/chain-v1-repair \
  --plan-input runs/preparation/repair-plan-input.json \
  --preflight runs/chain-v1-repair-preflight
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python scripts/run_algorithm_chain.py \
  retry --output runs/chain-v1-repair
```

The migration checks predecessor plan/state identity, both decisions, counted
forwards, exact data/scorer/model pins and the single authorized policy change.
It copies old attempts byte-for-byte and retains their history. It establishes a
new runtime fingerprint for the changed physical device. The resulting state is
still blocked with A=2; the existing `retry` operation grants A=3. Inherited cost:
1046 + 28 = 1074 forwards, leaving 198926 before the full A stage.

Actual host launch (through the existing authorized host connection):

```bash
cd /home/dbw/merit-feddg-chain-5066828
HF_HOME=/home/dbw/hf-shared HF_HUB_CACHE=/home/dbw/hf-shared/hub OMP_NUM_THREADS=1 \
  bash scripts/run_algorithm_chain_detached.sh \
  /home/dbw/.runtime/miniconda3/envs/huatuo/bin/python \
  /home/dbw/merit-feddg-chain-5066828/runs/chain-v1-repair
```

Controller PID 1839971, A worker PID 1841317 (host namespace). Controller log:
`runs/chain-v1-repair/controller-20260920T100801Z-1839967.log`.
The wrapper received exactly two positional arguments. The controller sets the
physical GPU UUID from the frozen plan for every worker and disables CUDA in the
separate scorer. No repeated baseline/expert cache generation or TEST launch.

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. /home/dbw/.runtime/miniconda3/envs/huatuo/bin/python -m pytest \
  tests/test_algorithm_chain.py tests/test_algorithm_chain_controller.py \
  tests/test_chain_repair.py tests/test_chain_device_migration.py \
  tests/test_pathway_restore.py -o addopts= -q --disable-warnings
```

128 passed in 3.85 seconds. Compilation, CLI help, wrapper shell syntax and diff
checks passed. Relevant new tests verify final-query-only hooks during complete
prefix rebuilding, exact cached-logit reproduction, actual forward counts,
strictly identified desktop-context admission, and inheritance without resetting
attempts, histories or costs.
