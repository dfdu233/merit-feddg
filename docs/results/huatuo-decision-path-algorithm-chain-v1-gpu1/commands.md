# Actual device migration and execution

Executed from `/home/dbw/merit-feddg-chain-5066828` on 2026-09-20 UTC.

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory --format=csv
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python scripts/migrate_chain_device.py \
  --source runs/chain-v1 --output runs/chain-v1-gpu1 \
  --gpu-uuid GPU-3846413a-4238-d307-b1f3-10c2dfbe002c
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python scripts/run_algorithm_chain.py \
  retry --output /home/dbw/merit-feddg-chain-5066828/runs/chain-v1-gpu1
HF_HOME=/home/dbw/hf-shared HF_HUB_CACHE=/home/dbw/hf-shared/hub OMP_NUM_THREADS=1 \
  bash scripts/run_algorithm_chain_detached.sh \
  /home/dbw/.runtime/miniconda3/envs/huatuo/bin/python \
  /home/dbw/merit-feddg-chain-5066828/runs/chain-v1-gpu1
```

The device migration script verifies the predecessor pins, zero model execution, exact single resource-failure history, and a new destination. It creates a device-only successor with a lineage certificate, unchanged scientific plan/code pins, and inherited **blocked** state. Only the existing `retry` command increments the inherited A attempt from 1 to 2 and grants execution. The original plan/state/artifacts are untouched; failure artifacts are copied byte-for-byte. This does not use `initial_state` or create fresh retry eligibility.

Controller PID 3360098; worker PID 3360541 (container namespace). Durable controller log:
`runs/chain-v1-gpu1/controller-20260920T094751Z-3360091.log`.

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. /home/dbw/.runtime/miniconda3/envs/huatuo/bin/python -m pytest \
  tests/test_algorithm_chain.py tests/test_algorithm_chain_controller.py \
  tests/test_chain_device_migration.py tests/test_pathway_restore.py \
  -o addopts= -q --disable-warnings
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python -m py_compile scripts/migrate_chain_device.py
bash -n scripts/run_algorithm_chain_detached.sh
git diff --check
```

Tests: 120 passed in 4.14 seconds. The migration-specific synthetic tests check attempt inheritance, refusal after exhaustion, prior-evidence preservation, and rejection of model work, holdout use, repeated attempts or existing destination. They are not model evidence.

Read-only status:

```bash
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python scripts/run_algorithm_chain.py \
  status --output /home/dbw/merit-feddg-chain-5066828/runs/chain-v1-gpu1
```

Both A attempts are consumed. No third retry, plan reset, B/C bypass, or TEST command is authorized by this frozen campaign. Full predictions, references, source identities and detailed caches remain local. The public per-case file includes hashed identifiers and numeric diagnostics, without answer text, reference text, images or expert payloads.
