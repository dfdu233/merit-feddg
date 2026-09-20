# Actual launch and recovery commands

Execution worktree: `/home/dbw/merit-feddg-chain-5066828`.
Python: `/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python`.
Original executable input plan: `/home/dbw/merit-feddg-chain-5066828/runs/preparation/plan-input.json`.
Frozen output: `/home/dbw/merit-feddg-chain-5066828/runs/chain-v1`.

The actual freeze command was executed on the host with existing shared assets:

```bash
ssh -i /root/.ssh/merit_host_gpu0_ed25519 -o BatchMode=yes merit-runner@172.17.0.1 \
 'cd /home/dbw/merit-feddg-chain-5066828 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HOME=/home/dbw/hf-shared HF_HUB_CACHE=/home/dbw/hf-shared/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/dbw/.runtime/miniconda3/envs/huatuo/bin/python scripts/run_algorithm_chain.py freeze --plan /home/dbw/merit-feddg-chain-5066828/runs/preparation/plan-input.json --output /home/dbw/merit-feddg-chain-5066828/runs/chain-v1'
```

**Do not repeat freeze on this existing directory.** The successful plan and failed attempt are immutable.

Actual detached submission (the wrapper has exactly two positional arguments):

```bash
ssh -i /root/.ssh/merit_host_gpu0_ed25519 -o BatchMode=yes merit-runner@172.17.0.1 \
 'cd /home/dbw/merit-feddg-chain-5066828 && export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HOME=/home/dbw/hf-shared HF_HUB_CACHE=/home/dbw/hf-shared/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; bash scripts/run_algorithm_chain_detached.sh /home/dbw/.runtime/miniconda3/envs/huatuo/bin/python /home/dbw/merit-feddg-chain-5066828/runs/chain-v1'
```

Read real state without starting work:

```bash
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python \
 /home/dbw/merit-feddg-chain-5066828/scripts/run_algorithm_chain.py status \
 --output /home/dbw/merit-feddg-chain-5066828/runs/chain-v1
```

Before using the **one remaining** technical retry, independently verify that no other compute context remains on the frozen UUID. No retry was executed in this delivery, and no process may be killed to satisfy the check:

```bash
ssh -i /root/.ssh/merit_host_gpu0_ed25519 -o BatchMode=yes merit-runner@172.17.0.1 \
 'nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv'
```

After the actual condition is resolved, the existing supported recovery command preserves the previous failure and forward budget:

```bash
ssh -i /root/.ssh/merit_host_gpu0_ed25519 -o BatchMode=yes merit-runner@172.17.0.1 \
 'cd /home/dbw/merit-feddg-chain-5066828 && /home/dbw/.runtime/miniconda3/envs/huatuo/bin/python scripts/run_algorithm_chain.py retry --output /home/dbw/merit-feddg-chain-5066828/runs/chain-v1'
```

Then use the same detached submission above. Do not edit plan/state, change the GPU UUID, reset attempts, or start a separate scorer. The controller will invoke its CUDA-disabled scoring child only after real complete predictions exist, and will apply the frozen policy itself.

Validation commands actually run:

```bash
cd /home/dbw/merit-feddg-chain-5066828
OMP_NUM_THREADS=1 PYTHONPATH=. /home/dbw/.runtime/miniconda3/envs/huatuo/bin/python -m pytest \
 tests/test_algorithm_chain.py tests/test_algorithm_chain_controller.py \
 tests/test_pathway_restore.py tests/test_pathway_runner.py \
 -o addopts= -q --disable-warnings
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python -m compileall -q \
 merit_feddg/algorithm_chain scripts/run_algorithm_chain.py
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python scripts/run_algorithm_chain.py --help
bash -n scripts/run_algorithm_chain_detached.sh
git diff --check
```

Raw logs, process audit, unsuccessful pre-freeze metadata-permission check, complete source/exposure inventories and fingerprints remain under `runs/preparation/`. Node logs and controller/worker exit state remain under `runs/chain-v1/`. They are intentionally not committed as raw medical data.
