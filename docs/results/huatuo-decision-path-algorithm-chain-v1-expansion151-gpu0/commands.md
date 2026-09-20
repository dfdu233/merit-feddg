# Actual GPU0 continuation commands

Executed on 2026-09-20 UTC in `/home/dbw/merit-feddg-chain-5066828`. The latest user explicitly authorized host GPU0. These are execution records, not instructions to rerun a terminal experiment.

The predecessor GPU1 C-attempt-2 failed ownership verification before model loading (0 new forwards). Migration preserved that failure, all earlier stages, A=3/C=2 and 10164 forwards. The standard retry alone granted C=3.

```bash
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python scripts/migrate_chain_device.py \
  --source runs/chain-v1-expansion151-gpu1 \
  --output runs/chain-v1-expansion151-gpu0-v2 \
  --gpu-uuid GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023 \
  --display-contexts runs/expansion-preparation/gpu0-display-contexts.json
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python scripts/run_algorithm_chain.py retry \
  --output runs/chain-v1-expansion151-gpu0-v2
```

The new run tree was made readable/writable by the existing host runner. Through the existing host SSH connection, executed:

```bash
cd /home/dbw/merit-feddg-chain-5066828
HF_HOME=/home/dbw/hf-shared HF_HUB_CACHE=/home/dbw/hf-shared/hub OMP_NUM_THREADS=1 \
  bash scripts/run_algorithm_chain_detached.sh \
  /home/dbw/.runtime/miniconda3/envs/huatuo/bin/python \
  /home/dbw/merit-feddg-chain-5066828/runs/chain-v1-expansion151-gpu0-v2
```

Actual controller PID2326084 and worker PID2326893 were checked against GPU UUID and live case artifacts. The container-visible GPU0 is host GPU1, so the host connection was necessary. No guessed index, second worker, model download, dependency upgrade, other-process termination, or source change during inference was used.

Only the previously verified Nautilus C+G context (PID2321122, 34 MiB, frozen maximum64 MiB) was allowed; ordinary competing compute processes remained rejected. Raw plans, original answers/references and medical images remain on the server. The earlier cancelled GPU0 queue and failed GPU1 attempt remain preserved.
