# Actual freeze and resource queue

Executed from `/home/dbw/merit-feddg-chain-5066828`:

```bash
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python scripts/run_algorithm_chain.py freeze-expansion \
  --plan runs/expansion-preparation/plan-input-gpu1.json \
  --source runs/chain-v1-repair \
  --output runs/chain-v1-expansion151-gpu1
```

The superseded GPU0 wait process group 2075033 was explicitly terminated on the
host before model launch. Cancellation is recorded in
`runs/chain-v1-expansion151/cancelled-before-launch.json`.
The old plan/state were not rewritten.

The following exact shell body was submitted locally with `nohup setsid bash -c`,
stdin redirected from `/dev/null`, and stdout/stderr redirected to
`runs/chain-v1-expansion151-gpu1/resource-wait.log`. Its PID was recorded in
`resource-wait.pid`. It has no model call until both resource checks pass.

```bash
set -eu
cd '/home/dbw/merit-feddg-chain-5066828'
exec 9>'/home/dbw/merit-feddg-chain-5066828/runs/chain-v1-expansion151-gpu1/.launch-queue.lock'
flock -n -E 73 9
export PYTHONPATH='/home/dbw/merit-feddg-chain-5066828' HF_HOME=/home/dbw/hf-shared HF_HUB_CACHE=/home/dbw/hf-shared/hub OMP_NUM_THREADS=1
queue_checks=0
while [ "$queue_checks" -lt 1440 ]; do
  date -u +%Y-%m-%dT%H:%M:%SZ
  if '/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python' -c 'import json,os,subprocess,sys
from merit_feddg.algorithm_chain.native import check_device
p=json.load(open("/home/dbw/merit-feddg-chain-5066828/runs/chain-v1-expansion151-gpu1/plan.json"));s=json.load(open("/home/dbw/merit-feddg-chain-5066828/runs/chain-v1-expansion151-gpu1/state.json"))
if s['\''node'\'']!='\''C'\'': sys.exit(42)
u=p['\''runtime'\'']['\''gpu_uuid'\'']
assert u=='\''GPU-3846413a-4238-d307-b1f3-10c2dfbe002c'\'','\''GPU1-only constraint'\''
os.environ['\''CUDA_VISIBLE_DEVICES'\'']=u
free=int(subprocess.check_output(['\''nvidia-smi'\'','\''--id='\''+u,'\''--query-gpu=memory.free'\'','\''--format=csv,noheader,nounits'\''],text=True).strip())
if free<30720: print('\''WAIT_GPU1 free_MiB='\''+str(free),flush=True);sys.exit(1)
check_device(u)
print('\''GPU1_RESOURCE_READY free_MiB='\''+str(free),flush=True)
'; then
    exec 9>&-
    exec bash scripts/run_algorithm_chain_detached.sh '/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python' '/home/dbw/merit-feddg-chain-5066828/runs/chain-v1-expansion151-gpu1'
  else
    queue_rc=$?
    if [ "$queue_rc" -eq 42 ]; then echo STATE_NO_LONGER_PENDING; exit 0; fi
  fi
  queue_checks=$((queue_checks+1))
  sleep 30
done
echo GPU1_RESOURCE_WAIT_TIMEOUT_AFTER_12_HOURS
exit 75
```

Read-only status, without claiming that queued C is already generating:

```bash
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python scripts/run_algorithm_chain.py \
  status --output runs/chain-v1-expansion151-gpu1
tail runs/chain-v1-expansion151-gpu1/resource-wait.log
```

The source directories were verified through the normal `source_inventory`
contract. Additional input selection and audit are preserved under
`runs/expansion-preparation/`. The actual per-cohort native canaries will execute
only inside the scheduled worker.

Validation:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. /home/dbw/.runtime/miniconda3/envs/huatuo/bin/python -m pytest \
  tests/test_algorithm_chain.py tests/test_algorithm_chain_controller.py \
  tests/test_chain_expansion.py tests/test_chain_repair.py \
  tests/test_chain_device_migration.py tests/test_pathway_restore.py \
  -o addopts= -q --disable-warnings
```

134 passed in 3.61 seconds. No dependency installation, model download, training,
official TEST execution or GPU0 model launch occurred for this expansion.
