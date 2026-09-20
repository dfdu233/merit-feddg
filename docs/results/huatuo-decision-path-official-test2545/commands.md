# Actual preparation and launch

Executed in `/home/dbw/merit-feddg-chain-5066828`, existing Huatuo Python environment. Prior failed outputs remain preserved.

```bash
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python scripts/prepare_algorithm_test.py --output runs/official-test2545-inputs-v2
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python scripts/run_algorithm_test.py freeze \
  --inputs runs/official-test2545-inputs-v2 \
  --predecessor runs/chain-v1-expansion151-gpu0-v2 \
  --previous-test runs/official-test2545-gpu0-v1 \
  --output runs/official-test2545-gpu0-v2 --max-total-forwards 200000
```

The new task directories were assigned to the existing host runner. On the host, with existing shared offline cache:

```bash
cd /home/dbw/merit-feddg-chain-5066828
HF_HOME=/home/dbw/hf-shared HF_HUB_CACHE=/home/dbw/hf-shared/hub OMP_NUM_THREADS=1 \
  nohup setsid /home/dbw/.runtime/miniconda3/envs/huatuo/bin/python scripts/run_algorithm_test.py run \
  --output /home/dbw/merit-feddg-chain-5066828/runs/official-test2545-gpu0-v2 \
  > runs/official-test2545-gpu0-v2/controller.log 2>&1 < /dev/null &
```

Actual controller2729171 exited on a competing-compute ownership failure before loading. Afterwards a single detached host shell (PID2740944) was launched with CUDA_VISIBLE_DEVICES set to the frozen GPU0 UUID, the same shared-cache variables and the following bounded wait body. It consumes the last attempt only after a successful read-only resource check:

```bash
for probe_index in {1..1440}; do
  if /home/dbw/.runtime/miniconda3/envs/huatuo/bin/python -c 'from merit_feddg.algorithm_chain.native import check_device; from merit_feddg.algorithm_chain.storage import read; p=read("runs/official-test2545-gpu0-v2/plan.json"); check_device(p["runtime"]["gpu_uuid"],p["runtime"].get("allowed_display_contexts",[]))'; then
    date -u
    exec /home/dbw/.runtime/miniconda3/envs/huatuo/bin/python scripts/run_algorithm_test.py run --output /home/dbw/merit-feddg-chain-5066828/runs/official-test2545-gpu0-v2
  fi
  sleep 30
done
printf 'Resource wait expired without another attempt\n'
```

Log `runs/official-test2545-gpu0-v2/resource-wait.log`. Do not launch a duplicate waiter/controller. The first failed v1 had identical commands without `--previous-test`, using inputs-v1 and run-v1. The source-order repair did not change native decoding or the old TRAIN plan.
