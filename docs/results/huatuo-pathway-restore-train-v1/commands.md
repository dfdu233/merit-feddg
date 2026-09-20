# Actual server commands

All commands run from `/home/dbw/merit-feddg-pathway-restore`. Generation code is frozen at `b51dc3f13b5911c7a0c699767110b06e81c9781f`; later report/evaluator/test-only commits are compatible. The wrapper verifies the generation-source diff against that commit. It records command, repository SHA, UUID, PID, phase, log and exit code, and uses an exclusive device lock plus the runner's output lock. Protocol files contain exact source fingerprints and run identity.

The actual completed SLAKE launch used `runs/pathway-audit-v1/slake-v5.sh` in tmux `merit-pathway-slake-v5`, wrapper PID 2818470 and full-run Python PID 2819141. It executed check, canary, then run. Check/canary/run all exited 0. Its 128-case completion marker counts processed cases, including explicitly failed mass arms.

The subsequent actual VQA launch used `runs/pathway-audit-v1/vqa-v5.sh` in tmux `merit-pathway-vqa-v5`. Check exited 0; canary exited 1 on an undefined mass ratio. Run was not invoked. Both sessions have ended.

Current-source verification and safe SLAKE resume (completed compatible cases are reused):

```bash
cd /home/dbw/merit-feddg-pathway-restore
bash scripts/run_pathway_train_v1.sh slake check
tmux new-session -d -s merit-pathway-slake-resume \
  'bash /home/dbw/merit-feddg-pathway-restore/scripts/run_pathway_train_v1.sh slake run'
```

The failed VQA canary is preserved. `bash scripts/run_pathway_train_v1.sh vqarad run` explicitly refuses a full run. Do not expand or select a different prefix to avoid the failure. A future change to mass semantics or kernel would be a different experiment.

To reproduce scoring independently into a fresh output directory (references are read only here):

```bash
cd /home/dbw/merit-feddg-pathway-restore
OMP_NUM_THREADS=1 PYTHONPATH=. \
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python \
  scripts/evaluate_huatuo_pathway.py \
  --run runs/pathway-slake128-v5 \
  --source-run /home/dbw/merit-feddg-huatuo-critic/runs/native-slake128-confirm-v2 \
  --references /home/dbw/merit-feddg-huatuo-critic/runs/inputs-slake128-confirm/references.json \
  --output runs/pathway-evaluation-replay-slake128
```

The published evaluation used the same command with `--output docs/results/huatuo-pathway-restore-train-v1/slake128`. Scoring refuses to overwrite prior output. A second scoring invocation may reuse identical local offline diagnostics but cannot replace changed diagnostics.

Validation actually executed:

```bash
cd /home/dbw/merit-feddg-pathway-restore
OMP_NUM_THREADS=1 PYTHONPATH=. \
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python -m pytest \
  tests/test_pathway_restore.py tests/test_pathway_runner.py \
  tests/test_capability_runtime.py tests/test_transport_runtime_integration.py \
  tests/test_evidence_permissions_transport.py \
  -o addopts='' -q --disable-warnings
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python -m py_compile \
  merit_feddg/pathway_restore.py merit_feddg/huatuo_pathway.py \
  scripts/run_huatuo_pathway.py scripts/evaluate_huatuo_pathway.py
bash -n scripts/run_pathway_train_v1.sh
git diff --check
```

No new models or core dependencies were installed. Local raw run and failed-attempt directories under `runs/` are deliberately excluded from Git.
