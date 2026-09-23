#!/usr/bin/env bash
set -euo pipefail

repo=/home/dbw/merit-tx-v3
python=/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python
run_dir="$repo/runs/v3-musk"
mkdir -p "$run_dir"
exec 8>"$run_dir/auth-watcher.lock"
flock -n 8 || { echo 'MUSK_AUTH_WATCHER_ALREADY_RUNNING' >&2; exit 1; }
cd "$repo"
echo "WAIT_HF_LOGIN $(date -u +%FT%TZ)" >&2

until "$python" - <<'PY'
from huggingface_hub import get_token
raise SystemExit(0 if get_token() else 1)
PY
do
  sleep 30
done
echo "HF_LOGIN_DETECTED $(date -u +%FT%TZ)" >&2

for attempt in 1 2 3; do
  if "$python" -u scripts/download_musk_weights.py; then
    echo "MUSK_DOWNLOAD_COMPLETE $(date -u +%FT%TZ)" >&2
    break
  fi
  echo "MUSK_DOWNLOAD_RETRY $attempt $(date -u +%FT%TZ)" >&2
  if [[ $attempt == 3 ]]; then exit 1; fi
  sleep 300
done

# Preserve headroom for the user's independent MIMIC producer on physical GPU1.
while true; do
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  if [[ $free_mib =~ ^[0-9]+$ ]] && (( free_mib >= 12000 )); then break; fi
  echo "WAIT_GPU1_HEADROOM free_mib=$free_mib $(date -u +%FT%TZ)" >&2
  sleep 60
done
echo "MUSK_SOURCE_RUN_START $(date -u +%FT%TZ)" >&2
nice -n 10 bash scripts/run_musk_source_qualification.sh
echo "MUSK_SOURCE_RUN_COMPLETE $(date -u +%FT%TZ)" >&2
