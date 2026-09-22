#!/usr/bin/env bash
set -euo pipefail
run=/home/dbw/merit-feddg-huatuo-spatial/runs/mimic-gpu1-v2
queue_pid=1496037
while kill -0 "$queue_pid" 2>/dev/null; do
  sleep 60
done
if ! grep -q 'QUEUE_V2_COMPLETE' "$run/queue.log"; then
  echo "QUEUE_DID_NOT_COMPLETE $(date -u +%FT%TZ)" >&2
  exit 1
fi
echo "SCORE_STARTED $(date -u +%FT%TZ)" >&2
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python /home/dbw/merit-tx-v3/scripts/score_mimic_gpu1_bard.py
echo "SCORE_COMPLETE $(date -u +%FT%TZ)" >&2
