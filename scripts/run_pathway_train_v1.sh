#!/bin/bash
# Frozen existing queues only. Use within tmux; no automatic GPU discovery.
set -euo pipefail
cd /home/dbw/merit-feddg-pathway-restore
DATASET=${1:?slake or vqarad}
STAGE=${2:?check, canary or run}
case "$DATASET" in
  slake) SOURCE=/home/dbw/merit-feddg-huatuo-critic/runs/native-slake128-confirm-v2; OUTPUT=runs/pathway-slake128-v5; PLANNED=128 ;;
  vqarad) SOURCE=/home/dbw/merit-feddg-huatuo-anchored/runs/native-vqa87-v1; OUTPUT=runs/pathway-vqa87-v5; PLANNED=87 ;;
  *) exit 2 ;;
esac
case "$STAGE" in check|canary|run) ;; *) exit 2 ;; esac
# Preserve the failed VQA canary. It does not authorize a full run.
if [ "$DATASET" = vqarad ] && [ "$STAGE" = run ]; then
  echo 'VQA-RAD87 canary failed with undefined restore_mass; full run is blocked.' >&2
  exit 3
fi
# Documentation/scoring commits may differ; generation sources must not.
git diff --exit-code b51dc3f13b5911c7a0c699767110b06e81c9781f -- merit_feddg scripts/run_huatuo_pathway.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=GPU-3846413a-4238-d307-b1f3-10c2dfbe002c
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p runs/pathway-audit-v1
exec 9>runs/pathway-audit-v1/device.lock
flock -n 9 || exit 4
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RECORD="runs/pathway-audit-v1/${DATASET}-${STAGE}-${STAMP}"
exec > >(tee "$RECORD.log") 2>&1
echo $$ > "$RECORD.pid"
printf 'planned=%s stage=%s output=%s\n' "$PLANNED" "$STAGE" "$OUTPUT"
git rev-parse HEAD
nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.used,memory.total --format=csv
trap 'code=$?; echo "$code" > "$RECORD.exit"; echo "exit=$code stage=$STAGE"' EXIT
set -x
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python -u scripts/run_huatuo_pathway.py \
  --source-run "$SOURCE" --generation-json docs/results/huatuo-pathway-restore-train-v1/generation.json \
  --output "$OUTPUT" --import-root /home/dbw/ANCHOR --layers -1 \
  --modes restore_mass restore_vector --stage "$STAGE"
