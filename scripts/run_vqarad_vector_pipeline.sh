#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/merit-feddg
export PYTHONPATH=/home/dbw/merit-feddg
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

python_bin=/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python
source_root=/home/dbw/merit-feddg/runs/matched-vector-gate-source
bridge_path="$source_root/bridge-seed0-epoch1.pt"
prepare_pid=$(cat "$source_root/prepare.pid")

while kill -0 "$prepare_pid" 2>/dev/null && \
      ! ps -o stat= -p "$prepare_pid" | grep -q '^Z'; do
  sleep 20
done
test -s "$source_root/source.jsonl"
test -s "$source_root/audit.json"

"$python_bin" -m merit_feddg.tensor_train \
  --generalist "$source_root/generalist-base.yaml" \
  --contract configs/vqarad_tensor_contract.json \
  --source "$source_root/source.jsonl" \
  --output "$bridge_path" \
  --epochs 1 \
  --learning-rate 0.0001 \
  --seed 0 \
  > "$source_root/train.log" 2>&1

test -s "$bridge_path"
export MERIT_TENSOR_BRIDGE="$bridge_path"
"$python_bin" -m merit_feddg.matched_evaluation \
  --protocol vector \
  --config configs/matched_vector_gate.yaml \
  --manifest /home/dbw/merit-feddg/runs/vqarad-official-full-test/manifest.jsonl \
  --output runs/matched-vector-gate \
  --artifacts artifacts \
  > runs/matched-vector-gate.background.log 2>&1
