#!/usr/bin/env bash
# One-command, offline, resumable LLaVA-Med full-method experiment on host GPU 0.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${MERIT_LLAVA_PYTHON:-/opt/miniconda3/envs/huatuo/bin/python}"
GPU_INDEX="${MERIT_GPU_INDEX:-0}"
ARTIFACTS="${MERIT_ARTIFACTS:-$ROOT_DIR/artifacts}"
OUTPUT="${MERIT_OUTPUT:-$ROOT_DIR/runs/native-v08-full-gpu0}"
SOURCE_PER_GROUP="${MERIT_SOURCE_PER_GROUP:-16}"
TARGET_LIMIT="${MERIT_TARGET_LIMIT:-16}"
SEED="${MERIT_SEED:-17}"
CHEXAGENT="${MERIT_CHEXAGENT:-on}"
CHECK_ONLY="${MERIT_CHECK_ONLY:-0}"
CONFIG="${MERIT_CONFIG:-$ROOT_DIR/configs/llava_med_capabilities.yaml}"

[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }
[[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || { echo "MERIT_GPU_INDEX must be an integer" >&2; exit 2; }
[[ "$SOURCE_PER_GROUP" =~ ^[1-9][0-9]*$ ]] || { echo "MERIT_SOURCE_PER_GROUP must be positive" >&2; exit 2; }
[[ "$TARGET_LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "MERIT_TARGET_LIMIT must be positive" >&2; exit 2; }
[[ "$CHEXAGENT" =~ ^(on|off|auto)$ ]] || { echo "MERIT_CHEXAGENT must be on, off or auto" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }

cd -- "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

COMMON_ARGS=(
  --python "$PYTHON_BIN"
  --config "$CONFIG"
  --artifacts "$ARTIFACTS"
  --skip-download
  --chexagent "$CHEXAGENT"
)

echo "Running offline preflight with GPU $GPU_INDEX and CheXagent=$CHEXAGENT"
bash "$ROOT_DIR/run_llava_med.sh" "${COMMON_ARGS[@]}" --check-only
if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "Preflight completed; MERIT_CHECK_ONLY=1, so inference was not started."
  exit 0
fi

IFS=',' read -r GPU_TOTAL_MIB GPU_FREE_MIB < <(
  nvidia-smi --id="$GPU_INDEX" \
    --query-gpu=memory.total,memory.free \
    --format=csv,noheader,nounits | head -n 1 | tr -d ' '
)
MIN_FREE_MIB=20480
if [[ "$CHEXAGENT" != "off" ]]; then
  MIN_FREE_MIB=28672
fi
echo "GPU $GPU_INDEX memory: total=${GPU_TOTAL_MIB} MiB, free=${GPU_FREE_MIB} MiB; required free=${MIN_FREE_MIB} MiB"
if (( GPU_FREE_MIB < MIN_FREE_MIB )); then
  echo "Insufficient free GPU memory. Stop other jobs or run MERIT_CHEXAGENT=off for the core suite." >&2
  exit 3
fi

mkdir -p -- "$OUTPUT"
echo "Starting/resuming full method matrix: source/group=$SOURCE_PER_GROUP, target/dataset=$TARGET_LIMIT, seed=$SEED"
echo "Output: $OUTPUT"
bash "$ROOT_DIR/run_llava_med.sh" "${COMMON_ARGS[@]}" \
  --dataset both \
  --source-per-group "$SOURCE_PER_GROUP" \
  --target-limit "$TARGET_LIMIT" \
  --seed "$SEED" \
  --output "$OUTPUT" \
  2>&1 | tee -a "$OUTPUT/host-gpu0.log"
