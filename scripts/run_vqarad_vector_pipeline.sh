#!/usr/bin/env bash
# Entire collaboration is training-free. No source split, cache building or trainer.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
# Respect the caller's container GPU mapping and Python environment.
python_bin="${MERIT_PYTHON:-python}"
manifest="${1:?Usage: run_vqarad_vector_pipeline.sh FULL_MANIFEST [additional runner arguments]}"
shift
exec "$python_bin" -m merit_feddg.matched_evaluation \
  --protocol spatial \
  --config configs/matched_vector_gate.yaml \
  --manifest "$manifest" \
  --output runs/matched-spatial \
  --artifacts artifacts "$@"
