#!/usr/bin/env bash
set -euo pipefail
# Reuse the server environment; do not install or upgrade shared packages.
cd /home/dbw/merit-feddg-evidence-gate-kb
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
exec /home/dbw/merit-feddg/.venv/bin/python "$@"
