#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: ./study.sh MANIFEST [RUN_NAME]"
  exit 2
fi

MANIFEST="$1"
RUN_NAME="${2:-real-lodo}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXE="$REPO_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON_EXE" ]] || ! "$PYTHON_EXE" -c "import torch, transformers, datasets" >/dev/null 2>&1; then
  "$REPO_ROOT/bootstrap.sh" research-2d "${INCLUDE_GATED:+--include-gated}"
fi

"$PYTHON_EXE" -m merit_feddg.cli extract \
  --manifest "$MANIFEST" \
  --config "$REPO_ROOT/configs/real_2d.yaml" \
  --artifacts "$REPO_ROOT/artifacts" \
  --output "$REPO_ROOT/cache/$RUN_NAME.jsonl"

"$PYTHON_EXE" -m merit_feddg.cli compare \
  --input "$REPO_ROOT/cache/$RUN_NAME.jsonl" \
  --config "$REPO_ROOT/configs/real_compare.example.yaml" \
  --output "$REPO_ROOT/runs/$RUN_NAME"

echo "Completed real-model study: $REPO_ROOT/runs/$RUN_NAME"
