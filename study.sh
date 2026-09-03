#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: ./study.sh MANIFEST [RUN_NAME] [--include-gated] [--install-system]"
  exit 2
fi

MANIFEST="$1"
shift
RUN_NAME="real-lodo"
if [[ $# -gt 0 && "$1" != -* ]]; then
  RUN_NAME="$1"
  shift
fi

INCLUDE_GATED="${INCLUDE_GATED:-0}"
INSTALL_SYSTEM="${INSTALL_SYSTEM:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --include-gated) INCLUDE_GATED=1; shift ;;
    --install-system) INSTALL_SYSTEM=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXE="$REPO_ROOT/.venv/bin/python"
bootstrap_args=(--profile research-2d)
verify_args=(--profile research-2d --root "$REPO_ROOT/artifacts")
if [[ "$INCLUDE_GATED" == "1" ]]; then
  bootstrap_args+=(--include-gated)
  verify_args+=(--include-gated)
fi
if [[ "$INSTALL_SYSTEM" == "1" ]]; then
  bootstrap_args+=(--install-system)
fi

if [[ ! -x "$PYTHON_EXE" ]] \
  || ! "$PYTHON_EXE" -c "import torch, transformers, datasets" >/dev/null 2>&1 \
  || ! "$PYTHON_EXE" -m merit_feddg.cli verify-assets "${verify_args[@]}" >/dev/null 2>&1; then
  "$REPO_ROOT/bootstrap.sh" "${bootstrap_args[@]}"
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
