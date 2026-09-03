#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-smoke}"
GATED_FLAG="${2:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXE="$REPO_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON_EXE" ]]; then
  python3 -m venv "$REPO_ROOT/.venv"
fi

"$PYTHON_EXE" -m pip install --upgrade pip
if [[ "$PROFILE" == "smoke" ]]; then
  "$PYTHON_EXE" -m pip install -e "$REPO_ROOT[dev]"
else
  "$PYTHON_EXE" -m pip install -e "$REPO_ROOT[research,dev]"
fi

DOWNLOAD_ARGS=(--profile "$PROFILE" --root "$REPO_ROOT/artifacts")
if [[ "$GATED_FLAG" == "--include-gated" ]]; then
  DOWNLOAD_ARGS+=(--include-gated)
fi
"$PYTHON_EXE" -m merit_feddg.cli download "${DOWNLOAD_ARGS[@]}"
"$PYTHON_EXE" -m pytest "$REPO_ROOT/tests"
"$PYTHON_EXE" -m merit_feddg.cli simulate --config "$REPO_ROOT/configs/smoke.yaml" --output "$REPO_ROOT/runs/smoke"
echo "MERIT-FedDG is ready. Profile: $PROFILE"
