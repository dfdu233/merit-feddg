#!/usr/bin/env bash
set -euo pipefail
SOURCE_PER_GROUP=16
TARGET_LIMIT=16
MIRROR=auto
CONFIG=configs/capability_generation.yaml
SKIP_BOOTSTRAP=0
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-per-group) SOURCE_PER_GROUP="${2:?missing size}"; shift 2 ;;
    --target-limit) TARGET_LIMIT="${2:?missing size}"; shift 2 ;;
    --mirror) MIRROR="${2:?missing mode}"; shift 2 ;;
    --config) CONFIG="${2:?missing config}"; shift 2 ;;
    --skip-bootstrap) SKIP_BOOTSTRAP=1; shift ;;
    --install-system) EXTRA+=(--install-system); shift ;;
    --conch-source) EXTRA+=(--conch-source "${2:?missing URL}"); shift 2 ;;
    -h|--help)
      echo 'Usage: ./run_capabilities.sh [--mirror cn] [--source-per-group 16] [--target-limit 16]'
      echo 'Options: --install-system --conch-source URL --skip-bootstrap --config PATH'
      echo 'Native capability collaboration on real free-text PathVQA; proxy domains are not hospitals.'
      echo 'Uses existing open-generation assets; custom config assets must be installed separately.'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
if [[ $SKIP_BOOTSTRAP -eq 0 ]]; then
  ./bootstrap.sh --profile open-generation --include-gated --mirror "$MIRROR" "${EXTRA[@]}"
fi
PYTHON="$REPO_ROOT/.venv/bin/python"
DATA_DIR="$REPO_ROOT/data/open-source${SOURCE_PER_GROUP}-target${TARGET_LIMIT}"
"$PYTHON" -m merit_feddg.cli prepare-open-vqa --artifacts "$REPO_ROOT/artifacts" \
  --output "$DATA_DIR" --source-per-group "$SOURCE_PER_GROUP" --target-limit "$TARGET_LIMIT"
"$PYTHON" -m merit_feddg.cli capability-study \
  --source "$DATA_DIR/source.jsonl" --target "$DATA_DIR/target.jsonl" \
  --references "$DATA_DIR/references.json" --config "$CONFIG" \
  --artifacts "$REPO_ROOT/artifacts" --output "$REPO_ROOT/runs/capability-generation"
echo 'Completed. See runs/capability-generation/latest.json for the exact run directory.'
