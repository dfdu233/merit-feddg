#!/usr/bin/env bash
set -euo pipefail

LIMIT_PER_CENTER=12
MIRROR="auto"
INSTALL_SYSTEM=0
FORCE_EXTRACT=0
CONCH_SOURCE=""

usage() {
  cat <<'EOF'
Usage: ./run_pathorob.sh [options]

Runs the real PathoROB six-class leave-one-medical-center-out study.

  --limit-per-center N        Label-blind patches per medical center (default: 12; 0=all)
  --mirror MODE               auto, cn, or global
  --install-system            Install required Ubuntu packages
  --conch-source URL          Override CONCH git source (useful behind a GitHub proxy)
  --force-extract             Re-run expensive model inference instead of reusing cache
  -h, --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit-per-center) LIMIT_PER_CENTER="${2:?missing limit}"; shift 2 ;;
    --mirror) MIRROR="${2:?missing mirror mode}"; shift 2 ;;
    --install-system) INSTALL_SYSTEM=1; shift ;;
    --conch-source) CONCH_SOURCE="${2:?missing CONCH source}"; shift 2 ;;
    --force-extract) FORCE_EXTRACT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

bootstrap_args=(--profile pathorob-real --include-gated --mirror "$MIRROR")
if [[ $INSTALL_SYSTEM -eq 1 ]]; then
  bootstrap_args+=(--install-system)
fi
if [[ -n "$CONCH_SOURCE" ]]; then
  bootstrap_args+=(--conch-source "$CONCH_SOURCE")
fi
./bootstrap.sh "${bootstrap_args[@]}"

PYTHON="$REPO_ROOT/.venv/bin/python"
DATA_DIR="$REPO_ROOT/data/pathorob-real-per-center${LIMIT_PER_CENTER}"
CACHE_DIR="$REPO_ROOT/cache"
RUN_DIR="$REPO_ROOT/runs/pathorob-real-per-center${LIMIT_PER_CENTER}"
CACHE="$CACHE_DIR/pathorob-real-per-center${LIMIT_PER_CENTER}.jsonl"

"$PYTHON" -m merit_feddg.cli prepare-pathorob \
  --artifacts "$REPO_ROOT/artifacts" \
  --output "$DATA_DIR" \
  --limit-per-center "$LIMIT_PER_CENTER"

MANIFEST="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["canonical_manifest"])' "$DATA_DIR/prepare-report.json")"
mkdir -p "$CACHE_DIR" "$RUN_DIR"

cache_ready=0
if [[ $FORCE_EXTRACT -ne 1 && -s "$CACHE" ]]; then
  if "$PYTHON" -m merit_feddg.cli verify-extract-cache \
    --cache "$CACHE" \
    --manifest "$MANIFEST" \
    --config "$REPO_ROOT/configs/pathorob_real.yaml" \
    --artifacts "$REPO_ROOT/artifacts"; then
    cache_ready=1
  else
    echo "Cached evidence is stale or incomplete; extracting it again." >&2
  fi
fi

if [[ $cache_ready -ne 1 ]]; then
  "$PYTHON" -m merit_feddg.cli extract \
    --manifest "$MANIFEST" \
    --config "$REPO_ROOT/configs/pathorob_real.yaml" \
    --artifacts "$REPO_ROOT/artifacts" \
    --output "$CACHE"
else
  echo "Reusing provenance-verified real-model cache: $CACHE"
fi

"$PYTHON" -m merit_feddg.cli real-multiclass \
  --input "$CACHE" \
  --model-config "$REPO_ROOT/configs/pathorob_real.yaml" \
  --study-config "$REPO_ROOT/configs/pathorob_study.yaml" \
  --artifacts "$REPO_ROOT/artifacts" \
  --output "$RUN_DIR/result.json"

echo "Completed: $RUN_DIR/result.json"
