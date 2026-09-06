#!/usr/bin/env bash
# Reuse the user's REMOTE LLaVA-Med environment; never run research bootstrap here.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${MERIT_LLAVA_PYTHON:-/opt/miniconda3/envs/huatuo/bin/python}"
ARGS=()
while (($#)); do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || { echo "--python requires a path" >&2; exit 2; }
      PYTHON_BIN="$2"; shift 2 ;;
    --help|-h)
      echo 'Usage: bash run_llava_med.sh [--python /path/to/python] [options]'
      echo 'Defaults reuse the existing remote huatuo environment and LLaVA-Med checkpoint.'
      echo 'Options: --check-only --prepare-only --skip-download --install-deps'
      echo '         --model-path PATH --llava-source PATH --vision-tower-path PATH'
      echo '         --mirror global|cn --dataset both|pathvqa|vqarad'
      echo '         --source-per-group N --target-limit N --artifacts PATH --output PATH'
      echo '         --generalist llava|openmed --config PATH --retrieval-answers on|off'
      echo '         --chexagent auto|on|off'
      echo 'No torch/transformers upgrades and no default 3B/7B model downloads.'
      exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN; supply --python" >&2; exit 2; }
cd -- "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m merit_feddg.llava_run "${ARGS[@]}"
