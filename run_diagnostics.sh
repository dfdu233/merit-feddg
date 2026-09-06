#!/usr/bin/env bash
# Real source-only canary. Existing weights/environment; no target generation.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT_DIR/run_llava_med.sh" --study diagnose --skip-download \
  --source-per-group 2 --target-limit 1 --output runs/native-v010-diagnostics "$@"
