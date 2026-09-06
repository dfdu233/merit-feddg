#!/usr/bin/env bash
# Reuse installed real models. No bootstrap, weights deletion or core-stack upgrade.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT_DIR/run_llava_med.sh" --study value --output runs/native-v09-value "$@"
