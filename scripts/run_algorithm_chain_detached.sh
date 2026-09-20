#!/usr/bin/env bash
# Starts only an already frozen, finite research campaign; no TEST launch.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: $0 /absolute/path/to/python /absolute/path/to/frozen-campaign" >&2
  exit 2
fi
python_bin="$1"
out="$2"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -x "$python_bin" && -f "$out/plan.json" && -f "$out/state.json" ]] || {
  echo "Existing Python and frozen plan/state are required" >&2
  exit 2
}
command -v setsid >/dev/null
stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
log="$out/controller-$stamp.log"
# Controller owns a nonblocking flock. Duplicate launches exit without model work.
nohup setsid "$python_bin" "$root/scripts/run_algorithm_chain.py" run --output "$out" \
  >"$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$out/controller-$stamp.pid"
printf 'Submitted PID=%s\nLog=%s\nThis is launch metadata, not evidence of model execution.\n' "$pid" "$log"
