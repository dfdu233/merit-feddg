#!/usr/bin/env bash
set -euo pipefail

repo=/home/dbw/merit-feddg-huatuo-spatial
run_root="$repo/runs/mimic-gpu1-v2"
cache="$repo/runs/llava-test-v1/mimic-expert-cache/d480df9b456b4cdc6955bd9f117dfe469da1f8b2bf0b890ed9c35951582c0947"
python=/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python
manifest=runs/llava-test-v1/mimic-manifest.jsonl
mkdir -p "$run_root"
exec 9>"$run_root/queue.lock"
flock -n 9 || { echo 'ANOTHER_MIMIC_V2_QUEUE'; exit 1; }
cd "$repo"
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:scripts
export HF_HOME=/home/dbw/ANCHOR/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
echo "QUEUE_V2_STARTED $(date -u +%FT%TZ)" >&2

verify() {
  "$python" - "$1" "$2" "$3" "$4" <<'PY'
import gzip, json, sys
from pathlib import Path
out, start, end, model = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rows = [json.loads(line) for line in Path('runs/llava-test-v1/mimic-manifest.jsonl').read_text().splitlines()]
expert_cases = 0
for row in rows[start:end]:
    case = out / row['id']
    with gzip.open(case / 'bard.json.gz', 'rt') as handle:
        answer = json.load(handle)
    with gzip.open(case / 'provenance.json.gz', 'rt') as handle:
        provenance = json.load(handle)
    assert provenance['row']['image_sha256'] == row['image_sha256'], row['id']
    assert provenance['methods'] == ['bard'], row['id']
    expert_cases += bool(answer['expert_branches'])
if model == 'llava':
    assert expert_cases > 0, (start, end, 'no expert branch in chunk')
print(f'VERIFIED {model} {start} {end} expert_cases={expert_cases}', flush=True)
PY
}

run_range() {
  local model=$1 start=$2 end=$3 config out log
  if [[ $model == llava ]]; then
    config=runs/llava-test-v1/mimic-config-v2.yaml
  else
    config=runs/huatuo-test-v1/mimic-config.yaml
  fi
  out="runs/$model-test-v1/mimic-gpu1-bard-v2-$start-$end"
  log="$run_root/$model-$start-$end.log"
  echo "START $model $start $end $(date -u +%FT%TZ)" >&2
  for attempt in 1 2 3; do
    if "$python" -u scripts/run_adaptive_bard_canary.py \
       --cache "$cache" --manifest "$manifest" --methods bard --skip-stress \
       --cached-receiver --compress-artifacts --reference-native-evidence \
       --max-forwards 200000000 --config "$config" --output "$out" \
       --start-index "$start" --end-index "$end" --shard-index 0 --shard-count 1 \
       >>"$log" 2>&1; then
      verify "$out" "$start" "$end" "$model"
      echo "COMPLETE $model $start $end $(date -u +%FT%TZ)" >&2
      return 0
    fi
    echo "RETRY $model $start $end $attempt $(date -u +%FT%TZ)" >&2
    sleep 30
  done
  echo "FAILED $model $start $end $(date -u +%FT%TZ)" >&2
  return 1
}

# The original Huatuo 0:64 child was already running when v1's wrapper was stopped.
while kill -0 1489492 2>/dev/null; do
  state=$(ps -o stat= -p 1489492 2>/dev/null || true)
  [[ $state != Z* ]] || break
  sleep 15
done
verify runs/huatuo-test-v1/mimic-gpu1-bard-0-64 0 64 huatuo
echo "INHERITED_HUATUO_COMPLETE $(date -u +%FT%TZ)" >&2

# The first LLaVA v1 batch had no expert branches; v2 uses a bounded 256-token
# report reserve. Validate a real case before committing to the full queue.
run_range llava 0 1
run_range llava 1 64

starts=(64 320 832 1344 1856 2368 2880 3392 3904 4416 4928)
ends=(320 832 1344 1856 2368 2880 3392 3904 4416 4928 5159)
for idx in "${!starts[@]}"; do
  start=${starts[$idx]}
  end=${ends[$idx]}
  run_range huatuo "$start" "$end"
  run_range llava "$start" "$end"
done
echo "QUEUE_V2_COMPLETE $(date -u +%FT%TZ)" >&2
