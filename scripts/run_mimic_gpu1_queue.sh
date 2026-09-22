#!/usr/bin/env bash
set -euo pipefail

repo=/home/dbw/merit-feddg-huatuo-spatial
run_root="$repo/runs/mimic-gpu1-v1"
cache="$repo/runs/llava-test-v1/mimic-expert-cache/d480df9b456b4cdc6955bd9f117dfe469da1f8b2bf0b890ed9c35951582c0947"
python=/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python
mkdir -p "$run_root"
exec 9>"$run_root/queue.lock"
flock -n 9 || { echo 'ANOTHER_MIMIC_QUEUE_OWNS_GPU1'; exit 1; }
cd "$repo"

echo "QUEUE_STARTED $(date -u +%FT%TZ)" >&2
while kill -0 1269809 2>/dev/null; do
  echo "WAIT_OMNI_GPU1 $(date -u +%FT%TZ)" >&2
  sleep 30
done

"$python" - <<'PY'
import json
from pathlib import Path
from merit_feddg.io import load_experiment_yaml

root = Path('runs/llava-test-v1')
cache = root / 'mimic-expert-cache/d480df9b456b4cdc6955bd9f117dfe469da1f8b2bf0b890ed9c35951582c0947'
protocol = json.loads((cache / 'protocol.json').read_text())
rows = [json.loads(line) for line in (root / 'mimic-manifest.jsonl').read_text().splitlines()]
assert len(rows) == len({row['id'] for row in rows}) == 5159
assert all(Path(row['image']).is_file() for row in rows)
for name in ('llava', 'huatuo'):
    config = load_experiment_yaml(f'runs/{name}-test-v1/mimic-config.yaml')
    assert config['experts'] == protocol['config']['experts']
print('PREFLIGHT_OK 5159 BOTH_MODELS', flush=True)
PY

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=.:scripts
export HF_HOME=/home/dbw/ANCHOR/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4

# Alternate models after bounded chunks so both have real outputs early.
starts=(0 64 320 832 1344 1856 2368 2880 3392 3904 4416 4928)
ends=(64 320 832 1344 1856 2368 2880 3392 3904 4416 4928 5159)
for idx in "${!starts[@]}"; do
  start=${starts[$idx]}
  end=${ends[$idx]}
  for model in llava huatuo; do
    out="runs/$model-test-v1/mimic-gpu1-bard-$start-$end"
    log="$run_root/$model-$start-$end.log"
    echo "START $model $start $end $(date -u +%FT%TZ)" >&2
    for attempt in 1 2 3; do
      if "$python" -u scripts/run_adaptive_bard_canary.py \
          --cache "$cache" --manifest runs/llava-test-v1/mimic-manifest.jsonl \
          --methods bard --skip-stress --cached-receiver \
          --compress-artifacts --reference-native-evidence \
          --max-forwards 200000000 --config "runs/$model-test-v1/mimic-config.yaml" \
          --output "$out" --start-index "$start" --end-index "$end" \
          --shard-index 0 --shard-count 1 >>"$log" 2>&1; then
        break
      fi
      echo "RETRY $model $start $end $attempt $(date -u +%FT%TZ)" >&2
      sleep 30
    done
    "$python" - "$out" "$start" "$end" <<'PY'
import gzip
import json
import sys
from pathlib import Path

out, start, end = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(line) for line in Path('runs/llava-test-v1/mimic-manifest.jsonl').read_text().splitlines()]
for row in rows[start:end]:
    case = out / row['id']
    for name in ('bard', 'provenance'):
        path = case / f'{name}.json.gz'
        assert path.is_file(), path
        with gzip.open(path, 'rt') as handle:
            artifact = json.load(handle)
        if name == 'provenance':
            assert artifact['row']['image_sha256'] == row['image_sha256']
            assert artifact['methods'] == ['bard']
print(f'VERIFIED {start} {end} {end-start}', flush=True)
PY
    echo "COMPLETE $model $start $end $(date -u +%FT%TZ)" >&2
  done
done
echo "QUEUE_COMPLETE $(date -u +%FT%TZ)" >&2
