#!/usr/bin/env bash
set -euo pipefail

repo=/home/dbw/merit-tx-v3
python=/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python
run_dir="$repo/runs/v3-musk"
report_dir="$repo/reports/v3-musk"
mkdir -p "$run_dir" "$report_dir"
exec 9>"$run_dir/qualification.lock"
flock -n 9 || { echo 'MUSK_QUALIFICATION_ALREADY_RUNNING' >&2; exit 1; }

cd "$repo"
# This container exposes host GPU1 as its only device, numbered cuda:0.
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/dbw/.runtime/musk-extra:$repo"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4

checkpoint=artifacts/models/xiangjx--musk/model.safetensors
test -s "$checkpoint" || { echo "MUSK_CHECKPOINT_MISSING $checkpoint" >&2; exit 1; }
test "$(git -C upstream/MUSK rev-parse HEAD)" = 714b666969c1911e5efe70d991140a21030f4ef3

echo "PREFLIGHT_START $(date -u +%FT%TZ)" >&2
"$python" -u scripts/preflight_musk_source.py
echo "PREFLIGHT_COMPLETE $(date -u +%FT%TZ)" >&2

for model in llava huatuo; do
  source_dir="$run_dir/split/qualification"
  echo "QUALIFICATION_START $model $(date -u +%FT%TZ)" >&2
  "$python" -u scripts/build_merit_tx_source_observations.py \
    --manifest "$source_dir/manifest.jsonl" \
    --baseline "$source_dir/$model-baseline.json" \
    --candidate "adaptive-bard=$source_dir/$model-candidate.json" \
    --references "$source_dir/shared-references.json" \
    --proposer-map "$source_dir/$model-proposers.json" \
    --config runs/v3/config.yaml --artifacts artifacts \
    --knockoff-controls 4 --metric token_f1 \
    --output "$run_dir/$model-Q.jsonl"
  "$python" -u scripts/fit_expert_qualification.py \
    --input "$run_dir/$model-Q.jsonl" \
    --output "$report_dir/$model-Q-qualification.json"
  echo "QUALIFICATION_COMPLETE $model $(date -u +%FT%TZ)" >&2
done

"$python" - "$report_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for model in ('llava', 'huatuo'):
    card = json.loads((root / f'{model}-Q-qualification.json').read_text())
    print(model, 'cards', len(card.get('cards', [])), flush=True)
    for item in card.get('cards', []):
        if item.get('expert_id') == 'musk_claim_verifier':
            print('MUSK_CARD', model, json.dumps(item, ensure_ascii=False), flush=True)
PY
echo "MUSK_SOURCE_QUALIFICATION_COMPLETE $(date -u +%FT%TZ)" >&2
