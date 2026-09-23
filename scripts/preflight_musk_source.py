"""Run the pinned MUSK checkpoint on one label-blind pathology source case."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from merit_feddg.experts.musk import MuskConceptExpert


ROOT = Path('/home/dbw/merit-tx-v3')
SOURCE = ROOT / 'upstream/MUSK'
WEIGHTS = ROOT / 'artifacts/models/xiangjx--musk/model.safetensors'
MANIFEST = Path('/home/dbw/merit-tx-v2/runs/tx-v2/qualification-routed.jsonl')
OUTPUT = ROOT / 'reports/v3/musk-real-preflight.json'
REVISION = '714b666969c1911e5efe70d991140a21030f4ef3'
PROPOSITIONS = (
    "Histopathology image showing Hashimoto's thyroiditis.",
    "Histopathology image showing no Hashimoto's thyroiditis.",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    row = next(
        row for row in rows
        if row['modality'] == 'pathology'
        and row['task'] == 'open_vqa'
        and row.get('split', row.get('role')) == 'source'
        and "hashimoto" in row['question'].casefold()
        and Path(row['image']).is_file()
    )
    assert row['id'] == 'pathvqa-train-00005-001881'
    if sha256(Path(row['image'])) != row['image_sha256']:
        raise ValueError('source image hash does not match the frozen manifest')
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f'MUSK weights are missing: {WEIGHTS}')
    start = time.monotonic()
    model = MuskConceptExpert(
        str(WEIGHTS), source_path=str(SOURCE), source_revision=REVISION,
        device='cuda:0', dtype='float16',
    )
    try:
        scores = model.score_claims(row['image'], row['question'], '', list(PROPOSITIONS))
    finally:
        model.close()
    if scores.shape != (2,) or not np.isfinite(scores).all():
        raise ValueError('MUSK returned invalid source proposition scores')
    result = {
        'source_id': row['id'],
        'source_split': 'source',
        'modality': row['modality'],
        'image_sha256': row['image_sha256'],
        'checkpoint_sha256': sha256(WEIGHTS),
        'source_revision': REVISION,
        'propositions': list(PROPOSITIONS),
        'image_minus_null_scores': scores.tolist(),
        'seconds': time.monotonic() - start,
        'physical_gpu': 1,
        'references_read': False,
        'purpose': 'real MUSK adapter execution and finite score check; not diagnostic accuracy',
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
