"""Download only the approved, revision-pinned MUSK checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import get_token, hf_hub_download


REPO = 'xiangjx/musk'
REVISION = 'de1ffed28608c197d2903f6fa42b491a3fbf0fb8'
ROOT = Path('/home/dbw/merit-tx-v3')
TARGET = ROOT / 'artifacts/models/xiangjx--musk'
AUDIT = ROOT / 'reports/v3-musk/checkpoint.json'


def main() -> None:
    if not get_token():
        raise RuntimeError('Hugging Face login is required in this environment')
    TARGET.mkdir(parents=True, exist_ok=True)
    path = Path(hf_hub_download(
        repo_id=REPO,
        filename='model.safetensors',
        revision=REVISION,
        local_dir=str(TARGET),
        token=True,
    ))
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    payload = {
        'repository': REPO,
        'revision': REVISION,
        'checkpoint': str(path),
        'size_bytes': path.stat().st_size,
        'sha256': digest.hexdigest(),
        'token_recorded': False,
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
