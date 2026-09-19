"""CPU checks: incomplete downloads cannot be accepted as model identity."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'critic_backend', Path(__file__).resolve().parents[1]/'scripts/llava_critic_backend.py')
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


def test_partial_checkpoint_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, 'CHECKPOINT', tmp_path)
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {str(i): name for i, name in enumerate(backend.WEIGHTS)}}))
    for name in backend.WEIGHTS:
        (tmp_path/name).write_bytes(b'incomplete download')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        backend.identity()


def test_unexpected_shard_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, 'CHECKPOINT', tmp_path)
    (tmp_path/'model.safetensors.index.json').write_text(
        json.dumps({'weight_map': {'weight': 'other.safetensors'}}))
    with pytest.raises(ValueError, match='Unexpected checkpoint shard index'):
        backend.identity()
