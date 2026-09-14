import importlib
import json
from pathlib import Path

import pytest


def scheduler(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('run_soft_guidance_shard')


def test_shards_cover_full_manifests_without_overlap(monkeypatch):
    mod = scheduler(monkeypatch)
    for n in (451, 2094):
        a = {i for i in range(n) if mod.assigned(i, 0)}
        b = {i for i in range(n) if mod.assigned(i, 1)}
        assert not a & b
        assert a | b == set(range(n))


def test_merge_requires_every_case_and_nonempty_arms(monkeypatch, tmp_path):
    mod = scheduler(monkeypatch)
    frozen = {'datasets': {'data': {'rows': [{'id': 'a'}, {'id': 'b'}]}}}
    dest = tmp_path / 'data'
    dest.mkdir()
    assert not mod.merge(tmp_path, frozen)
    assert not (tmp_path / 'complete.json').exists()
    for key in ('a', 'b'):
        record = {'identity': tmp_path.name, 'id': key,
            'arms': {a: {'text': 'yes', 'token_ids': [1]} for a in mod.ARMS}}
        (dest / (key + '.json')).write_text(json.dumps(record))
    assert mod.merge(tmp_path, frozen)
    record['arms']['text_soft']['token_ids'] = []
    (dest / 'b.json').write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match='empty'):
        mod.merge(tmp_path, frozen)
