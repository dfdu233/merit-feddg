"""Protocol and native-expansion checks without external weights or references."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from merit_feddg.huatuo_pathway import prepare_native, prepare_pair

_spec = importlib.util.spec_from_file_location('pathway_runner',
    Path(__file__).resolve().parents[1] / 'scripts/run_huatuo_pathway.py')
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def settings():
    return dict(do_sample=False, num_beams=1, max_new_tokens=64,
                min_new_tokens=0, repetition_penalty=1.)


def test_generation_settings_not_guessed():
    assert runner.validate_generation_kwargs(settings(), 64) == settings()
    with pytest.raises(ValueError):
        runner.validate_generation_kwargs({'max_new_tokens': 64}, 64)


@pytest.mark.parametrize('update', [dict(do_sample=True), dict(num_beams=2),
    dict(max_new_tokens=128), dict(temperature=.2), dict(min_new_tokens=-1),
    dict(repetition_penalty=float('nan')), dict(use_cache=False)])
def test_unsupported_or_changed_generation_rejected(update):
    with pytest.raises(ValueError):
        runner.validate_generation_kwargs(settings() | update, 64)


def test_atomic_write_never_replaces(tmp_path):
    file = tmp_path / 'out.json'
    runner.write_new(file, {'n': 1})
    with pytest.raises(FileExistsError):
        runner.write_new(file, {'n': 2})
    assert runner.read(file) == {'n': 1}
    assert list(tmp_path.glob('.pathway-*')) == []


def make_source(tmp_path):
    image = tmp_path / 'image.bin'
    image.write_bytes(b'not a clinical image: protocol test only')
    row = dict(id='train-1', image=str(image), image_sha256=runner.digest(image),
               question='question', benchmark_prompt='frozen prompt')
    source = dict(identity='source', rows=[row], selection='existing TRAIN schedule',
                  test_image_exclusion_sha256='prior-audit', generation_config={})
    runner.write_new(tmp_path / 'protocol.json', source)
    runner.write_new(tmp_path / 'complete.json', dict(identity='source', n=1))
    runner.write_new(tmp_path / 'cases/train-1.json', dict(id='train-1', identity='source', complete=True,
        arms={k: dict(text='output', token_ids=[1]) for k in ('generalist', 'compact')}))
    return source


def test_source_audit_and_changed_pixels(tmp_path):
    source = make_source(tmp_path)
    assert runner.frozen_source(tmp_path)[0] == source
    Path(source['rows'][0]['image']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='image changed'):
        runner.frozen_source(tmp_path)


@pytest.mark.parametrize('mutation', ['labels', 'duplicate', 'test', 'unsafe', 'incomplete'])
def test_source_rejects_invalid_inputs(tmp_path, mutation):
    source = make_source(tmp_path)
    if mutation == 'labels':
        source['rows'][0]['reference'] = 'forbidden'
    elif mutation == 'duplicate':
        source['rows'].append(dict(source['rows'][0]))
    elif mutation == 'test':
        source['selection'] = 'official TEST'
    elif mutation == 'unsafe':
        source['rows'][0]['id'] = '../escape'
    else:
        (tmp_path / 'complete.json').write_text(json.dumps(dict(identity='wrong', n=1)))
    (tmp_path / 'protocol.json').write_text(json.dumps(source))
    with pytest.raises(ValueError):
        runner.frozen_source(tmp_path)


class FakeAdapter:
    def __init__(self, truncate=False):
        self.truncate = truncate
        self.model = self

    def _inputs(self, image, prompt):
        ids = torch.tensor([[7, -200, 3, 4] + ([9] if prompt == 'evidence' else [])])
        return ids, torch.ones(1, 3, 4, 4)

    def prepare_inputs_labels_for_multimodal_new(self, ids, positions, mask, past, labels, pixels):
        expanded = torch.cat((labels[:, :1], torch.full((1, 3), -100), labels[:, 2:]), 1)
        embeds = torch.ones(1, expanded.shape[1], 8)
        if self.truncate:
            expanded, embeds = expanded[:, :-1], embeds[:, :-1]
        return None, None, None, None, embeds, expanded


@pytest.fixture
def llava_constants(monkeypatch):
    monkeypatch.setitem(sys.modules, 'llava.constants', SimpleNamespace(IMAGE_TOKEN_INDEX=-200))


def test_span_from_native_expansion_not_fixed_patch_count(llava_constants):
    p, _ = prepare_native(FakeAdapter(), object(), 'base')
    assert (p.visual_span.start, p.visual_span.stop, p.length) == (1, 4, 6)
    a, b = prepare_pair(FakeAdapter(), object(), 'base', 'evidence')
    assert a.length + 1 == b.length


def test_native_truncation_is_detected(llava_constants):
    with pytest.raises(RuntimeError, match='truncated'):
        prepare_native(FakeAdapter(truncate=True), object(), 'base')


@pytest.mark.parametrize('config', [SimpleNamespace(no_repeat_ngram_size=2),
    SimpleNamespace(forced_eos_token_id=3), SimpleNamespace(stop_strings=['stop'])])
def test_inherited_processors_are_not_silently_ignored(config):
    with pytest.raises(ValueError, match='inherited'):
        runner.check_native_generation_defaults(config)


def test_all_bypass_canary_cannot_unlock_full_run(tmp_path):
    row = {'id': 'case'}
    runner.write_new(tmp_path / 'cases/case.json', {'parity': {'attention_path_exercised': False}})
    with pytest.raises(RuntimeError, match='no expert-exposed'):
        runner.assert_exposed_canary(tmp_path, [row])
