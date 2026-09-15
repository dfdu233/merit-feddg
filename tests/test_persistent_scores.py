from types import SimpleNamespace
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from merit_feddg.persistent_scores import PersistentScores


class Model:
    generation_config = SimpleNamespace()
    config = SimpleNamespace(is_encoder_decoder=False)

    def _update_model_kwargs_for_generation(self, output, kwargs, **unused):
        return dict(past_key_values=output.cache)

    def prepare_inputs_for_generation(self, ids, **kwargs):
        return dict(ids=ids, **kwargs)

    def __call__(self, ids, past_key_values, **unused):
        value = past_key_values + int(ids[0, -1])
        return SimpleNamespace(cache=value, logits=torch.tensor([[[float(value), 0.]]]))


def session():
    model = Model()

    def scores(prefix):
        assert not prefix
        model._update_model_kwargs_for_generation(SimpleNamespace(cache=10), {})
        return np.array([10., 0.])

    return SimpleNamespace(tensor_packet=None,
        generalist=SimpleNamespace(model=model, torch=torch),
        inputs={'inputs': torch.tensor([[0]])}, next_scores=scores)


def test_cache_extension_reset_and_branch_isolation():
    a, b = PersistentScores(session()), PersistentScores(session())
    assert a.next_scores(())[0] == 10
    assert a.next_scores((2, 3))[0] == 15
    assert a.next_scores((2, 3))[0] == 15
    assert a.incremental_calls == 2
    assert b.next_scores((4,))[0] == 14
    assert a.next_scores(())[0] == 10
    assert a.next_scores((7,))[0] == 17
    assert a.next_scores((8,))[0] == 18
    assert a.prefills == 3
    assert '_update_model_kwargs_for_generation' not in a.generalist.model.__dict__


def test_exception_restores_model_method():
    s = session()
    def fail(prefix):
        raise RuntimeError('prefill failure')
    s.next_scores = fail
    with pytest.raises(RuntimeError, match='prefill failure'):
        PersistentScores(s).next_scores(())
    assert '_update_model_kwargs_for_generation' not in s.generalist.model.__dict__


def test_reject_unsupported_evidence_and_processors():
    s = session()
    s.tensor_packet = []
    with pytest.raises(ValueError, match='semantic'):
        PersistentScores(s)
    s = session()
    s.generalist.model.generation_config = SimpleNamespace(renormalize_logits=True)
    with pytest.raises(ValueError, match='renormalize'):
        PersistentScores(s)


@pytest.mark.parametrize('audit', [
    {'status': 'parity_failed', 'cases': [1]*4},
    {'status': 'exact_parity_passed', 'cases': [1]*4,
     'checks': [{'exact': False, 'max_abs_delta': .001}]},
])
def test_continuation_rejects_failed_audit(monkeypatch, tmp_path, audit):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    wrapper = importlib.import_module('run_class_text_cached')
    path = tmp_path/'audit.json'
    path.write_text(json.dumps(audit))
    monkeypatch.setattr(sys, 'argv', ['cached', '--cache-audit', str(path)])
    with pytest.raises(RuntimeError):
        wrapper.main()
