import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('run_class_text_guidance')


def test_cad_equation_and_zero(monkeypatch):
    m = module(monkeypatch)
    a, b = np.array([1., 3.]), np.array([4., 2.])
    assert np.array_equal(m.cad_scores(a, b), 1.5*b-.5*a)
    assert np.array_equal(m.cad_scores(a, b, 0), b)
    with pytest.raises(ValueError):
        m.cad_scores(a, b, -1)
    with pytest.raises(ValueError):
        m.cad_scores(a, [float('nan'), 0])


def test_cad_same_prefix_and_zero_skips_without(monkeypatch):
    m = module(monkeypatch)
    histories = [[], []]

    def branch(index):
        def scores(prefix):
            histories[index].append(prefix)
            return np.array([0., 2., 1.]) if not prefix else np.array([0., 1., 3.])
        return SimpleNamespace(next_scores=scores, decode=lambda ids: str(ids),
            propose=lambda *a, **k: [SimpleNamespace(text='zero', tokens=(1, 2))])
    result = m.cad_decode(branch(0), branch(1), 4, {2})
    assert histories[0] == histories[1] == [(), (1,)]
    assert result['token_ids'] == [1, 2]
    assert m.cad_decode(None, branch(1), 4, {2}, 0)['token_ids'] == [1, 2]
