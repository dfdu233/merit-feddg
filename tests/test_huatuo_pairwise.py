import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('pairwise', Path(__file__).resolve().parents[1]/'scripts/run_huatuo_pairwise.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_order_mapping():
    assert module.selection('B', 'A') == ('compact', 'order_consistent_compact')
    assert module.selection('A', 'B') == ('generalist', 'order_consistent_generalist')
    for pair in [('A','A'),('B','B'),('C','C'),('C','A'),('B','C')]:
        assert module.selection(*pair) == ('generalist', 'tie_or_order_inconsistent')


@pytest.mark.parametrize('text', ['A', '[[A]] or [[B]]', '[[C]] then [[C]]', '', '[[D]]'])
def test_malformed_not_keep(text):
    with pytest.raises(ValueError):
        module.verdict(text)


def test_valid_verdict():
    assert module.verdict('The second answer fits the image.\n[[B]]\n') == 'B'
    assert module.verdict('[[C]] for a tie. The two answers agree.') == 'C'
