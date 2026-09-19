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


def test_channel_preserves_candidate_content():
    import json
    for channel in ('finite_choice', 'free_text'):
        p = module.comparison_prompt('图像?', '原答案', '候选', channel)
        assert json.loads(p.split('\n', 1)[1]) == {'question': '图像?', 'A': '原答案', 'B': '候选'}
    with pytest.raises(ValueError):
        module.comparison_prompt('q', 'a', 'b', 'guess')


def test_reasoned_critic_preserves_candidates_and_explicit_verdict():
    prompt = module.comparison_prompt('图像?', '原答案', '候选', 'critic_reasoned')
    assert 'Question: [图像?]' in prompt
    assert 'The first response: [原答案]' in prompt
    assert 'The second response: [候选]' in prompt
    assert 'Neither response is a reference answer' in prompt
    assert all(label in prompt for label in ('[[A]]', '[[B]]', '[[C]]'))
