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


def test_single_order_is_deterministic_and_maps_actual_order():
    for case_id in ['case1', 'case2', 'case3']:
        order = module.comparison_orders(case_id, 'single')
        assert order == module.comparison_orders(case_id, 'single')
        assert len(order) == 1
        assert module.single_selection('A', order[0])[0] == order[0][0]
        assert module.single_selection('B', order[0])[0] == order[0][1]
        assert module.single_selection('C', order[0])[0] == 'generalist'
    with pytest.raises(ValueError):
        module.single_selection('bad', ('generalist', 'compact'))


@pytest.mark.parametrize('text', ['A', '[[A]] or [[B]]', '[[C]] then [[C]]', '', '[[D]]'])
def test_malformed_not_keep(text):
    with pytest.raises(ValueError):
        module.verdict(text)


def test_valid_verdict():
    assert module.verdict('The second answer fits the image.\n[[B]]\n') == 'B'
    assert module.verdict('[[C]] for a tie. The two answers agree.') == 'C'
    assert module.verdict('Hence, the better answer is [A] for the first response.') == 'A'
    assert module.verdict('Verdict: [B].') == 'B'
    assert module.verdict('- [A] if first is better\n- [B] if second is better\n'
                          '- [C] if tied\nThe verdict is [[C]].') == 'C'
    assert module.verdict('The correct answer is [B].\n\nVerdict: [[B]]') == 'B'
    with pytest.raises(ValueError):
        module.verdict('The correct answer is [A].\n\nVerdict: [[B]]')


@pytest.mark.parametrize('text', ['[A] then [[B]]', '[A] [A]', '[[[A]]]', '[A]]', '[[A]', 'A', '[D]'])
def test_ambiguous_brackets_rejected(text):
    with pytest.raises(ValueError):
        module.verdict(text)


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
