from copy import deepcopy

import pytest

from merit_feddg.capabilities import EvidenceItem
from merit_feddg.context_admission import admission_prompt, admitted_view


def packet(index=0):
    return EvidenceItem(str(index), 'expert', 'classification', 'anatomy',
                        {'score': 0.2, 'label': 'organ'}, 'unverified observation')


def test_rejected_context_cannot_leak_and_inputs_unchanged():
    items = tuple(packet(i) for i in range(4))
    original = deepcopy(items)
    assert admitted_view(items, ('A', 'B', 'C', 'D')) == items[:1]
    assert admitted_view(items, ('A', 'B', 'C', 'D'), complement=True) == items[1:]
    assert items == original
    assert admitted_view(items, ('D',) * 4) == ()
    with pytest.raises(ValueError):
        admitted_view(items, ('A',))
    with pytest.raises(ValueError):
        admitted_view(items, ('A', 'B', 'C', 'invalid'))


def test_prompt_has_no_answer_or_question_type_rule():
    spec = {'description': 'Classify anatomical structure only', 'scope': 'anatomy'}
    for question in ('Where is the structure?', 'Which disease?', 'Is it visible?'):
        prompt = admission_prompt(question, packet(), spec, 'scope')
        assert question in prompt
        assert 'Disagreement' in prompt
        assert 'Classify anatomical structure only' in prompt
        assert 'reference_answer' not in prompt
        assert 'generalist_answer' not in prompt
    with pytest.raises(ValueError):
        admission_prompt('Q', packet(), spec, 'new_policy')
