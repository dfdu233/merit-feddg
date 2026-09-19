import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from critic_verdict_completion import continuation_prefix


def test_exact_prefix_except_eos_within_original_budget():
    prefix, budget = continuation_prefix([20,30,99], [99], [4,5], 12)
    assert prefix == [20,30,4,5]
    assert budget == 8


def test_no_truncation_when_budget_exhausted():
    with pytest.raises(ValueError, match='No original budget'):
        continuation_prefix([20,30,99], [99], [4,5], 5)
