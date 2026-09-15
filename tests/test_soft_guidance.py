from types import SimpleNamespace

import numpy as np
import pytest

from merit_feddg.soft_guidance import decode, mix_scores


def test_interpolation_endpoints_and_shift_invariance():
    a, b = np.array([1., 2., 3.]), np.array([4., 3., 1.])
    assert np.array_equal(mix_scores(a,b,0),a)
    assert np.array_equal(mix_scores(a,b,1),b)
    assert np.argmax(mix_scores(a,b,.5)) == np.argmax(mix_scores(a+40,b-2,.5))
    with pytest.raises(ValueError): mix_scores(a,b,2)
    with pytest.raises(ValueError): mix_scores(a,b[:2],.5)


def test_zero_skips_evidence_and_preserves_tokens():
    base = SimpleNamespace(propose=lambda *a, **k: [SimpleNamespace(text='yes',tokens=(1,2))])
    assert decode(base,None,alpha=0,max_tokens=5,eos_ids={2})['token_ids'] == [1,2]


def test_branches_receive_identical_chosen_prefix():
    prefixes = [[],[]]
    def branch(i):
        def scores(prefix):
            prefixes[i].append(prefix)
            return np.array([0.,2.,1.]) if not prefix else np.array([0.,1.,3.])
        return SimpleNamespace(next_scores=scores,decode=lambda ids:str(ids))
    result=decode(branch(0),branch(1),alpha=.5,max_tokens=4,eos_ids={2})
    assert prefixes[0] == prefixes[1] == [(),(1,)]
    assert result['token_ids'] == [1,2]
