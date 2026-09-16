import numpy as np
import pytest

from merit_feddg.authority_projection import binary_finding_group, project_binary_authority


def test_underflow_and_invariants():
    result = project_binary_authority([0, -1000], ['positive', 'negative'], .5)
    np.testing.assert_allclose(result['projected_probabilities'], [.5, .5])
    assert np.isfinite(result['kl_q_to_p'])
    result = project_binary_authority([0, 1, 2, 3], ['positive', 'positive', 'negative', 'unknown'], .7)
    p, q = result['base_probabilities'], result['projected_probabilities']
    assert q[3] == p[3]
    assert q[0] / q[1] == pytest.approx(p[0] / p[1])
    assert q[:2].sum() / q[:3].sum() == pytest.approx(.7)


@pytest.mark.parametrize('text,expected', [
    ('No pneumothorax. Pleural effusion is present.', 'unknown'),
    ('Pneumothorax is absent. Pleural effusion is present.', 'positive'),
    ('Pleural effusion is possible.', 'unknown'),
    ('History of pleural effusion.', 'unknown'),
    ('Pleural effusion is not present.', 'negative'),
    ('Yes', 'positive'), ('No', 'negative'),
    ('No evidence of pneumothorax; effusion is present.', 'unknown'),
])
def test_conservative_mapping(text, expected):
    assert binary_finding_group(text, ['effusion']) == expected


def test_missing_side():
    result = project_binary_authority([0, 1], ['positive', 'unknown'], .2)
    assert result['status'] == 'infeasible_missing_semantic_side'
    np.testing.assert_array_equal(result['base_probabilities'], result['projected_probabilities'])
