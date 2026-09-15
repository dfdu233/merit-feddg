from types import SimpleNamespace

import numpy as np
import pytest

from merit_feddg.control_evidence import (
    constrained_scores,
    decode_controlled,
    residual_direction,
    spatial_controls,
)


def test_controls_cancel_spurious_and_duplicate_evidence():
    base, evidence = np.array([2., 0., -1.]), np.array([0., 2., -1.])
    assert not residual_direction(base, evidence, [evidence]).any()
    assert not residual_direction(base, base, [evidence]).any()
    assert not residual_direction(base, evidence, []).any()
    np.testing.assert_allclose(residual_direction(base, evidence, [base]),
                               residual_direction(base, evidence, [base, base]))


def test_additive_logits_are_unidentifiable_and_controls_only_shrink():
    rng = np.random.default_rng(41)
    for _ in range(30):
        base, evidence, c1, c2 = rng.normal(size=(4, 31))
        first = residual_direction(base, evidence, [c1])
        both = residual_direction(base, evidence, [c1, c2])
        assert np.all(np.abs(both) <= np.abs(first) + 1e-12)
        assert np.all(first * both >= 0)
        np.testing.assert_allclose(both, residual_direction(base + 10, evidence - 8,
                                                          [c1 + 4, c2 - 3]), atol=1e-12)


@pytest.mark.parametrize("budget", [0., 1e-8, .001, .05, 1.])
def test_kl_budget_and_maximal_feasible_strength(budget):
    base, direction = np.array([4., 1., -2.]), np.array([-8., 5., 2.])
    scores, audit = constrained_scores(base, direction, kl_budget=budget)
    assert audit["kl"] <= budget + 1e-12
    p0 = np.exp(base - base.max()); p0 /= p0.sum()
    p = np.exp(scores - scores.max()); p /= p.sum()
    assert np.sum(p * np.log(p / p0)) <= budget + 1e-12
    if 0 < audit["strength"] < 1:
        bumped = base + (audit["strength"] + 1e-6) * direction
        pb = np.exp(bumped - bumped.max()); pb /= pb.sum()
        assert np.sum(pb * np.log(pb / p0)) > budget
    if budget == 0:
        np.testing.assert_array_equal(scores, base)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.])
def test_bad_options_and_scores_fail(value):
    with pytest.raises(ValueError):
        constrained_scores([1., 0.], [0., 1.], kl_budget=value)
    with pytest.raises(ValueError):
        constrained_scores([1., 0.], [0., 1.], max_strength=value)
    with pytest.raises(ValueError):
        residual_direction([1., 0.], [0., 1.], [[1., 2., 3.]])


class Session:
    def __init__(self, scores):
        self.scores, self.prefixes = np.asarray(scores), []

    def next_scores(self, prefix):
        self.prefixes.append(prefix)
        return self.scores

    def decode(self, prefix):
        return " ".join(map(str, prefix))

    def propose(self, prefix, *, count, length):
        return [SimpleNamespace(text="exact production baseline", tokens=(9,))]


def test_same_prefix_even_when_each_branch_would_choose_different_tokens():
    base, real, c1, c2 = [Session(v) for v in ([.1, 0., -1.], [0., 2., -1.],
                                             [.2, 0., -1.], [0., -.1, 2.])]
    out = decode_controlled(base, real, [c1, c2], max_tokens=4, eos_ids=set(), kl_budget=1.)
    assert out["score_calls"] == 16
    assert base.prefixes == real.prefixes == c1.prefixes == c2.prefixes
    assert base.prefixes == [tuple(out["token_ids"][:i]) for i in range(4)]


def test_unknown_and_zero_fast_paths_preserve_exact_production_tokens():
    for kwargs in ({"controls": []}, {"controls": [Session([0., 1.])], "kl_budget": 0.}):
        base = Session([1., 0.])
        real = Session([0., 1.])
        out = decode_controlled(base, real, max_tokens=8, eos_ids={9}, **kwargs)
        assert out["token_ids"] == [9]
        assert out["text"] == "exact production baseline"
        assert not base.prefixes and not real.prefixes
        assert out["guidance_applied"] is False


def packet(regions):
    return SimpleNamespace(regions=np.asarray(regions, dtype=np.float32).reshape(-1, 64),
                           importance=np.array([.5, .5]), labels=("a", "b"),
                           numeric=np.ones((2, 10)), sources=(("s", "a"), ("s", "b")))


def test_controls_preserve_histograms_overlap_sources_weights_and_padding():
    values = np.zeros((2, 8, 8), dtype=np.float32)
    values[0, 2:4, 1:3] = .5
    values[1, 3:5, 2:5] = 1.
    original = packet(values)
    controls, audit = spatial_controls(original, (80, 40))
    assert len(controls) == 2 and audit["reason"] == "available"
    np.testing.assert_array_equal(original.regions, values.reshape(2, -1))
    for control in controls:
        assert control is not original and control.sources == original.sources
        np.testing.assert_array_equal(control.importance, original.importance)
        np.testing.assert_array_equal(np.sort(control.regions, axis=1), np.sort(original.regions, axis=1))
        np.testing.assert_allclose(control.regions @ control.regions.T, original.regions @ original.regions.T)
        reshaped = control.regions.reshape(2, 8, 8)
        assert not reshaped[:, :2].any() and not reshaped[:, 6:].any()


def test_constant_or_thin_geometry_is_unknown_not_fake_control():
    for obj, size in ((packet(np.ones((2, 8, 8))), (80, 80)),
                      (packet(np.zeros((2, 8, 8))), (80, 5))):
        controls, audit = spatial_controls(obj, size)
        assert not controls and audit["reason"] != "available"


def test_agreeing_false_experts_are_not_a_correctness_certificate():
    # A deliberate counterexample: all observed score evidence favours token 1,
    # while external ground truth could be token 0. The policy MUST NOT claim
    # it can identify this failure from the same observations without labels.
    direction = residual_direction([.01, 0.], [0., 4.], [[2., 0.], [1., 0.]])
    scores, audit = constrained_scores([.01, 0.], direction, kl_budget=.05)
    assert np.argmax(scores) == 1 and audit["kl"] <= .05
