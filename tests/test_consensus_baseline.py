import numpy as np
import pytest

from merit_feddg.consensus_baseline import (
    decode_strict_base_relative_consensus,
    strict_base_relative_consensus_step,
)


def test_strict_probability_rule_matches_unanimous_up_and_down_formula():
    base = np.array([0.20, 0.20, 0.60])
    refs = [np.array([0.40, 0.10, 0.50]), np.array([0.30, 0.05, 0.65])]
    token, audit = strict_base_relative_consensus_step(np.log(base), [np.log(p) for p in refs])
    # Raw per-token masses: weakest common rise .30, weakest common fall .10,
    # and the base .60 where the references disagree.
    assert token == 2
    assert audit["quorum"] == 2
    assert audit["unanimous_up_tokens"] == 1
    assert audit["unanimous_down_tokens"] == 1
    assert audit["raw_mass_before_renormalization"] == pytest.approx(1.0)


def test_disagreement_returns_base_and_logit_offsets_do_not_matter():
    base = np.array([0.10, 0.30, 0.60])
    refs = [np.array([0.05, 0.92, 0.03]), np.array([0.15, 0.05, 0.80])]
    token, audit = strict_base_relative_consensus_step(np.log(base), [np.log(p) for p in refs])
    shifted, _ = strict_base_relative_consensus_step(
        np.log(base) + 200,
        [np.log(refs[0]) - 900, np.log(refs[1]) + 500],
    )
    assert token == shifted == 2
    assert audit["unanimous_up_tokens"] == audit["unanimous_down_tokens"] == 0


def test_rejects_mismatched_vocabulary_support():
    with pytest.raises(ValueError, match="exact vocabulary support"):
        strict_base_relative_consensus_step(
            np.array([1.0, 0.0, -np.inf]), [np.array([0.0, 1.0, 2.0])]
        )


def test_free_decode_uses_its_own_committed_prefix():
    class Session:
        eos_ids = {2}

        def __init__(self, first):
            self.first = first
            self.seen = []

        def next_scores(self, prefix):
            self.seen.append(tuple(prefix))
            return np.log(self.first if not prefix else [0.01, 0.01, 0.98])

        def decode(self, tokens):
            return "answer" if tokens else ""

    base = Session([0.60, 0.39, 0.01])
    a = Session([0.10, 0.89, 0.01])
    b = Session([0.10, 0.89, 0.01])
    output = decode_strict_base_relative_consensus(base, {"a": a, "b": b}, max_tokens=4)
    assert output["token_ids"] == [1, 2]
    assert base.seen == a.seen == b.seen == [(), (1,)]
