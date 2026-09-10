from types import SimpleNamespace

import numpy as np
import pytest

from merit_feddg.block_decode import Block
from merit_feddg.vector_gate import VectorGateConfig, assess_visual_contrast


def test_full_budget_detects_late_change_with_four_sequence_forwards():
    class Candidate:
        def __init__(self, end):
            self.tokens = (0,) * 8 + (end,)

        def propose(self, prefix, count, length):
            return [Block(self.tokens[:length], "", 0.)]

    class Verifier:
        def __init__(self, real):
            self.real, self.calls = real, []

        def sequence_mean_logp(self, prefix, tokens):
            self.calls.append((prefix, tokens))
            return -1. if self.real and tokens[-1] == 2 else -2.

        def next_scores(self, prefix):
            pytest.fail("must not replay each token")

    image, control = Verifier(True), Verifier(False)
    audit = assess_visual_contrast(Candidate(1), Candidate(2), image, control, (),
                                   config=VectorGateConfig(64), remaining_tokens=64)
    assert audit["accepted"] and audit["verifier_queries"] == 4
    assert audit["verifier_scoring"] == "raw_teacher_forced_sequence"
    assert len(image.calls) == len(control.calls) == 2
    short = assess_visual_contrast(Candidate(1), Candidate(2), image, control, (),
                                   config=VectorGateConfig(8), remaining_tokens=64)
    assert not short["accepted"] and short["reason"] == "no_change_within_probe_horizon"


def test_teacher_forced_verifier_alignment_after_image_expansion():
    torch = pytest.importorskip("torch")
    from merit_feddg.llava_generalist import LlavaMedAnswerSession

    calls = []

    def forward(**inputs):
        calls.append(inputs)
        # Model expands an image placeholder into four visual tokens.
        n = inputs["input_ids"].shape[1] + 3
        logits = torch.stack([torch.tensor([float(i), -float(i), 0.]) for i in range(n)])
        return SimpleNamespace(logits=logits[None])

    probe = SimpleNamespace(torch=torch, model=forward,
        _inputs=lambda image, prompt: {"inputs": torch.tensor([[0, -200, 0]]),
                                       "attention_mask": torch.ones(1, 3)},
        _validate_context=lambda inputs, n: None)
    session = LlavaMedAnswerSession(probe, None, "question")
    score = session.sequence_mean_logp((0,), (1, 2))
    expected = (torch.log_softmax(torch.tensor([6., -6., 0.]), 0)[1]
                + torch.log_softmax(torch.tensor([7., -7., 0.]), 0)[2]) / 2
    assert score == pytest.approx(float(expected))
    assert len(calls) == 1 and not calls[0]["use_cache"]
    assert calls[0]["input_ids"].tolist() == [[0, -200, 0, 0, 1]]
    assert "labels" not in calls[0]
    assert np.isfinite(score)
