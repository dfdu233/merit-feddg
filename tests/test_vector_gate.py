"""Training-free decisions and actual vector/runtime integration, no medical labels."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from merit_feddg.block_decode import Block
from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
from merit_feddg.matched_evaluation import experiment_arms
from merit_feddg.vector_gate import (
    VectorGateConfig,
    assess_visual_contrast,
    mean_color_control,
    token_log_probability,
)


class Candidate:
    def __init__(self, tokens):
        self.tokens, self.calls = tokens, []

    def propose(self, prefix, count, length):
        self.calls.append((prefix, count, length))
        return [Block(self.tokens[:length], "", -0.1)]


class Verifier:
    def __init__(self, probabilities):
        with np.errstate(divide="ignore"):
            self.scores, self.calls = np.log(probabilities), []

    def next_scores(self, prefix):
        self.calls.append(prefix)
        return self.scores


def evaluate(original=(0.1, 0.2, 0.7), control=(0.1, 0.7, 0.2), baseline=(1,), candidate=(2,)):
    sessions = (Candidate(baseline), Candidate(candidate), Verifier(original), Verifier(control))
    audit = assess_visual_contrast(*sessions, (7, 8), config=VectorGateConfig(3), remaining_tokens=2)
    return audit, sessions


def test_frozen_visual_contrast_can_accept_a_correction_without_baseline_agreement():
    audit, sessions = evaluate()
    assert audit["accepted"] and audit["gain"] > 0
    assert not audit["gate_trained"] and not audit["verifier_uses_expert_evidence"]
    assert sessions[0].calls == sessions[1].calls == [((7, 8), 1, 2)]
    # Two candidates share the initial prefix; reuse both image distributions.
    assert audit["verifier_queries"] == 2
    assert audit["estimated_replayed_forward_steps"] == 6
    assert sessions[2].calls == sessions[3].calls == [(7, 8)]


def test_rejects_language_prior_without_extra_image_support():
    audit, _ = evaluate(original=(0.1, 0.2, 0.7), control=(0.1, 0.2, 0.7))
    assert not audit["accepted"] and audit["gain"] == pytest.approx(0)


def test_lower_visual_support_is_rejected():
    audit, _ = evaluate(original=(0.1, 0.7, 0.2), control=(0.1, 0.2, 0.7))
    assert not audit["accepted"] and audit["gain"] < 0


def test_identical_continuations_skip_verifier_and_abstain():
    audit, sessions = evaluate(candidate=(1,))
    assert not audit["accepted"] and audit["reason"] == "no_change_within_probe_horizon"
    assert not sessions[2].calls and audit["verifier_queries"] == 0


def test_nonfinite_likelihood_fails_closed_with_cost_audit():
    audit, _ = evaluate(original=(0.1, 0.9, 0.0))
    assert not audit["accepted"] and audit["reason"].startswith("gate_invalid:")
    assert audit["candidate_generated_tokens"] == 2 and audit["seconds"] >= 0


def test_control_preserves_original_and_dimensions():
    original = Image.new("RGB", (6, 4), (10, 30, 50))
    original.putpixel((0, 0), (250, 210, 170))
    before = original.tobytes()
    neutral = mean_color_control(original)
    assert original.tobytes() == before and neutral.size == original.size
    assert len(np.unique(np.asarray(neutral).reshape(-1, 3), axis=0)) == 1 and neutral.tobytes() != before


@pytest.mark.parametrize("scores", [[float("nan"), 0], [float("inf"), 0], [-np.inf, -np.inf]])
def test_invalid_distribution(scores):
    with pytest.raises(ValueError):
        token_log_probability(scores, 0)


def test_log_probability_is_shift_invariant_and_handles_large_logits():
    assert token_log_probability([10000, 10001], 1) == pytest.approx(
        token_log_probability([0, 1], 1))


def test_matched_arms_only_gate_differs_and_no_question_type_is_needed():
    decoder = ValueGenerationConfig(evidence_style="tensor", visual_views=0, vector_gate="visual_contrast")
    arms = experiment_arms(decoder, "vector")
    assert set(arms) == {"generalist", "tensor_all", "tensor_gate"}
    assert replace(arms["tensor_gate"], vector_gate="off") == arms["tensor_all"] == arms["generalist"]
    assert arms["tensor_gate"].max_new_tokens == arms["tensor_all"].max_new_tokens
    with pytest.raises(ValueError):
        ValueGenerationConfig(vector_gate="visual_contrast")


def test_native_session_probes_do_not_mutate_live_decoder_or_commit_tokens():
    calls = []

    class Probe:
        tensor_bridge = object()

        def new_tensor_answer_session(self, image, prompt, items):
            calls.append(("tensor", image, prompt, items))
            return Candidate((2,) if items else (1,))

        def new_answer_session(self, image, prompt):
            calls.append(("plain", image, prompt))
            return Verifier((0.1, 0.2, 0.7) if image is original else (0.1, 0.7, 0.2))

    original = Image.new("RGB", (4, 4), "gray")
    config = ValueGenerationConfig(evidence_style="tensor", visual_views=0, vector_gate="visual_contrast")
    session = NativeSession(Probe(), original, "Any question", "Any question", config)
    state = NativeState(prefix=(4,))
    marker = object()
    session._session, session._key = marker, "live"
    from test_tensor_evidence import classification

    audit = session.assess_vector_evidence(state, (classification(),))
    assert audit["accepted"] and state.prefix == (4,) and state.items == ()
    assert session._session is marker and session._key == "live"
    assert all(call[2] == "Any question" for call in calls)
    assert [v[0] for v in calls] == ["tensor", "tensor", "plain", "plain"]


def test_unified_embedding_and_fusion_preserve_checkpoint_behavior():
    from test_tensor_evidence import bridge_fixture

    torch, bridge, packet, visual = bridge_fixture()
    with torch.no_grad():
        bridge.gate.fill_(0.5)
    evidence = bridge.encode_evidence(visual, packet)
    assert evidence.shape == (1, len(packet), bridge.config["width"])
    torch.testing.assert_close(bridge.fuse_evidence(visual, evidence), bridge(visual, packet))
    assert bridge.fuse_evidence(visual, evidence[:, :0]) is visual


def test_admission_detects_eviction_of_existing_vectors():
    from test_tensor_evidence import classification, contract, segmentation

    from merit_feddg.tensor_evidence import compile_tensor_evidence, preserves_tensor_records

    before = compile_tensor_evidence([segmentation()], (4, 2), contract(1))
    after = compile_tensor_evidence([classification(), segmentation()], (4, 2), contract(1))
    assert not preserves_tensor_records(before, after)
    assert preserves_tensor_records(before, before)
