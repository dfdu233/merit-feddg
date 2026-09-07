"""Numerical/protocol tests, not medical efficacy experiments."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from merit_feddg.block_decode import Block
from merit_feddg.bounded_session import BoundedNativeSession
from merit_feddg.capabilities import EvidenceItem
from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
from merit_feddg.evidence_calibration import calibration_partition, fit_evidence_calibration
from merit_feddg.evidence_decode import GuidanceConfig, bounded_distribution
from merit_feddg.evidence_packets import active_packet


@pytest.mark.parametrize("field,value", [("strength", -1), ("clip", float("nan")),
    ("token_kl", float("inf")), ("case_kl", True), ("evidence_tokens", 0),
    ("evidence_tokens", 1.5), ("control", "unknown")])
def test_invalid_guidance(field, value):
    with pytest.raises(ValueError):
        GuidanceConfig(**{field: value})


@pytest.mark.parametrize("change", [{"strength": 0}, {"clip": 0}, {"token_kl": 0}, {"case_kl": 0}])
def test_zero_budget_is_exact_base(change):
    baseline, _ = bounded_distribution([0, 1, -np.inf], None, GuidanceConfig())
    output, trace = bounded_distribution([0, 1, -np.inf], [4, 0, -np.inf], GuidanceConfig(**change))
    assert np.array_equal(baseline, output)
    assert trace["kl"] == 0


def test_numeric_budget_random_vocabularies_and_conservation():
    rng = np.random.default_rng(14)
    config = GuidanceConfig(strength=4, clip=3, token_kl=0.005, case_kl=0.011)
    spent = 0.0
    for _ in range(20):
        base, evidence = rng.normal(size=(2, 123)) * 5
        output, trace = bounded_distribution(base, evidence, config, spent=spent)
        normalized, _ = bounded_distribution(base, None, config)
        actual = np.sum(np.exp(output) * (output - normalized))
        assert np.isclose(np.exp(output).sum(), 1)
        assert actual <= min(config.token_kl, config.case_kl - spent) + 1e-10
        assert abs(actual - trace["kl"]) < 1e-10
        spent = trace["spent"]
        assert spent <= config.case_kl + 1e-10


def test_same_distribution_and_mask_are_preserved():
    output, trace = bounded_distribution([1, 2, -np.inf], [1001, 1002, -np.inf], GuidanceConfig())
    base, _ = bounded_distribution([1, 2, -np.inf], None, GuidanceConfig())
    assert np.allclose(output, base)
    assert np.isneginf(output[-1]) and trace["kl"] < 1e-10


@pytest.mark.parametrize("base,evidence", [([0, -np.inf], [0, 1]), ([0, 1], [0]),
    ([np.nan, 0], [1, 0]), ([np.inf, 0], [1, 0]), ([-np.inf], [-np.inf]),
    ([[0, 1]], [[0, 1]])])
def test_invalid_vocabulary(base, evidence):
    with pytest.raises(ValueError):
        bounded_distribution(base, evidence, GuidanceConfig())


def item(name="e", expert="small", **extra):
    return EvidenceItem(name, expert, "classification", "appearance",
                        {"catalog": [{"concept": "lung", "similarity": 0.2}]},
                        provenance={"merit_acquired_token": 0, **extra})


def test_packet_dedup_expiry_and_no_cross_expert_agreement():
    config = ValueGenerationConfig(visual_views=0, evidence_style="scoped")
    packet, audit = active_packet([item(), item("copy"), item("other", "different")],
                                 "What organ?", config, prefix_tokens=1, lifetime=2)
    assert len(packet) == 2
    assert audit[1]["reason"] == "duplicate"
    packet, audit = active_packet([item()], "What organ?", config, prefix_tokens=2, lifetime=2)
    assert not packet and audit[0]["reason"] == "expired"


def test_format_control_removes_all_observations_not_negation():
    packet, _ = active_packet([item()], "What organ?", ValueGenerationConfig(visual_views=0),
                              prefix_tokens=0, lifetime=16, control="format_only")
    assert "lung" not in str(packet) and "similarity" not in str(packet)
    assert packet[0]["payload"]["status"] == "unknown"
    with pytest.raises(ValueError):
        active_packet([item(merit_acquired_token=10)], "q", ValueGenerationConfig(),
                      prefix_tokens=0, lifetime=16)


class TinyProbe:
    """Small deterministic fake, never presented as real medical generation."""
    def __init__(self):
        self.calls = []
        self.processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=self.decode))
        self.torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

    def decode(self, tokens, **kwargs):
        return " ".join("evidence" if t == 1 else "base" for t in tokens if t != 2)

    def new_answer_session(self, image, prompt):
        owner = self

        class Session:
            eos_ids = frozenset({2})

            def next_scores(self, prefix):
                owner.calls.append((tuple(prefix), prompt))
                if len(prefix) >= 3:
                    return np.array([-5., -5., 5.])
                if "TOOL OBSERVATIONS" in prompt and "withheld control" not in prompt:
                    return np.array([-1., 5., -3.])
                return np.array([2., 1., -3.])

            def propose(self, prefix, count, length):
                ids = []
                for _ in range(length):
                    next_id = int(np.argmax(self.next_scores(tuple(prefix) + tuple(ids))))
                    ids.append(next_id)
                    if next_id == 2:
                        break
                return [Block(tuple(ids), owner.decode(ids), -0.1, ids[-1] == 2)]
        return Session()


def test_live_protocol_same_prefix_expiry_and_single_image():
    probe = TinyProbe()
    config = ValueGenerationConfig(visual_views=0, max_new_tokens=8)
    guidance = GuidanceConfig(strength=1, clip=10, token_kl=4, case_kl=10, evidence_tokens=2)
    session = BoundedNativeSession(probe, "original", "question", "question", config, guidance)
    block = session.propose(NativeState(items=(item(),)), 5)
    assert block.tokens == (1, 1, 0, 2) and block.finished
    assert probe.calls[0][0] == probe.calls[1][0] == ()
    assert probe.calls[2][0] == probe.calls[3][0] == (1,)
    assert len(session.last_guidance_trace) == 3
    assert session.last_guidance_trace[-1]["event"] == "guidance_none"
    with pytest.raises(ValueError):
        BoundedNativeSession(probe, ["original", "overlay"], "q", "q", config)


def test_no_evidence_or_zero_strength_never_evidence_prefills():
    probe = TinyProbe()
    config = ValueGenerationConfig(visual_views=0)
    session = BoundedNativeSession(probe, "original", "q", "q", config, GuidanceConfig(strength=0))
    with_packet = session.propose(NativeState(items=(item(),)), 8)
    without = session.propose(NativeState(), 8)
    assert with_packet.tokens == without.tokens
    assert all(prompt == "q" for _, prompt in probe.calls)


def test_direct_and_guided_evidence_use_identical_prompt_serialization():
    probe = TinyProbe()
    config = ValueGenerationConfig(visual_views=0, evidence_style="scoped")
    state = NativeState(items=(item(),))
    direct = NativeSession(probe, "original", "q", "q", config)
    image, expected = direct.context(state)
    guided = BoundedNativeSession(probe, image, "q", "q", config)
    guided.propose(state, 1)
    assert probe.calls[0][1] == "q"
    assert probe.calls[1][1] == expected


def records(gain=0.1, kind="hospital"):
    output = []
    for domain in ("one", "two"):
        for index in range(80):
            group = f"{domain}-patient-{index}"
            for strength in (0.25, 0.5):
                output.append({"role": "source", "scope": "s", "sample_id": group,
                               "group_id": group, "domain": domain, "domain_kind": kind,
                               "strength": strength, "gain": gain})
    return output


def test_calibration_independent_groups_tie_smallest_and_proxy_refusal():
    card = fit_evidence_calibration(records())["cards"]["s"]
    assert card["qualified"] and card["strength"] == 0.25
    assert fit_evidence_calibration(records(kind="proxy"))["cards"]["s"]["status"] == "proxy_or_unverified_domains"
    assert not fit_evidence_calibration(records(gain=-0.1))["cards"]["s"]["qualified"]


def test_confirmation_cannot_reselect_strength_to_hide_failure():
    data = records()
    for row in data:
        if calibration_partition(row["group_id"]) == "select":
            row["gain"] = 0.2 if row["strength"] == 0.25 else 0.1
        else:
            row["gain"] = -0.1 if row["strength"] == 0.25 else 0.9
    card = fit_evidence_calibration(data)["cards"]["s"]
    assert not card["qualified"] and card["selected_strength"] == 0.25


def test_calibration_rejects_target_and_incomplete_or_duplicate_cohort():
    data = records()
    with pytest.raises(ValueError, match="matching"):
        fit_evidence_calibration(data[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        fit_evidence_calibration(data + [data[0]])
    with pytest.raises(ValueError, match="only source"):
        fit_evidence_calibration([{**data[0], "role": "target"}])


def test_spent_is_carried_by_state_not_mutated_between_paired_replays():
    probe = TinyProbe()
    session = BoundedNativeSession(probe, "original", "q", "q", ValueGenerationConfig(visual_views=0))
    state = NativeState(items=(item(),))
    first = session.propose(state, 1)
    spent = session.last_guidance_spent
    second = session.propose(state, 1)
    assert first == second and session.last_guidance_spent == spent and state.guidance_spent == 0
    session.propose(replace(state, prefix=first.tokens, guidance_spent=spent), 1)
    assert session.last_guidance_spent >= spent
