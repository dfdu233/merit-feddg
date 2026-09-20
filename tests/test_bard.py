from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from merit_feddg.bard import (
    BARDConfig,
    bard_step,
    decode_bard,
    geometric_median,
    single_fault_probe,
)
from merit_feddg.bard_protocol import (
    acquire_expert_groups,
    build_isolated_sessions,
    run_bard_method,
)
from merit_feddg.capabilities import EvidenceItem
from merit_feddg.capability_runtime import NativeState, ValueGenerationConfig
from merit_feddg.matched_evaluation import experiment_arms


def item(expert, evidence):
    return EvidenceItem(
        evidence_id=evidence,
        expert_id=expert,
        capability="classification",
        scope="x",
        payload={"value": evidence},
    )


def test_geometric_median_rejects_one_extreme_outlier():
    points = np.array([[1.0, 0.0], [1.1, 0.1], [0.9, -0.1], [-100.0, 100.0]])
    median = geometric_median(points)
    assert np.linalg.norm(median - np.array([1.0, 0.0])) < 0.25


def test_additive_logit_offset_does_not_change_decision():
    config = BARDConfig(fault_budget=0)
    base = np.array([3.0, 2.0, 0.0])
    experts = [np.array([2.0, 4.0, 0.0]), np.array([2.1, 4.1, 0.1])]
    token1, _ = bard_step(base, experts, config, bounded_commit=False)
    token2, _ = bard_step(
        base,
        [experts[0] + 500.0, experts[1] - 900.0],
        config,
        bounded_commit=False,
    )
    assert token1 == token2 == 1


def test_f1_requires_four_branches():
    config = BARDConfig(fault_budget=1)
    token, audit = bard_step(
        np.array([4.0, 3.0]),
        [np.array([1.0, 8.0])] * 3,
        config,
    )
    assert token == 0
    assert audit["reason"] == "insufficient_byzantine_redundancy"


def test_f1_commits_with_three_of_four_supporters_despite_one_outlier():
    config = BARDConfig(fault_budget=1)
    experts = [
        np.array([1.0, 8.0]),
        np.array([2.0, 7.0]),
        np.array([2.0, 6.0]),
        np.array([20.0, -20.0]),
    ]
    token, audit = bard_step(np.array([4.0, 3.0]), experts, config)
    assert token == 1
    assert audit["committed"]
    assert audit["supporters"] == 3
    assert audit["conservative_margin"] > 0


def test_f1_rejects_two_of_four_support():
    config = BARDConfig(fault_budget=1)
    experts = [
        np.array([1.0, 8.0]),
        np.array([1.0, 8.0]),
        np.array([8.0, 1.0]),
        np.array([8.0, 1.0]),
    ]
    token, audit = bard_step(np.array([4.0, 3.0]), experts, config)
    assert token == 0
    assert not audit["committed"]


def test_single_fault_probe_costs_no_model_calls():
    config = BARDConfig(fault_budget=1)
    experts = [
        np.array([1.0, 8.0, 0.0]),
        np.array([2.0, 7.0, 0.0]),
        np.array([2.0, 6.0, 0.0]),
        np.array([3.0, 5.0, 0.0]),
    ]
    audit = single_fault_probe(np.array([4.0, 3.0, 0.0]), experts, config)
    assert len(audit["single_faults"]) == 4
    assert audit["extra_model_forwards"] == 0


def test_mask_mismatch_fails_closed():
    config = BARDConfig(fault_budget=0)
    with pytest.raises(ValueError, match="vocabulary support"):
        bard_step(
            np.array([1.0, 0.0, -np.inf]),
            [np.array([1.0, -np.inf, 0.0])],
            config,
        )


class ScoreSession:
    eos_ids = frozenset({2})

    def __init__(self, rows):
        self.rows = rows
        self.prefixes = []

    def next_scores(self, prefix):
        self.prefixes.append(tuple(prefix))
        return np.asarray(self.rows[len(prefix)], dtype=float)

    def decode(self, tokens):
        return " ".join(map(str, tokens))


def test_decode_uses_exact_same_prefix_for_all_branches():
    config = BARDConfig(fault_budget=1)
    base = ScoreSession([[5, 4, -5], [5, 4, 9]])
    experts = {
        f"e{i}": ScoreSession([[1, 8, -5], [1, 8, 9]])
        for i in range(4)
    }
    result = decode_bard(
        base, experts, max_tokens=2, config=config, fault_probe=True
    )
    assert result["token_ids"] == [1, 2]
    assert base.prefixes == [(), (1,)]
    assert all(session.prefixes == [(), (1,)] for session in experts.values())
    assert result["trace"][0]["fault_probe"]["extra_model_forwards"] == 0


def test_structural_fallback_does_not_evaluate_expert_branches():
    config = BARDConfig(fault_budget=1)
    base = ScoreSession([[5, 4, 9]])
    experts = {
        f"e{i}": ScoreSession([[1, 8, 9]])
        for i in range(3)
    }
    result = decode_bard(base, experts, max_tokens=1, config=config)
    assert result["structural_fallback"]
    assert result["token_ids"] == [2]
    assert result["trace"][0]["reason"] == "insufficient_byzantine_redundancy"
    assert all(not session.prefixes for session in experts.values())


class FakeRuntime:
    def __init__(self):
        self.config = SimpleNamespace(max_expert_calls=4)
        self.calls = []

    def descriptors(self, _state):
        return [
            {"expert": "a", "capability": "c1"},
            {"expert": "a", "capability": "c2"},
            {"expert": "b", "capability": "c"},
            {"expert": "c", "capability": "c"},
            {"expert": "d", "capability": "c"},
        ]

    def execute(self, state, descriptor):
        self.calls.append((descriptor["expert"], descriptor["capability"], state.items))
        evidence = item(
            descriptor["expert"],
            descriptor["expert"] + "-" + descriptor["capability"],
        )
        return NativeState(items=(evidence,)), {
            "expert": descriptor["expert"],
            "executed": True,
        }


def test_acquisition_uses_clean_states_and_prioritizes_distinct_experts():
    runtime = FakeRuntime()
    result = acquire_expert_groups(runtime)
    assert list(result["groups"]) == ["a", "b", "c", "d"]
    assert len(result["groups"]["a"]) == 1
    assert result["native_requests"] == 4
    assert [expert for expert, _, _ in runtime.calls] == ["a", "b", "c", "d"]
    assert all(items == () for _, _, items in runtime.calls)


class FakeNative:
    def __init__(self, probe, image, prompt, question, config):
        self.probe = probe
        self.image = image
        self.prompt = prompt
        self.question = question
        self.config = config
        self.last_transport = {}

    def context(self, state):
        self.last_transport = {
            "presented": [
                {"expert_id": value.expert_id, "evidence_id": value.evidence_id}
                for value in state.items
            ]
        }
        suffix = ",".join(value.expert_id for value in state.items)
        return self.image, self.prompt + ((" " + suffix) if suffix else "")


class FakeProbe:
    def new_answer_session(self, _image, prompt):
        expert = prompt != "Q"
        rows = [[1, 5, -5], [0, 0, 10]] if expert else [[5, 4, -5], [0, 0, 10]]
        return ScoreSession(rows)


def make_native():
    config = SimpleNamespace(
        evidence_style="semantic",
        semantic_spatial=False,
        visual_views=0,
        token_budgeted_evidence=True,
        max_new_tokens=2,
    )
    return FakeNative(FakeProbe(), "img", "Q", "q", config)


def test_isolated_transport_has_one_expert_per_receiver_prompt():
    native = make_native()
    groups = {name: (item(name, name),) for name in "abcd"}
    _, sessions, audits = build_isolated_sessions(native, groups)
    assert set(sessions) == set("abcd")
    assert all(value["presented_items"] == 1 for value in audits.values())


def test_run_bard_method_records_fault_model_and_transport():
    native = make_native()
    groups = {name: (item(name, name),) for name in "abcd"}
    acquisition = {
        "groups": groups,
        "events": [],
        "native_requests": 4,
        "seconds": 0.1,
        "selection": "frozen",
    }
    result = run_bard_method(
        native, acquisition, {"fault_budget": 1}, "bard"
    )
    assert result["token_ids"][0] == 1
    assert result["byzantine_model"]["required_nodes_for_commit"] == 4
    assert result["presented_evidence_count"] == 4
    assert result["trace"][0]["reason"] == "bounded_fault_consensus"


def test_bard_matched_arms_fix_transport_and_disable_old_gates():
    decoder = ValueGenerationConfig(
        max_new_tokens=4,
        block_tokens=4,
        max_expert_calls=4,
        max_decisions=4,
        max_evidence_chars=1000,
        visual_views=0,
        evidence_style="semantic",
        token_budgeted_evidence=True,
        evidence_order="acquisition",
    )
    arms = experiment_arms(decoder, "bard")
    assert list(arms) == [
        "generalist",
        "joint_all",
        "isolated_mean",
        "isolated_geomedian",
        "bard",
    ]
    assert all(value.compact_native for value in arms.values())
    assert all(value.compact_columns for value in arms.values())
    assert all(value.vector_gate == "off" for value in arms.values())
    assert all(not value.semantic_spatial for value in arms.values())


def test_bard_matched_arm_rejects_spatial_or_entry_transport():
    decoder = ValueGenerationConfig(
        max_new_tokens=4,
        block_tokens=4,
        max_expert_calls=4,
        max_decisions=4,
        max_evidence_chars=1000,
        visual_views=0,
        evidence_style="semantic",
        token_budgeted_evidence=True,
        evidence_order="acquisition",
    )
    with pytest.raises(ValueError, match="semantic-only"):
        experiment_arms(replace(decoder, semantic_spatial=True), "bard")
    with pytest.raises(ValueError, match="semantic-only"):
        experiment_arms(replace(decoder, native_entry_transport=True), "bard")
