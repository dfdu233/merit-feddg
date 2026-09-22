import json

import numpy as np
import pytest

from merit_feddg.block_ablation import build_block_ablation_plan
from merit_feddg.block_decode import Block
from merit_feddg.block_intervention import (
    BlockInterventionConfig,
    ExpertBlockBranch,
    decode_counterfactual_blocks,
    evaluate_expert_block,
    measure_same_prefix_context_drift,
)


class ScoreSession:
    def __init__(self, proposals, sequence_scores, *, token_text=None, logits=None):
        self.proposals = proposals
        self.sequence_scores = {
            tuple(tokens): float(value) for tokens, value in sequence_scores.items()
        }
        self.token_text = token_text or {}
        self.logits = logits
        self.propose_calls = []
        self.score_calls = []
        self.eos_ids = {99}

    def propose(self, prefix, count, length):
        assert count == 1
        self.propose_calls.append((tuple(prefix), length))
        value = self.proposals[min(len(prefix), len(self.proposals) - 1)]
        tokens = tuple(value[:length])
        return [Block(tokens, self.decode(tokens), -0.1, bool(tokens and tokens[-1] == 99))]

    def decode(self, tokens):
        if self.token_text:
            return "".join(self.token_text.get(token, str(token)) for token in tokens)
        return " ".join(str(token) for token in tokens)

    def sequence_mean_logp(self, prefix, tokens):
        self.score_calls.append((tuple(prefix), tuple(tokens)))
        return self.sequence_scores[tuple(tokens)]

    def next_scores(self, prefix):
        if callable(self.logits):
            return np.asarray(self.logits(tuple(prefix)), dtype=float)
        if self.logits is not None:
            return np.asarray(self.logits, dtype=float)
        return np.log(np.asarray([0.1, 0.8, 0.1], dtype=float))


def pair(real_margin, control_margins):
    base = Block((1,), "base", -0.1)
    expert = Block((2,), "expert", -0.1)
    real = ScoreSession([(2,)], {(1,): 0.0, (2,): real_margin})
    controls = tuple(
        ScoreSession([(2,)], {(1,): 0.0, (2,): value})
        for value in control_margins
    )
    branch = ExpertBlockBranch("expert", real, controls)
    return base, expert, branch


def test_conjunction_fixes_positive_gamma_with_negative_real_support():
    base, expert, branch = pair(-0.1, (-2.0, -2.0, -2.0))
    full = evaluate_expert_block(
        prefix=(),
        base_block=base,
        expert_block=expert,
        branch=branch,
        config=BlockInterventionConfig(decision_rule="conjunction"),
    )
    gamma_only = evaluate_expert_block(
        prefix=(),
        base_block=base,
        expert_block=expert,
        branch=branch,
        config=BlockInterventionConfig(decision_rule="gamma_only"),
    )
    assert full["gamma"] == pytest.approx(1.9)
    assert full["real_margin"] == pytest.approx(-0.1)
    assert not full["accepted"]
    assert full["reason"] == "missing_absolute_support"
    assert gamma_only["accepted"]


def test_full_rule_requires_absolute_support_and_counterfactual_specificity():
    base, expert, branch = pair(0.8, (-0.2, 0.0, 0.1))
    audit = evaluate_expert_block(
        prefix=(7,),
        base_block=base,
        expert_block=expert,
        branch=branch,
        config=BlockInterventionConfig(decision_rule="conjunction"),
    )
    assert audit["accepted"]
    assert audit["absolute_support"]
    assert audit["counterfactual_specificity"]
    assert audit["gamma"] == pytest.approx(0.8)


def test_real_only_is_explicit_no_control_ablation():
    base, expert, branch = pair(0.4, ())
    branch = ExpertBlockBranch("expert", branch.real_session, (), "none")
    real_only = evaluate_expert_block(
        prefix=(),
        base_block=base,
        expert_block=expert,
        branch=branch,
        config=BlockInterventionConfig(decision_rule="real_only"),
    )
    full = evaluate_expert_block(
        prefix=(),
        base_block=base,
        expert_block=expert,
        branch=branch,
        config=BlockInterventionConfig(decision_rule="conjunction"),
    )
    assert real_only["accepted"]
    assert not full["accepted"]
    assert full["reason"] == "insufficient_counterfactual_controls"


def test_multiple_experts_selects_largest_specificity_gain():
    base_session = ScoreSession([(1,)], {(1,): 0.0, (2,): 0.0, (3,): 0.0})
    e1 = ScoreSession([(2,)], {(1,): 0.0, (2,): 0.6, (3,): 0.0})
    e2 = ScoreSession([(3,)], {(1,): 0.0, (2,): 0.0, (3,): 1.2})
    c1 = tuple(
        ScoreSession([(2,)], {(1,): 0.0, (2,): 0.1, (3,): 0.0})
        for _ in range(3)
    )
    c2 = tuple(
        ScoreSession([(3,)], {(1,): 0.0, (2,): 0.0, (3,): 0.0})
        for _ in range(3)
    )
    result = decode_counterfactual_blocks(
        base_session,
        {
            "e1": ExpertBlockBranch("e1", e1, c1),
            "e2": ExpertBlockBranch("e2", e2, c2),
        },
        config=BlockInterventionConfig(max_new_tokens=1, block_tokens=1),
    )
    assert result["token_ids"] == [3]
    assert result["trace"][0]["selected_expert"] == "e2"


def _two_step_sessions():
    base = ScoreSession(
        [(1,), (3,)],
        {(1,): 0.0, (2,): 0.0, (3,): 0.0, (4,): 0.0},
    )
    expert = ScoreSession(
        [(2,), (4,)],
        {(1,): 0.0, (2,): 1.0, (3,): 0.0, (4,): 1.0},
    )
    controls = tuple(
        ScoreSession(
            [(2,), (4,)],
            {(1,): 0.0, (2,): 0.0, (3,): 0.0, (4,): 0.0},
        )
        for _ in range(3)
    )
    return base, expert, controls


def test_ephemeral_context_returns_to_evidence_free_generalist_next_block():
    base, expert, controls = _two_step_sessions()
    result = decode_counterfactual_blocks(
        base,
        {"e": ExpertBlockBranch("e", expert, controls)},
        config=BlockInterventionConfig(
            max_new_tokens=2,
            block_tokens=1,
            context_policy="ephemeral",
        ),
    )
    assert result["trace"][0]["expert_block_committed"]
    assert result["trace"][1]["base_candidate"]["token_ids"] == [3]
    assert result["trace"][1]["incumbent_context_expert"] is None
    assert result["trace"][0]["expert_context_discarded_after_commit"]


def test_persistent_context_ablation_keeps_last_expert_as_incumbent():
    base, expert, controls = _two_step_sessions()
    result = decode_counterfactual_blocks(
        base,
        {"e": ExpertBlockBranch("e", expert, controls)},
        config=BlockInterventionConfig(
            max_new_tokens=2,
            block_tokens=1,
            context_policy="persistent",
        ),
    )
    assert result["trace"][0]["expert_block_committed"]
    assert result["trace"][1]["base_candidate"]["token_ids"] == [4]
    assert result["trace"][1]["incumbent_context_expert"] == "e"
    assert not result["trace"][0]["expert_context_discarded_after_commit"]


def test_sentence_mode_commits_exact_token_prefix_without_retokenizing():
    text = {10: "Mild ", 11: "edema.", 12: " More", 13: " text"}
    base = ScoreSession(
        [(1, 99)],
        {(1, 99): 0.0, (10, 11): 0.0},
        token_text={1: "No finding", 99: ""},
    )
    expert = ScoreSession(
        [(10, 11, 12, 13)],
        {(1, 99): 0.0, (10, 11): 0.9},
        token_text=text,
    )
    controls = tuple(
        ScoreSession(
            [(10, 11, 12, 13)],
            {(1, 99): 0.0, (10, 11): 0.0},
            token_text=text,
        )
        for _ in range(3)
    )
    result = decode_counterfactual_blocks(
        base,
        {"e": ExpertBlockBranch("e", expert, controls)},
        config=BlockInterventionConfig(
            max_new_tokens=4,
            block_mode="sentence",
            sentence_max_tokens=4,
        ),
    )
    assert result["trace"][0]["experts"]["e"]["candidate"]["token_ids"] == [10, 11]
    assert result["trace"][0]["selected_block"]["token_ids"] == [10, 11]


def test_context_drift_probe_holds_prefix_fixed_and_measures_js():
    base = ScoreSession(
        [(1,)],
        {(1,): 0.0},
        logits=lambda _prefix: np.log([0.05, 0.9, 0.05]),
    )
    expert = ScoreSession(
        [(1,)],
        {(1,): 0.0},
        logits=lambda _prefix: np.log([0.05, 0.2, 0.75]),
    )
    result = measure_same_prefix_context_drift(
        base,
        expert,
        prefix=(7, 8),
        horizon=3,
    )
    assert len(result["steps"]) == 3
    assert result["mean_js"] > 0
    assert result["greedy_disagreement_rate"] == pytest.approx(1.0)
    assert [row["prefix_length"] for row in result["steps"]] == [2, 3, 4]


def test_sum_vs_mean_sequence_scoring_is_explicit_oe_ablation():
    base = Block((1,), "short", -0.1)
    expert = Block((2, 3), "long", -0.1)
    real = ScoreSession(
        [(2, 3)],
        {(1,): -0.2, (2, 3): -0.15},
    )
    controls = tuple(
        ScoreSession([(2, 3)], {(1,): -0.2, (2, 3): -0.4})
        for _ in range(3)
    )
    branch = ExpertBlockBranch("e", real, controls)
    mean = evaluate_expert_block(
        prefix=(),
        base_block=base,
        expert_block=expert,
        branch=branch,
        config=BlockInterventionConfig(score_reduction="mean"),
    )
    total = evaluate_expert_block(
        prefix=(),
        base_block=base,
        expert_block=expert,
        branch=branch,
        config=BlockInterventionConfig(score_reduction="sum"),
    )
    assert mean["real_margin"] == pytest.approx(0.05)
    assert total["real_margin"] == pytest.approx(-0.1)
    assert mean["accepted"] and not total["accepted"]


def test_control_builder_is_label_blind_group_disjoint_and_deterministic():
    from scripts.build_merit_block_controls import build_controls

    rows = [
        {
            "id": "a",
            "image_sha256": "sha-a",
            "group_id": "p1",
            "modality": "cxr",
            "task": "open_vqa",
            "experts": {"e": [{"evidence_id": "a", "expert_id": "e", "capability": "classification", "scope": "finding"}]},
        },
        {
            "id": "b",
            "image_sha256": "sha-b",
            "group_id": "p2",
            "modality": "cxr",
            "task": "open_vqa",
            "experts": {"e": [{"evidence_id": "b", "expert_id": "e", "capability": "classification", "scope": "finding"}]},
        },
        {
            "id": "c",
            "image_sha256": "sha-c",
            "group_id": "p3",
            "modality": "cxr",
            "task": "open_vqa",
            "experts": {"e": [{"evidence_id": "c", "expert_id": "e", "capability": "classification", "scope": "finding"}]},
        },
        {
            "id": "d",
            "image_sha256": "sha-d",
            "group_id": "p4",
            "modality": "pathology",
            "task": "open_vqa",
            "experts": {"e": [{"evidence_id": "d", "expert_id": "e", "capability": "classification", "scope": "finding"}]},
        },
    ]
    first = build_controls(rows, count=2, seed=7)
    second = build_controls(rows, count=2, seed=7)
    assert first == second
    a = first[0]["experts"]["e"]
    assert set(a["control_provenance"]["matched_ids"]) == {"b", "c"}
    assert "a" not in a["control_provenance"]["matched_ids"]
    assert a["control_provenance"]["labels_used"] is False
    assert a["fixtures"]["shuffled"] in (
        rows[1]["experts"]["e"],
        rows[2]["experts"]["e"],
        rows[3]["experts"]["e"],
    )


def test_control_builder_rejects_reference_fields(tmp_path):
    from scripts.build_merit_block_controls import read_rows

    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "a",
                "image_sha256": "sha-a",
                "group_id": "p1",
                "modality": "cxr",
                "task": "open_vqa",
                "experts": {},
                "answer": "yes",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden"):
        read_rows(path)


def test_ablation_plan_is_one_factor_and_keeps_primary_arm_fixed():
    from merit_feddg.io import load_yaml

    plan = build_block_ablation_plan(load_yaml("configs/merit_block.yaml"))
    assert plan["primary_arm"] == "full"
    assert plan["selection_on_target_forbidden"]
    assert plan["one_factor_at_a_time"]
    full = plan["arms"]["full"]
    assert full["decision_rule"] == "conjunction"
    assert full["context_policy"] == "ephemeral"
    assert full["control_kind"] == "matched_wrong_patient"
    assert "decision__gamma_only" in plan["arms"]
    assert "locality__persistent" in plan["arms"]
    assert "control__random_control" in plan["arms"]
    assert "granularity__sentence" in plan["arms"]


def test_optional_drift_probe_is_diagnostic_only():
    base, expert, controls = _two_step_sessions()
    result = decode_counterfactual_blocks(
        base,
        {"e": ExpertBlockBranch("e", expert, controls)},
        config=BlockInterventionConfig(
            max_new_tokens=1,
            block_tokens=1,
            drift_probe_horizon=2,
        ),
    )
    trace = result["trace"][0]
    assert trace["expert_block_committed"]
    assert trace["same_prefix_context_drift"]["horizon"] == 2
    assert result["method"]["drift_probe_horizon"] == 2


def test_label_free_mechanism_summary_reports_acceptance_drift_and_cost():
    from scripts.summarize_merit_block_runs import summarize

    rows = [
        {
            "id": "x",
            "seconds": 1.25,
            "references_read": False,
            "trace": [
                {
                    "expert_block_committed": True,
                    "selected_expert": "e",
                    "same_prefix_context_drift": {
                        "mean_js": 0.2,
                        "greedy_disagreement_rate": 0.5,
                    },
                    "experts": {
                        "e": {
                            "reason": "absolute_support_and_counterfactual_specificity",
                            "real_margin": 0.8,
                            "gamma": 0.6,
                            "control_count": 3,
                            "next_score_queries": 4,
                            "sequence_score_calls": 0,
                        }
                    },
                }
            ],
        }
    ]
    result = summarize(rows)
    assert result["cases"] == 1
    assert result["accepted_blocks"] == 1
    assert result["block_accept_rate"] == pytest.approx(1.0)
    assert result["mean_same_prefix_js"] == pytest.approx(0.2)
    assert result["mean_greedy_disagreement_rate"] == pytest.approx(0.5)
    assert result["selected_experts"] == {"e": 1}
    assert result["wall_seconds"] == pytest.approx(1.25)
    assert result["references_read"] is False


def test_merit_block_receiver_configs_exclude_old_admission_policies():
    from merit_feddg.io import load_experiment_yaml

    llava = load_experiment_yaml("configs/merit_block_llava.yaml")
    huatuo = load_experiment_yaml("configs/merit_block_huatuo.yaml")
    for config in (llava, huatuo):
        generation = config["capability_value"]["generation"]
        assert config["prompt_contract"] == "anchor-task-v1"
        assert generation["evidence_style"] == "semantic"
        assert generation["token_budgeted_evidence"]
        assert generation["vector_gate"] == "off"
        assert "merit_tx" not in config
        assert "bard" not in config
    assert huatuo["generalist"]["backend"] == "huatuo_vision"


def _packet_item(expert_id):
    return {
        "evidence_id": f"{expert_id}-1",
        "expert_id": expert_id,
        "capability": "classification",
        "scope": "finding",
        "payload": {"finding": "x", "score": 0.5},
    }


def test_expert_count_ablation_requires_explicit_subset_and_honors_ids():
    from scripts.run_merit_block import _branch_packets

    experts = {
        "alpha": {
            "real": [_packet_item("alpha")],
            "matched_controls": [],
            "random_controls": [],
        },
        "beta": {
            "real": [_packet_item("beta")],
            "matched_controls": [],
            "random_controls": [],
        },
    }
    arm = {
        "changed_axis": "expert_count",
        "expert_limit": 1,
        "evidence_fixture": "native_real",
        "control_kind": "none",
        "controls": 0,
    }
    with pytest.raises(ValueError, match="explicit --experts"):
        _branch_packets(experts, arm)
    real, _controls, _kind = _branch_packets(
        experts,
        arm,
        explicit_experts=["beta"],
    )
    assert list(real) == ["beta"]


def test_control_builder_excludes_duplicate_image_sha_even_across_groups():
    from scripts.build_merit_block_controls import build_controls

    rows = [
        {
            "id": "a",
            "image_sha256": "same",
            "group_id": "p1",
            "modality": "cxr",
            "task": "open_vqa",
            "experts": {"e": [_packet_item("e")]},
        },
        {
            "id": "b",
            "image_sha256": "same",
            "group_id": "p2",
            "modality": "cxr",
            "task": "open_vqa",
            "experts": {"e": [_packet_item("e")]},
        },
        {
            "id": "c",
            "image_sha256": "different",
            "group_id": "p3",
            "modality": "cxr",
            "task": "open_vqa",
            "experts": {"e": [_packet_item("e")]},
        },
    ]
    result = build_controls(rows, count=2, seed=3)
    matched = result[0]["experts"]["e"]["control_provenance"]["matched_ids"]
    assert matched == ["c"]
