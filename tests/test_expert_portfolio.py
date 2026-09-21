import json

import pytest

from merit_feddg.expert_policy import SourceQualificationCard
from merit_feddg.expert_portfolio import (
    fit_expert_portfolios,
    load_expert_portfolio,
    portfolio_experts_for,
    require_disjoint_source_groups,
)
from scripts.fit_expert_qualification import fit as fit_qualification
from scripts.split_merit_tx_source import assign_units, build_units


def _spec(expert, group):
    return {
        "id": expert,
        "fault_group": group,
        "evidence_role": "direct_visual_verifier",
        "commit_authority": "source_qualified",
        "modalities": ["pathology"],
        "tasks": ["open_vqa"],
        "capabilities": ["classification"],
        "scope": "classification",
        "literature": ["peer-reviewed"],
    }


def _card(expert, group, *, veto=True):
    del group
    return SourceQualificationCard(
        expert_id=expert,
        capability="classification",
        scope="classification",
        modality="pathology",
        task="open_vqa",
        claim_type="*",
        n=80,
        domains=("site-a", "site-b"),
        utility_lcb=0.2,
        harm_ucb=0.1,
        specificity_lcb=0.7,
        action_rate_lcb=0.5,
        support_n=40,
        support_domains=("site-a", "site-b"),
        support_consequential_n=30,
        support_help_n=27,
        support_harm_n=3,
        support_neutral_n=10,
        support_precision_lcb=0.7,
        veto_precision_lcb=0.7 if veto else 0.0,
        veto_n=20 if veto else 0,
        veto_domains=("site-a", "site-b") if veto else (),
        veto_consequential_n=20 if veto else 0,
        veto_harm_n=18 if veto else 0,
        veto_help_n=2 if veto else 0,
        veto_neutral_n=0,
    )


def _row(index, expert, effect, outcome, *, split="source"):
    return {
        "expert_id": expert,
        "capability": "classification",
        "scope": "classification",
        "modality": "pathology",
        "task": "open_vqa",
        "claim_type": "*",
        "domain": "site-a" if index % 2 == 0 else "site-b",
        "group_id": f"patient-{index}",
        "transaction_id": f"tx-{index}",
        "outcome_delta": outcome,
        "real_effect": effect,
        "knockoff_effect": 0.0,
        "differential_effect": effect,
        "proposer_expert_ids": [],
        "split": split,
        "target_test_selection": False,
    }


def _policy():
    return {
        "qualification_min_domains": 2,
        "qualification_max_harm_ucb": 0.25,
        "qualification_min_support_precision_lcb": 0.5,
        "qualification_min_veto_precision_lcb": 0.5,
        "qualification_min_consequential": 4,
        "min_support_groups": 1,
        "require_independent_validator": True,
        "reject_on_qualified_contradiction": True,
        "portfolio_max_experts": 3,
        "portfolio_min_actions": 4,
        "portfolio_min_domains": 2,
        "portfolio_max_harm_ucb": 0.25,
        "portfolio_min_precision_lcb": 0.5,
    }


def test_v3_directional_precision_excludes_neutral_source_transactions():
    rows = []
    for index in range(20):
        rows.append(
            {
                "expert_id": "visual",
                "capability": "classification",
                "scope": "classification",
                "modality": "pathology",
                "task": "open_vqa",
                "claim_type": "*",
                "domain": "site-a" if index < 10 else "site-b",
                "group_id": f"patient-{index}",
                "outcome_delta": 0.5 if index < 4 else 0.0,
                "real_effect": 0.8,
                "knockoff_effect": 0.0,
            }
        )
    card = fit_qualification(rows)["cards"][0]
    assert card["support_n"] == 20
    assert card["support_consequential_n"] == 4
    assert card["support_help_n"] == 4
    assert card["support_harm_n"] == 0
    assert card["support_neutral_n"] == 16
    # v2 divided directional successes by all 20 rows. v3 estimates precision
    # only on the four source transactions whose task score actually changed.
    assert card["support_precision_lcb"] > 0.5
    assert card["specificity_lcb"] > 0.5
    assert card["action_rate_lcb"] > 0


def test_portfolio_selection_detects_negative_expert_interaction():
    specs = {
        "expert_a": _spec("expert_a", "group-a"),
        "expert_b": _spec("expert_b", "group-b"),
    }
    cards = {
        card.key: card
        for card in (
            _card("expert_a", "group-a", veto=False),
            _card("expert_b", "group-b", veto=True),
        )
    }
    rows = []
    # A safely supports the first half. B's separately-qualified negative signal
    # vetoes those useful A actions when both are present.
    for index in range(20):
        rows.extend(
            (
                _row(index, "expert_a", 0.8, 0.5),
                _row(index, "expert_b", -0.8, 0.5),
            )
        )
    # B also has its own useful support region, so it is not globally "bad".
    for index in range(20, 40):
        rows.extend(
            (
                _row(index, "expert_a", 0.0, 0.4),
                _row(index, "expert_b", 0.8, 0.4),
            )
        )

    payload = fit_expert_portfolios(
        rows,
        specs=specs,
        qualification_cards=cards,
        policy=_policy(),
    )
    bucket = payload["buckets"][0]
    assert bucket["selected"]["expert_ids"] == ["expert_a"]
    pair = next(
        row
        for row in bucket["pair_interactions"]
        if row["experts"] == ["expert_a", "expert_b"]
    )
    assert pair["interaction_utility"] < 0

    candidates = {
        tuple(row["expert_ids"]): row for row in bucket["candidate_portfolios"]
    }
    assert candidates[("expert_a",)]["utility_mean"] > candidates[
        ("expert_a", "expert_b")
    ]["utility_mean"]
    add_back = next(
        row
        for row in bucket["excluded_add_one"]
        if row["expert_id"] == "expert_b"
    )
    assert add_back["adding_expert_reduces_mean_utility"]
    assert add_back["utility_mean_delta_if_added"] < 0
    add_b = next(
        row for row in bucket["add_one"] if row["expert_id"] == "expert_b"
    )
    assert add_b["adding_expert_reduces_mean_utility"]
    assert add_b["utility_mean_delta_when_added"] < 0


def test_portfolio_can_choose_empty_instead_of_forcing_coverage():
    specs = {"expert_a": _spec("expert_a", "group-a")}
    qcard = _card("expert_a", "group-a", veto=False)
    rows = [
        _row(index, "expert_a", 0.8, -0.5)
        for index in range(20)
    ]
    payload = fit_expert_portfolios(
        rows,
        specs=specs,
        qualification_cards={qcard.key: qcard},
        policy=_policy(),
    )
    bucket = payload["buckets"][0]
    assert bucket["selection_reason"] == "no-safe-nonempty-source-portfolio"
    assert bucket["selected"]["expert_ids"] == []


def test_portfolio_policy_fails_closed_on_unseen_cell(tmp_path):
    payload = {
        "schema": "merit-expert-portfolio-v1",
        "source_only": True,
        "buckets": [
            {
                "modality": "pathology",
                "task": "open_vqa",
                "claim_type": "*",
                "selected": {"expert_ids": ["conch"]},
            }
        ],
    }
    path = tmp_path / "portfolio.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    policy = load_expert_portfolio(path)
    assert portfolio_experts_for(
        policy,
        modality="pathology",
        task="open_vqa",
        claim_type="diagnosis",
    ) == ("conch",)
    assert portfolio_experts_for(
        policy,
        modality="ct",
        task="open_vqa",
        claim_type="diagnosis",
    ) == ()


def test_portfolio_fit_rejects_target_rows():
    specs = {"expert_a": _spec("expert_a", "group-a")}
    card = _card("expert_a", "group-a")
    with pytest.raises(ValueError, match="target/test"):
        fit_expert_portfolios(
            [_row(0, "expert_a", 0.8, 0.5, split="test")],
            specs=specs,
            qualification_cards={card.key: card},
            policy=_policy(),
        )


def test_monet_claim_phrasing_is_candidate_specific_without_loading_weights():
    from merit_feddg.experts.monet import MonetConceptExpert

    assert MonetConceptExpert._claim_phrase("melanoma") == (
        "Dermatology image showing melanoma."
    )
    assert MonetConceptExpert._claim_phrase(
        "The skin image shows an ulcerated lesion."
    ) == "The skin image shows an ulcerated lesion."


def test_musk_claim_phrasing_is_candidate_specific_without_loading_weights():
    from merit_feddg.experts.musk import MuskConceptExpert

    assert MuskConceptExpert._phrase("adenocarcinoma") == (
        "Histopathology image showing adenocarcinoma."
    )
    assert MuskConceptExpert._phrase(
        "The histopathology image shows adenocarcinoma."
    ) == "The histopathology image shows adenocarcinoma."

def test_source_policy_stages_require_group_disjointness():
    result = require_disjoint_source_groups(
        qualification={"patient-a", "patient-b"},
        portfolio_selection={"patient-c", "patient-d"},
        fresh_canary={"patient-e"},
    )
    assert result["qualification"] == ("patient-a", "patient-b")
    with pytest.raises(ValueError, match="overlap"):
        require_disjoint_source_groups(
            qualification={"patient-a", "patient-b"},
            portfolio_selection={"patient-b", "patient-c"},
        )


def test_v3_qualification_records_source_group_identity():
    rows = []
    for index in range(8):
        rows.append(
            {
                "expert_id": "visual",
                "capability": "classification",
                "scope": "classification",
                "modality": "pathology",
                "task": "open_vqa",
                "claim_type": "*",
                "domain": "site-a" if index % 2 == 0 else "site-b",
                "group_id": f"patient-{index}",
                "outcome_delta": 0.5 if index < 4 else 0.0,
                "real_effect": 0.8,
                "knockoff_effect": 0.0,
            }
        )
    payload = fit_qualification(rows)
    assert payload["source_group_ids"] == [f"patient-{index}" for index in range(8)]
    assert len(payload["source_groups_sha256"]) == 64



def test_three_way_source_units_merge_duplicate_images_even_with_different_groups(tmp_path):
    shared = tmp_path / "shared.png"
    other = tmp_path / "other.png"
    shared.write_bytes(b"same-image-bytes")
    other.write_bytes(b"different-image-bytes")
    rows = [
        {
            "id": "a",
            "image": str(shared),
            "domain": "site-a",
            "group_id": "patient-a",
        },
        {
            "id": "b",
            "image": str(shared),
            "domain": "site-b",
            "group_id": "patient-b",
        },
        {
            "id": "c",
            "image": str(other),
            "domain": "site-a",
            "group_id": "patient-c",
        },
    ]
    units = build_units(rows)
    assert len(units) == 2
    merged = next(unit for unit in units if set(unit["sample_ids"]) == {"a", "b"})
    assert merged["group_ids"] == ["patient-a", "patient-b"]
    assignment = assign_units(
        units,
        seed="frozen",
        ratios=(0.5, 0.25, 0.25),
    )
    assert assignment[merged["root"]] in {"qualification", "portfolio", "canary"}
