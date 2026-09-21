import json
import pytest

from merit_feddg.expert_policy import (
    SourceQualificationCard,
    select_expert_descriptors,
    transaction_descriptors,
)
from merit_feddg.io import load_experiment_yaml
from merit_feddg.merit_tx import (
    MeritTxConfig,
    TransactionEvidence,
    decide_transaction,
    differential_margin,
)
from merit_feddg.transactional_claims import (
    AtomicClinicalClaim,
    ClaimTransaction,
    TransactionDecision,
    apply_transactions,
    claimize_vqa,
)
from scripts.fit_expert_qualification import fit, read_rows


def _spec(role, capability, *, group, authority="source_qualified", **kwargs):
    return {
        "id": group,
        "fault_group": group,
        "evidence_role": role,
        "commit_authority": authority,
        "modalities": ["pathology"],
        "tasks": ["open_vqa"],
        "capabilities": [capability],
        "scope": kwargs.pop("scope", capability),
        "literature": ["peer-reviewed"],
        **kwargs,
    }


def _descriptor(name, capability, scope=None):
    return {
        "expert": name,
        "capability": capability,
        "scope": scope or capability,
        "description": name,
        "requires_region": False,
        "question_type": "diagnosis",
    }


def _card(expert, capability, scope, *, utility=0.2, harm=0.1, specificity=0.7):
    return SourceQualificationCard(
        expert_id=expert,
        capability=capability,
        scope=scope,
        modality="pathology",
        task="open_vqa",
        claim_type="diagnosis",
        n=80,
        domains=("site-a", "site-b"),
        utility_lcb=utility,
        harm_ucb=harm,
        specificity_lcb=specificity,
    )


def test_expert_pool_prefers_complementary_patient_specific_roles_before_knowledge():
    specs = {
        "visual": _spec("direct_visual_verifier", "classification", group="visual"),
        "spatial": _spec("spatial_localizer", "segmentation", group="spatial"),
        "proposal": _spec(
            "proposal_generator", "generation", group="proposal", authority="never"
        ),
        "knowledge": _spec(
            "knowledge_retriever", "retrieval", group="knowledge", authority="never"
        ),
    }
    descriptors = [
        _descriptor("knowledge", "retrieval"),
        _descriptor("proposal", "generation"),
        _descriptor("spatial", "segmentation"),
        _descriptor("visual", "classification"),
    ]
    selected, audit = select_expert_descriptors(
        descriptors,
        specs,
        modality="pathology",
        task="open_vqa",
        claim_type="diagnosis",
        max_calls=4,
    )
    assert [row["expert"] for row in selected] == [
        "visual",
        "spatial",
        "proposal",
        "knowledge",
    ]
    assert audit[0]["patient_specific"] is True
    assert audit[-1]["commit_authority"] == "never"


def test_source_qualification_blocks_harmful_expert_commit_authority():
    specs = {
        "good": _spec("direct_visual_verifier", "classification", group="good"),
        "harmful": _spec("direct_visual_verifier", "classification", group="harmful"),
    }
    cards = {
        _card("good", "classification", "classification").key:
            _card("good", "classification", "classification"),
        _card(
            "harmful", "classification", "classification", utility=-0.01, harm=0.6
        ).key:
            _card(
                "harmful", "classification", "classification", utility=-0.01, harm=0.6
            ),
    }
    descriptors = [
        _descriptor("good", "classification"),
        _descriptor("harmful", "classification"),
    ]
    _, audit = select_expert_descriptors(
        descriptors,
        specs,
        modality="pathology",
        task="open_vqa",
        claim_type="diagnosis",
        qualification_cards=cards,
        max_calls=2,
    )
    authority = {row["expert"]: row["commit_authorized"] for row in audit}
    assert authority == {"good": True, "harmful": False}


def test_transaction_only_experts_do_not_enter_legacy_routing_but_enter_tx_pool():
    specs = {
        "legacy_off": {
            **_spec("direct_visual_verifier", "classification", group="legacy"),
            "enabled": False,
        },
        "tx": {
            **_spec("direct_visual_verifier", "classification", group="tx"),
            "transaction_only": True,
        },
    }
    row = {
        "question": "What diagnosis is shown?",
        "modality": "pathology",
        "task": "open_vqa",
    }
    descriptors = transaction_descriptors(specs, row)
    assert [value["expert"] for value in descriptors] == ["tx"]


def test_rejected_transaction_returns_exact_incumbent():
    baseline = "No pleural effusion."
    claims = (
        AtomicClinicalClaim(
            "c0",
            "The image does not show pleural effusion.",
            (0, len(baseline)),
        ),
    )
    transaction = ClaimTransaction(
        "t0",
        "REPLACE",
        AtomicClinicalClaim("candidate", "The image shows pleural effusion.", None),
        "Small left pleural effusion.",
        baseline_claim_id="c0",
        proposer_expert_id="proposal",
    )
    result = apply_transactions(
        baseline,
        claims,
        (transaction,),
        (TransactionDecision("t0", False, "insufficient-proof"),),
    )
    assert result["text"] == baseline
    assert result["fallback_exact"] is True


def test_report_transaction_only_changes_committed_span():
    baseline = "Heart size is normal. No pleural effusion."
    start = baseline.index("No pleural")
    claims = (
        AtomicClinicalClaim("heart", "Heart size is normal.", (0, start - 1)),
        AtomicClinicalClaim(
            "effusion",
            "The image does not show pleural effusion.",
            (start, len(baseline)),
        ),
    )
    transaction = ClaimTransaction(
        "t1",
        "REPLACE",
        AtomicClinicalClaim("new-effusion", "The image shows pleural effusion.", None),
        "Small left pleural effusion.",
        baseline_claim_id="effusion",
        proposer_expert_id="proposal",
    )
    decision = TransactionDecision(
        "t1",
        True,
        "proof-carrying-transaction",
        verifier_fault_groups=("visual",),
        patient_specific_support=True,
        source_qualified_support=True,
        differential_effect=0.4,
    )
    result = apply_transactions(baseline, claims, (transaction,), (decision,))
    assert result["text"] == "Heart size is normal. Small left pleural effusion."
    assert result["text"].startswith("Heart size is normal.")


def test_proposer_cannot_self_validate_transaction():
    specs = {
        "proposal": _spec(
            "direct_visual_verifier", "classification", group="same-group"
        ),
    }
    qcard = _card("proposal", "classification", "classification")
    transaction = ClaimTransaction(
        "t",
        "REPLACE",
        AtomicClinicalClaim("candidate", "The image shows adenocarcinoma.", None),
        "adenocarcinoma",
        baseline_claim_id="base",
        proposer_expert_id="proposal",
    )
    evidence = TransactionEvidence(
        expert_id="proposal",
        capability="classification",
        scope="classification",
        fault_group="same-group",
        evidence_role="direct_visual_verifier",
        differential_effect=0.5,
        support_direction=1,
        patient_specific=True,
    )
    decision = decide_transaction(
        transaction,
        (evidence,),
        specs=specs,
        qualification_cards={qcard.key: qcard},
        modality="pathology",
        task="open_vqa",
        claim_type="diagnosis",
    )
    assert not decision.commit
    assert decision.reason == "proposer-has-no-independent-qualified-validator"


def test_independent_source_qualified_visual_verifier_can_commit():
    specs = {
        "proposal": _spec(
            "proposal_generator", "generation", group="proposal", authority="never"
        ),
        "visual": _spec(
            "direct_visual_verifier", "classification", group="visual"
        ),
    }
    qcard = _card("visual", "classification", "classification")
    transaction = ClaimTransaction(
        "t",
        "REPLACE",
        AtomicClinicalClaim("candidate", "The image shows adenocarcinoma.", None),
        "adenocarcinoma",
        baseline_claim_id="base",
        proposer_expert_id="proposal",
    )
    evidence = TransactionEvidence(
        expert_id="visual",
        capability="classification",
        scope="classification",
        fault_group="visual",
        evidence_role="direct_visual_verifier",
        differential_effect=0.3,
        support_direction=1,
        patient_specific=True,
    )
    decision = decide_transaction(
        transaction,
        (evidence,),
        specs=specs,
        qualification_cards={qcard.key: qcard},
        modality="pathology",
        task="open_vqa",
        claim_type="diagnosis",
        config=MeritTxConfig(),
    )
    assert decision.commit
    assert decision.verifier_fault_groups == ("visual",)


def test_vqa_claimization_is_self_contained():
    claim = claimize_vqa("Is there cardiomegaly?", "No")[0]
    assert claim.proposition == "The image does not show cardiomegaly."
    assert claim.source_span == (0, 2)


def test_source_qualification_fitter_rejects_test_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps(
            {
                "expert_id": "e",
                "capability": "classification",
                "scope": "s",
                "modality": "pathology",
                "task": "open_vqa",
                "claim_type": "diagnosis",
                "domain": "d",
                "group_id": "g",
                "outcome_delta": 1,
                "real_effect": 1,
                "knockoff_effect": 0,
                "split": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="target/test"):
        read_rows(path)


def test_source_qualification_fitter_emits_conservative_cards():
    rows = []
    for domain in ("a", "b"):
        for index in range(20):
            rows.append(
                {
                    "expert_id": "e",
                    "capability": "classification",
                    "scope": "s",
                    "modality": "pathology",
                    "task": "open_vqa",
                    "claim_type": "diagnosis",
                    "domain": domain,
                    "group_id": f"{domain}-{index}",
                    "outcome_delta": 0.5,
                    "real_effect": 0.8,
                    "knockoff_effect": 0.1,
                }
            )
    payload = fit(rows)
    card = payload["cards"][0]
    assert payload["source_only"] is True
    assert card["utility_lcb"] > 0
    assert card["harm_ucb"] < 0.2
    assert card["specificity_lcb"] > 0.8


def test_region_prompted_expert_is_not_counted_without_real_region():
    specs = {
        "automatic": _spec(
            "spatial_localizer", "segmentation", group="automatic"
        ),
        "prompted": {
            **_spec("spatial_localizer", "segmentation", group="prompted"),
            "requires_region": True,
        },
    }
    descriptors = [
        _descriptor("automatic", "segmentation"),
        {**_descriptor("prompted", "segmentation"), "requires_region": True},
    ]
    selected, _audit = select_expert_descriptors(
        descriptors,
        specs,
        modality="pathology",
        task="open_vqa",
        max_calls=2,
        region_available=False,
    )
    assert [row["expert"] for row in selected] == ["automatic"]
    selected, _audit = select_expert_descriptors(
        descriptors,
        specs,
        modality="pathology",
        task="open_vqa",
        max_calls=2,
        region_available=True,
    )
    assert {row["expert"] for row in selected} == {"automatic", "prompted"}


def test_differential_margin_removes_shared_context_shift():
    # Both real and wrong-patient evidence add a +2 common shift to candidate
    # and incumbent. Only the patient-specific relative margin remains.
    value = differential_margin(
        incumbent_real=7.0,
        candidate_real=9.0,
        incumbent_knockoff=7.5,
        candidate_knockoff=8.0,
    )
    assert value["real_margin"] == 2.0
    assert value["knockoff_margin"] == 0.5
    assert value["differential_effect"] == 1.5
    assert value["support_direction"] == 1


def test_role_diversity_precedes_second_same_role_verifier():
    specs = {
        "visual_a": _spec("direct_visual_verifier", "classification", group="visual-a"),
        "visual_b": _spec("direct_visual_verifier", "classification", group="visual-b"),
        "spatial": _spec("spatial_localizer", "segmentation", group="spatial"),
    }
    descriptors = [
        _descriptor("visual_a", "classification"),
        _descriptor("visual_b", "classification"),
        _descriptor("spatial", "segmentation"),
    ]
    selected, _audit = select_expert_descriptors(
        descriptors,
        specs,
        modality="pathology",
        task="open_vqa",
        max_calls=2,
    )
    assert [row["expert"] for row in selected] == ["visual_a", "spatial"]


def test_fixed_conch_catalog_is_disabled_only_in_merit_tx():
    legacy = load_experiment_yaml("configs/llava_med_capabilities.yaml")
    tx = load_experiment_yaml("configs/merit_tx.yaml")
    assert legacy["experts"]["conch_tissue"].get("enabled", True) is True
    assert tx["experts"]["conch_tissue"]["enabled"] is False
    assert tx["experts"]["conch_claim_verifier"]["transaction_only"] is True
