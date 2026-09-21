import pytest

from merit_feddg.expert_policy import SourceQualificationCard, select_expert_descriptors
from merit_feddg.med_defer import NativeEvidence
from merit_feddg.merit_tx import MeritTxConfig, TransactionEvidence, decide_transaction, differential_margin_controls
from merit_feddg.transactional_claims import (
    AtomicClinicalClaim,
    ClaimTransaction,
    candidate_transactions,
)
from merit_feddg.transactional_runtime import (
    counterfactual_addition_proposition,
    transaction_claim_spec,
    verify_transaction,
)
from scripts.build_merit_tx_source_observations import report_outcome
from scripts.fit_expert_qualification import fit


def _spec(role, capability, *, group, modality="cxr", task="report_generation"):
    return {
        "id": group,
        "fault_group": group,
        "evidence_role": role,
        "commit_authority": (
            "never" if role in {"proposal_generator", "knowledge_retriever"}
            else "source_qualified"
        ),
        "modalities": [modality],
        "tasks": [task],
        "capabilities": [capability],
        "scope": capability,
        "literature": ["peer-reviewed"],
    }


def _card(expert, *, modality="cxr", task="report_generation", claim_type="*"):
    return SourceQualificationCard(
        expert_id=expert,
        capability="classification",
        scope="classification",
        modality=modality,
        task=task,
        claim_type=claim_type,
        n=80,
        domains=("site-a", "site-b"),
        utility_lcb=0.2,
        harm_ucb=0.1,
        specificity_lcb=0.7,
    )


def _descriptor(name, capability):
    return {
        "expert": name,
        "capability": capability,
        "scope": capability,
        "description": name,
        "requires_region": capability == "segmentation",
        "question_type": "diagnosis",
    }


def test_differential_direction_is_defined_by_matched_control_effect():
    support = differential_margin_controls(
        incumbent_real=1.0,
        candidate_real=0.9,
        knockoff_pairs=((1.0, 0.1), (1.0, 0.2), (1.0, 0.0)),
    )
    assert support["real_margin"] == pytest.approx(-0.1)
    assert support["differential_effect"] > 0
    assert support["support_direction"] == 1

    contradiction = differential_margin_controls(
        incumbent_real=1.0,
        candidate_real=1.3,
        knockoff_pairs=((1.0, 2.0), (1.0, 2.1), (1.0, 1.9)),
    )
    assert contradiction["real_margin"] > 0
    assert contradiction["differential_effect"] < 0
    assert contradiction["support_direction"] == -1


def test_negative_patient_specific_differential_is_a_qualified_veto():
    specs = {
        "support": _spec("direct_visual_verifier", "classification", group="support"),
        "veto": _spec("direct_visual_verifier", "classification", group="veto"),
    }
    support_card = _card("support")
    veto_card = _card("veto")
    transaction = ClaimTransaction(
        "tx",
        "REPLACE",
        AtomicClinicalClaim("candidate", "The image shows pleural effusion.", None),
        "Small pleural effusion.",
        baseline_claim_id="baseline",
    )
    decision = decide_transaction(
        transaction,
        (
            TransactionEvidence(
                expert_id="support",
                capability="classification",
                scope="classification",
                fault_group="support",
                evidence_role="direct_visual_verifier",
                differential_effect=0.4,
                support_direction=1,
                patient_specific=True,
            ),
            TransactionEvidence(
                expert_id="veto",
                capability="classification",
                scope="classification",
                fault_group="veto",
                evidence_role="direct_visual_verifier",
                differential_effect=-0.3,
                support_direction=-1,
                patient_specific=True,
            ),
        ),
        specs=specs,
        qualification_cards={
            support_card.key: support_card,
            veto_card.key: veto_card,
        },
        modality="cxr",
        task="report_generation",
        claim_type="*",
        config=MeritTxConfig(reject_on_qualified_contradiction=True),
    )
    assert not decision.commit
    assert decision.reason == "qualified-patient-specific-contradiction"
    assert set(decision.verifier_fault_groups) == {"support", "veto"}


def test_report_add_has_explicit_presence_counterfactual():
    candidate = AtomicClinicalClaim(
        "candidate",
        "The image shows pleural effusion at left lung.",
        None,
        grounding={
            "schema": "radgraph-xl",
            "observation": "pleural effusion",
            "tags": ["definitely present"],
            "located_at": ["left lung"],
            "suggestive_of": [],
        },
    )
    assert counterfactual_addition_proposition(candidate) == (
        "The image does not show pleural effusion at left lung."
    )
    transaction = ClaimTransaction(
        "add",
        "ADD",
        candidate,
        "Small left pleural effusion.",
    )
    claim = transaction_claim_spec(
        {
            "question": "Generate a radiology report.",
            "modality": "cxr",
            "task": "report_generation",
        },
        transaction,
        (),
    )
    assert claim.propositions[0].proposition == (
        "The image does not show pleural effusion at left lung."
    )
    assert claim.propositions[1].proposition == candidate.proposition
    assert claim.metadata["explicit_add_counterfactual"] is True


def test_uncertain_report_add_fails_closed_without_inventing_a_null():
    candidate = AtomicClinicalClaim(
        "candidate",
        "The image may show edema.",
        None,
        grounding={
            "schema": "radgraph-xl",
            "observation": "edema",
            "tags": ["uncertain"],
            "located_at": [],
            "suggestive_of": [],
        },
    )
    with pytest.raises(ValueError, match="uncertain"):
        counterfactual_addition_proposition(candidate)


def test_verifier_filters_are_applied_before_call_budget():
    specs = {
        "visual_a": _spec("direct_visual_verifier", "classification", group="visual-a"),
        "spatial": _spec("spatial_localizer", "segmentation", group="spatial"),
        "proposal": _spec("proposal_generator", "generation", group="proposal"),
        "visual_b": _spec("direct_visual_verifier", "classification", group="visual-b"),
    }
    cards = {}
    for expert in ("visual_a", "visual_b"):
        card = _card(expert)
        cards[card.key] = card
    selected, audit = select_expert_descriptors(
        [
            _descriptor("visual_a", "classification"),
            _descriptor("spatial", "segmentation"),
            _descriptor("proposal", "generation"),
            _descriptor("visual_b", "classification"),
        ],
        specs,
        modality="cxr",
        task="report_generation",
        claim_type="*",
        qualification_cards=cards,
        max_calls=2,
        require_commit_authority=True,
        allowed_evidence_roles=("direct_visual_verifier",),
        allowed_capabilities=("classification",),
    )
    assert [row["expert"] for row in selected] == ["visual_a", "visual_b"]
    assert [row["fault_group"] for row in audit] == ["visual-a", "visual-b"]


def test_report_add_is_verified_end_to_end_with_real_vs_knockoff_evidence():
    specs = {
        "visual": _spec("direct_visual_verifier", "classification", group="visual"),
    }
    qcard = _card("visual")

    class FakePool:
        def evidence_function(self, expert_id, image):
            assert expert_id == "visual"

            def infer(claim, _prefix):
                propositions = [item.proposition for item in claim.propositions]
                candidate_score = 2.0 if image == "target.png" else 0.2
                return NativeEvidence(
                    expert_id="visual",
                    capability="classification",
                    concept_scores={
                        propositions[0]: 0.0,
                        propositions[1]: candidate_score,
                    },
                    confidence=1.0,
                )

            return infer

    row = {
        "id": "target",
        "image": "target.png",
        "question": "Generate a radiology report.",
        "modality": "cxr",
        "task": "report_generation",
        "domain": "site-target",
        "group_id": "patient-target",
        "question_type": "finding",
    }
    controls = [
        {
            "id": f"control-{index}",
            "image": f"control-{index}.png",
            "question": "Generate a radiology report.",
            "modality": "cxr",
            "task": "report_generation",
            "domain": f"site-{index % 2}",
            "group_id": f"patient-{index}",
            "question_type": "finding",
        }
        for index in range(4)
    ]
    candidate = AtomicClinicalClaim(
        "candidate",
        "The image shows pleural effusion.",
        None,
        grounding={
            "schema": "radgraph-xl",
            "observation": "pleural effusion",
            "tags": ["definitely present"],
            "located_at": [],
            "suggestive_of": [],
        },
    )
    transaction = ClaimTransaction(
        "tx-add",
        "ADD",
        candidate,
        "Small pleural effusion.",
    )
    decision, evidence, audit = verify_transaction(
        row=row,
        transaction=transaction,
        baseline_claims=(),
        source_controls=controls,
        specs=specs,
        qualification_cards={qcard.key: qcard},
        pool=FakePool(),
        tx_policy={
            "qualification_min_domains": 2,
            "qualification_max_harm_ucb": 0.25,
            "qualification_min_specificity_lcb": 0.5,
            "min_support_groups": 1,
            "require_independent_validator": True,
            "require_patient_specific_support": True,
            "reject_on_qualified_contradiction": True,
        },
        claim_type="*",
        max_calls=2,
        knockoff_count=4,
    )
    assert decision.commit
    assert decision.reason == "proof-carrying-transaction"
    assert evidence[0]["differential_effect"] == pytest.approx(1.8)
    assert audit["controls"]["visual"]["count"] == 4


def test_source_report_add_counts_unsupported_addition_as_harm():
    candidate = AtomicClinicalClaim(
        "candidate",
        "The image shows pleural effusion.",
        None,
        grounding={
            "schema": "radgraph-xl",
            "observation": "pleural effusion",
            "tags": ["definitely present"],
            "located_at": [],
            "suggestive_of": [],
        },
    )
    transaction = ClaimTransaction(
        "tx-add",
        "ADD",
        candidate,
        "Small pleural effusion.",
    )
    assert report_outcome(transaction, (), ()) == -1.0
    reference = AtomicClinicalClaim(
        "reference",
        candidate.proposition,
        None,
        grounding=dict(candidate.grounding),
    )
    assert report_outcome(transaction, (), (reference,)) == 1.0


def _qualification_row(index, *, outcome_delta, effect, domain):
    return {
        "expert_id": "visual",
        "capability": "classification",
        "scope": "classification",
        "modality": "cxr",
        "task": "report_generation",
        "claim_type": "*",
        "domain": domain,
        "group_id": f"patient-{index}",
        "outcome_delta": outcome_delta,
        "real_effect": effect,
        "knockoff_effect": 0.0,
    }


def test_qualification_is_expert_action_conditional_not_candidate_average():
    rows = [
        _qualification_row(
            index,
            outcome_delta=0.5,
            effect=-0.2,
            domain="site-a" if index < 20 else "site-b",
        )
        for index in range(40)
    ]
    card = fit(rows)["cards"][0]
    # The candidate generator is uniformly beneficial, but this expert never
    # supports a transaction.  It must not inherit the generator's utility.
    assert card["utility_lcb"] == -1.0
    assert card["harm_ucb"] == 1.0
    assert card["specificity_lcb"] == 0.0


def test_qualification_rewards_bidirectional_differential_alignment():
    rows = []
    for index in range(20):
        rows.append(
            _qualification_row(
                index,
                outcome_delta=0.5,
                effect=0.8,
                domain="site-a",
            )
        )
    for index in range(20, 40):
        rows.append(
            _qualification_row(
                index,
                outcome_delta=-0.5,
                effect=-0.8,
                domain="site-b",
            )
        )
    card = fit(rows)["cards"][0]
    assert card["utility_lcb"] > 0
    assert card["harm_ucb"] < 0.25
    assert card["specificity_lcb"] > 0.8


def _rad_claim(claim_id, observation, *, present=True, span=None, location=()):
    tag = "definitely present" if present else "definitely absent"
    proposition = (
        f"The image shows {observation}"
        if present
        else f"The image does not show {observation}"
    )
    if location:
        proposition += " at " + ", ".join(location)
    proposition += "."
    return AtomicClinicalClaim(
        claim_id,
        proposition,
        span,
        grounding={
            "schema": "radgraph-xl",
            "observation": observation,
            "tags": [tag],
            "located_at": list(location),
            "suggestive_of": [],
        },
    )


def test_report_compiler_blocks_mixed_replace_and_add_in_same_candidate_sentence():
    baseline_text = "No pleural effusion."
    candidate_text = "Small pleural effusion with adjacent atelectasis."
    baseline = (
        _rad_claim(
            "base-effusion",
            "pleural effusion",
            present=False,
            span=(0, len(baseline_text)),
        ),
    )
    candidate = (
        _rad_claim(
            "candidate-effusion",
            "pleural effusion",
            present=True,
            span=(0, len(candidate_text)),
        ),
        _rad_claim(
            "candidate-atelectasis",
            "atelectasis",
            present=True,
            span=(0, len(candidate_text)),
        ),
    )
    transactions = candidate_transactions(
        task="report_generation",
        question="Generate report",
        baseline_text=baseline_text,
        candidate_text=candidate_text,
        baseline_claims=baseline,
        candidate_claims=candidate,
        proposer_expert_id="proposal",
    )
    assert transactions == ()


def test_report_compiler_pure_add_uses_atomic_propositions_not_whole_sentence():
    baseline_text = "Heart size is normal."
    candidate_text = "Heart size is normal. Small pleural effusion with atelectasis."
    baseline = (
        _rad_claim(
            "heart",
            "normal heart size",
            present=True,
            span=(0, len("Heart size is normal.")),
        ),
    )
    added_sentence_start = candidate_text.index("Small")
    candidate = (
        _rad_claim(
            "heart-candidate",
            "normal heart size",
            present=True,
            span=(0, len("Heart size is normal.")),
        ),
        _rad_claim(
            "effusion",
            "pleural effusion",
            present=True,
            span=(added_sentence_start, len(candidate_text)),
        ),
        _rad_claim(
            "atelectasis",
            "atelectasis",
            present=True,
            span=(added_sentence_start, len(candidate_text)),
        ),
    )
    transactions = candidate_transactions(
        task="report_generation",
        question="Generate report",
        baseline_text=baseline_text,
        candidate_text=candidate_text,
        baseline_claims=baseline,
        candidate_claims=candidate,
        proposer_expert_id="proposal",
    )
    assert len(transactions) == 2
    assert all(tx.operation == "ADD" for tx in transactions)
    assert {tx.replacement_text for tx in transactions} == {
        "The image shows pleural effusion.",
        "The image shows atelectasis.",
    }
    assert all("with" not in tx.replacement_text for tx in transactions)


def test_report_compiler_does_not_replace_sentence_if_candidate_omits_baseline_sibling():
    baseline_text = "No pleural effusion or pneumothorax."
    candidate_text = "Small pleural effusion."
    shared_span = (0, len(baseline_text))
    baseline = (
        _rad_claim(
            "base-effusion",
            "pleural effusion",
            present=False,
            span=shared_span,
        ),
        _rad_claim(
            "base-pneumothorax",
            "pneumothorax",
            present=False,
            span=shared_span,
        ),
    )
    candidate = (
        _rad_claim(
            "candidate-effusion",
            "pleural effusion",
            present=True,
            span=(0, len(candidate_text)),
        ),
    )
    transactions = candidate_transactions(
        task="report_generation",
        question="Generate report",
        baseline_text=baseline_text,
        candidate_text=candidate_text,
        baseline_claims=baseline,
        candidate_claims=candidate,
    )
    assert transactions == ()


def test_report_compiler_does_not_merge_multiple_incumbent_sentences_into_one_patch():
    baseline_text = "No pleural effusion. No pneumothorax."
    first_end = len("No pleural effusion.")
    second_start = first_end + 1
    candidate_text = "Small pleural effusion and pneumothorax."
    baseline = (
        _rad_claim(
            "base-effusion",
            "pleural effusion",
            present=False,
            span=(0, first_end),
        ),
        _rad_claim(
            "base-pneumothorax",
            "pneumothorax",
            present=False,
            span=(second_start, len(baseline_text)),
        ),
    )
    candidate = (
        _rad_claim(
            "candidate-effusion",
            "pleural effusion",
            present=True,
            span=(0, len(candidate_text)),
        ),
        _rad_claim(
            "candidate-pneumothorax",
            "pneumothorax",
            present=True,
            span=(0, len(candidate_text)),
        ),
    )
    transactions = candidate_transactions(
        task="report_generation",
        question="Generate report",
        baseline_text=baseline_text,
        candidate_text=candidate_text,
        baseline_claims=baseline,
        candidate_claims=candidate,
    )
    assert transactions == ()
