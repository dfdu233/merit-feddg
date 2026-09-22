import json

import pytest

from merit_feddg.expert_policy import (
    SourceQualificationCard,
    load_qualification_cards,
    select_expert_descriptors,
)
from merit_feddg.med_defer import NativeEvidence
from merit_feddg.merit_tx import (
    MeritTxConfig,
    TransactionEvidence,
    decide_transaction,
    differential_margin_controls,
)
from merit_feddg.transactional_claims import (
    AtomicClinicalClaim,
    ClaimTransaction,
    candidate_transactions,
)
from merit_feddg.transactional_runtime import (
    counterfactual_addition_proposition,
    normalize_proposer_expert_ids,
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


def _card(
    expert,
    *,
    modality="cxr",
    task="report_generation",
    claim_type="*",
    veto_precision=0.7,
    veto_n=20,
):
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
        action_rate_lcb=0.6,
        support_n=40,
        support_domains=("site-a", "site-b"),
        support_consequential_n=30,
        support_help_n=24,
        support_harm_n=6,
        support_neutral_n=10,
        support_precision_lcb=0.6,
        veto_precision_lcb=veto_precision,
        veto_n=veto_n,
        veto_domains=("site-a", "site-b") if veto_n else (),
        veto_consequential_n=veto_n,
        veto_harm_n=veto_n,
        veto_help_n=0,
        veto_neutral_n=0,
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


def test_v4_certificate_rejects_relative_only_support_and_veto():
    # OmniMedVQA exposed this exact failure class: D>0 even though the real
    # image still prefers the incumbent. v4 must abstain.
    relative_only = differential_margin_controls(
        incumbent_real=1.0,
        candidate_real=0.9,
        knockoff_pairs=((1.0, 0.1), (1.0, 0.2), (1.0, 0.0)),
    )
    assert relative_only["real_margin"] == pytest.approx(-0.1)
    assert relative_only["differential_effect"] > 0
    assert relative_only["support_direction"] == 0
    assert relative_only["certificate"] == "abstain"

    reverse_relative_only = differential_margin_controls(
        incumbent_real=1.0,
        candidate_real=1.3,
        knockoff_pairs=((1.0, 2.0), (1.0, 2.1), (1.0, 1.9)),
    )
    assert reverse_relative_only["real_margin"] > 0
    assert reverse_relative_only["differential_effect"] < 0
    assert reverse_relative_only["support_direction"] == 0


def test_v4_rejects_omnmed_relative_only_false_support_regression():
    # Frozen OmniMedVQA pilot: real image preferred the incumbent by -0.88933,
    # but the median wrong-patient margin was -1.04280, so legacy D=+0.15347
    # incorrectly admitted the candidate. Absolute support must dominate first.
    value = differential_margin_controls(
        incumbent_real=0.0,
        candidate_real=-0.88933,
        knockoff_pairs=(
            (0.0, -1.20),
            (0.0, -1.10),
            (0.0, -0.98560),
            (0.0, -0.90),
        ),
    )
    assert value["knockoff_margin"] == pytest.approx(-1.04280)
    assert value["differential_effect"] == pytest.approx(0.15347)
    assert value["real_margin"] < 0
    assert value["support_direction"] == 0
    assert value["certificate"] == "abstain"


def test_v4_certificate_requires_absolute_preference_and_control_dominance():
    support = differential_margin_controls(
        incumbent_real=1.0,
        candidate_real=2.0,
        knockoff_pairs=((1.0, 1.2), (1.0, 1.3), (1.0, 1.1), (1.0, 1.4)),
    )
    assert support["real_margin"] == pytest.approx(1.0)
    assert support["support_direction"] == 1
    assert support["specificity_pvalue"] == pytest.approx(0.2)

    veto = differential_margin_controls(
        incumbent_real=2.0,
        candidate_real=1.0,
        knockoff_pairs=((1.0, 0.8), (1.0, 0.9), (1.0, 0.7), (1.0, 0.6)),
    )
    assert veto["real_margin"] == pytest.approx(-1.0)
    assert veto["support_direction"] == -1
    assert veto["specificity_pvalue"] == pytest.approx(0.2)


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
            "qualification_min_support_precision_lcb": 0.5,
            "qualification_min_consequential": 4,
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
        "support_direction": 1 if effect > 0 else -1 if effect < 0 else 0,
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
    assert card["support_n"] == 0
    assert card["support_domains"] == []
    assert card["specificity_lcb"] == 0.0
    assert card["veto_precision_lcb"] == 0.0
    assert card["veto_n"] == 40


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
    assert card["support_n"] == 20
    assert card["support_domains"] == ["site-a"]
    assert card["specificity_lcb"] > 0.8
    assert card["veto_precision_lcb"] > 0.8
    assert card["veto_n"] == 20
    assert card["veto_domains"] == ["site-b"]


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


def test_support_authority_does_not_implicitly_authorize_veto():
    specs = {
        "visual": _spec("direct_visual_verifier", "classification", group="visual"),
    }
    support_only = _card("visual", veto_precision=0.0, veto_n=0)
    transaction = ClaimTransaction(
        "tx-no-veto",
        "REPLACE",
        AtomicClinicalClaim("candidate", "The image shows pleural effusion.", None),
        "Small pleural effusion.",
        baseline_claim_id="baseline",
    )
    decision = decide_transaction(
        transaction,
        (
            TransactionEvidence(
                expert_id="visual",
                capability="classification",
                scope="classification",
                fault_group="visual",
                evidence_role="direct_visual_verifier",
                differential_effect=-0.9,
                support_direction=-1,
                patient_specific=True,
            ),
        ),
        specs=specs,
        qualification_cards={support_only.key: support_only},
        modality="cxr",
        task="report_generation",
        claim_type="*",
        config=MeritTxConfig(),
    )
    assert not decision.commit
    assert decision.reason == "insufficient-qualified-patient-specific-support"
    assert decision.verifier_fault_groups == ()


def test_action_domain_coverage_cannot_be_borrowed_from_inactive_domains():
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
                outcome_delta=0.0,
                effect=0.0,
                domain="site-b",
            )
        )
    card_payload = fit(rows)["cards"][0]
    card = SourceQualificationCard(
        **{
            **card_payload,
            "domains": tuple(card_payload["domains"]),
            "support_domains": tuple(card_payload["support_domains"]),
            "veto_domains": tuple(card_payload["veto_domains"]),
        }
    )
    assert card.domains == ("site-a", "site-b")
    assert card.support_domains == ("site-a",)
    assert not card.authorizes_commit(min_domains=2)


def test_v4_qualification_schema_rejects_legacy_cards(tmp_path):
    rows = [
        _qualification_row(
            index,
            outcome_delta=0.5,
            effect=0.8,
            domain="site-a" if index < 20 else "site-b",
        )
        for index in range(40)
    ]
    payload = fit(rows)
    assert payload["schema"] == "merit-expert-qualification-v4"
    path = tmp_path / "cards.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_qualification_cards(path)
    assert len(loaded) == 1

    payload["schema"] = "merit-expert-qualification-v1"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported expert qualification"):
        load_qualification_cards(path)


def test_multi_proposer_normalization_is_stable_and_deduplicated():
    assert normalize_proposer_expert_ids(
        "chexagent",
        ("xrv,biomedclip_claim_verifier", "chexagent"),
    ) == (
        "chexagent",
        "xrv",
        "biomedclip_claim_verifier",
    )


def test_every_proposer_fault_group_is_excluded_from_commit_validation():
    specs = {
        "proposal_a": _spec(
            "direct_visual_verifier",
            "classification",
            group="proposal-a",
        ),
        "proposal_b": _spec(
            "direct_visual_verifier",
            "classification",
            group="proposal-b",
        ),
        "independent": _spec(
            "direct_visual_verifier",
            "classification",
            group="independent",
        ),
    }
    cards = {
        card.key: card
        for card in (
            _card("proposal_a"),
            _card("proposal_b"),
            _card("independent"),
        )
    }
    transaction = ClaimTransaction(
        "tx-multi-proposer",
        "REPLACE",
        AtomicClinicalClaim("candidate", "The image shows pleural effusion.", None),
        "Small pleural effusion.",
        baseline_claim_id="baseline",
        proposer_expert_ids=("proposal_a", "proposal_b"),
    )
    proposer_evidence = (
        TransactionEvidence(
            expert_id="proposal_a",
            capability="classification",
            scope="classification",
            fault_group="proposal-a",
            evidence_role="direct_visual_verifier",
            differential_effect=0.8,
            support_direction=1,
            patient_specific=True,
        ),
        TransactionEvidence(
            expert_id="proposal_b",
            capability="classification",
            scope="classification",
            fault_group="proposal-b",
            evidence_role="direct_visual_verifier",
            differential_effect=0.7,
            support_direction=1,
            patient_specific=True,
        ),
    )
    rejected = decide_transaction(
        transaction,
        proposer_evidence,
        specs=specs,
        qualification_cards=cards,
        modality="cxr",
        task="report_generation",
        claim_type="*",
        config=MeritTxConfig(),
    )
    assert not rejected.commit
    assert rejected.reason == "proposer-has-no-independent-qualified-validator"

    independent = TransactionEvidence(
        expert_id="independent",
        capability="classification",
        scope="classification",
        fault_group="independent",
        evidence_role="direct_visual_verifier",
        differential_effect=0.6,
        support_direction=1,
        patient_specific=True,
    )
    accepted = decide_transaction(
        transaction,
        proposer_evidence + (independent,),
        specs=specs,
        qualification_cards=cards,
        modality="cxr",
        task="report_generation",
        claim_type="*",
        config=MeritTxConfig(),
    )
    assert accepted.commit
    assert accepted.verifier_fault_groups == ("independent",)


def test_candidate_transactions_preserve_multi_proposer_provenance():
    baseline = AtomicClinicalClaim(
        "base",
        "The image does not show pleural effusion.",
        (0, 2),
    )
    candidate = AtomicClinicalClaim(
        "candidate",
        "The image shows pleural effusion.",
        (0, 3),
    )
    transactions = candidate_transactions(
        task="open_vqa",
        question="Is there pleural effusion?",
        baseline_text="No",
        candidate_text="Yes",
        baseline_claims=(baseline,),
        candidate_claims=(candidate,),
        proposer_expert_ids=("proposal_a", "proposal_b"),
    )
    assert transactions[0].proposer_expert_ids == ("proposal_a", "proposal_b")
