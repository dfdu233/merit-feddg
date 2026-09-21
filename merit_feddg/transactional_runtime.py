"""Runtime helpers for source-qualified MERIT-Tx transactions.

The runtime is task agnostic: VQA and report generation differ only in claimization.
Candidate generation is deliberately external/frozen before this module verifies
transactions, preventing qualification outcomes from steering the proposal itself.
"""
from dataclasses import asdict

from .claims import CandidateProposition, ClaimSpec
from .knockoff import select_matched_knockoffs
from .merit_tx import (
    MeritTxConfig,
    decide_transaction,
    native_transaction_evidence_controls,
)
from .transactional_protocol import claimize_incumbent, plan_transaction_experts


def merit_tx_config(mapping):
    mapping = dict(mapping or {})
    keys = {
        "require_independent_validator",
        "require_patient_specific_support",
        "reject_on_qualified_contradiction",
        "min_support_groups",
        "qualification_min_domains",
        "qualification_max_harm_ucb",
        "qualification_min_specificity_lcb",
    }
    return MeritTxConfig(**{key: mapping[key] for key in keys if key in mapping})


def transaction_claim_spec(row, transaction, baseline_claims):
    """Pair one immutable incumbent claim with one explicit replacement claim."""
    if transaction.operation != "REPLACE":
        raise ValueError("native differential verification currently requires REPLACE")
    baseline = {
        claim.claim_id: claim
        for claim in baseline_claims
    }.get(transaction.baseline_claim_id)
    if baseline is None:
        raise ValueError("transaction baseline claim is missing")
    return ClaimSpec(
        claim_id=transaction.transaction_id,
        question=str(row["question"]),
        modality=str(row["modality"]),
        required_capabilities=("classification",),
        propositions=(
            CandidateProposition(
                candidate_id="incumbent",
                answer=baseline.proposition,
                proposition=baseline.proposition,
                polarity="open",
            ),
            CandidateProposition(
                candidate_id="candidate",
                answer=transaction.proposed_claim.proposition,
                proposition=transaction.proposed_claim.proposition,
                polarity="open",
            ),
        ),
        closed_set=True,
        metadata={"merit_tx": True, "task": row["task"]},
    )


def native_scores(pool, expert_id, image, claim):
    evidence = pool.evidence_function(expert_id, image)(claim, "")
    queries = [item.proposition for item in claim.propositions]
    try:
        return tuple(float(evidence.concept_scores[query]) for query in queries)
    except KeyError as exc:
        raise ValueError(
            f"{expert_id} did not return native scores for both transaction claims"
        ) from exc


def claimize_pair(row, baseline_text, candidate_text, *, radgraph_claimizer=None):
    baseline_claims = claimize_incumbent(
        task=row["task"],
        question=row["question"],
        text=baseline_text,
        radgraph_claimizer=radgraph_claimizer,
    )
    candidate_claims = claimize_incumbent(
        task=row["task"],
        question=row["question"],
        text=candidate_text,
        radgraph_claimizer=radgraph_claimizer,
    )
    return tuple(baseline_claims), tuple(candidate_claims)


def selected_native_verifiers(
    row,
    specs,
    qualification_cards,
    *,
    claim_type,
    tx_policy,
    max_calls,
):
    plan = plan_transaction_experts(
        row,
        specs,
        qualification_cards=qualification_cards,
        claim_type=claim_type,
        max_calls=max_calls,
        region_available=False,
        qualification_min_domains=int(tx_policy.get("qualification_min_domains", 2)),
        qualification_max_harm_ucb=float(
            tx_policy.get("qualification_max_harm_ucb", 0.25)
        ),
        qualification_min_specificity_lcb=float(
            tx_policy.get("qualification_min_specificity_lcb", 0.5)
        ),
    )
    selected = []
    for descriptor, audit in zip(plan["descriptors"], plan["audit"], strict=True):
        if (
            audit["evidence_role"] == "direct_visual_verifier"
            and audit["patient_specific"]
            and audit["commit_authorized"]
            and descriptor["capability"] == "classification"
        ):
            selected.append(descriptor)
    return selected, plan["audit"]


def verify_transaction(
    *,
    row,
    transaction,
    baseline_claims,
    source_controls,
    specs,
    qualification_cards,
    pool,
    tx_policy,
    claim_type,
    max_calls,
    knockoff_count,
):
    """Verify one transaction without reading a reference answer."""
    if transaction.operation != "REPLACE":
        from .transactional_claims import TransactionDecision

        return (
            TransactionDecision(
                transaction.transaction_id,
                False,
                "native-verification-requires-explicit-replacement",
            ),
            [],
            {"selected": [], "skipped": [], "controls": {}},
        )

    claim = transaction_claim_spec(row, transaction, baseline_claims)
    verifiers, plan_audit = selected_native_verifiers(
        row,
        specs,
        qualification_cards,
        claim_type=claim_type,
        tx_policy=tx_policy,
        max_calls=max_calls,
    )
    evidences = []
    skipped = []
    control_audit = {}
    match_fields = ("question_type",) if row.get("question_type") else ()
    for descriptor in verifiers:
        expert_id = descriptor["expert"]
        try:
            incumbent_real, candidate_real = native_scores(
                pool, expert_id, row["image"], claim
            )
            controls = select_matched_knockoffs(
                source_controls,
                row,
                expert_id=expert_id,
                count=knockoff_count,
                match_fields=match_fields,
            )
            pairs = [
                native_scores(pool, expert_id, control["image"], claim)
                for control in controls
            ]
            evidence = native_transaction_evidence_controls(
                expert_id=expert_id,
                capability=descriptor["capability"],
                scope=descriptor["scope"],
                incumbent_real=incumbent_real,
                candidate_real=candidate_real,
                knockoff_pairs=pairs,
                specs=specs,
            )
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            skipped.append(
                {
                    "expert_id": expert_id,
                    "reason": "native-verification-unavailable",
                    "detail": str(exc),
                }
            )
            continue
        evidences.append(evidence)
        control_audit[expert_id] = {
            "ids": [control["id"] for control in controls],
            "count": len(controls),
            "matching": controls[0]["selection"] if controls else None,
        }

    decision = decide_transaction(
        transaction,
        evidences,
        specs=specs,
        qualification_cards=qualification_cards,
        modality=row["modality"],
        task=row["task"],
        claim_type=claim_type,
        config=merit_tx_config(tx_policy),
    )
    return (
        decision,
        [asdict(evidence) for evidence in evidences],
        {
            "selected": plan_audit,
            "skipped": skipped,
            "controls": control_audit,
        },
    )
