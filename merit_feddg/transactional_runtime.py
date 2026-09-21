"""Runtime helpers for source-qualified MERIT-Tx transactions.

The runtime is task agnostic: VQA and report generation differ only in claimization.
Candidate generation is deliberately external/frozen before this module verifies
transactions, preventing qualification outcomes from steering the proposal itself.
"""
from dataclasses import asdict

from .claims import CandidateProposition, ClaimSpec
from .expert_policy import select_expert_descriptors, transaction_descriptors
from .knockoff import select_matched_knockoffs
from .merit_tx import (
    MeritTxConfig,
    decide_transaction,
    native_transaction_evidence_controls,
)
from .transactional_protocol import claimize_incumbent


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


def counterfactual_addition_proposition(claim):
    """Build an explicit, answer-blind null for a report ADD transaction.

    ADD has no incumbent claim to score.  We therefore compare the proposed
    observation against the same observation with flipped presence polarity,
    rather than against an empty string or an unrelated baseline sentence.
    Uncertain propositions fail closed because they do not define a clean
    binary counterfactual.
    """
    grounding = claim.grounding or {}
    if grounding.get("schema") == "radgraph-xl":
        observation = str(grounding.get("observation", "")).strip()
        if not observation:
            raise ValueError("RadGraph ADD requires an observation")
        tags = [str(value).casefold() for value in grounding.get("tags", ())]
        if any("uncertain" in value for value in tags):
            raise ValueError("uncertain report ADD has no deterministic counterfactual")
        candidate_absent = any("absent" in value for value in tags)
        prefix = "The image shows " if candidate_absent else "The image does not show "
        proposition = prefix + observation
        located = [
            str(value).strip()
            for value in grounding.get("located_at", ())
            if str(value).strip()
        ]
        if located:
            proposition += " at " + ", ".join(located)
        return proposition.rstrip(".") + "."

    proposition = " ".join(str(claim.proposition).split())
    positive = "The image shows "
    negative = "The image does not show "
    if proposition.startswith(negative):
        return positive + proposition[len(negative):]
    if proposition.startswith(positive):
        return negative + proposition[len(positive):]
    raise ValueError(
        "ADD verification requires an explicit presence/absence proposition"
    )


def transaction_claim_spec(row, transaction, baseline_claims):
    """Compile one transaction into a two-proposition native verifier query."""
    if transaction.operation == "DELETE":
        raise ValueError("MERIT-Tx does not authorize DELETE from report omission")

    if transaction.operation == "REPLACE":
        baseline = {
            claim.claim_id: claim
            for claim in baseline_claims
        }.get(transaction.baseline_claim_id)
        if baseline is None:
            raise ValueError("transaction baseline claim is missing")
        incumbent_proposition = baseline.proposition
    elif transaction.operation == "ADD":
        incumbent_proposition = counterfactual_addition_proposition(
            transaction.proposed_claim
        )
    else:
        raise ValueError(f"unsupported transaction operation: {transaction.operation}")

    return ClaimSpec(
        claim_id=transaction.transaction_id,
        question=str(row["question"]),
        modality=str(row["modality"]),
        required_capabilities=("classification",),
        propositions=(
            CandidateProposition(
                candidate_id="incumbent",
                answer=incumbent_proposition,
                proposition=incumbent_proposition,
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
        metadata={
            "merit_tx": True,
            "task": row["task"],
            "operation": transaction.operation,
            "explicit_add_counterfactual": transaction.operation == "ADD",
        },
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
    """Reserve the verification budget for commit-capable native verifiers.

    Generic expert planning remains coverage-first, but commit verification is a
    different stage.  Spatial/proposal/knowledge tools are filtered *before*
    max_calls so they cannot evict an independent visual verifier and then be
    discarded after budgeting.
    """
    descriptors = transaction_descriptors(specs, row)
    selected, audit = select_expert_descriptors(
        descriptors,
        specs,
        modality=row["modality"],
        task=row["task"],
        claim_type=claim_type,
        qualification_cards=qualification_cards,
        max_calls=max_calls,
        require_commit_authority=True,
        region_available=False,
        qualification_min_domains=int(tx_policy.get("qualification_min_domains", 2)),
        qualification_max_harm_ucb=float(
            tx_policy.get("qualification_max_harm_ucb", 0.25)
        ),
        qualification_min_specificity_lcb=float(
            tx_policy.get("qualification_min_specificity_lcb", 0.5)
        ),
        allowed_evidence_roles=("direct_visual_verifier",),
        allowed_capabilities=("classification",),
    )
    return selected, audit


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
    if transaction.operation == "DELETE":
        from .transactional_claims import TransactionDecision

        return (
            TransactionDecision(
                transaction.transaction_id,
                False,
                "delete-not-authorized-by-omission",
            ),
            [],
            {"selected": [], "skipped": [], "controls": {}},
        )

    try:
        claim = transaction_claim_spec(row, transaction, baseline_claims)
    except ValueError as exc:
        from .transactional_claims import TransactionDecision

        return (
            TransactionDecision(
                transaction.transaction_id,
                False,
                "transaction-has-no-verifiable-counterfactual",
            ),
            [],
            {
                "selected": [],
                "skipped": [
                    {
                        "expert_id": None,
                        "reason": "counterfactual-unavailable",
                        "detail": str(exc),
                    }
                ],
                "controls": {},
            },
        )
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
