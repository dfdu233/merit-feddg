"""MERIT-Tx commit policy: proof-carrying edits over an immutable incumbent.

This module is deliberately agnostic to VQA versus report generation.  Both are
represented as atomic clinical claims plus proposed ADD/DELETE/REPLACE
transactions.

The policy does not average all experts.  It separates proposal authority from
commit authority and uses source-only qualification plus patient-specific
differential evidence.  General medical knowledge may corroborate a proposal
but cannot change the incumbent on its own.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .expert_policy import (
    qualification_for,
    role_card,
)
from .transactional_claims import ClaimTransaction, TransactionDecision


@dataclass(frozen=True)
class TransactionEvidence:
    expert_id: str
    capability: str
    scope: str
    fault_group: str
    evidence_role: str
    differential_effect: float
    support_direction: int
    patient_specific: bool

    def __post_init__(self):
        if not self.expert_id or not self.capability or not self.scope or not self.fault_group:
            raise ValueError("transaction evidence identity cannot be empty")
        if self.support_direction not in {-1, 0, 1}:
            raise ValueError("support_direction must be -1, 0, or 1")
        if not math.isfinite(self.differential_effect):
            raise ValueError("differential effect must be finite")


@dataclass(frozen=True)
class MeritTxConfig:
    require_independent_validator: bool = True
    require_patient_specific_support: bool = True
    reject_on_qualified_contradiction: bool = True
    min_support_groups: int = 1

    def __post_init__(self):
        if type(self.min_support_groups) is not int or self.min_support_groups < 1:
            raise ValueError("min_support_groups must be a positive integer")


def evidence_from_expert(
    *,
    expert_id,
    capability,
    scope,
    real_effect,
    knockoff_effect,
    support_direction,
    specs,
):
    """Construct one differential verification observation from an expert role card."""
    card = role_card(expert_id, specs[expert_id])
    differential = float(real_effect) - float(knockoff_effect)
    if not math.isfinite(differential):
        raise ValueError("real/knockoff evidence effects must be finite")
    return TransactionEvidence(
        expert_id=expert_id,
        capability=capability,
        scope=scope,
        fault_group=card.fault_group,
        evidence_role=card.evidence_role,
        differential_effect=differential,
        support_direction=int(support_direction),
        patient_specific=card.patient_specific,
    )


def decide_transaction(
    transaction: ClaimTransaction,
    evidences,
    *,
    specs,
    qualification_cards,
    modality,
    task,
    claim_type="*",
    config: MeritTxConfig | None = None,
):
    """Decide one atomic transaction without changing the incumbent trajectory."""
    config = config or MeritTxConfig()
    evidences = tuple(evidences)
    proposer_group = None
    if transaction.proposer_expert_id:
        proposer_group = role_card(
            transaction.proposer_expert_id,
            specs[transaction.proposer_expert_id],
        ).fault_group

    qualified_support = {}
    qualified_contradiction = {}
    for evidence in evidences:
        qcard = qualification_for(
            qualification_cards,
            expert_id=evidence.expert_id,
            capability=evidence.capability,
            scope=evidence.scope,
            modality=modality,
            task=task,
            claim_type=claim_type,
        )
        source_qualified = (
            qcard is not None
            and qcard.authorizes_commit()
        )
        # A negative/zero real-vs-knockoff effect is not patient-specific proof,
        # regardless of whether the raw expert output agrees with the candidate.
        if evidence.differential_effect <= 0:
            continue

        # Knowledge/proposal experts remain useful in the audit, but they do not
        # become patient-specific proof merely because they agree with a candidate.
        if not evidence.patient_specific:
            continue
        if not source_qualified:
            continue
        if evidence.support_direction > 0:
            qualified_support.setdefault(evidence.fault_group, []).append(evidence)
        elif evidence.support_direction < 0:
            qualified_contradiction.setdefault(evidence.fault_group, []).append(evidence)

    support_groups = set(qualified_support)
    contradiction_groups = set(qualified_contradiction)
    independent_support = set(support_groups)
    if config.require_independent_validator and proposer_group is not None:
        independent_support.discard(proposer_group)

    if config.reject_on_qualified_contradiction and contradiction_groups:
        return TransactionDecision(
            transaction_id=transaction.transaction_id,
            commit=False,
            reason="qualified-patient-specific-contradiction",
            verifier_fault_groups=tuple(sorted(support_groups | contradiction_groups)),
            patient_specific_support=bool(support_groups),
            source_qualified_support=bool(support_groups),
            differential_effect=min(
                evidence.differential_effect
                for rows in qualified_contradiction.values()
                for evidence in rows
            ),
        )

    required_groups = config.min_support_groups
    usable_groups = independent_support if config.require_independent_validator else support_groups
    if config.require_patient_specific_support and len(usable_groups) < required_groups:
        reason = (
            "proposer-has-no-independent-qualified-validator"
            if support_groups and proposer_group is not None and not independent_support
            else "insufficient-qualified-patient-specific-support"
        )
        return TransactionDecision(
            transaction_id=transaction.transaction_id,
            commit=False,
            reason=reason,
            verifier_fault_groups=tuple(sorted(support_groups)),
            patient_specific_support=bool(support_groups),
            source_qualified_support=bool(support_groups),
            differential_effect=(
                max(
                    evidence.differential_effect
                    for rows in qualified_support.values()
                    for evidence in rows
                )
                if support_groups
                else None
            ),
        )

    if len(usable_groups) >= required_groups:
        effect = min(
            max(abs(evidence.differential_effect) for evidence in qualified_support[group])
            for group in usable_groups
        )
        return TransactionDecision(
            transaction_id=transaction.transaction_id,
            commit=True,
            reason="proof-carrying-transaction",
            verifier_fault_groups=tuple(sorted(usable_groups)),
            patient_specific_support=True,
            source_qualified_support=True,
            differential_effect=float(effect),
        )

    return TransactionDecision(
        transaction_id=transaction.transaction_id,
        commit=False,
        reason="no-commit-proof",
        verifier_fault_groups=(),
        patient_specific_support=False,
        source_qualified_support=False,
        differential_effect=None,
    )
