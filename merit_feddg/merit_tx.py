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
from statistics import median

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
        expected_direction = (
            1
            if self.differential_effect > 0
            else -1
            if self.differential_effect < 0
            else 0
        )
        if self.support_direction != expected_direction:
            raise ValueError(
                "support_direction must equal sign(differential_effect)"
            )


@dataclass(frozen=True)
class MeritTxConfig:
    require_independent_validator: bool = True
    require_patient_specific_support: bool = True
    reject_on_qualified_contradiction: bool = True
    min_support_groups: int = 1
    qualification_min_domains: int = 2
    qualification_max_harm_ucb: float = 0.25
    qualification_min_support_precision_lcb: float = 0.5
    qualification_min_veto_precision_lcb: float = 0.5
    qualification_min_consequential: int = 4

    def __post_init__(self):
        if type(self.min_support_groups) is not int or self.min_support_groups < 1:
            raise ValueError("min_support_groups must be a positive integer")
        if type(self.qualification_min_domains) is not int or self.qualification_min_domains < 1:
            raise ValueError("qualification_min_domains must be positive")
        if not 0 <= self.qualification_max_harm_ucb <= 1:
            raise ValueError("qualification_max_harm_ucb must be in [0,1]")
        if not 0 <= self.qualification_min_support_precision_lcb <= 1:
            raise ValueError(
                "qualification_min_support_precision_lcb must be in [0,1]"
            )
        if not 0 <= self.qualification_min_veto_precision_lcb <= 1:
            raise ValueError(
                "qualification_min_veto_precision_lcb must be in [0,1]"
            )
        if (
            type(self.qualification_min_consequential) is not int
            or self.qualification_min_consequential < 0
        ):
            raise ValueError("qualification_min_consequential must be nonnegative")


def differential_margin(
    *,
    incumbent_real,
    candidate_real,
    incumbent_knockoff,
    candidate_knockoff,
):
    """Current-patient candidate margin minus matched-control candidate margin."""
    values = (
        incumbent_real,
        candidate_real,
        incumbent_knockoff,
        candidate_knockoff,
    )
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise TypeError("differential margins require numeric expert-native scores")
    values = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("differential margins require finite scores")
    real_margin = values[1] - values[0]
    knockoff_margin = values[3] - values[2]
    differential = real_margin - knockoff_margin
    return {
        "real_margin": real_margin,
        "knockoff_margin": knockoff_margin,
        "differential_effect": differential,
        # Direction is defined by the patient-specific contrast D_e, not by
        # the expert's absolute candidate-vs-incumbent margin.  A biased
        # verifier may prefer the incumbent on every image while still moving
        # specifically toward the candidate on the current patient.
        "support_direction": 1 if differential > 0 else -1 if differential < 0 else 0,
    }


def differential_margin_controls(
    *,
    incumbent_real,
    candidate_real,
    knockoff_pairs,
):
    """Real candidate margin minus the median matched-control candidate margin."""
    pairs = tuple(knockoff_pairs)
    if not pairs:
        raise ValueError("at least one matched knockoff control is required")
    real = differential_margin(
        incumbent_real=incumbent_real,
        candidate_real=candidate_real,
        incumbent_knockoff=pairs[0][0],
        candidate_knockoff=pairs[0][1],
    )
    margins = []
    for incumbent_knockoff, candidate_knockoff in pairs:
        value = differential_margin(
            incumbent_real=incumbent_real,
            candidate_real=candidate_real,
            incumbent_knockoff=incumbent_knockoff,
            candidate_knockoff=candidate_knockoff,
        )
        margins.append(value["knockoff_margin"])
    knockoff_margin = float(median(margins))
    real_margin = float(real["real_margin"])
    differential = real_margin - knockoff_margin
    return {
        "real_margin": real_margin,
        "knockoff_margin": knockoff_margin,
        "knockoff_margins": tuple(float(value) for value in margins),
        "controls": len(margins),
        "differential_effect": differential,
        "support_direction": 1 if differential > 0 else -1 if differential < 0 else 0,
    }


def native_transaction_evidence_controls(
    *,
    expert_id,
    capability,
    scope,
    incumbent_real,
    candidate_real,
    knockoff_pairs,
    specs,
):
    """Construct native transaction evidence using a robust matched-control median."""
    value = differential_margin_controls(
        incumbent_real=incumbent_real,
        candidate_real=candidate_real,
        knockoff_pairs=knockoff_pairs,
    )
    return evidence_from_expert(
        expert_id=expert_id,
        capability=capability,
        scope=scope,
        real_effect=value["real_margin"],
        knockoff_effect=value["knockoff_margin"],
        support_direction=value["support_direction"],
        specs=specs,
    )


def native_transaction_evidence(
    *,
    expert_id,
    capability,
    scope,
    incumbent_real,
    candidate_real,
    incumbent_knockoff,
    candidate_knockoff,
    specs,
):
    """Prefer expert-native claim margins over a shared receiver residual."""
    value = differential_margin(
        incumbent_real=incumbent_real,
        candidate_real=candidate_real,
        incumbent_knockoff=incumbent_knockoff,
        candidate_knockoff=candidate_knockoff,
    )
    return evidence_from_expert(
        expert_id=expert_id,
        capability=capability,
        scope=scope,
        real_effect=value["real_margin"],
        knockoff_effect=value["knockoff_margin"],
        support_direction=value["support_direction"],
        specs=specs,
    )


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
    if int(support_direction) not in {-1, 0, 1}:
        raise ValueError("support direction must be -1, 0, or 1")
    # Keep the argument for backward-compatible callers, but normalize the
    # stored direction to the signed differential effect.  This makes support
    # and contradiction two sides of the same matched-control statistic.
    direction = 1 if differential > 0 else -1 if differential < 0 else 0
    return TransactionEvidence(
        expert_id=expert_id,
        capability=capability,
        scope=scope,
        fault_group=card.fault_group,
        evidence_role=card.evidence_role,
        differential_effect=differential,
        support_direction=direction,
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
    proposer_ids = list(transaction.proposer_expert_ids)
    if transaction.proposer_expert_id:
        proposer_ids.append(transaction.proposer_expert_id)
    proposer_groups = {
        role_card(expert_id, specs[expert_id]).fault_group
        for expert_id in proposer_ids
    }

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
        support_authorized = (
            qcard is not None
            and qcard.authorizes_commit(
                min_domains=config.qualification_min_domains,
                max_harm_ucb=config.qualification_max_harm_ucb,
                min_support_precision_lcb=(
                    config.qualification_min_support_precision_lcb
                ),
                min_consequential=config.qualification_min_consequential,
            )
        )
        veto_authorized = (
            qcard is not None
            and qcard.authorizes_veto(
                min_domains=config.qualification_min_domains,
                min_veto_precision_lcb=config.qualification_min_veto_precision_lcb,
                min_consequential=config.qualification_min_consequential,
            )
        )
        # Knowledge/proposal experts remain useful in the audit, but they do not
        # become patient-specific proof merely because they agree with a candidate.
        if not evidence.patient_specific:
            continue
        if evidence.differential_effect > 0 and support_authorized:
            qualified_support.setdefault(evidence.fault_group, []).append(evidence)
        elif evidence.differential_effect < 0 and veto_authorized:
            # Negative D_e can veto only when negative decisions themselves
            # have a conservative source-only precision bound.  Safe positive
            # support does not imply safe contradiction.
            qualified_contradiction.setdefault(evidence.fault_group, []).append(evidence)

    support_groups = set(qualified_support)
    contradiction_groups = set(qualified_contradiction)
    independent_support = set(support_groups)
    if config.require_independent_validator:
        independent_support.difference_update(proposer_groups)

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
            if support_groups and proposer_groups and not independent_support
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
