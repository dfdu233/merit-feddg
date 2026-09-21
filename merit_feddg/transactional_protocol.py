"""Unified MERIT-Tx planning for VQA and report generation."""
from __future__ import annotations

from .expert_policy import (
    select_expert_descriptors,
    transaction_descriptors,
)
from .transactional_claims import RadGraphClaimizer, claimize_vqa


def claimize_incumbent(*, task, question, text, radgraph_claimizer=None):
    """Use one algorithmic interface across short-answer VQA and long reports."""
    if task == "report_generation":
        claimizer = radgraph_claimizer or RadGraphClaimizer()
        return claimizer(text)
    return claimize_vqa(question, text)


def plan_transaction_experts(
    row,
    specs,
    *,
    qualification_cards=None,
    claim_type="*",
    max_calls=6,
    region_available=False,
    qualification_min_domains=2,
    qualification_max_harm_ucb=0.25,
    qualification_min_specificity_lcb=0.5,
):
    descriptors = transaction_descriptors(specs, row)
    selected, audit = select_expert_descriptors(
        descriptors,
        specs,
        modality=row["modality"],
        task=row["task"],
        claim_type=claim_type,
        qualification_cards=qualification_cards,
        max_calls=max_calls,
        region_available=region_available,
        qualification_min_domains=qualification_min_domains,
        qualification_max_harm_ucb=qualification_max_harm_ucb,
        qualification_min_specificity_lcb=qualification_min_specificity_lcb,
    )
    return {
        "descriptors": selected,
        "audit": audit,
        "selection": (
            "literature-role coverage first; distinct patient-specific verification "
            "and spatial roles before proposal/knowledge tools; commit authority "
            "requires source-only qualification"
        ),
        "target_labels_used": False,
    }
