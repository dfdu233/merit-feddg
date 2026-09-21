"""Literature-grounded expert-pool policy for MERIT-Tx.

The pool is organized by clinical capability and evidence role, not by the
number of available models. Target labels are never consulted here. An expert
may propose evidence without being authorized to change the incumbent answer;
commit authority requires a source-only qualification card.

Design references:
- MedRAX (ICML 2025): explicit tool roles for classification, segmentation,
  grounding, VQA and report generation.
- VILA-M3 (CVPR 2025): model cards describing modality/task/specialist scope.
- DnR (CVPR 2026): question-conditioned use of external visual experts.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

_ALLOWED_EVIDENCE_ROLES = frozenset(
    {
        "direct_visual_verifier",
        "spatial_localizer",
        "proposal_generator",
        "knowledge_retriever",
        "broad_embedding_fallback",
    }
)
_ALLOWED_AUTHORITY = frozenset({"never", "source_qualified"})
_PATIENT_SPECIFIC_ROLES = frozenset(
    {"direct_visual_verifier", "spatial_localizer"}
)


@dataclass(frozen=True)
class ExpertRoleCard:
    expert_id: str
    fault_group: str
    evidence_role: str
    modalities: tuple[str, ...]
    tasks: tuple[str, ...]
    capabilities: tuple[str, ...]
    scope: str
    commit_authority: str
    literature: tuple[str, ...]

    def __post_init__(self):
        if not self.expert_id or not self.fault_group or not self.scope:
            raise ValueError("expert role identity cannot be empty")
        if self.evidence_role not in _ALLOWED_EVIDENCE_ROLES:
            raise ValueError(f"unsupported expert evidence role: {self.evidence_role}")
        if self.commit_authority not in _ALLOWED_AUTHORITY:
            raise ValueError(f"unsupported commit authority: {self.commit_authority}")
        if not self.capabilities:
            raise ValueError("expert role card needs at least one capability")
        if not self.literature:
            raise ValueError("expert role card needs a literature/model-card basis")

    @property
    def patient_specific(self) -> bool:
        return self.evidence_role in _PATIENT_SPECIFIC_ROLES


@dataclass(frozen=True)
class SourceQualificationCard:
    """Source-only permission to let one expert influence a claim transaction."""

    expert_id: str
    capability: str
    scope: str
    modality: str
    task: str
    claim_type: str
    n: int
    domains: tuple[str, ...]
    utility_lcb: float
    harm_ucb: float
    specificity_lcb: float
    source_only: bool = True

    def __post_init__(self):
        if not self.source_only:
            raise ValueError("target outcomes cannot enter an expert qualification card")
        if self.n < 1 or len(set(self.domains)) < 1:
            raise ValueError("qualification card needs source observations")
        for value in (self.utility_lcb, self.harm_ucb, self.specificity_lcb):
            if not math.isfinite(value):
                raise ValueError("qualification statistics must be finite")

    @property
    def key(self):
        return (
            self.expert_id,
            self.capability,
            self.scope,
            self.modality,
            self.task,
            self.claim_type,
        )

    def authorizes_commit(
        self,
        *,
        min_domains=2,
        max_harm_ucb=0.5,
        min_specificity_lcb=0.5,
    ) -> bool:
        return (
            len(set(self.domains)) >= min_domains
            and self.utility_lcb > 0
            and self.harm_ucb <= max_harm_ucb
            and self.specificity_lcb >= min_specificity_lcb
        )


def role_card(expert_id, spec) -> ExpertRoleCard:
    role = spec.get("evidence_role")
    capability = tuple(spec.get("capabilities", ()))
    if role is None:
        if capability == ("retrieval",):
            role = "knowledge_retriever"
        elif capability == ("generation",):
            role = "proposal_generator"
        elif "segmentation" in capability or "detection" in capability:
            role = "spatial_localizer"
        elif "classification" in capability:
            role = "direct_visual_verifier"
        else:
            role = "broad_embedding_fallback"
    literature = spec.get("literature", ())
    if isinstance(literature, str):
        literature = (literature,)
    if not literature:
        literature = ("legacy-unreviewed-role-card",)
    authority = spec.get(
        "commit_authority",
        "never" if role in {"proposal_generator", "knowledge_retriever"} else "source_qualified",
    )
    return ExpertRoleCard(
        expert_id=expert_id,
        fault_group=str(spec.get("fault_group", expert_id)).strip(),
        evidence_role=role,
        modalities=tuple(spec.get("modalities", ())),
        tasks=tuple(spec.get("tasks", ())),
        capabilities=capability,
        scope=str(spec.get("scope", capability[0] if capability else "unknown")),
        commit_authority=authority,
        literature=tuple(str(value) for value in literature),
    )


def load_qualification_cards(path):
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "merit-expert-qualification-v1":
        raise ValueError("unsupported expert qualification card schema")
    if payload.get("source_only") is not True:
        raise ValueError("expert qualification must be source-only")
    result = {}
    for row in payload.get("cards", []):
        card = SourceQualificationCard(**{**row, "domains": tuple(row["domains"])})
        if card.key in result:
            raise ValueError(f"duplicate expert qualification card: {card.key}")
        result[card.key] = card
    return result


def qualification_for(
    cards,
    *,
    expert_id,
    capability,
    scope,
    modality,
    task,
    claim_type,
):
    exact = (
        expert_id,
        capability,
        scope,
        modality,
        task,
        claim_type,
    )
    if exact in cards:
        return cards[exact]
    wildcard = (
        expert_id,
        capability,
        scope,
        modality,
        task,
        "*",
    )
    return cards.get(wildcard)


def transaction_descriptors(specs, row):
    """Build claim-transaction descriptors, including transaction-only experts."""
    from .capabilities import tool_descriptors

    transaction_specs = {
        name: {
            **spec,
            "enabled": True,
            "transaction_only": False,
        }
        for name, spec in specs.items()
        if spec.get("expert_pool_enabled", True) is not False
    }
    return tool_descriptors(transaction_specs, row)


def select_expert_descriptors(
    descriptors,
    specs,
    *,
    modality,
    task,
    claim_type="*",
    qualification_cards=None,
    max_calls=6,
    require_commit_authority=False,
):
    """Coverage-first, diversity-aware expert selection with source-only authority."""
    if type(max_calls) is not int or max_calls < 1:
        raise ValueError("max_calls must be a positive integer")
    qualification_cards = qualification_cards or {}
    role_priority = {
        "direct_visual_verifier": 0,
        "spatial_localizer": 1,
        "proposal_generator": 2,
        "knowledge_retriever": 3,
        "broad_embedding_fallback": 4,
    }

    rows = []
    for index, descriptor in enumerate(descriptors):
        expert = descriptor["expert"]
        card = role_card(expert, specs[expert])
        if specs[expert].get("expert_pool_enabled", True) is False:
            continue
        capability_name = descriptor["capability"]
        qcard = qualification_for(
            qualification_cards,
            expert_id=expert,
            capability=capability_name,
            scope=descriptor["scope"],
            modality=modality,
            task=task,
            claim_type=claim_type,
        )
        commit_authorized = (
            card.commit_authority == "source_qualified"
            and qcard is not None
            and qcard.authorizes_commit()
        )
        if require_commit_authority and not commit_authorized:
            continue
        rows.append(
            {
                "descriptor": descriptor,
                "card": card,
                "qualification": qcard,
                "commit_authorized": commit_authorized,
                "original_index": index,
            }
        )

    selected = []
    seen_cells = set()
    seen_fault_groups = set()
    while rows and len(selected) < max_calls:
        def key(row):
            card = row["card"]
            descriptor = row["descriptor"]
            cell = (
                card.evidence_role,
                descriptor["capability"],
                card.fault_group,
            )
            new_cell = cell not in seen_cells
            new_fault = card.fault_group not in seen_fault_groups
            return (
                role_priority[card.evidence_role],
                not new_cell,
                not new_fault,
                not row["commit_authorized"],
                row["original_index"],
            )

        rows.sort(key=key)
        row = rows.pop(0)
        selected.append(row)
        card = row["card"]
        descriptor = row["descriptor"]
        seen_cells.add(
            (card.evidence_role, descriptor["capability"], card.fault_group)
        )
        seen_fault_groups.add(card.fault_group)

    audit = []
    for row in selected:
        card = row["card"]
        qcard = row["qualification"]
        audit.append(
            {
                **row["descriptor"],
                "fault_group": card.fault_group,
                "evidence_role": card.evidence_role,
                "patient_specific": card.patient_specific,
                "commit_authority": card.commit_authority,
                "commit_authorized": row["commit_authorized"],
                "qualification": (
                    {
                        "n": qcard.n,
                        "domains": list(qcard.domains),
                        "utility_lcb": qcard.utility_lcb,
                        "harm_ucb": qcard.harm_ucb,
                        "specificity_lcb": qcard.specificity_lcb,
                    }
                    if qcard is not None
                    else None
                ),
                "literature": list(card.literature),
            }
        )
    return [row["descriptor"] for row in selected], audit
