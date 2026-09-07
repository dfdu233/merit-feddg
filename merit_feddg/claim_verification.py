"""Pre-commit verification for one atomic clinical claim.

This module does not generate a diagnosis and does not average experts as if
they were independent annotators.  It decides whether a proposed claim has
qualified, scoped support, should be revised because of contradiction, or must
remain uncommitted.  It is the semantic contract for the next live decoder.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .claims import ClaimSpec
from .evidence_bridges import ClaimEvidence

_REAL_DOMAIN_KINDS = frozenset({"hospital", "center", "dataset"})


@dataclass(frozen=True)
class EvidenceCertificate:
    """Source-only permission for one expert/capability evidence channel."""

    expert_id: str
    capability: str
    scope: str
    qualified: bool
    source_only: bool
    lower_content_gain: float
    domains: tuple[str, ...]
    domain_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.expert_id or not self.capability or not self.scope:
            raise ValueError("certificate identity cannot be empty")
        if not self.source_only:
            raise ValueError("target outcomes cannot enter an evidence certificate")
        if not math.isfinite(self.lower_content_gain):
            raise ValueError("certificate content gain must be finite")
        if self.qualified and (
            len(self.domains) < 2
            or not self.domain_kinds
            or set(self.domain_kinds) - _REAL_DOMAIN_KINDS
            or self.lower_content_gain <= 0
        ):
            raise ValueError("qualified evidence needs positive support from real source domains")

    @property
    def key(self) -> tuple[str, str, str]:
        return self.expert_id, self.capability, self.scope


@dataclass(frozen=True)
class VerificationConfig:
    min_support: float = 0.55
    contradiction_threshold: float = 0.55
    min_coverage: float = 0.5
    require_certificate: bool = True

    def __post_init__(self) -> None:
        values = (self.min_support, self.contradiction_threshold, self.min_coverage)
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("verification thresholds must lie in [0, 1]")


@dataclass(frozen=True)
class VerificationDecision:
    action: str
    reason: str
    claim_id: str
    candidate_id: str
    support: float
    contradiction: float
    coverage: float
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.action not in {"COMMIT", "REVISE", "ABSTAIN"}:
            raise ValueError("unknown claim verification action")


class ClaimCommitVerifier:
    """Fail-closed verification before an answer clause is exposed to the user."""

    def __init__(self, config: VerificationConfig | None = None) -> None:
        self.config = config or VerificationConfig()

    def verify(
        self,
        claim: ClaimSpec,
        candidate_id: str,
        evidences: Sequence[ClaimEvidence],
        certificates: Mapping[tuple[str, str, str], EvidenceCertificate] | None = None,
    ) -> VerificationDecision:
        proposition = next(
            (candidate for candidate in claim.propositions if candidate.candidate_id == candidate_id),
            None,
        )
        if proposition is None:
            raise ValueError("candidate does not belong to this claim")
        certificates = dict(certificates or {})
        accepted = []
        for evidence in evidences:
            if evidence.claim_id != claim.claim_id or evidence.abstained:
                continue
            if evidence.expert_id is None or evidence.capability is None:
                continue
            if self.config.require_certificate:
                scope = str(evidence.provenance.get("scope", "")).strip()
                certificate = certificates.get((evidence.expert_id, evidence.capability, scope))
                if certificate is None or not certificate.qualified:
                    continue
            row = next(
                (candidate for candidate in evidence.candidates
                 if candidate.candidate_id == candidate_id),
                None,
            )
            if row is not None:
                accepted.append((evidence, row))
        if not accepted:
            return self._decision("ABSTAIN", "no-qualified-scoped-evidence", claim, candidate_id)

        # Max pooling is intentional: correlated tools do not acquire fictitious
        # confidence through averaging or multiplication. A strong contradiction
        # remains visible even when another expert supports the claim.
        support = max(evidence.confidence * row.support for evidence, row in accepted)
        contradiction = max(
            evidence.confidence * row.contradiction for evidence, row in accepted
        )
        coverage = max(row.coverage for _, row in accepted)
        evidence_ids = tuple(
            sorted(
                {
                    str(evidence.provenance.get("evidence_id", evidence.expert_id))
                    for evidence, _ in accepted
                }
            )
        )
        spatial_required = bool(claim.metadata.get("spatial_required", False))
        has_spatial = any(row.spatial for _, row in accepted)
        if spatial_required and not has_spatial:
            return self._decision(
                "ABSTAIN",
                "missing-spatial-support",
                claim,
                candidate_id,
                support,
                contradiction,
                coverage,
                evidence_ids,
            )
        if proposition.polarity == "negated" and not any(
            evidence.provenance.get("coverage_semantics") == "exhaustive"
            for evidence, _ in accepted
        ):
            return self._decision(
                "ABSTAIN",
                "negative-claim-without-exhaustive-coverage",
                claim,
                candidate_id,
                support,
                contradiction,
                coverage,
                evidence_ids,
            )
        if support >= self.config.min_support and contradiction >= self.config.contradiction_threshold:
            action, reason = "REVISE", "qualified-evidence-conflict"
        elif contradiction >= self.config.contradiction_threshold:
            action, reason = "REVISE", "qualified-evidence-contradicts-claim"
        elif support >= self.config.min_support and coverage >= self.config.min_coverage:
            action, reason = "COMMIT", "qualified-evidence-supports-claim"
        else:
            action, reason = "ABSTAIN", "insufficient-support-or-coverage"
        return self._decision(
            action,
            reason,
            claim,
            candidate_id,
            support,
            contradiction,
            coverage,
            evidence_ids,
        )

    @staticmethod
    def _decision(
        action,
        reason,
        claim,
        candidate_id,
        support=0.0,
        contradiction=0.0,
        coverage=0.0,
        evidence_ids=(),
    ):
        return VerificationDecision(
            action=action,
            reason=reason,
            claim_id=claim.claim_id,
            candidate_id=candidate_id,
            support=float(support),
            contradiction=float(contradiction),
            coverage=float(coverage),
            evidence_ids=tuple(evidence_ids),
        )
