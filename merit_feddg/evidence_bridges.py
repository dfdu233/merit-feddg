from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .claims import ClaimSpec
from .med_defer import NativeEvidence


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _validate_mask(mask: Any) -> None:
    if isinstance(mask, Mapping):
        if not mask:
            raise ValueError("mask mappings cannot be empty")
        recognized = {"rle", "counts", "size", "area_fraction", "mask", "bbox", "polygon"}
        if not recognized.intersection(mask):
            raise ValueError("mask mapping has no recognized spatial payload")
        if "area_fraction" in mask:
            area = float(mask["area_fraction"])
            if not math.isfinite(area) or not 0.0 <= area <= 1.0:
                raise ValueError("mask area_fraction must be in [0, 1]")
        for key in ("rle", "counts"):
            if key in mask and not mask[key]:
                raise ValueError(f"mask {key} cannot be empty")
        return
    if isinstance(mask, np.ndarray) or (
        isinstance(mask, Sequence) and not isinstance(mask, (str, bytes, bytearray))
    ):
        array = np.asarray(mask)
        if array.size == 0 or array.ndim not in {2, 3}:
            raise ValueError("dense masks must be non-empty 2D or 3D arrays")
        if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
            raise ValueError("dense masks must contain finite numeric values")
        return
    raise ValueError("unsupported mask payload")


def _validate_box(box: Any, coordinate_space: str) -> None:
    if coordinate_space not in {"normalized", "pixel"}:
        raise ValueError("box_coordinate_space must be normalized or pixel")
    if not isinstance(box, Sequence) or isinstance(box, (str, bytes, bytearray)):
        raise TypeError("a box must be a four-value sequence")
    if len(box) != 4:
        raise ValueError("a box must contain x1, y1, x2, y2")
    values = tuple(float(value) for value in box)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("box coordinates must be finite")
    x1, y1, x2, y2 = values
    if x1 > x2 or y1 > y2:
        raise ValueError("box corners are reversed")
    if coordinate_space == "normalized" and any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("normalized box coordinates must be in [0, 1]")
    if coordinate_space == "pixel" and any(value < 0.0 for value in values):
        raise ValueError("pixel box coordinates cannot be negative")


@dataclass(frozen=True)
class SpatialProvenance:
    kind: str
    payload: Any


@dataclass(frozen=True)
class CandidateClaimEvidence:
    candidate_id: str
    proposition: str
    support: float
    contradiction: float
    coverage: float
    spatial: tuple[SpatialProvenance, ...] = ()
    rationale: str | None = None

    def __post_init__(self) -> None:
        for name in ("support", "contradiction", "coverage"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")

    @property
    def signed_support(self) -> float:
        return self.coverage * (self.support - self.contradiction)


@dataclass(frozen=True)
class ClaimEvidence:
    claim_id: str
    expert_id: str | None
    capability: str | None
    candidates: tuple[CandidateClaimEvidence, ...]
    confidence: float
    abstained: bool
    reason: str
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def abstain(cls, claim: ClaimSpec, reason: str, expert_id: str | None = None) -> ClaimEvidence:
        return cls(
            claim_id=claim.claim_id,
            expert_id=expert_id,
            capability=None,
            candidates=tuple(
                CandidateClaimEvidence(
                    candidate_id=item.candidate_id,
                    proposition=item.proposition,
                    support=0.0,
                    contradiction=0.0,
                    coverage=0.0,
                )
                for item in claim.propositions
            ),
            confidence=0.0,
            abstained=True,
            reason=reason,
        )

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not self.candidates:
            raise ValueError("claim evidence must retain every requested proposition")

    def concept_scores(self) -> dict[str, float]:
        """Return bounded semantic-proposition scores for the existing decoder."""

        if self.abstained:
            return {item.proposition: 0.0 for item in self.candidates}
        return {item.proposition: self.confidence * item.signed_support for item in self.candidates}


class EvidenceBridge(Protocol):
    capability: str

    def convert(self, claim: ClaimSpec, evidence: NativeEvidence) -> ClaimEvidence: ...


def _semantic_raw_scores(
    claim: ClaimSpec, evidence: NativeEvidence
) -> tuple[np.ndarray, np.ndarray]:
    """Look up scores only by semantic propositions, not bare answer strings."""

    candidate_support = evidence.provenance.get("candidate_support", {})
    if not isinstance(candidate_support, Mapping):
        raise TypeError("candidate_support must be a mapping")
    values: list[float] = []
    covered: list[float] = []
    for item in claim.propositions:
        if item.proposition in evidence.concept_scores:
            value = float(evidence.concept_scores[item.proposition])
            if not math.isfinite(value):
                raise ValueError("semantic proposition scores must be finite")
            values.append(value)
            covered.append(1.0)
        elif item.candidate_id in candidate_support:
            value = float(candidate_support[item.candidate_id])
            if not math.isfinite(value):
                raise ValueError("candidate_support values must be finite")
            values.append(value)
            covered.append(1.0)
        else:
            values.append(0.0)
            covered.append(0.0)
    return np.asarray(values, dtype=float), np.asarray(covered, dtype=float)


def _supports(claim: ClaimSpec, evidence: NativeEvidence) -> tuple[np.ndarray, np.ndarray]:
    values, coverage = _semantic_raw_scores(claim, evidence)
    semantics = str(evidence.provenance.get("score_semantics", "logit"))
    if semantics not in {"logit", "probability"}:
        raise ValueError("score_semantics must be 'logit' or 'probability'")
    if semantics == "probability":
        if np.any((values[coverage > 0] < 0.0) | (values[coverage > 0] > 1.0)):
            raise ValueError("probability scores must be in [0, 1]")
        supports = values
    elif claim.closed_set and len(values) > 1:
        shifted = values - float(np.max(values))
        exp = np.exp(np.clip(shifted, -60.0, 60.0))
        supports = exp / max(float(np.sum(exp)), 1e-12)
    else:
        supports = 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))
    # A missing proposition must never inherit probability mass from softmax.
    return supports * coverage, coverage


def _candidate_rows(
    claim: ClaimSpec,
    evidence: NativeEvidence,
    *,
    coverage_scale: float = 1.0,
    spatial: tuple[SpatialProvenance, ...] = (),
    rationale: str | None = None,
) -> tuple[CandidateClaimEvidence, ...]:
    supports, present = _supports(claim, evidence)
    rows: list[CandidateClaimEvidence] = []
    for item, support, covered in zip(claim.propositions, supports, present):
        coverage = _clip01(float(covered) * coverage_scale)
        rows.append(
            CandidateClaimEvidence(
                candidate_id=item.candidate_id,
                proposition=item.proposition,
                support=_clip01(support),
                contradiction=_clip01((1.0 - support) if covered else 0.0),
                coverage=coverage,
                spatial=spatial,
                rationale=rationale,
            )
        )
    return tuple(rows)


def _result(
    claim: ClaimSpec,
    evidence: NativeEvidence,
    candidates: tuple[CandidateClaimEvidence, ...],
    reason: str,
) -> ClaimEvidence:
    covered = [item.coverage > 0.0 for item in candidates]
    if not any(covered):
        return ClaimEvidence.abstain(claim, "no-semantic-proposition-score", evidence.expert_id)
    if not all(covered):
        return ClaimEvidence.abstain(claim, "incomplete-semantic-coverage", evidence.expert_id)
    return ClaimEvidence(
        claim_id=claim.claim_id,
        expert_id=evidence.expert_id,
        capability=evidence.capability,
        candidates=candidates,
        confidence=evidence.confidence,
        abstained=False,
        reason=reason,
        provenance=dict(evidence.provenance),
    )


class ClassificationEvidenceBridge:
    capability = "classification"

    def convert(self, claim: ClaimSpec, evidence: NativeEvidence) -> ClaimEvidence:
        return _result(claim, evidence, _candidate_rows(claim, evidence), "classification-evidence")


class RetrievalEvidenceBridge:
    capability = "retrieval"

    def convert(self, claim: ClaimSpec, evidence: NativeEvidence) -> ClaimEvidence:
        references = evidence.provenance.get("references", ())
        rationale = evidence.generated_text
        if not rationale and references:
            rationale = f"Retrieved {len(references)} source record(s)."
        if not rationale and not references:
            return ClaimEvidence.abstain(claim, "retrieval-without-provenance", evidence.expert_id)
        return _result(
            claim,
            evidence,
            _candidate_rows(claim, evidence, rationale=rationale),
            "retrieval-evidence",
        )


class SegmentationEvidenceBridge:
    capability = "segmentation"

    def convert(self, claim: ClaimSpec, evidence: NativeEvidence) -> ClaimEvidence:
        empty_is_valid = bool(evidence.provenance.get("empty_mask_is_valid", False))
        if not evidence.masks and not empty_is_valid:
            return ClaimEvidence.abstain(claim, "segmentation-without-mask", evidence.expert_id)
        for mask in evidence.masks:
            _validate_mask(mask)
        spatial = tuple(SpatialProvenance("mask", mask) for mask in evidence.masks)
        return _result(
            claim,
            evidence,
            _candidate_rows(claim, evidence, spatial=spatial),
            "segmentation-evidence",
        )


class DetectionEvidenceBridge:
    capability = "detection"

    def convert(self, claim: ClaimSpec, evidence: NativeEvidence) -> ClaimEvidence:
        empty_is_valid = bool(evidence.provenance.get("empty_detection_is_valid", False))
        if not evidence.boxes and not empty_is_valid:
            return ClaimEvidence.abstain(claim, "detection-without-box", evidence.expert_id)
        coordinate_space = str(evidence.provenance.get("box_coordinate_space", "normalized"))
        for box in evidence.boxes:
            _validate_box(box, coordinate_space)
        spatial = tuple(SpatialProvenance("box", box) for box in evidence.boxes)
        return _result(
            claim,
            evidence,
            _candidate_rows(claim, evidence, spatial=spatial),
            "detection-evidence",
        )


class GenerationEvidenceBridge:
    capability = "generation"

    def convert(self, claim: ClaimSpec, evidence: NativeEvidence) -> ClaimEvidence:
        if not evidence.generated_text or not evidence.generated_text.strip():
            return ClaimEvidence.abstain(claim, "generation-without-text", evidence.expert_id)
        return _result(
            claim,
            evidence,
            _candidate_rows(claim, evidence, rationale=evidence.generated_text.strip()),
            "generation-evidence",
        )


class EvidenceBridgeRegistry:
    """Capability registry with an explicit abstention path for unknown bridges."""

    def __init__(self, *, include_defaults: bool = True) -> None:
        self._bridges: dict[str, EvidenceBridge] = {}
        if include_defaults:
            for bridge in (
                ClassificationEvidenceBridge(),
                RetrievalEvidenceBridge(),
                SegmentationEvidenceBridge(),
                DetectionEvidenceBridge(),
                GenerationEvidenceBridge(),
            ):
                self.register(bridge)

    def register(self, bridge: EvidenceBridge) -> None:
        if bridge.capability in self._bridges:
            raise ValueError(f"duplicate evidence bridge: {bridge.capability}")
        self._bridges[bridge.capability] = bridge

    def convert(self, claim: ClaimSpec, evidence: NativeEvidence) -> ClaimEvidence:
        if evidence.capability not in claim.required_capabilities:
            return ClaimEvidence.abstain(
                claim, "capability-not-requested", expert_id=evidence.expert_id
            )
        bridge = self._bridges.get(evidence.capability)
        if bridge is None:
            return ClaimEvidence.abstain(
                claim,
                f"no-evidence-bridge:{evidence.capability}",
                expert_id=evidence.expert_id,
            )
        try:
            return bridge.convert(claim, evidence)
        except (TypeError, ValueError, OverflowError):
            return ClaimEvidence.abstain(
                claim, "invalid-native-evidence", expert_id=evidence.expert_id
            )


def semantic_score_map(
    claim: ClaimSpec,
    evidence: NativeEvidence,
    registry: EvidenceBridgeRegistry | None = None,
) -> dict[str, float]:
    """Convert native evidence to decoder scores, returning zeros on abstention."""

    converted = (registry or EvidenceBridgeRegistry()).convert(claim, evidence)
    return converted.concept_scores()
