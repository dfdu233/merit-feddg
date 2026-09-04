from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

SUPPORTED_CAPABILITIES = frozenset(
    {"classification", "detection", "segmentation", "retrieval", "generation"}
)


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


@dataclass(frozen=True)
class DomainSignal:
    """Label-free target-study signals used to discount an expert at inference time."""

    ood_score: float = 0.0
    image_quality: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.ood_score <= 1.0:
            raise ValueError("ood_score must be in [0, 1]")
        if not 0.0 <= self.image_quality <= 1.0:
            raise ValueError("image_quality must be in [0, 1]")


@dataclass(frozen=True)
class ExpertCard:
    """Static capability and source-domain card; no target labels are stored here."""

    expert_id: str
    modalities: tuple[str, ...]
    capabilities: tuple[str, ...]
    source_reliability_lcb: float
    validation_domain_scores: tuple[float, ...] = ()
    expected_gain: float = 1.0
    latency_ms: float = 100.0

    def __post_init__(self) -> None:
        unknown = set(self.capabilities) - SUPPORTED_CAPABILITIES
        if unknown:
            raise ValueError(f"unsupported capabilities: {sorted(unknown)}")
        if not self.modalities or not self.capabilities:
            raise ValueError("an expert needs at least one modality and capability")
        if not 0.0 <= self.source_reliability_lcb <= 1.0:
            raise ValueError("source_reliability_lcb must be in [0, 1]")
        if any(not 0.0 <= score <= 1.0 for score in self.validation_domain_scores):
            raise ValueError("validation_domain_scores must be in [0, 1]")
        if self.expected_gain < 0.0 or self.latency_ms < 0.0:
            raise ValueError("expected_gain and latency_ms cannot be negative")


@dataclass(frozen=True)
class NativeEvidence:
    """Common envelope that preserves heterogeneous expert-native evidence.

    ``concept_scores`` is the only field translated into language-token guidance.
    Spatial and free-text outputs remain available for provenance and later adapters.
    """

    expert_id: str
    capability: str
    concept_scores: Mapping[str, float]
    confidence: float
    ood_score: float = 0.0
    image_quality: float = 1.0
    masks: tuple[Any, ...] = ()
    boxes: tuple[tuple[float, float, float, float], ...] = ()
    generated_text: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.capability not in SUPPORTED_CAPABILITIES:
            raise ValueError(f"unsupported capability: {self.capability}")
        if not self.concept_scores:
            raise ValueError("concept_scores cannot be empty")
        if not all(math.isfinite(float(value)) for value in self.concept_scores.values()):
            raise ValueError("concept scores must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 <= self.ood_score <= 1.0:
            raise ValueError("ood_score must be in [0, 1]")
        if not 0.0 <= self.image_quality <= 1.0:
            raise ValueError("image_quality must be in [0, 1]")


@dataclass(frozen=True)
class ClaimRequest:
    sample_id: str
    claim_id: str
    modality: str
    required_capabilities: tuple[str, ...]
    concepts: tuple[str, ...]
    base_logits: tuple[float, ...]
    uncertainty: float
    router_probs: Mapping[str, float]
    domain_signals: Mapping[str, DomainSignal] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.concepts) < 2 or len(self.base_logits) != len(self.concepts):
            raise ValueError("concepts and base_logits must have the same length >= 2")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("uncertainty must be in [0, 1]")
        if not set(self.required_capabilities) <= SUPPORTED_CAPABILITIES:
            raise ValueError("request contains an unsupported capability")


@dataclass(frozen=True)
class DomainTrust:
    source_lcb: float
    lower_tail_stability: float
    ood_discount: float
    quality: float
    score: float


@dataclass(frozen=True)
class DeferralDecision:
    expert_id: str | None
    utility: float
    route_confidence: float
    pre_call_trust: float
    reason: str


@dataclass(frozen=True)
class ClaimTrace:
    sample_id: str
    claim_id: str
    selected_expert: str | None
    capability: str | None
    utility: float
    trust: float
    gate: float
    expert_called: bool
    cache_hit: bool
    reason: str
    base_logits: tuple[float, ...]
    guided_logits: tuple[float, ...]
    concept_delta: tuple[float, ...]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class DomainTrustCalibrator:
    """Conservative source-only trust: LCB x lower-tail stability x OOD x quality."""

    def __init__(self, cvar_alpha: float = 0.25, ood_temperature: float = 2.0) -> None:
        if not 0.0 < cvar_alpha <= 1.0:
            raise ValueError("cvar_alpha must be in (0, 1]")
        if ood_temperature < 0.0:
            raise ValueError("ood_temperature cannot be negative")
        self.cvar_alpha = float(cvar_alpha)
        self.ood_temperature = float(ood_temperature)

    def lower_tail(self, scores: Sequence[float]) -> float:
        if not scores:
            return 1.0
        ordered = np.sort(np.asarray(scores, dtype=float))
        count = max(1, math.ceil(len(ordered) * self.cvar_alpha))
        return _clip01(float(np.mean(ordered[:count])))

    def score(
        self,
        card: ExpertCard,
        signal: DomainSignal,
        evidence: NativeEvidence | None = None,
    ) -> DomainTrust:
        ood = max(signal.ood_score, evidence.ood_score if evidence else 0.0)
        quality = min(signal.image_quality, evidence.image_quality if evidence else 1.0)
        lower_tail = self.lower_tail(card.validation_domain_scores)
        ood_discount = math.exp(-self.ood_temperature * ood)
        score = card.source_reliability_lcb * lower_tail * ood_discount * quality
        return DomainTrust(
            source_lcb=card.source_reliability_lcb,
            lower_tail_stability=lower_tail,
            ood_discount=float(ood_discount),
            quality=float(quality),
            score=_clip01(score),
        )


class ClaimDeferralController:
    """Select NONE or one qualified specialist before the next clinical claim."""

    def __init__(
        self,
        trust_calibrator: DomainTrustCalibrator,
        uncertainty_threshold: float = 0.35,
        minimum_utility: float = 0.08,
        cost_weight: float = 0.05,
    ) -> None:
        self.trust_calibrator = trust_calibrator
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.minimum_utility = float(minimum_utility)
        self.cost_weight = float(cost_weight)

    def select(self, request: ClaimRequest, cards: Sequence[ExpertCard]) -> DeferralDecision:
        if request.uncertainty < self.uncertainty_threshold:
            return DeferralDecision(None, 0.0, 0.0, 0.0, "generalist-confident")

        ranked: list[tuple[float, str, float, float]] = []
        required = set(request.required_capabilities)
        for card in cards:
            if request.modality not in card.modalities and "*" not in card.modalities:
                continue
            overlap = required.intersection(card.capabilities)
            if not overlap:
                continue
            capability_match = len(overlap) / max(len(required), 1)
            route_confidence = _clip01(request.router_probs.get(request.modality, 0.0))
            signal = request.domain_signals.get(card.expert_id, DomainSignal())
            trust = self.trust_calibrator.score(card, signal).score
            benefit = (
                request.uncertainty
                * route_confidence
                * capability_match
                * trust
                * card.expected_gain
            )
            utility = benefit - self.cost_weight * card.latency_ms / 1000.0
            ranked.append((utility, card.expert_id, route_confidence, trust))

        if not ranked:
            return DeferralDecision(None, 0.0, 0.0, 0.0, "no-compatible-expert")
        utility, expert_id, route_confidence, trust = max(ranked)
        if utility < self.minimum_utility:
            return DeferralDecision(
                None, utility, route_confidence, trust, "utility-below-threshold"
            )
        return DeferralDecision(expert_id, utility, route_confidence, trust, "expert-deferred")


EvidenceProvider = Callable[[ClaimRequest], NativeEvidence]


class LazyExpertPool:
    """Lazily calls and caches only the expert selected for a clinical claim."""

    def __init__(self) -> None:
        self.cards: dict[str, ExpertCard] = {}
        self.providers: dict[str, EvidenceProvider] = {}
        self.cache: dict[tuple[str, str, str], NativeEvidence] = {}
        self.call_counts: dict[str, int] = {}

    def register(self, card: ExpertCard, provider: EvidenceProvider) -> None:
        if card.expert_id in self.cards:
            raise ValueError(f"duplicate expert_id: {card.expert_id}")
        self.cards[card.expert_id] = card
        self.providers[card.expert_id] = provider
        self.call_counts[card.expert_id] = 0

    def get(self, expert_id: str, request: ClaimRequest) -> tuple[NativeEvidence, bool]:
        key = (request.sample_id, request.claim_id, expert_id)
        if key in self.cache:
            return self.cache[key], True
        evidence = self.providers[expert_id](request)
        if evidence.expert_id != expert_id:
            raise ValueError("evidence expert_id does not match the selected expert")
        self.cache[key] = evidence
        self.call_counts[expert_id] += 1
        return evidence, False


class MedDeferEngine:
    """Claim-level conditional computation kernel for medical VLM decoding."""

    def __init__(
        self,
        controller: ClaimDeferralController,
        guidance_strength: float = 0.8,
        max_bias_norm: float = 1.25,
        minimum_post_call_trust: float = 0.05,
    ) -> None:
        self.controller = controller
        self.guidance_strength = float(guidance_strength)
        self.max_bias_norm = float(max_bias_norm)
        self.minimum_post_call_trust = float(minimum_post_call_trust)

    @staticmethod
    def _normalized_concept_delta(request: ClaimRequest, evidence: NativeEvidence) -> np.ndarray:
        raw = np.asarray(
            [float(evidence.concept_scores.get(concept, 0.0)) for concept in request.concepts],
            dtype=float,
        )
        centered = raw - float(np.mean(raw))
        norm = float(np.linalg.norm(centered))
        return centered / norm if norm > 1e-12 else np.zeros_like(centered)

    def guide(self, request: ClaimRequest, pool: LazyExpertPool) -> ClaimTrace:
        base = np.asarray(request.base_logits, dtype=float)
        decision = self.controller.select(request, list(pool.cards.values()))
        if decision.expert_id is None:
            zeros = tuple(0.0 for _ in request.concepts)
            return ClaimTrace(
                request.sample_id,
                request.claim_id,
                None,
                None,
                decision.utility,
                decision.pre_call_trust,
                0.0,
                False,
                False,
                decision.reason,
                tuple(base),
                tuple(base),
                zeros,
            )

        card = pool.cards[decision.expert_id]
        evidence, cache_hit = pool.get(decision.expert_id, request)
        signal = request.domain_signals.get(decision.expert_id, DomainSignal())
        trust = self.controller.trust_calibrator.score(card, signal, evidence).score
        if evidence.capability not in request.required_capabilities:
            trust = 0.0
            reason = "capability-mismatch"
        elif trust < self.minimum_post_call_trust:
            reason = "post-call-trust-too-low"
        else:
            reason = "bounded-expert-guidance"

        direction = self._normalized_concept_delta(request, evidence)
        gate = _clip01(
            request.uncertainty * decision.route_confidence * trust * evidence.confidence
        )
        if reason != "bounded-expert-guidance":
            gate = 0.0
        delta = self.guidance_strength * gate * direction
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > self.max_bias_norm:
            delta *= self.max_bias_norm / delta_norm
        guided = base + delta
        return ClaimTrace(
            request.sample_id,
            request.claim_id,
            decision.expert_id,
            evidence.capability,
            decision.utility,
            trust,
            gate,
            True,
            cache_hit,
            reason,
            tuple(float(value) for value in base),
            tuple(float(value) for value in guided),
            tuple(float(value) for value in delta),
        )
