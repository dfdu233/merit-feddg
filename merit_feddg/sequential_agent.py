from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .intervention_risk import (
    InterventionAction,
    InterventionRiskController,
    InterventionSignals,
    RiskDecision,
)
from .med_defer import (
    ClaimRequest,
    DomainTrustCalibrator,
    ExpertCard,
    LazyExpertPool,
    NativeEvidence,
)

SignalBuilder = Callable[
    [ClaimRequest, ExpertCard, NativeEvidence, tuple["AcquisitionStep", ...]],
    InterventionSignals,
]


@dataclass(frozen=True)
class AcquisitionStep:
    step_index: int
    expert_id: str
    capability: str
    utility: float
    evidence: NativeEvidence
    risk_decision: RiskDecision
    cache_hit: bool


@dataclass(frozen=True)
class SequentialAgentTrace:
    sample_id: str
    claim_id: str
    action: InterventionAction
    selected_expert: str | None
    base_logits: tuple[float, ...]
    guided_logits: tuple[float, ...]
    steps: tuple[AcquisitionStep, ...] = ()
    reason: str = ""
    failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class SequentialAgentConfig:
    max_expert_calls: int = 2
    cost_weight: float = 0.05
    guidance_strength: float = 0.8
    max_bias_norm: float = 1.25

    def __post_init__(self) -> None:
        if self.max_expert_calls < 1:
            raise ValueError("max_expert_calls must be at least one")
        if not np.isfinite([self.cost_weight, self.guidance_strength, self.max_bias_norm]).all():
            raise ValueError("finite configuration required")
        if self.cost_weight < 0.0:
            raise ValueError("cost_weight cannot be negative")
        if self.guidance_strength < 0.0 or self.max_bias_norm < 0.0:
            raise ValueError("guidance parameters cannot be negative")


def default_signal_builder(
    request: ClaimRequest,
    card: ExpertCard,
    evidence: NativeEvidence,
    history: tuple[AcquisitionStep, ...],
) -> InterventionSignals:
    """Build conservative risk signals from already-available metadata.

    Missing verification remains unknown. This historical logits-only path
    cannot certify joint free answers. The opt-in medcave.run_case path generates
    actual candidates and re-evaluates the accumulated evidence state.
    """

    signal = request.domain_signals.get(card.expert_id)
    pre_ood = signal.ood_score if signal is not None else None
    source_reliability = (float(card.source_reliability_lcb)
                          if card.qualification_artifact else None)
    semantic_valid = evidence.provenance.get("semantic_bridge_validated")
    coverage = evidence.provenance.get("coverage") if semantic_valid is True else None
    if request.expert_queries and semantic_valid is not True:
        coverage = 0.0

    conflict = evidence.provenance.get("conflict")
    instability = evidence.provenance.get("instability")
    visual_consistency = evidence.provenance.get("visual_consistency")

    # Independent corroboration may be supplied by a verifier.  A compatible
    # second expert can reduce conflict, but agreement is never inferred merely
    # from having multiple calls.
    if history and "cross_expert_conflict" in evidence.provenance:
        conflict = max(conflict or 0, float(evidence.provenance["cross_expert_conflict"]))

    return InterventionSignals(
        applicability=1.0,
        source_reliability=source_reliability,
        pre_ood=pre_ood,
        post_ood=evidence.ood_score if evidence.provenance.get("post_ood_artifact") else None,
        conflict=conflict,
        instability=instability,
        coverage=coverage,
        visual_consistency=visual_consistency,
        expert_confidence=None,  # Native confidence is not a calibrated correctness probability.
        metadata={
            "expert_id": card.expert_id,
            "history_length": len(history),
            "raw_confidence": evidence.confidence,
        },
    )


class SequentialSpecialistAgent:
    """Training-free ACCEPT/ACQUIRE/FALLBACK loop over registered specialists.

    The language model is not asked to choose a checkpoint by name.  Candidate
    experts are filtered deterministically by modality and capability, ranked
    using source-only qualification/cost signals, and then passed through the
    intervention-risk controller after each observation.
    """

    def __init__(
        self,
        risk_controller: InterventionRiskController,
        *,
        trust_calibrator: DomainTrustCalibrator | None = None,
        config: SequentialAgentConfig | None = None,
        signal_builder: SignalBuilder | None = None,
    ) -> None:
        self.risk_controller = risk_controller
        self.trust_calibrator = trust_calibrator or DomainTrustCalibrator()
        self.config = config or SequentialAgentConfig()
        self.signal_builder = signal_builder or default_signal_builder

    def _rank_candidates(
        self,
        request: ClaimRequest,
        pool: LazyExpertPool,
        used: set[str],
    ) -> list[tuple[float, str]]:
        required = set(request.required_capabilities)
        ranked: list[tuple[float, str]] = []
        for expert_id, card in pool.cards.items():
            if expert_id in used:
                continue
            if not card.checkpoint_fingerprint:
                continue
            if any(pool.cards[k].checkpoint_fingerprint == card.checkpoint_fingerprint for k in used):
                continue
            if request.modality not in card.modalities and "*" not in card.modalities:
                continue
            overlap = required.intersection(card.capabilities)
            if not overlap or not card.validation_domain_scores:
                continue
            signal = request.domain_signals.get(expert_id)
            if signal is None:
                continue
            capability_match = len(overlap) / max(len(required), 1)
            trust = self.trust_calibrator.score(card, signal).score
            value = capability_match * trust * card.expected_gain
            utility = value - self.config.cost_weight * card.latency_ms / 1000.0
            if np.isfinite(utility) and utility > 0:
                ranked.append((float(utility), expert_id))
        ranked.sort(reverse=True)
        return ranked

    @staticmethod
    def _normalized_delta(request: ClaimRequest, evidence: NativeEvidence) -> np.ndarray:
        raw = np.asarray(
            [float(evidence.concept_scores.get(concept, 0.0)) for concept in request.concepts],
            dtype=float,
        )
        centered = raw - float(np.mean(raw))
        norm = float(np.linalg.norm(centered))
        if norm <= 1e-12:
            return np.zeros_like(centered)
        return centered / norm

    def _guided_logits(
        self,
        request: ClaimRequest,
        evidence: NativeEvidence,
    ) -> tuple[float, ...]:
        base = np.asarray(request.base_logits, dtype=float)
        direction = self._normalized_delta(request, evidence)
        delta = self.config.guidance_strength * direction
        norm = float(np.linalg.norm(delta))
        if norm > self.config.max_bias_norm:
            delta *= self.config.max_bias_norm / norm
        return tuple(float(value) for value in base + delta)

    def run(self, request: ClaimRequest, pool: LazyExpertPool) -> SequentialAgentTrace:
        base = tuple(float(value) for value in request.base_logits)
        used: set[str] = set()
        steps: list[AcquisitionStep] = []

        for step_index in range(self.config.max_expert_calls):
            ranked = self._rank_candidates(request, pool, used)
            if not ranked:
                return SequentialAgentTrace(
                    sample_id=request.sample_id,
                    claim_id=request.claim_id,
                    action=InterventionAction.FALLBACK,
                    selected_expert=None,
                    base_logits=base,
                    guided_logits=base,
                    steps=tuple(steps),
                    reason="no-qualified-independent-expert",
                )

            utility, expert_id = ranked[0]
            used.add(expert_id)
            card = pool.cards[expert_id]
            try:
                evidence, cache_hit = pool.get(expert_id, request)
            except (RuntimeError, ValueError, TypeError, OSError, KeyError, ArithmeticError) as exc:
                return SequentialAgentTrace(request.sample_id, request.claim_id,
                    InterventionAction.FALLBACK, None, base, base, tuple(steps),
                    "tool-failure", (type(exc).__name__,))

            # Capability mismatch is a hard structural failure and must not be
            # rescued by confidence, source performance or visual saliency.
            if (evidence.capability not in request.required_capabilities
                    or evidence.capability not in card.capabilities
                    or evidence.expert_id != expert_id
                    or evidence.provenance.get("semantic_bridge_validated") is not True):
                decision = self.risk_controller.decide(
                    InterventionSignals(applicability=0.0)
                )
            else:
                decision = self.risk_controller.decide(
                    self.signal_builder(request, card, evidence, tuple(steps))
                )

            steps.append(
                AcquisitionStep(
                    step_index=step_index,
                    expert_id=expert_id,
                    capability=evidence.capability,
                    utility=utility,
                    evidence=evidence,
                    risk_decision=decision,
                    cache_hit=cache_hit,
                )
            )

            if decision.action is InterventionAction.ACCEPT:
                if steps[:-1]:
                    # This historical logits-only path has no joint semantic verifier.
                    # The free-generation path below re-evaluates the accumulated state.
                    return SequentialAgentTrace(request.sample_id, request.claim_id,
                        InterventionAction.FALLBACK, None, base, base, tuple(steps),
                        "joint-verification-unavailable")
                return SequentialAgentTrace(
                    sample_id=request.sample_id,
                    claim_id=request.claim_id,
                    action=InterventionAction.ACCEPT,
                    selected_expert=expert_id,
                    base_logits=base,
                    guided_logits=self._guided_logits(request, evidence),
                    steps=tuple(steps),
                    reason=decision.reason,
                )

            if decision.action is InterventionAction.FALLBACK:
                return SequentialAgentTrace(
                    sample_id=request.sample_id,
                    claim_id=request.claim_id,
                    action=InterventionAction.FALLBACK,
                    selected_expert=None,
                    base_logits=base,
                    guided_logits=base,
                    steps=tuple(steps),
                    reason=decision.reason,
                )

            # ACQUIRE continues to an independent specialist.  If the budget is
            # exhausted, fail back to the untouched generalist rather than force
            # an uncertain intervention into the answer.

        return SequentialAgentTrace(
            sample_id=request.sample_id,
            claim_id=request.claim_id,
            action=InterventionAction.FALLBACK,
            selected_expert=None,
            base_logits=base,
            guided_logits=base,
            steps=tuple(steps),
            reason="acquisition-budget-exhausted",
        )
