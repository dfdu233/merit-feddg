from __future__ import annotations

from merit_feddg.intervention_risk import (
    CalibrationRecord,
    InterventionAction,
    InterventionRiskController,
    InterventionRiskScorer,
    InterventionSignals,
    RiskThresholds,
    fit_source_risk_thresholds,
)
from merit_feddg.med_defer import (
    ClaimRequest,
    DomainSignal,
    ExpertCard,
    LazyExpertPool,
    NativeEvidence,
)
from merit_feddg.sequential_agent import SequentialAgentConfig, SequentialSpecialistAgent


def _thresholds(accept: float = 0.10, acquire: float = 0.30) -> RiskThresholds:
    return RiskThresholds(
        accept_max_risk=accept,
        acquire_max_risk=acquire,
        accept_empirical_harm=0.0,
        accept_harm_upper_bound=0.05,
        acquire_empirical_harm=0.05,
        acquire_harm_upper_bound=0.20,
        calibration_size=100,
        acceptance_disabled=False,  # Explicit synthetic controller certificate fixture.
    )


def _observed(**kwargs):
    return InterventionSignals(**{"applicability": 1.0, "source_reliability": 1.0,
        "pre_ood": 0.0, "post_ood": 0.0, "conflict": 0.0, "instability": 0.0, "coverage": 1.0,
        "visual_consistency": 1.0, "expert_confidence": 1.0} | kwargs)


def test_max_risk_cannot_be_averaged_away() -> None:
    scorer = InterventionRiskScorer()
    result = scorer.score(
        InterventionSignals(
            applicability=0.0,
            source_reliability=1.0,
            pre_ood=0.0,
            post_ood=0.0,
            conflict=0.0,
            instability=0.0,
            coverage=1.0,
            visual_consistency=1.0,
            expert_confidence=1.0,
        )
    )
    assert result.score == 1.0
    assert result.dominant_component == "inapplicability"


def test_controller_has_accept_acquire_fallback_regions() -> None:
    controller = InterventionRiskController(_thresholds())

    assert (
        controller.decide(_observed(source_reliability=0.95)).action
        is InterventionAction.ACCEPT
    )
    assert (
        controller.decide(_observed(source_reliability=0.80)).action
        is InterventionAction.ACQUIRE
    )
    assert (
        controller.decide(_observed(source_reliability=0.40)).action
        is InterventionAction.FALLBACK
    )


def test_source_calibration_fails_closed_before_enough_support() -> None:
    records = [CalibrationRecord(score=0.05, harmful=False) for _ in range(10)]
    thresholds = fit_source_risk_thresholds(records, minimum_support=20)
    assert thresholds.accept_max_risk == 0.0
    assert thresholds.acquire_max_risk == 0.0


def test_source_calibration_separates_low_risk_safe_from_high_risk_harm() -> None:
    records = [CalibrationRecord(score=0.10, harmful=False) for _ in range(100)]
    records += [CalibrationRecord(score=0.80, harmful=True) for _ in range(100)]
    thresholds = fit_source_risk_thresholds(
        records,
        accept_target_harm=0.05,
        acquire_target_harm=0.20,
        minimum_support=20,
    )
    assert thresholds.accept_max_risk == 0.10
    assert thresholds.acquire_max_risk == 0.10
    assert thresholds.accept_harm_upper_bound <= 0.05


def _request() -> ClaimRequest:
    return ClaimRequest(
        sample_id="sample-1",
        claim_id="claim-1",
        modality="xray",
        required_capabilities=("classification",),
        concepts=("negative", "positive"),
        base_logits=(1.0, 0.0),
        uncertainty=0.7,
        router_probs={"xray": 1.0},
        domain_signals={
            "expert-risky": DomainSignal(ood_score=0.0, image_quality=1.0),
            "expert-safe": DomainSignal(ood_score=0.0, image_quality=1.0),
        },
    )


def _evidence(expert_id: str, confidence: float) -> NativeEvidence:
    return NativeEvidence(
        expert_id=expert_id,
        capability="classification",
        concept_scores={"negative": -1.0, "positive": 1.0},
        confidence=confidence,
        provenance={"coverage": 1.0, "semantic_bridge_validated": True},
    )


def test_legacy_logits_agent_cannot_claim_joint_free_answer_verification() -> None:
    pool = LazyExpertPool()
    # Higher expected gain deliberately makes the riskier expert get queried first.
    pool.register(
        ExpertCard(
            expert_id="expert-risky",
            checkpoint_fingerprint="model-A",
            modalities=("xray",),
            capabilities=("classification",),
            source_reliability_lcb=0.80,
            validation_domain_scores=(0.80, 0.82),
            expected_gain=2.0,
            latency_ms=1.0,
        ),
        lambda request: _evidence("expert-risky", 0.90),
    )
    pool.register(
        ExpertCard(
            expert_id="expert-safe",
            checkpoint_fingerprint="model-B",
            modalities=("xray",),
            capabilities=("classification",),
            source_reliability_lcb=0.95,
            validation_domain_scores=(0.95, 0.96),
            expected_gain=1.0,
            latency_ms=1.0,
        ),
        lambda request: _evidence("expert-safe", 0.98),
    )

    agent = SequentialSpecialistAgent(
        InterventionRiskController(_thresholds()),
        config=SequentialAgentConfig(max_expert_calls=2),
        signal_builder=lambda request, card, evidence, history:
            _observed(source_reliability=card.source_reliability_lcb),
    )
    trace = agent.run(_request(), pool)

    assert len(trace.steps) == 2
    assert trace.steps[0].expert_id == "expert-risky"
    assert trace.steps[0].risk_decision.action is InterventionAction.ACQUIRE
    assert trace.steps[1].expert_id == "expert-safe"
    assert trace.steps[1].risk_decision.action is InterventionAction.ACCEPT
    assert trace.action is InterventionAction.FALLBACK
    assert trace.reason == "joint-verification-unavailable"
    assert trace.guided_logits == trace.base_logits


def test_sequential_agent_preserves_baseline_when_budget_expires() -> None:
    pool = LazyExpertPool()
    pool.register(
        ExpertCard(
            expert_id="expert-risky",
            checkpoint_fingerprint="model-A",
            modalities=("xray",),
            capabilities=("classification",),
            source_reliability_lcb=0.80,
            validation_domain_scores=(0.80, 0.82),
            expected_gain=1.0,
            latency_ms=1.0,
        ),
        lambda request: _evidence("expert-risky", 0.90),
    )
    request = _request()
    request = ClaimRequest(
        **{
            **request.__dict__,
            "domain_signals": {
                "expert-risky": DomainSignal(ood_score=0.0, image_quality=1.0)
            },
        }
    )

    agent = SequentialSpecialistAgent(
        InterventionRiskController(_thresholds()),
        config=SequentialAgentConfig(max_expert_calls=1),
    )
    trace = agent.run(request, pool)

    assert trace.action is InterventionAction.FALLBACK
    assert trace.guided_logits == trace.base_logits
    assert trace.reason == "acquisition-budget-exhausted"
