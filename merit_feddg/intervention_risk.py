from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping

import numpy as np


def _clip01(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("risk signals must be finite")
    return float(np.clip(value, 0.0, 1.0))


class InterventionAction(str, Enum):
    """Action taken after evaluating a specialist intervention."""

    ACCEPT = "accept"
    ACQUIRE = "acquire"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class InterventionSignals:
    """Label-free signals describing one proposed specialist intervention.

    All fields use a [0, 1] convention.  Positive quantities such as
    ``applicability`` and ``source_reliability`` are converted to risk by taking
    their complement; quantities already expressing risk, such as ``pre_ood``
    and ``conflict``, are used directly.

    The structure intentionally keeps the axes separate.  A high-confidence
    expert is not allowed to cancel an anatomy/capability mismatch or severe
    domain shift merely because another scalar is favorable.
    """

    applicability: float = 1.0
    source_reliability: float = 1.0
    pre_ood: float = 0.0
    post_ood: float = 0.0
    conflict: float = 0.0
    instability: float = 0.0
    coverage: float = 1.0
    visual_consistency: float = 1.0
    expert_confidence: float = 1.0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "applicability",
            "source_reliability",
            "pre_ood",
            "post_ood",
            "conflict",
            "instability",
            "coverage",
            "visual_consistency",
            "expert_confidence",
        ):
            object.__setattr__(self, name, _clip01(getattr(self, name)))

    def risk_components(self) -> dict[str, float]:
        return {
            "inapplicability": 1.0 - self.applicability,
            "source_unreliability": 1.0 - self.source_reliability,
            "pre_ood": self.pre_ood,
            "post_ood": self.post_ood,
            "conflict": self.conflict,
            "instability": self.instability,
            "missing_coverage": 1.0 - self.coverage,
            "visual_inconsistency": 1.0 - self.visual_consistency,
            "expert_uncertainty": 1.0 - self.expert_confidence,
        }


@dataclass(frozen=True)
class RiskScore:
    score: float
    dominant_component: str
    components: Mapping[str, float]


@dataclass(frozen=True)
class InterventionRiskConfig:
    """Training-free aggregation rule for intervention nonconformity.

    ``max`` is the conservative default: one severe risk axis vetoes an
    intervention instead of being averaged away by several benign axes.
    ``top2_mean`` is provided only as a sensitivity ablation.
    """

    aggregation: str = "max"
    enabled_components: tuple[str, ...] = (
        "inapplicability",
        "source_unreliability",
        "pre_ood",
        "post_ood",
        "conflict",
        "instability",
        "missing_coverage",
        "visual_inconsistency",
        "expert_uncertainty",
    )

    def __post_init__(self) -> None:
        if self.aggregation not in {"max", "top2_mean"}:
            raise ValueError("aggregation must be 'max' or 'top2_mean'")
        if not self.enabled_components:
            raise ValueError("at least one risk component must be enabled")


class InterventionRiskScorer:
    """Compute a deterministic nonconformity score without fitting parameters."""

    def __init__(self, config: InterventionRiskConfig | None = None) -> None:
        self.config = config or InterventionRiskConfig()

    def score(self, signals: InterventionSignals) -> RiskScore:
        all_components = signals.risk_components()
        unknown = set(self.config.enabled_components) - set(all_components)
        if unknown:
            raise ValueError(f"unknown risk components: {sorted(unknown)}")
        components = {
            name: float(all_components[name]) for name in self.config.enabled_components
        }
        ordered = sorted(components.items(), key=lambda item: item[1], reverse=True)
        dominant_name, dominant_value = ordered[0]
        if self.config.aggregation == "max" or len(ordered) == 1:
            score = dominant_value
        else:
            score = float(np.mean([value for _, value in ordered[:2]]))
        return RiskScore(
            score=_clip01(score),
            dominant_component=dominant_name,
            components=components,
        )


@dataclass(frozen=True)
class CalibrationRecord:
    """One frozen source-domain intervention outcome.

    ``harmful`` is evaluated only on source calibration data after both the base
    and guided predictions have been frozen.  Target labels must never be used
    to construct these records.
    """

    score: float
    harmful: bool
    group_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", _clip01(self.score))


@dataclass(frozen=True)
class RiskThresholds:
    accept_max_risk: float
    acquire_max_risk: float
    accept_empirical_harm: float
    accept_harm_upper_bound: float
    acquire_empirical_harm: float
    acquire_harm_upper_bound: float
    calibration_size: int
    method: str = "source-wilson-risk-control"

    def __post_init__(self) -> None:
        if not 0.0 <= self.accept_max_risk <= self.acquire_max_risk <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= accept <= acquire <= 1")


def _wilson_upper(harmful: int, total: int, z: float) -> float:
    if total <= 0:
        return 1.0
    p = harmful / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = p + z2 / (2.0 * total)
    radius = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
    return _clip01((center + radius) / denom)


def _select_threshold(
    records: list[CalibrationRecord],
    *,
    target_harm: float,
    confidence_z: float,
    minimum_support: int,
) -> tuple[float, float, float]:
    """Choose the widest source-supported threshold satisfying a harm bound.

    This is deliberately a small-sample source calibration rule rather than a
    learned gate.  It scans frozen risk scores and accepts the largest threshold
    whose one-sided Wilson upper confidence bound on harmful interventions does
    not exceed ``target_harm``.  If no threshold has enough support, it fails
    closed at zero risk.
    """

    if not records:
        return 0.0, 0.0, 1.0
    candidates = sorted({0.0, *(record.score for record in records)})
    best = (0.0, 0.0, 1.0)
    for threshold in candidates:
        selected = [record for record in records if record.score <= threshold]
        if len(selected) < minimum_support:
            continue
        harmful = sum(record.harmful for record in selected)
        empirical = harmful / len(selected)
        upper = _wilson_upper(harmful, len(selected), confidence_z)
        if upper <= target_harm:
            best = (threshold, empirical, upper)
    return best


def fit_source_risk_thresholds(
    records: Iterable[CalibrationRecord],
    *,
    accept_target_harm: float = 0.05,
    acquire_target_harm: float = 0.20,
    confidence_z: float = 1.645,
    minimum_support: int = 20,
) -> RiskThresholds:
    """Calibrate ACCEPT/ACQUIRE/FALLBACK thresholds on source interventions.

    No model parameters are fitted.  The complete downstream agent policy should
    be frozen before collecting ``records`` so that calibration evaluates the
    actual intervention policy rather than an isolated component.
    """

    if not 0.0 < accept_target_harm <= acquire_target_harm < 1.0:
        raise ValueError(
            "require 0 < accept_target_harm <= acquire_target_harm < 1"
        )
    if confidence_z <= 0.0:
        raise ValueError("confidence_z must be positive")
    if minimum_support < 1:
        raise ValueError("minimum_support must be at least one")

    frozen = list(records)
    accept_t, accept_emp, accept_upper = _select_threshold(
        frozen,
        target_harm=accept_target_harm,
        confidence_z=confidence_z,
        minimum_support=minimum_support,
    )
    acquire_t, acquire_emp, acquire_upper = _select_threshold(
        frozen,
        target_harm=acquire_target_harm,
        confidence_z=confidence_z,
        minimum_support=minimum_support,
    )
    acquire_t = max(acquire_t, accept_t)
    return RiskThresholds(
        accept_max_risk=accept_t,
        acquire_max_risk=acquire_t,
        accept_empirical_harm=accept_emp,
        accept_harm_upper_bound=accept_upper,
        acquire_empirical_harm=acquire_emp,
        acquire_harm_upper_bound=acquire_upper,
        calibration_size=len(frozen),
    )


@dataclass(frozen=True)
class RiskDecision:
    action: InterventionAction
    risk: RiskScore
    reason: str


class InterventionRiskController:
    """Map a proposed intervention to ACCEPT, ACQUIRE or FALLBACK."""

    def __init__(
        self,
        thresholds: RiskThresholds,
        scorer: InterventionRiskScorer | None = None,
    ) -> None:
        self.thresholds = thresholds
        self.scorer = scorer or InterventionRiskScorer()

    def decide(self, signals: InterventionSignals) -> RiskDecision:
        risk = self.scorer.score(signals)
        if risk.score <= self.thresholds.accept_max_risk:
            action = InterventionAction.ACCEPT
            reason = "risk-within-accept-envelope"
        elif risk.score <= self.thresholds.acquire_max_risk:
            action = InterventionAction.ACQUIRE
            reason = "risk-requires-independent-evidence"
        else:
            action = InterventionAction.FALLBACK
            reason = f"risk-too-high:{risk.dominant_component}"
        return RiskDecision(action=action, risk=risk, reason=reason)
