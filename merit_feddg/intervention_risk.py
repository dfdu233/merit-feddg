from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


def _clip01(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("risk signals must be finite")
    if not 0 <= value <= 1:
        raise ValueError("risk signals must be in [0, 1]")
    return value


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

    applicability: float | None = None
    source_reliability: float | None = None
    pre_ood: float | None = None
    post_ood: float | None = None
    conflict: float | None = None
    instability: float | None = None
    coverage: float | None = None
    visual_consistency: float | None = None
    expert_confidence: float | None = None
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
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _clip01(getattr(self, name)))

    def risk_components(self) -> dict[str, float | None]:
        values = {
            "inapplicability": None if self.applicability is None else 1-self.applicability,
            "source_unreliability": None if self.source_reliability is None else 1-self.source_reliability,
            "pre_ood": self.pre_ood,
            "post_ood": self.post_ood,
            "conflict": self.conflict,
            "instability": self.instability,
            "missing_coverage": None if self.coverage is None else 1-self.coverage,
            "visual_inconsistency": None if self.visual_consistency is None else 1-self.visual_consistency,
            "expert_uncertainty": None if self.expert_confidence is None else 1-self.expert_confidence,
        }
        return values


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
            name: (1.0 if all_components[name] is None else float(all_components[name]))
            for name in self.config.enabled_components
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
    accept_empirical_harm: float | None
    accept_harm_upper_bound: float
    acquire_empirical_harm: float | None
    acquire_harm_upper_bound: float
    calibration_size: int
    method: str = "source-dev-wilson-selection-only"
    acceptance_disabled: bool = True

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
        return 0.0, None, 1.0
    candidates = sorted({0.0, *(record.score for record in records)})
    best = (0.0, None, 1.0)
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
    """Select source-dev thresholds; this function always disables ACCEPT.

    No model parameters are fitted. Independent full-trajectory source-cal
    validation must follow policy freezing; searching Wilson bounds on development
    records is not a deployment or conformal guarantee.
    """

    if not 0.0 < accept_target_harm <= acquire_target_harm < 1.0:
        raise ValueError(
            "require 0 < accept_target_harm <= acquire_target_harm < 1"
        )
    if not math.isfinite(confidence_z) or confidence_z <= 0.0:
        raise ValueError("confidence_z must be positive")
    if type(minimum_support) is not int or minimum_support < 1:
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
        acceptance_disabled=True,  # Development selection is never a deployment certificate.
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
        if signals.applicability == 0 or signals.metadata.get("hard_failures"):
            return RiskDecision(InterventionAction.FALLBACK, risk, "structural-mismatch")
        missing = [k for k in self.scorer.config.enabled_components
                   if signals.risk_components()[k] is None]
        if missing:
            return RiskDecision(InterventionAction.ACQUIRE, risk, "unknown:" + ",".join(missing))
        if not self.thresholds.acceptance_disabled and risk.score <= self.thresholds.accept_max_risk:
            action = InterventionAction.ACCEPT
            reason = "risk-within-accept-envelope"
        elif risk.score <= self.thresholds.acquire_max_risk:
            action = InterventionAction.ACQUIRE
            reason = "risk-requires-additional-model-source"
        else:
            action = InterventionAction.FALLBACK
            reason = f"risk-too-high:{risk.dominant_component}"
        return RiskDecision(action=action, risk=risk, reason=reason)
