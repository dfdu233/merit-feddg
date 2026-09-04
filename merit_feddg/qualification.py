"""Source-only expert qualification and continuous-feature OOD assessment.

This module deliberately does not learn a binary defer/error predictor.  An
expert is qualified with its real multi-class source-validation probabilities,
and distribution shift is measured from frozen continuous representations.
The public fitting API only accepts explicitly named ``source_*`` arrays; target
labels are neither accepted nor stored.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

OODStage = Literal["pre_call", "post_call"]


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _require_nonempty(value: str, field_name: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned


def _contains_target_labels(payload: Any) -> bool:
    """Detect accidental target-ground-truth material in provenance metadata."""

    forbidden = {"target_label", "target_labels", "target_ground_truth", "target_y"}
    if isinstance(payload, Mapping):
        return any(
            str(key).strip().lower() in forbidden or _contains_target_labels(value)
            for key, value in payload.items()
        )
    if isinstance(payload, (list, tuple)):
        return any(_contains_target_labels(value) for value in payload)
    return False


def fingerprint_payload(payload: bytes | str | Mapping[str, Any]) -> str:
    """Return a reproducible SHA-256 fingerprint for a model/adapter manifest."""

    if isinstance(payload, bytes):
        encoded = payload
    elif isinstance(payload, str):
        encoded = payload.encode("utf-8")
    else:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class ExpertIdentity:
    """Runtime identity that must exactly match a qualification artifact."""

    expert_id: str
    model_fingerprint: str
    adapter_fingerprint: str
    modality: str
    capability: str
    task: str

    def __post_init__(self) -> None:
        for name in (
            "expert_id",
            "model_fingerprint",
            "adapter_fingerprint",
            "modality",
            "capability",
            "task",
        ):
            _require_nonempty(getattr(self, name), name)

    def to_dict(self) -> dict[str, str]:
        return {
            "expert_id": self.expert_id,
            "model_fingerprint": self.model_fingerprint,
            "adapter_fingerprint": self.adapter_fingerprint,
            "modality": self.modality,
            "capability": self.capability,
            "task": self.task,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExpertIdentity:
        return cls(**{key: str(payload[key]) for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class DomainMetrics:
    """True multi-class metrics for one source validation domain."""

    domain: str
    sample_count: int
    class_support: tuple[int, ...]
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    negative_log_likelihood: float
    multiclass_brier: float
    expected_calibration_error: float
    mean_predictive_entropy: float
    per_class_f1: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_nonempty(self.domain, "domain")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if sum(self.class_support) != self.sample_count:
            raise ValueError("class_support must sum to sample_count")
        if len(self.class_support) < 2 or len(self.per_class_f1) != len(self.class_support):
            raise ValueError("multi-class metrics require aligned class vectors")
        for name in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "multiclass_brier",
            "expected_calibration_error",
            "mean_predictive_entropy",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not math.isfinite(self.negative_log_likelihood) or self.negative_log_likelihood < 0:
            raise ValueError("negative_log_likelihood must be finite and non-negative")
        if any(not 0.0 <= value <= 1.0 for value in self.per_class_f1):
            raise ValueError("per_class_f1 values must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "sample_count": self.sample_count,
            "class_support": list(self.class_support),
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "negative_log_likelihood": self.negative_log_likelihood,
            "multiclass_brier": self.multiclass_brier,
            "expected_calibration_error": self.expected_calibration_error,
            "mean_predictive_entropy": self.mean_predictive_entropy,
            "per_class_f1": list(self.per_class_f1),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DomainMetrics:
        return cls(
            domain=str(payload["domain"]),
            sample_count=int(payload["sample_count"]),
            class_support=tuple(int(value) for value in payload["class_support"]),
            accuracy=float(payload["accuracy"]),
            balanced_accuracy=float(payload["balanced_accuracy"]),
            macro_f1=float(payload["macro_f1"]),
            negative_log_likelihood=float(payload["negative_log_likelihood"]),
            multiclass_brier=float(payload["multiclass_brier"]),
            expected_calibration_error=float(payload["expected_calibration_error"]),
            mean_predictive_entropy=float(payload["mean_predictive_entropy"]),
            per_class_f1=tuple(float(value) for value in payload["per_class_f1"]),
        )


@dataclass(frozen=True)
class LatencyMetrics:
    sample_count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("at least one source latency observation is required")
        if any(
            not math.isfinite(value) or value < 0
            for value in (self.mean_ms, self.p50_ms, self.p95_ms)
        ):
            raise ValueError("latency values must be finite and non-negative")

    @classmethod
    def fit(cls, source_latency_ms: Sequence[float]) -> LatencyMetrics:
        values = np.asarray(source_latency_ms, dtype=float)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("source_latency_ms must be a non-empty finite vector")
        if np.any(values < 0):
            raise ValueError("source_latency_ms cannot contain negative values")
        return cls(
            sample_count=int(values.size),
            mean_ms=float(np.mean(values)),
            p50_ms=float(np.quantile(values, 0.50)),
            p95_ms=float(np.quantile(values, 0.95)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LatencyMetrics:
        return cls(
            sample_count=int(payload["sample_count"]),
            mean_ms=float(payload["mean_ms"]),
            p50_ms=float(payload["p50_ms"]),
            p95_ms=float(payload["p95_ms"]),
        )


@dataclass(frozen=True)
class DiagonalGaussianReference:
    """Shrinkage-diagonal reference and its empirical source distances."""

    class_index: int | None
    sample_count: int
    mean: tuple[float, ...]
    variance: tuple[float, ...]
    calibration_distances: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("reference sample_count must be positive")
        if not self.mean or len(self.variance) != len(self.mean):
            raise ValueError("reference mean and variance dimensions must agree")
        if any(not math.isfinite(value) for value in self.mean):
            raise ValueError("reference means must be finite")
        if any(not math.isfinite(value) or value <= 0 for value in self.variance):
            raise ValueError("reference variances must be finite and positive")
        if len(self.calibration_distances) != self.sample_count:
            raise ValueError("one calibration distance is required per reference sample")
        if any(not math.isfinite(value) or value < 0 for value in self.calibration_distances):
            raise ValueError("calibration distances must be finite and non-negative")
        if tuple(sorted(self.calibration_distances)) != self.calibration_distances:
            raise ValueError("calibration distances must be sorted")

    def squared_mahalanobis(self, feature: np.ndarray) -> float:
        mean = np.asarray(self.mean, dtype=float)
        variance = np.asarray(self.variance, dtype=float)
        return float(np.sum(np.square(feature - mean) / variance))

    def empirical_ood(self, distance: float) -> float:
        calibration = np.asarray(self.calibration_distances, dtype=float)
        # Empirical upper percentile: values beyond every source observation map
        # to 1.0, while values below the source range map to 0.0.
        rank = int(np.searchsorted(calibration, float(distance), side="left"))
        return _clip01(rank / max(calibration.size, 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_index": self.class_index,
            "sample_count": self.sample_count,
            "mean": list(self.mean),
            "variance": list(self.variance),
            "calibration_distances": list(self.calibration_distances),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DiagonalGaussianReference:
        class_index = payload.get("class_index")
        return cls(
            class_index=None if class_index is None else int(class_index),
            sample_count=int(payload["sample_count"]),
            mean=tuple(float(value) for value in payload["mean"]),
            variance=tuple(float(value) for value in payload["variance"]),
            calibration_distances=tuple(float(value) for value in payload["calibration_distances"]),
        )


@dataclass(frozen=True)
class FeatureReference:
    """Frozen-feature OOD model fitted only on source observations."""

    stage: OODStage
    encoder_fingerprint: str
    dimension: int
    shrinkage: float
    minimum_variance: float
    global_reference: DiagonalGaussianReference
    class_references: tuple[DiagonalGaussianReference, ...] = ()
    calibration_method: str = "source-only-cross-fit"

    def __post_init__(self) -> None:
        if self.stage not in ("pre_call", "post_call"):
            raise ValueError("stage must be pre_call or post_call")
        _require_nonempty(self.encoder_fingerprint, "encoder_fingerprint")
        if self.dimension <= 0 or len(self.global_reference.mean) != self.dimension:
            raise ValueError("feature dimension does not match the global reference")
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        if self.minimum_variance <= 0 or not math.isfinite(self.minimum_variance):
            raise ValueError("minimum_variance must be finite and positive")
        _require_nonempty(self.calibration_method, "calibration_method")
        seen: set[int] = set()
        for reference in self.class_references:
            if reference.class_index is None or reference.class_index in seen:
                raise ValueError("class references need unique class indices")
            if len(reference.mean) != self.dimension:
                raise ValueError("class reference has the wrong feature dimension")
            seen.add(reference.class_index)

    def score(
        self, feature: Sequence[float] | np.ndarray, class_index: int | None = None
    ) -> tuple[float, float, int | None, str]:
        vector = np.asarray(feature, dtype=float)
        if vector.shape != (self.dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"feature must be a finite vector of shape ({self.dimension},)")

        references = {reference.class_index: reference for reference in self.class_references}
        if class_index is not None and class_index in references:
            selected = references[class_index]
            reason = "class-conditional"
        elif class_index is not None:
            selected = self.global_reference
            reason = "global-fallback-unseen-class"
        elif self.class_references:
            selected = min(
                self.class_references,
                key=lambda reference: reference.squared_mahalanobis(vector),
            )
            reason = "nearest-class"
        else:
            selected = self.global_reference
            reason = "global"

        distance = selected.squared_mahalanobis(vector)
        return selected.empirical_ood(distance), distance, selected.class_index, reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "encoder_fingerprint": self.encoder_fingerprint,
            "dimension": self.dimension,
            "shrinkage": self.shrinkage,
            "minimum_variance": self.minimum_variance,
            "global_reference": self.global_reference.to_dict(),
            "class_references": [reference.to_dict() for reference in self.class_references],
            "calibration_method": self.calibration_method,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FeatureReference:
        return cls(
            stage=str(payload["stage"]),  # type: ignore[arg-type]
            encoder_fingerprint=str(payload["encoder_fingerprint"]),
            dimension=int(payload["dimension"]),
            shrinkage=float(payload["shrinkage"]),
            minimum_variance=float(payload["minimum_variance"]),
            global_reference=DiagonalGaussianReference.from_dict(payload["global_reference"]),
            class_references=tuple(
                DiagonalGaussianReference.from_dict(item)
                for item in payload.get("class_references", ())
            ),
            calibration_method=str(payload.get("calibration_method", "legacy-in-sample")),
        )


@dataclass(frozen=True)
class QualificationArtifact:
    """Versioned, source-only evidence that an expert is safe to consider."""

    schema_version: int
    identity: ExpertIdentity
    class_names: tuple[str, ...]
    aggregate_metrics: DomainMetrics
    per_domain_metrics: tuple[DomainMetrics, ...]
    performance_metric: str
    performance_lcb: float
    performance_cvar: float
    cvar_alpha: float
    latency: LatencyMetrics
    cheap_feature_reference: FeatureReference
    native_feature_reference: FeatureReference
    target_labels_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported qualification schema version")
        if len(self.class_names) < 2 or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must contain at least two unique classes")
        if any(not name.strip() for name in self.class_names):
            raise ValueError("class_names cannot contain empty values")
        if not self.per_domain_metrics:
            raise ValueError("at least one real source validation domain is required")
        if self.aggregate_metrics.sample_count != sum(
            metrics.sample_count for metrics in self.per_domain_metrics
        ):
            raise ValueError("aggregate and per-domain sample counts do not agree")
        if any(
            len(metrics.class_support) != len(self.class_names)
            for metrics in (self.aggregate_metrics, *self.per_domain_metrics)
        ):
            raise ValueError("metric class dimensions do not match class_names")
        if self.performance_metric not in {
            "macro_f1",
            "balanced_accuracy",
            "accuracy",
        }:
            raise ValueError("performance_metric is not supported")
        if not 0.0 <= self.performance_lcb <= 1.0:
            raise ValueError("performance_lcb must be in [0, 1]")
        if not 0.0 <= self.performance_cvar <= 1.0:
            raise ValueError("performance_cvar must be in [0, 1]")
        if not 0.0 < self.cvar_alpha <= 1.0:
            raise ValueError("cvar_alpha must be in (0, 1]")
        if self.cheap_feature_reference.stage != "pre_call":
            raise ValueError("cheap features must be a pre_call reference")
        if self.native_feature_reference.stage != "post_call":
            raise ValueError("native features must be a post_call reference")
        if self.target_labels_used:
            raise ValueError("qualification artifacts must never use target labels")
        if _contains_target_labels(self.metadata):
            raise ValueError("metadata must not contain target labels")
        try:
            json.dumps(self.metadata, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("metadata must be JSON serializable") from error

    @property
    def source_domain_count(self) -> int:
        return len(self.per_domain_metrics)

    @property
    def source_sample_count(self) -> int:
        return self.aggregate_metrics.sample_count

    @property
    def expert_id(self) -> str:
        return self.identity.expert_id

    @property
    def model_fingerprint(self) -> str:
        return self.identity.model_fingerprint

    @property
    def adapter_fingerprint(self) -> str:
        return self.identity.adapter_fingerprint

    @property
    def modality(self) -> str:
        return self.identity.modality

    @property
    def capability(self) -> str:
        return self.identity.capability

    @property
    def task(self) -> str:
        return self.identity.task

    @property
    def artifact_fingerprint(self) -> str:
        return fingerprint_payload(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "class_names": list(self.class_names),
            "aggregate_metrics": self.aggregate_metrics.to_dict(),
            "per_domain_metrics": [metrics.to_dict() for metrics in self.per_domain_metrics],
            "performance_metric": self.performance_metric,
            "performance_lcb": self.performance_lcb,
            "performance_cvar": self.performance_cvar,
            "cvar_alpha": self.cvar_alpha,
            "latency": self.latency.to_dict(),
            "cheap_feature_reference": self.cheap_feature_reference.to_dict(),
            "native_feature_reference": self.native_feature_reference.to_dict(),
            "target_labels_used": False,
            "metadata": dict(self.metadata),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> QualificationArtifact:
        if payload.get("target_labels_used", False) or _contains_target_labels(payload):
            raise ValueError("target-label qualification artifacts are forbidden")
        return cls(
            schema_version=int(payload["schema_version"]),
            identity=ExpertIdentity.from_dict(payload["identity"]),
            class_names=tuple(str(value) for value in payload["class_names"]),
            aggregate_metrics=DomainMetrics.from_dict(payload["aggregate_metrics"]),
            per_domain_metrics=tuple(
                DomainMetrics.from_dict(item) for item in payload["per_domain_metrics"]
            ),
            performance_metric=str(payload["performance_metric"]),
            performance_lcb=float(payload["performance_lcb"]),
            performance_cvar=float(payload["performance_cvar"]),
            cvar_alpha=float(payload["cvar_alpha"]),
            latency=LatencyMetrics.from_dict(payload["latency"]),
            cheap_feature_reference=FeatureReference.from_dict(payload["cheap_feature_reference"]),
            native_feature_reference=FeatureReference.from_dict(
                payload["native_feature_reference"]
            ),
            target_labels_used=False,
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, payload: str) -> QualificationArtifact:
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise TypeError("qualification JSON must contain an object")
        return cls.from_dict(parsed)


@dataclass(frozen=True)
class QualificationPolicy:
    minimum_source_domains: int = 2
    minimum_samples_per_domain: int = 1
    minimum_samples_per_class: int = 1
    minimum_performance_lcb: float = 0.05
    minimum_performance_cvar: float = 0.05
    maximum_p95_latency_ms: float | None = None

    def __post_init__(self) -> None:
        if (
            self.minimum_source_domains < 1
            or self.minimum_samples_per_domain < 1
            or self.minimum_samples_per_class < 1
        ):
            raise ValueError("minimum source counts must be positive")
        if not 0.0 <= self.minimum_performance_lcb <= 1.0:
            raise ValueError("minimum_performance_lcb must be in [0, 1]")
        if not 0.0 <= self.minimum_performance_cvar <= 1.0:
            raise ValueError("minimum_performance_cvar must be in [0, 1]")
        if self.maximum_p95_latency_ms is not None and self.maximum_p95_latency_ms < 0:
            raise ValueError("maximum_p95_latency_ms cannot be negative")


@dataclass(frozen=True)
class QualificationDecision:
    allowed: bool
    reason: str
    expert_id: str
    performance_lcb: float = 0.0
    performance_cvar: float = 0.0


@dataclass(frozen=True)
class OODAssessment:
    stage: OODStage
    allowed: bool
    score: float
    squared_distance: float | None
    selected_class: int | None
    reason: str

    def __post_init__(self) -> None:
        if self.stage not in ("pre_call", "post_call"):
            raise ValueError("invalid OOD stage")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("OOD score must be in [0, 1]")


class QualificationGate:
    """Fail-closed runtime validator and two-stage OOD scorer."""

    def __init__(self, policy: QualificationPolicy | None = None) -> None:
        self.policy = policy or QualificationPolicy()

    def authorize(
        self,
        runtime_identity: ExpertIdentity,
        artifact: QualificationArtifact | None,
    ) -> QualificationDecision:
        if artifact is None:
            return QualificationDecision(False, "missing-qualification", runtime_identity.expert_id)

        expected = runtime_identity.to_dict()
        observed = artifact.identity.to_dict()
        for field_name in expected:
            if expected[field_name] != observed[field_name]:
                return QualificationDecision(
                    False,
                    f"{field_name.replace('_', '-')}-mismatch",
                    runtime_identity.expert_id,
                    artifact.performance_lcb,
                    artifact.performance_cvar,
                )

        policy = self.policy
        if any(
            reference.calibration_method != "source-only-cross-fit"
            for reference in (
                artifact.cheap_feature_reference,
                artifact.native_feature_reference,
            )
        ):
            reason = "non-cross-fitted-ood-calibration"
        elif artifact.source_domain_count < policy.minimum_source_domains:
            reason = "insufficient-source-domains"
        elif min(item.sample_count for item in artifact.per_domain_metrics) < (
            policy.minimum_samples_per_domain
        ):
            reason = "insufficient-source-samples"
        elif min(artifact.aggregate_metrics.class_support) < policy.minimum_samples_per_class:
            # The class vocabulary is frozen before qualification.  A global
            # fallback for a class never observed often enough in the source
            # centers would manufacture trust rather than estimate it.
            reason = "insufficient-source-class-support"
        elif artifact.performance_lcb < policy.minimum_performance_lcb:
            reason = "performance-lcb-below-policy"
        elif artifact.performance_cvar < policy.minimum_performance_cvar:
            reason = "performance-cvar-below-policy"
        elif (
            policy.maximum_p95_latency_ms is not None
            and artifact.latency.p95_ms > policy.maximum_p95_latency_ms
        ):
            reason = "latency-above-policy"
        else:
            return QualificationDecision(
                True,
                "qualified",
                runtime_identity.expert_id,
                artifact.performance_lcb,
                artifact.performance_cvar,
            )
        return QualificationDecision(
            False,
            reason,
            runtime_identity.expert_id,
            artifact.performance_lcb,
            artifact.performance_cvar,
        )

    def _score_ood(
        self,
        stage: OODStage,
        runtime_identity: ExpertIdentity,
        artifact: QualificationArtifact | None,
        feature: Sequence[float] | np.ndarray,
        encoder_fingerprint: str,
        class_index: int | None,
    ) -> OODAssessment:
        decision = self.authorize(runtime_identity, artifact)
        if not decision.allowed or artifact is None:
            return OODAssessment(stage, False, 1.0, None, None, decision.reason)

        if class_index is not None:
            if not isinstance(class_index, (int, np.integer)) or not (
                0 <= int(class_index) < len(artifact.class_names)
            ):
                return OODAssessment(stage, False, 1.0, None, None, "class-outside-frozen-taxonomy")
            if (
                artifact.aggregate_metrics.class_support[int(class_index)]
                < self.policy.minimum_samples_per_class
            ):
                return OODAssessment(
                    stage,
                    False,
                    1.0,
                    None,
                    int(class_index),
                    "unqualified-source-class",
                )

        reference = (
            artifact.cheap_feature_reference
            if stage == "pre_call"
            else artifact.native_feature_reference
        )
        if reference.encoder_fingerprint != encoder_fingerprint:
            return OODAssessment(
                stage, False, 1.0, None, None, "feature-encoder-fingerprint-mismatch"
            )
        try:
            score, distance, selected_class, detail = reference.score(feature, class_index)
        except (TypeError, ValueError):
            return OODAssessment(stage, False, 1.0, None, None, "invalid-feature")
        return OODAssessment(
            stage,
            True,
            score,
            distance,
            selected_class,
            f"qualified-{detail}",
        )

    def pre_call_ood(
        self,
        runtime_identity: ExpertIdentity,
        artifact: QualificationArtifact | None,
        cheap_feature: Sequence[float] | np.ndarray,
        cheap_encoder_fingerprint: str,
        class_index: int | None = None,
    ) -> OODAssessment:
        """Cheap, shared-encoder OOD available before loading the expert."""

        return self._score_ood(
            "pre_call",
            runtime_identity,
            artifact,
            cheap_feature,
            cheap_encoder_fingerprint,
            class_index,
        )

    def post_call_native_ood(
        self,
        runtime_identity: ExpertIdentity,
        artifact: QualificationArtifact | None,
        native_feature: Sequence[float] | np.ndarray,
        native_encoder_fingerprint: str,
        class_index: int | None = None,
    ) -> OODAssessment:
        """Expert-native OOD evaluated only after sparse expert invocation."""

        return self._score_ood(
            "post_call",
            runtime_identity,
            artifact,
            native_feature,
            native_encoder_fingerprint,
            class_index,
        )


def _validate_multiclass_source_data(
    class_names: Sequence[str],
    source_labels: Sequence[int] | np.ndarray,
    source_probabilities: Sequence[Sequence[float]] | np.ndarray,
    source_domains: Sequence[str] | np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    names = tuple(str(name).strip() for name in class_names)
    if len(names) < 2 or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("class_names must contain at least two unique non-empty classes")
    labels = np.asarray(source_labels)
    probabilities = np.asarray(source_probabilities, dtype=float)
    domains = np.asarray(source_domains, dtype=str)
    if labels.ndim != 1 or domains.ndim != 1 or probabilities.ndim != 2:
        raise ValueError("source labels/domains must be vectors and probabilities a matrix")
    if probabilities.shape != (labels.size, len(names)) or domains.size != labels.size:
        raise ValueError("source arrays have incompatible sample or class dimensions")
    if labels.size == 0 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("source_labels must be a non-empty integer vector")
    labels = labels.astype(int, copy=False)
    if np.any(labels < 0) or np.any(labels >= len(names)):
        raise ValueError("source_labels contain an invalid class index")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("source_probabilities must be finite and non-negative")
    if not np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-5):
        raise ValueError("source_probabilities rows must sum to one")
    if any(not str(domain).strip() for domain in domains):
        raise ValueError("source_domains cannot contain empty values")
    return names, labels, probabilities, domains


def _multiclass_metrics(
    domain: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_count: int,
    ece_bins: int,
) -> DomainMetrics:
    predictions = np.argmax(probabilities, axis=1)
    support = np.bincount(labels, minlength=class_count)
    per_class_f1: list[float] = []
    recalls: list[float] = []
    for class_index in range(class_count):
        truth = labels == class_index
        predicted = predictions == class_index
        true_positive = int(np.sum(truth & predicted))
        false_positive = int(np.sum(~truth & predicted))
        false_negative = int(np.sum(truth & ~predicted))
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        denominator = precision + recall
        per_class_f1.append(2.0 * precision * recall / denominator if denominator else 0.0)
        if recall_denominator:
            recalls.append(recall)

    # The task vocabulary is frozen. A small label-blind source sample may omit
    # a class, but silently deleting that class from macro-F1 would inflate the
    # qualification score and make it inconsistent with target evaluation.
    macro_f1 = float(np.mean(per_class_f1))
    confidence = np.max(probabilities, axis=1)
    correct = predictions == labels
    ece = 0.0
    edges = np.linspace(0.0, 1.0, ece_bins + 1)
    for bin_index in range(ece_bins):
        if bin_index == ece_bins - 1:
            selected = (confidence >= edges[bin_index]) & (confidence <= edges[bin_index + 1])
        else:
            selected = (confidence >= edges[bin_index]) & (confidence < edges[bin_index + 1])
        if np.any(selected):
            weight = float(np.mean(selected))
            ece += weight * abs(
                float(np.mean(correct[selected])) - float(np.mean(confidence[selected]))
            )

    clipped = np.clip(probabilities[np.arange(labels.size), labels], 1e-12, 1.0)
    one_hot = np.eye(class_count, dtype=float)[labels]
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
    normalized_entropy = entropy / math.log(class_count)
    return DomainMetrics(
        domain=domain,
        sample_count=int(labels.size),
        class_support=tuple(int(value) for value in support),
        accuracy=_clip01(float(np.mean(correct))),
        balanced_accuracy=_clip01(float(np.mean(recalls)) if recalls else 0.0),
        macro_f1=_clip01(macro_f1),
        negative_log_likelihood=float(-np.mean(np.log(clipped))),
        multiclass_brier=_clip01(
            float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1)) / 2.0)
        ),
        expected_calibration_error=_clip01(ece),
        mean_predictive_entropy=_clip01(float(np.mean(normalized_entropy))),
        per_class_f1=tuple(_clip01(value) for value in per_class_f1),
    )


def _fit_diagonal_reference(
    features: np.ndarray,
    class_index: int | None,
    shrinkage: float,
    minimum_variance: float,
    variance_target: np.ndarray | None = None,
    calibration_distances: np.ndarray | None = None,
) -> DiagonalGaussianReference:
    mean = np.mean(features, axis=0)
    if features.shape[0] > 1:
        raw_variance = np.var(features, axis=0, ddof=1)
    elif variance_target is not None:
        raw_variance = np.asarray(variance_target, dtype=float)
    else:
        raw_variance = np.ones(features.shape[1], dtype=float)
    if variance_target is None:
        positive = raw_variance[raw_variance > minimum_variance]
        scalar_target = float(np.median(positive)) if positive.size else 1.0
        target = np.full(features.shape[1], scalar_target, dtype=float)
    else:
        target = np.asarray(variance_target, dtype=float)
    # In the intended n << d frozen-embedding regime, a fixed 0.1 shrinkage
    # still leaves hundreds of independently estimated variances effectively
    # unconstrained. Treat the configured value as a floor and increase the
    # target weight when dimensionality dominates the available degrees of
    # freedom. This also keeps cross-fit and full-source distance scales stable.
    degrees_of_freedom = max(features.shape[0] - 1, 0)
    data_weight = (1.0 - shrinkage) * degrees_of_freedom / (degrees_of_freedom + features.shape[1])
    variance = data_weight * raw_variance + (1.0 - data_weight) * target
    variance = np.maximum(variance, minimum_variance)
    if calibration_distances is None:
        distances = np.sum(np.square(features - mean) / variance, axis=1)
    else:
        distances = np.asarray(calibration_distances, dtype=float)
        if distances.shape != (features.shape[0],) or not np.all(np.isfinite(distances)):
            raise ValueError("calibration distances must align with reference samples")
        if np.any(distances < 0):
            raise ValueError("calibration distances cannot be negative")
    return DiagonalGaussianReference(
        class_index=class_index,
        sample_count=int(features.shape[0]),
        mean=tuple(float(value) for value in mean),
        variance=tuple(float(value) for value in variance),
        calibration_distances=tuple(float(value) for value in np.sort(distances)),
    )


def _cross_fitted_distances(
    features: np.ndarray,
    *,
    source_labels: np.ndarray | None,
    source_domains: np.ndarray | None,
    class_index: int | None,
    shrinkage: float,
    minimum_variance: float,
) -> np.ndarray:
    """Compute source nonconformity without scoring any row against itself."""

    if class_index is None:
        selected_indices = np.arange(features.shape[0], dtype=int)
    else:
        if source_labels is None:
            raise ValueError("class-conditional cross-fitting requires source labels")
        selected_indices = np.flatnonzero(source_labels == class_index)
    if selected_indices.size < 2:
        raise ValueError("cross-fitted class references require at least two source samples")

    use_domain_folds = (
        source_domains is not None and np.unique(source_domains[selected_indices]).size >= 2
    )
    if use_domain_folds:
        held_out_groups = [
            selected_indices[source_domains[selected_indices] == domain]
            for domain in sorted(np.unique(source_domains[selected_indices]))
        ]
    else:
        held_out_groups = [np.asarray([index], dtype=int) for index in selected_indices]

    distances_by_index: dict[int, float] = {}
    all_indices = np.arange(features.shape[0], dtype=int)
    for held_out in held_out_groups:
        if use_domain_folds:
            held_domain = source_domains[int(held_out[0])]
            training_indices = all_indices[source_domains != held_domain]
        else:
            training_indices = all_indices[~np.isin(all_indices, held_out)]
        if training_indices.size == 0:
            raise ValueError("cross-fitting left no source observations for a reference")

        global_training_reference = _fit_diagonal_reference(
            features[training_indices],
            None,
            shrinkage,
            minimum_variance,
        )
        if class_index is None:
            scoring_reference = global_training_reference
        else:
            class_training_indices = training_indices[
                source_labels[training_indices] == class_index
            ]
            if class_training_indices.size == 0:
                raise ValueError("cross-fitting left no source observations for a class")
            scoring_reference = _fit_diagonal_reference(
                features[class_training_indices],
                class_index,
                shrinkage,
                minimum_variance,
                np.asarray(global_training_reference.variance, dtype=float),
            )
        for index in held_out:
            distances_by_index[int(index)] = scoring_reference.squared_mahalanobis(
                features[int(index)]
            )

    return np.asarray([distances_by_index[int(index)] for index in selected_indices], dtype=float)


def fit_feature_reference(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    stage: OODStage,
    encoder_fingerprint: str,
    source_labels: Sequence[int] | np.ndarray | None = None,
    source_domains: Sequence[str] | np.ndarray | None = None,
    shrinkage: float = 0.1,
    minimum_variance: float = 1e-6,
) -> FeatureReference:
    """Fit a shrinkage diagonal, optionally class-conditional Mahalanobis model."""

    features = np.asarray(source_features, dtype=float)
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] == 0:
        raise ValueError("source_features must contain at least two feature vectors")
    if not np.all(np.isfinite(features)):
        raise ValueError("source_features must be finite")
    if stage not in ("pre_call", "post_call"):
        raise ValueError("stage must be pre_call or post_call")
    _require_nonempty(encoder_fingerprint, "encoder_fingerprint")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    if minimum_variance <= 0 or not math.isfinite(minimum_variance):
        raise ValueError("minimum_variance must be finite and positive")

    labels: np.ndarray | None = None
    if source_labels is not None:
        labels = np.asarray(source_labels)
        if labels.shape != (features.shape[0],) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("source_labels must be an integer vector aligned with features")
        labels = labels.astype(int, copy=False)
        if np.any(labels < 0):
            raise ValueError("source feature class indices cannot be negative")
    domains: np.ndarray | None = None
    if source_domains is not None:
        domains = np.asarray(source_domains, dtype=str)
        if domains.shape != (features.shape[0],):
            raise ValueError("source_domains must be a vector aligned with features")
        if any(not str(domain).strip() for domain in domains):
            raise ValueError("source_domains cannot contain empty values")

    global_calibration = _cross_fitted_distances(
        features,
        source_labels=labels,
        source_domains=domains,
        class_index=None,
        shrinkage=shrinkage,
        minimum_variance=minimum_variance,
    )
    global_reference = _fit_diagonal_reference(
        features,
        None,
        shrinkage,
        minimum_variance,
        calibration_distances=global_calibration,
    )
    class_references: list[DiagonalGaussianReference] = []
    if labels is not None:
        for class_index in sorted(int(value) for value in np.unique(labels)):
            selected = features[labels == class_index]
            # A one-shot class cannot supply a self-excluded empirical null.
            # Keep the full-source global reference available, while the
            # qualification policy decides whether one sample is sufficient.
            if selected.shape[0] < 2:
                continue
            class_calibration = _cross_fitted_distances(
                features,
                source_labels=labels,
                source_domains=domains,
                class_index=class_index,
                shrinkage=shrinkage,
                minimum_variance=minimum_variance,
            )
            class_references.append(
                _fit_diagonal_reference(
                    selected,
                    class_index,
                    shrinkage,
                    minimum_variance,
                    np.asarray(global_reference.variance, dtype=float),
                    class_calibration,
                )
            )

    return FeatureReference(
        stage=stage,
        encoder_fingerprint=encoder_fingerprint,
        dimension=int(features.shape[1]),
        shrinkage=float(shrinkage),
        minimum_variance=float(minimum_variance),
        global_reference=global_reference,
        class_references=tuple(class_references),
        calibration_method="source-only-cross-fit",
    )


def fit_qualification_artifact(
    identity: ExpertIdentity,
    *,
    class_names: Sequence[str],
    source_labels: Sequence[int] | np.ndarray,
    source_probabilities: Sequence[Sequence[float]] | np.ndarray,
    source_domains: Sequence[str] | np.ndarray,
    source_latency_ms: Sequence[float],
    source_cheap_features: Sequence[Sequence[float]] | np.ndarray,
    cheap_encoder_fingerprint: str,
    source_native_features: Sequence[Sequence[float]] | np.ndarray,
    native_encoder_fingerprint: str,
    performance_metric: str = "macro_f1",
    cvar_alpha: float = 0.25,
    lcb_z: float = 1.645,
    shrinkage: float = 0.1,
    minimum_variance: float = 1e-6,
    ece_bins: int = 10,
    metadata: Mapping[str, Any] | None = None,
) -> QualificationArtifact:
    """Build an expert qualification artifact from source validation data only.

    ``performance_lcb`` is a lower confidence bound over continuous per-domain
    multi-class scores, not a binary correctness-risk model.  ``performance_cvar``
    is the mean of the worst source-domain tail.
    """

    names, labels, probabilities, domains = _validate_multiclass_source_data(
        class_names, source_labels, source_probabilities, source_domains
    )
    if not 0.0 < cvar_alpha <= 1.0:
        raise ValueError("cvar_alpha must be in (0, 1]")
    if lcb_z < 0 or not math.isfinite(lcb_z):
        raise ValueError("lcb_z must be finite and non-negative")
    if ece_bins < 2:
        raise ValueError("ece_bins must be at least two")

    aggregate = _multiclass_metrics(
        "__all_source_domains__", labels, probabilities, len(names), ece_bins
    )
    per_domain = tuple(
        _multiclass_metrics(
            str(domain),
            labels[domains == domain],
            probabilities[domains == domain],
            len(names),
            ece_bins,
        )
        for domain in sorted(np.unique(domains))
    )
    if performance_metric not in {"macro_f1", "balanced_accuracy", "accuracy"}:
        raise ValueError("performance_metric is not supported")
    domain_scores = np.asarray(
        [float(getattr(metrics, performance_metric)) for metrics in per_domain], dtype=float
    )
    standard_error = (
        float(np.std(domain_scores, ddof=1) / math.sqrt(domain_scores.size))
        if domain_scores.size > 1
        else 0.0
    )
    performance_lcb = _clip01(float(np.mean(domain_scores)) - lcb_z * standard_error)
    tail_count = max(1, math.ceil(domain_scores.size * cvar_alpha))
    performance_cvar = _clip01(float(np.mean(np.sort(domain_scores)[:tail_count])))

    cheap = np.asarray(source_cheap_features, dtype=float)
    native = np.asarray(source_native_features, dtype=float)
    if cheap.shape[0] != labels.size or native.shape[0] != labels.size:
        raise ValueError("source feature matrices must align with source validation labels")

    return QualificationArtifact(
        schema_version=1,
        identity=identity,
        class_names=names,
        aggregate_metrics=aggregate,
        per_domain_metrics=per_domain,
        performance_metric=performance_metric,
        performance_lcb=performance_lcb,
        performance_cvar=performance_cvar,
        cvar_alpha=float(cvar_alpha),
        latency=LatencyMetrics.fit(source_latency_ms),
        cheap_feature_reference=fit_feature_reference(
            cheap,
            stage="pre_call",
            encoder_fingerprint=cheap_encoder_fingerprint,
            source_labels=labels,
            source_domains=domains,
            shrinkage=shrinkage,
            minimum_variance=minimum_variance,
        ),
        native_feature_reference=fit_feature_reference(
            native,
            stage="post_call",
            encoder_fingerprint=native_encoder_fingerprint,
            source_labels=labels,
            source_domains=domains,
            shrinkage=shrinkage,
            minimum_variance=minimum_variance,
        ),
        target_labels_used=False,
        metadata=dict(metadata or {}),
    )
