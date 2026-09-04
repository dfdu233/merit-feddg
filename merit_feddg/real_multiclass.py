"""Real multi-class, leave-one-medical-center-out Med-DEFER experiments.

The inference policy in this module never receives a target label.  Source
labels are used only to qualify a frozen specialist on its native task; the
target label is joined back after prediction solely for evaluation.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .io import save_json
from .math_utils import softmax
from .qualification import (
    ExpertIdentity,
    QualificationArtifact,
    QualificationGate,
    QualificationPolicy,
    fingerprint_payload,
    fit_qualification_artifact,
)
from .types import EvidenceRecord

METHODS = (
    "generalist",
    "specialist",
    "routed_fusion",
    "uncertainty_only",
    "med_defer_no_dg",
    "mean_domain_trust",
    "med_defer_full",
    "shuffled_evidence",
    "wrong_capability",
)

FIXED_TAXONOMY_MACRO_F1_DEFINITION = (
    "unweighted mean of per-class F1 over the complete frozen task taxonomy; "
    "a class with zero target support and zero predictions contributes 0 rather than being removed"
)


@dataclass(frozen=True)
class RealMulticlassPrediction:
    method: str
    sample_id: str
    cluster_id: str
    domain: str
    label: int
    predicted: int
    confidence: float
    scores: tuple[float, ...]
    gate: float
    expert_called: bool
    pre_call_ood: float | None
    post_call_ood: float | None
    decision_reason: str
    target_label_used_by_decision: bool = False


def _model_fingerprint(spec: dict, artifact_root: str | Path | None, kind: str) -> str:
    payload: dict[str, Any] = {
        "id": spec["id"],
        "revision": spec.get("revision", "main"),
        "adapter": spec.get("adapter"),
    }
    if artifact_root is not None:
        directory = Path(artifact_root) / "models" / str(spec["id"]).replace("/", "--")
        marker = directory / ".merit-download-complete.json"
        try:
            with marker.open("r", encoding="utf-8") as handle:
                completed = json.load(handle)
            payload["snapshot_fingerprint"] = completed.get("fingerprint")
            payload["snapshot_bytes"] = completed.get("bytes")
        except (OSError, TypeError, ValueError):
            payload["snapshot_fingerprint"] = "registry-only"
    payload["kind"] = kind
    return fingerprint_payload(payload)


def identity_from_configs(
    model_config: dict,
    study_config: dict,
    artifact_root: str | Path | None = None,
) -> tuple[ExpertIdentity, str, str]:
    task = study_config["task"]
    expert_id = str(task["expert_id"])
    expert_spec = model_config["experts"][expert_id]
    broad_spec = model_config["broad_specialist"]
    identity = ExpertIdentity(
        expert_id=expert_id,
        model_fingerprint=_model_fingerprint(expert_spec, artifact_root, "expert-model"),
        adapter_fingerprint=fingerprint_payload(
            {"adapter": expert_spec["adapter"], "contract": "semantic-claim-v1"}
        ),
        modality=str(task["modality"]),
        capability=str(task.get("capability", "classification")),
        task=str(task["id"]),
    )
    cheap_fingerprint = _model_fingerprint(broad_spec, artifact_root, "pre-call-encoder")
    native_fingerprint = _model_fingerprint(expert_spec, artifact_root, "native-encoder")
    return identity, cheap_fingerprint, native_fingerprint


def _validated_candidates(records: list[EvidenceRecord], minimum: int = 3) -> tuple[str, ...]:
    if not records:
        raise ValueError("at least one record is required")
    candidates = tuple(records[0].candidates)
    if len(candidates) < minimum:
        raise ValueError(f"real study requires at least {minimum} classes")
    for record in records:
        if tuple(record.candidates) != candidates:
            raise ValueError("all records in one task must use the same frozen class order")
        kind = str((record.metadata or {}).get("domain_kind", ""))
        if kind != "real_medical_center":
            raise ValueError("real study requires explicit medical-center domains")
    return candidates


def _class_support(records: list[EvidenceRecord], class_names: tuple[str, ...]) -> dict[str, int]:
    counts = np.zeros(len(class_names), dtype=int)
    for record in records:
        if not 0 <= int(record.label) < len(class_names):
            raise ValueError(f"record {record.sample_id} has a label outside the frozen taxonomy")
        counts[int(record.label)] += 1
    return {name: int(counts[index]) for index, name in enumerate(class_names)}


def _target_class_coverage(
    target: list[EvidenceRecord],
    class_names: tuple[str, ...],
    held_out_center: str,
    study_config: dict,
) -> dict[str, Any]:
    """Describe target coverage after inference without feeding labels to decisions."""

    support = _class_support(target, class_names)
    configured = study_config.get("task", {}).get("structurally_unavailable_classes_by_center", {})
    if not isinstance(configured, dict):
        raise TypeError("structurally_unavailable_classes_by_center must be a mapping")
    raw_structural = configured.get(held_out_center, [])
    if not isinstance(raw_structural, (list, tuple)):
        raise TypeError("structurally unavailable classes must be a list per center")
    structural = tuple(str(value) for value in raw_structural)
    unknown = sorted(set(structural) - set(class_names))
    if unknown:
        raise ValueError(f"structural-unavailability config names unknown classes: {unknown}")
    contradicted = [name for name in structural if support[name] > 0]
    if contradicted:
        raise ValueError(
            f"structurally unavailable classes are present in the target records: {contradicted}"
        )
    observed = [name for name in class_names if support[name] > 0]
    unobserved_sampled = [
        name for name in class_names if support[name] == 0 and name not in structural
    ]
    return {
        "target_class_support": support,
        "observed_target_classes": observed,
        "structurally_unavailable_target_classes": [
            name for name in class_names if name in structural
        ],
        "unobserved_sampled_target_classes": unobserved_sampled,
    }


def _metadata_vector(record: EvidenceRecord, key: str) -> np.ndarray:
    metadata = record.metadata or {}
    value = metadata.get(key)
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"record {record.sample_id} has no valid {key}")
    return vector


def _native_vector(record: EvidenceRecord, expert_id: str) -> np.ndarray:
    metadata = record.metadata or {}
    value = (metadata.get("expert_native_features") or {}).get(expert_id)
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"record {record.sample_id} has no native feature for {expert_id}")
    return vector


def fit_multiclass_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit one source-only temperature against the real multiclass NLL."""

    values = np.asarray(logits, dtype=float)
    targets = np.asarray(labels, dtype=int)
    if values.ndim != 2 or values.shape[0] != targets.size or values.shape[1] < 3:
        raise ValueError("temperature fitting requires aligned multi-class source logits")
    if np.any(targets < 0) or np.any(targets >= values.shape[1]):
        raise ValueError("temperature fitting received an invalid source label")
    temperatures = np.geomspace(0.05, 20.0, 161)
    losses = []
    for temperature in temperatures:
        probabilities = np.stack([softmax(row / temperature) for row in values])
        truth = np.clip(probabilities[np.arange(targets.size), targets], 1e-12, 1.0)
        losses.append(float(-np.mean(np.log(truth))))
    return float(temperatures[int(np.argmin(losses))])


def fit_real_multiclass_qualification(
    source_records: list[EvidenceRecord],
    model_config: dict,
    study_config: dict,
    artifact_root: str | Path | None = None,
) -> QualificationArtifact:
    """Qualify one expert from explicitly pre-filtered real source centers."""

    class_names = _validated_candidates(source_records)
    source_domains = {record.domain for record in source_records}
    if len(source_domains) < 2:
        raise ValueError("qualification needs at least two real source medical centers")
    task = study_config["task"]
    expert_id = str(task["expert_id"])
    identity, cheap_fingerprint, native_fingerprint = identity_from_configs(
        model_config, study_config, artifact_root
    )
    source_labels = np.asarray([record.label for record in source_records], dtype=int)
    source_expert_logits = np.stack(
        [np.asarray(record.expert_scores[expert_id], dtype=float) for record in source_records]
    )
    source_generalist_logits = np.stack(
        [np.asarray(record.general_final_logits, dtype=float) for record in source_records]
    )
    expert_temperature = fit_multiclass_temperature(source_expert_logits, source_labels)
    generalist_temperature = fit_multiclass_temperature(source_generalist_logits, source_labels)
    probabilities = np.stack([softmax(row / expert_temperature) for row in source_expert_logits])
    cheap_features = np.stack(
        [_metadata_vector(record, "cheap_domain_feature") for record in source_records]
    )
    native_features = np.stack([_native_vector(record, expert_id) for record in source_records])
    latencies = [
        float((record.metadata or {}).get("expert_latency_ms", {}).get(expert_id, 0.0))
        for record in source_records
    ]
    settings = study_config.get("qualification", {})
    return fit_qualification_artifact(
        identity,
        class_names=class_names,
        source_labels=source_labels,
        source_probabilities=probabilities,
        source_domains=np.asarray([record.domain for record in source_records]),
        source_latency_ms=latencies,
        source_cheap_features=cheap_features,
        cheap_encoder_fingerprint=cheap_fingerprint,
        source_native_features=native_features,
        native_encoder_fingerprint=native_fingerprint,
        performance_metric=str(settings.get("performance_metric", "macro_f1")),
        cvar_alpha=float(settings.get("cvar_alpha", 0.25)),
        lcb_z=float(settings.get("lcb_z", 1.645)),
        shrinkage=float(settings.get("shrinkage", 0.1)),
        minimum_variance=float(settings.get("minimum_variance", 1e-6)),
        metadata={
            "protocol": "leave-one-real-medical-center-out",
            "source_centers": sorted(source_domains),
            "labels": "real native multiclass task",
            "expert_logit_temperature": expert_temperature,
            "generalist_logit_temperature": generalist_temperature,
        },
    )


def normalized_uncertainty(scores: np.ndarray) -> float:
    probabilities = softmax(np.asarray(scores, dtype=float))
    entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))))
    return float(entropy / math.log(len(probabilities)))


def _expert_confidence(scores: np.ndarray) -> float:
    probabilities = np.sort(softmax(np.asarray(scores, dtype=float)))
    margin = float(probabilities[-1] - probabilities[-2])
    return float(np.clip(2.0 * margin, 0.05, 1.0))


def bounded_candidate_fusion(
    base_scores: np.ndarray,
    evidence_scores: np.ndarray,
    gate: float,
    *,
    strength: float,
    max_delta_norm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject only a bounded direction; never replace the generalist scores."""

    base = np.asarray(base_scores, dtype=float)
    evidence = np.asarray(evidence_scores, dtype=float)
    if base.shape != evidence.shape or base.ndim != 1 or base.size < 3:
        raise ValueError("aligned multi-class base and evidence vectors are required")
    direction = softmax(evidence) - 1.0 / evidence.size
    norm = float(np.linalg.norm(direction))
    if norm > 1e-12:
        direction /= norm
    else:
        direction = np.zeros_like(direction)
    delta = float(strength) * float(np.clip(gate, 0.0, 1.0)) * direction
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm > max_delta_norm:
        delta *= max_delta_norm / delta_norm
    return base + delta, delta


def _gate_settings(study_config: dict) -> dict:
    return study_config.get("med_defer", {}).get("domain_trust", {})


def _qualification_policy(study_config: dict) -> QualificationPolicy:
    settings = study_config.get("qualification", {})
    return QualificationPolicy(
        minimum_source_domains=int(settings.get("minimum_source_domains", 2)),
        minimum_samples_per_domain=int(settings.get("minimum_samples_per_domain", 1)),
        minimum_samples_per_class=int(settings.get("minimum_samples_per_class", 1)),
        minimum_performance_lcb=float(settings.get("minimum_performance_lcb", 0.05)),
        minimum_performance_cvar=float(settings.get("minimum_performance_cvar", 0.05)),
        maximum_p95_latency_ms=settings.get("maximum_p95_latency_ms"),
    )


def _method_decision(
    *,
    method: str,
    sample_id: str,
    base_scores: np.ndarray,
    expert_scores: np.ndarray,
    cheap_feature: np.ndarray,
    native_feature: np.ndarray,
    image_quality: float,
    identity: ExpertIdentity,
    cheap_fingerprint: str,
    native_fingerprint: str,
    artifact: QualificationArtifact,
    study_config: dict,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Label-free inference decision used by every target example."""

    if method not in METHODS:
        raise ValueError(f"unsupported real-study method: {method}")
    base_temperature = float(artifact.metadata.get("generalist_logit_temperature", 1.0))
    expert_temperature = float(artifact.metadata.get("expert_logit_temperature", 1.0))
    base = np.asarray(base_scores, dtype=float) / max(base_temperature, 1e-6)
    expert = np.asarray(expert_scores, dtype=float) / max(expert_temperature, 1e-6)
    if method == "generalist":
        return base.copy(), {
            "gate": 0.0,
            "expert_called": False,
            "reason": "generalist-only",
            "pre_call_ood": None,
            "post_call_ood": None,
        }
    if method == "specialist":
        return expert.copy(), {
            "gate": 1.0,
            "expert_called": True,
            "reason": "specialist-only-control",
            "pre_call_ood": None,
            "post_call_ood": None,
        }

    settings = study_config.get("med_defer", {})
    guidance = settings.get("guidance", {})
    strength = float(guidance.get("strength", 0.8))
    cap = float(guidance.get("max_bias_norm", 1.25))
    uncertainty_threshold = float(settings.get("controller", {}).get("uncertainty_threshold", 0.35))
    confidence = _expert_confidence(expert)
    called = True
    pre_score: float | None = None
    post_score: float | None = None

    if method == "routed_fusion":
        gate_value = 1.0
        reason = "route-only-fusion"
    elif method == "uncertainty_only":
        triggered = normalized_uncertainty(base) >= uncertainty_threshold
        gate_value = confidence if triggered else 0.0
        called = triggered
        reason = "entropy-triggered" if triggered else "confident-none"
    elif method == "med_defer_no_dg":
        gate_value = confidence
        reason = "qualified-first-claim-without-domain-gate"
    elif method == "wrong_capability":
        gate_value = 0.0
        called = False
        reason = "capability-mismatch-fail-closed"
    else:
        gate = QualificationGate(_qualification_policy(study_config))
        authorization = gate.authorize(identity, artifact)
        if not authorization.allowed:
            gate_value = 0.0
            called = False
            reason = authorization.reason
        elif method == "mean_domain_trust":
            mean_source = float(
                np.mean(
                    [
                        getattr(item, artifact.performance_metric)
                        for item in artifact.per_domain_metrics
                    ]
                )
            )
            gate_value = mean_source * confidence
            reason = "mean-source-domain-trust"
        else:
            pre = gate.pre_call_ood(
                identity,
                artifact,
                cheap_feature,
                cheap_fingerprint,
                # Never condition a hard pre-call veto on the generalist's
                # argmax: the method is specifically meant to catch confident
                # wrong classes. Nearest known source class is label-free and
                # separates domain shift from a base-model class mismatch.
                class_index=None,
            )
            pre_score = pre.score
            maximum_pre = float(_gate_settings(study_config).get("maximum_pre_call_ood", 1.0))
            if not pre.allowed or pre.score > maximum_pre:
                gate_value = 0.0
                called = False
                reason = "pre-call-ood-reject" if pre.allowed else pre.reason
            else:
                predicted_by_expert = int(np.argmax(expert))
                post = gate.post_call_native_ood(
                    identity,
                    artifact,
                    native_feature,
                    native_fingerprint,
                    class_index=predicted_by_expert,
                )
                post_score = post.score
                if not post.allowed:
                    gate_value = 0.0
                    reason = post.reason
                else:
                    temperature = float(_gate_settings(study_config).get("ood_temperature", 2.0))
                    ood_discount = math.exp(-temperature * max(pre.score, post.score))
                    source_robustness = math.sqrt(
                        artifact.performance_lcb * artifact.performance_cvar
                    )
                    gate_value = (
                        source_robustness
                        * ood_discount
                        * float(np.clip(image_quality, 0.0, 1.0))
                        * confidence
                    )
                    reason = "geomean-lcb-cvar-native-ood-guidance"

    fused, _ = bounded_candidate_fusion(
        base,
        expert,
        gate_value,
        strength=strength,
        max_delta_norm=cap,
    )
    return fused, {
        "sample_id": sample_id,
        "gate": float(np.clip(gate_value, 0.0, 1.0)),
        "expert_called": called,
        "reason": reason,
        "pre_call_ood": pre_score,
        "post_call_ood": post_score,
    }


def _shuffle_indices(records: list[EvidenceRecord], seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    if len(records) < 2:
        raise ValueError("shuffled-evidence control requires at least two target records")
    cycle = rng.permutation(len(records))
    mapping = {
        int(cycle[index]): int(cycle[(index + 1) % len(cycle)]) for index in range(len(cycle))
    }
    return {record.sample_id: mapping[index] for index, record in enumerate(records)}


def _ece(predictions: list[RealMulticlassPrediction], bins: int = 10) -> float:
    confidence = np.asarray([item.confidence for item in predictions])
    correct = np.asarray([item.predicted == item.label for item in predictions])
    total = max(len(predictions), 1)
    value = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if np.any(selected):
            value += float(np.sum(selected) / total) * abs(
                float(np.mean(confidence[selected])) - float(np.mean(correct[selected]))
            )
    return value


def _macro_f1(predictions: list[RealMulticlassPrediction], class_count: int) -> float:
    values = []
    for label in range(class_count):
        truth = np.asarray([item.label == label for item in predictions])
        guess = np.asarray([item.predicted == label for item in predictions])
        tp = int(np.sum(truth & guess))
        fp = int(np.sum(~truth & guess))
        fn = int(np.sum(truth & ~guess))
        denominator = 2 * tp + fp + fn
        values.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(values))


def _paired_bootstrap_delta(
    predictions: list[RealMulticlassPrediction],
    base: list[RealMulticlassPrediction],
    *,
    seed: int,
    repetitions: int = 2000,
) -> dict[str, float | int | str]:
    _validate_paired_predictions(predictions, base)
    now = np.asarray([item.predicted == item.label for item in predictions], dtype=float)
    old = np.asarray([item.predicted == item.label for item in base], dtype=float)
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(predictions):
        clusters[item.cluster_id].append(index)
    cluster_ids = sorted(clusters)
    rng = np.random.default_rng(seed)
    deltas = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled_clusters = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sample = np.asarray(
            [row for cluster_id in sampled_clusters for row in clusters[str(cluster_id)]],
            dtype=int,
        )
        deltas[index] = float(np.mean(now[sample] - old[sample]))
    return {
        "unit": "slide_id cluster",
        "clusters": len(cluster_ids),
        "accuracy_delta": float(np.mean(now - old)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
    }


def _cluster_paired_test(
    predictions: list[RealMulticlassPrediction], base: list[RealMulticlassPrediction]
) -> dict[str, float | int | str]:
    _validate_paired_predictions(predictions, base)
    rescued = harmed = 0
    for item, reference in zip(predictions, base):
        now = item.predicted == item.label
        before = reference.predicted == reference.label
        rescued += int(now and not before)
        harmed += int(before and not now)
    cluster_deltas: dict[str, list[float]] = defaultdict(list)
    for item, reference in zip(predictions, base):
        cluster_deltas[item.cluster_id].append(
            float(item.predicted == item.label) - float(reference.predicted == reference.label)
        )
    positive = sum(float(np.mean(values)) > 0 for values in cluster_deltas.values())
    negative = sum(float(np.mean(values)) < 0 for values in cluster_deltas.values())
    nonzero = positive + negative
    if nonzero == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(nonzero, index) for index in range(min(positive, negative) + 1))
        p_value = min(1.0, 2.0 * tail / (2**nonzero))
    return {
        "unit": "slide_id mean patch-accuracy delta",
        "patch_rescued": rescued,
        "patch_harmed": harmed,
        "positive_clusters": positive,
        "negative_clusters": negative,
        "tied_clusters": len(cluster_deltas) - nonzero,
        "p_value": p_value,
    }


def _validate_paired_predictions(
    predictions: list[RealMulticlassPrediction], base: list[RealMulticlassPrediction]
) -> None:
    if len(predictions) != len(base) or not predictions:
        raise ValueError("paired comparison requires equally sized non-empty predictions")
    for item, reference in zip(predictions, base):
        if (
            item.sample_id != reference.sample_id
            or item.cluster_id != reference.cluster_id
            or item.label != reference.label
        ):
            raise ValueError("paired comparison received misaligned samples or slide clusters")


def _full_vs_shuffled_report(
    full: list[RealMulticlassPrediction],
    shuffled: list[RealMulticlassPrediction],
    *,
    seed: int,
) -> dict[str, Any]:
    """Direct, equal-budget mechanism contrast; never an adaptation signal."""

    return {
        "contrast": "med_defer_full-minus-shuffled_evidence",
        "estimand": "paired patch-accuracy delta with slide_id as the resampling unit",
        "paired_bootstrap": _paired_bootstrap_delta(full, shuffled, seed=seed),
        "paired_sign_test": _cluster_paired_test(full, shuffled),
        "analysis_role": "descriptive mechanism falsification only; not for target tuning",
    }


def summarize_real_predictions(
    predictions: list[RealMulticlassPrediction],
    class_count: int,
    base: list[RealMulticlassPrediction] | None = None,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    if not predictions:
        raise ValueError("cannot summarize empty target predictions")
    by_domain: dict[str, list[bool]] = defaultdict(list)
    for item in predictions:
        by_domain[item.domain].append(item.predicted == item.label)
    fixed_taxonomy_macro_f1 = _macro_f1(predictions, class_count)
    result: dict[str, Any] = {
        "n": len(predictions),
        "accuracy": float(np.mean([item.predicted == item.label for item in predictions])),
        # Keep the historical key for machine compatibility, but make the
        # estimand explicit in every new report.
        "macro_f1": fixed_taxonomy_macro_f1,
        "fixed_taxonomy_macro_f1": fixed_taxonomy_macro_f1,
        "ece": _ece(predictions),
        "domain_accuracy": {key: float(np.mean(value)) for key, value in sorted(by_domain.items())},
        "worst_domain_accuracy": min(float(np.mean(value)) for value in by_domain.values()),
        "expert_call_rate": float(np.mean([item.expert_called for item in predictions])),
        "mean_gate": float(np.mean([item.gate for item in predictions])),
        "target_label_used_by_decision": any(
            item.target_label_used_by_decision for item in predictions
        ),
    }
    if base is not None:
        result["paired_bootstrap"] = _paired_bootstrap_delta(predictions, base, seed=seed)
        result["cluster_paired_test"] = _cluster_paired_test(predictions, base)
        result["same_as_generalist"] = float(
            np.mean([left.predicted == right.predicted for left, right in zip(predictions, base)])
        )
    return result


def evaluate_real_multiclass_fold(
    records: list[EvidenceRecord],
    held_out_center: str,
    model_config: dict,
    study_config: dict,
    *,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Fit on source centers; join the held-out center labels only for evaluation."""

    class_names = _validated_candidates(records)
    source = [record for record in records if record.domain != held_out_center]
    target = [record for record in records if record.domain == held_out_center]
    if not source or not target:
        raise ValueError("held-out center must leave non-empty source and target records")
    source_slides = {(record.metadata or {}).get("slide_id") for record in source}
    target_slides = {(record.metadata or {}).get("slide_id") for record in target}
    leaked = sorted(str(value) for value in source_slides & target_slides if value is not None)
    if leaked:
        raise ValueError(f"slide leakage across source/target: {leaked[:3]}")

    artifact = fit_real_multiclass_qualification(source, model_config, study_config, artifact_root)
    identity, cheap_fingerprint, native_fingerprint = identity_from_configs(
        model_config, study_config, artifact_root
    )
    expert_id = identity.expert_id
    seed = int(study_config.get("seed", 42))
    donors = _shuffle_indices(target, seed)
    predictions: dict[str, list[RealMulticlassPrediction]] = {method: [] for method in METHODS}
    for index, record in enumerate(target):
        base = np.asarray(record.general_final_logits, dtype=float)
        own_expert = np.asarray(record.expert_scores[expert_id], dtype=float)
        cheap = _metadata_vector(record, "cheap_domain_feature")
        native = _native_vector(record, expert_id)
        quality = float((record.metadata or {}).get("image_quality", 1.0))
        full_scores, full_decision = _method_decision(
            method="med_defer_full",
            sample_id=record.sample_id,
            base_scores=base,
            expert_scores=own_expert,
            cheap_feature=cheap,
            native_feature=native,
            image_quality=quality,
            identity=identity,
            cheap_fingerprint=cheap_fingerprint,
            native_fingerprint=native_fingerprint,
            artifact=artifact,
            study_config=study_config,
        )
        for method in METHODS:
            if method == "med_defer_full":
                # Reuse the exact decision below for the matched shuffled control.
                scores, decision = full_scores, dict(full_decision)
            elif method == "shuffled_evidence":
                # An equal-budget falsification control: keep this sample's
                # qualification, pre/post OOD, gate and call decision unchanged,
                # but replace only its semantic expert scores with a different
                # sample's scores.  The derangement guarantees no fixed points.
                donor = target[donors[record.sample_id]]
                donor_expert = np.asarray(donor.expert_scores[expert_id], dtype=float)
                base_temperature = float(artifact.metadata.get("generalist_logit_temperature", 1.0))
                expert_temperature = float(artifact.metadata.get("expert_logit_temperature", 1.0))
                guidance = study_config.get("med_defer", {}).get("guidance", {})
                scores, _ = bounded_candidate_fusion(
                    base / max(base_temperature, 1e-6),
                    donor_expert / max(expert_temperature, 1e-6),
                    float(full_decision["gate"]),
                    strength=float(guidance.get("strength", 0.8)),
                    max_delta_norm=float(guidance.get("max_bias_norm", 1.25)),
                )
                decision = {
                    **full_decision,
                    "reason": "shuffled-scores-with-matched-full-gate-and-call",
                }
            else:
                scores, decision = _method_decision(
                    method=method,
                    sample_id=record.sample_id,
                    base_scores=base,
                    expert_scores=own_expert,
                    cheap_feature=cheap,
                    native_feature=native,
                    image_quality=quality,
                    identity=identity,
                    cheap_fingerprint=cheap_fingerprint,
                    native_fingerprint=native_fingerprint,
                    artifact=artifact,
                    study_config=study_config,
                )
            probabilities = softmax(scores)
            predicted = int(np.argmax(probabilities))
            predictions[method].append(
                RealMulticlassPrediction(
                    method=method,
                    sample_id=record.sample_id,
                    cluster_id=str((record.metadata or {}).get("slide_id", record.sample_id)),
                    domain=record.domain,
                    label=record.label,
                    predicted=predicted,
                    confidence=float(probabilities[predicted]),
                    scores=tuple(float(value) for value in scores),
                    gate=float(decision["gate"]),
                    expert_called=bool(decision["expert_called"]),
                    pre_call_ood=decision["pre_call_ood"],
                    post_call_ood=decision["post_call_ood"],
                    decision_reason=str(decision["reason"]),
                    target_label_used_by_decision=False,
                )
            )

    base = predictions["generalist"]
    metrics = {
        method: summarize_real_predictions(
            items,
            len(class_names),
            None if method == "generalist" else base,
            seed=seed,
        )
        for method, items in predictions.items()
    }
    # Target labels are inspected only after all method decisions have been
    # materialized. These fields are evaluation diagnostics, never gate inputs.
    target_coverage = _target_class_coverage(target, class_names, held_out_center, study_config)
    source_class_support = _class_support(source, class_names)
    minimum_class_support = _qualification_policy(study_config).minimum_samples_per_class
    full_vs_shuffled = _full_vs_shuffled_report(
        predictions["med_defer_full"],
        predictions["shuffled_evidence"],
        seed=seed,
    )
    return {
        "protocol": "leave-one-real-medical-center-out",
        "execution_mode": "frozen-real-model-evidence; selected-call counts are counterfactual",
        "held_out_center": held_out_center,
        "source_centers": sorted({record.domain for record in source}),
        "source_samples": len(source),
        "target_samples": len(target),
        "class_names": list(class_names),
        "macro_f1_definition": FIXED_TAXONOMY_MACRO_F1_DEFINITION,
        "source_class_support": source_class_support,
        "minimum_source_samples_per_frozen_class": minimum_class_support,
        "source_classes_below_qualification_support": [
            name for name in class_names if source_class_support[name] < minimum_class_support
        ],
        **target_coverage,
        "task_type": "real_multiclass",
        "binary_error_estimator": False,
        "target_labels_used_during_qualification_or_inference": False,
        "qualification": artifact.to_dict(),
        "metrics": metrics,
        "full_vs_shuffled": full_vs_shuffled,
        "predictions": {
            method: [asdict(item) for item in items] for method, items in predictions.items()
        },
    }


def _pooled_fold_predictions(
    folds: list[dict[str, Any]], method: str
) -> list[RealMulticlassPrediction]:
    pooled: list[RealMulticlassPrediction] = []
    for fold in folds:
        center = str(fold["held_out_center"])
        for raw in fold["predictions"][method]:
            payload = dict(raw)
            payload["scores"] = tuple(float(value) for value in payload["scores"])
            prediction = RealMulticlassPrediction(**payload)
            # slide_id is the biological cluster within a center. Prefixing
            # prevents an accidental identifier collision across institutions.
            pooled.append(replace(prediction, cluster_id=f"{center}::{prediction.cluster_id}"))
    return pooled


def run_real_multiclass_loco(
    records: list[EvidenceRecord],
    model_config: dict,
    study_config: dict,
    output: str | Path,
    *,
    artifact_root: str | Path | None = None,
    held_out_center: str | None = None,
) -> dict[str, Any]:
    centers = sorted({record.domain for record in records})
    selected = [held_out_center] if held_out_center else centers
    folds = [
        evaluate_real_multiclass_fold(
            records,
            center,
            model_config,
            study_config,
            artifact_root=artifact_root,
        )
        for center in selected
    ]
    aggregate: dict[str, Any] = {}
    for method in METHODS:
        values = [fold["metrics"][method] for fold in folds]
        aggregate[method] = {
            "accuracy": float(np.mean([item["accuracy"] for item in values])),
            "macro_f1": float(np.mean([item["macro_f1"] for item in values])),
            "fixed_taxonomy_macro_f1": float(
                np.mean([item["fixed_taxonomy_macro_f1"] for item in values])
            ),
            "ece": float(np.mean([item["ece"] for item in values])),
            # One LOCO fold has one target center. The cross-fold worst-center
            # statistic is therefore the minimum target-center accuracy, not
            # the mean of the per-fold one-center values.
            "worst_domain_accuracy": float(min(item["accuracy"] for item in values)),
            "expert_call_rate": float(np.mean([item["expert_call_rate"] for item in values])),
        }
    pooled_full = _pooled_fold_predictions(folds, "med_defer_full")
    pooled_shuffled = _pooled_fold_predictions(folds, "shuffled_evidence")
    full_vs_shuffled = _full_vs_shuffled_report(
        pooled_full,
        pooled_shuffled,
        seed=int(study_config.get("seed", 42)),
    )
    posthoc_diagnostics = []
    for fold in folds:
        full = fold["metrics"]["med_defer_full"]
        direct = fold["full_vs_shuffled"]
        paired_to_generalist = full["cluster_paired_test"]
        paired_to_shuffled = direct["paired_sign_test"]
        reasons = []
        if direct["paired_bootstrap"]["accuracy_delta"] <= 0:
            reasons.append("full-not-better-than-shuffled")
        if paired_to_generalist["patch_rescued"] <= paired_to_generalist["patch_harmed"]:
            reasons.append("rescued-not-greater-than-harmed")
        if paired_to_shuffled["positive_clusters"] <= paired_to_shuffled["negative_clusters"]:
            reasons.append("full-not-positive-in-slide-sign-contrast-to-shuffled")
        if full["expert_call_rate"] > 0 and full["same_as_generalist"] >= 0.95:
            reasons.append("expert-called-but-predictions-almost-never-change")
        posthoc_diagnostics.append(
            {
                "held_out_center": fold["held_out_center"],
                "mechanism_signal_present": not reasons,
                "diagnostic_flags": reasons,
                "full_vs_shuffled": direct,
            }
        )
    report = {
        "protocol": "leave-one-real-medical-center-out",
        "execution_mode": "frozen-real-model-evidence; selected-call counts are counterfactual",
        "analysis_role": str(
            study_config.get("evaluation", {}).get("analysis_role", "exploratory_mechanism_pilot")
        ),
        "centers": selected,
        "folds": folds,
        "aggregate": aggregate,
        "macro_f1_definition": FIXED_TAXONOMY_MACRO_F1_DEFINITION,
        "macro_f1_aggregation": "unweighted mean of fixed-taxonomy center-fold macro-F1",
        "target_class_support_by_center": {
            fold["held_out_center"]: fold["target_class_support"] for fold in folds
        },
        "observed_target_classes_by_center": {
            fold["held_out_center"]: fold["observed_target_classes"] for fold in folds
        },
        "structurally_unavailable_target_classes_by_center": {
            fold["held_out_center"]: fold["structurally_unavailable_target_classes"]
            for fold in folds
        },
        "unobserved_sampled_target_classes_by_center": {
            fold["held_out_center"]: fold["unobserved_sampled_target_classes"] for fold in folds
        },
        "full_vs_shuffled": full_vs_shuffled,
        "posthoc_mechanism_diagnostics": posthoc_diagnostics,
        "target_label_selection_policy": (
            "descriptive-only; never use these target-label diagnostics to tune the method, "
            "thresholds, sample size, or select a reported run"
        ),
        "target_labels_used_during_qualification_or_inference": False,
    }
    output_path = Path(output)
    save_json(output_path, report)
    _write_real_markdown(output_path.with_suffix(".md"), report)
    return report


def _write_real_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Real PathoROB leave-one-center-out comparison",
        "",
        "No target-center label was used by qualification, OOD scoring, routing, or inference.",
        (
            "Expert-call rates are selected-call counterfactuals on one frozen real-model "
            "cache; they do not establish live latency savings."
        ),
        (
            "Fixed-taxonomy macro-F1 is the unweighted mean over every frozen class; "
            "zero-support classes are not silently removed."
        ),
        "",
        "| Method | Accuracy | Fixed-taxonomy Macro-F1 | ECE ↓ | Worst center | Expert call rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in report["aggregate"].items():
        lines.append(
            f"| {method} | {metrics['accuracy']:.4f} | "
            f"{metrics['fixed_taxonomy_macro_f1']:.4f} | "
            f"{metrics['ece']:.4f} | {metrics['worst_domain_accuracy']:.4f} | "
            f"{metrics['expert_call_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Target-class coverage (evaluation only)",
            "",
            (
                "Class support is reported after prediction and is never supplied to qualification, "
                "OOD scoring, routing, or inference."
            ),
            "",
        ]
    )
    for fold in report["folds"]:
        support = ", ".join(
            f"{name}={count}" for name, count in fold["target_class_support"].items()
        )
        structural = ", ".join(fold["structurally_unavailable_target_classes"]) or "none"
        sampled = ", ".join(fold["unobserved_sampled_target_classes"]) or "none"
        lines.append(
            f"- {fold['held_out_center']}: {support}; structurally unavailable: "
            f"{structural}; absent from label-blind sample: {sampled}."
        )
    direct = report["full_vs_shuffled"]
    bootstrap = direct["paired_bootstrap"]
    sign = direct["paired_sign_test"]
    lines.extend(
        [
            "",
            "## Direct full-vs-shuffled mechanism contrast",
            "",
            (
                "This equal-budget, slide-cluster-paired contrast is descriptive mechanism "
                "falsification only and is not used to tune on target labels."
            ),
            "",
            (
                f"- Aggregate accuracy delta: {bootstrap['accuracy_delta']:.4f} "
                f"(slide-cluster bootstrap 95% CI "
                f"[{bootstrap['ci95_low']:.4f}, {bootstrap['ci95_high']:.4f}]); "
                f"paired sign-test p={sign['p_value']:.4g}."
            ),
        ]
    )
    for fold in report["folds"]:
        direct = fold["full_vs_shuffled"]
        bootstrap = direct["paired_bootstrap"]
        sign = direct["paired_sign_test"]
        lines.append(
            f"- {fold['held_out_center']}: delta={bootstrap['accuracy_delta']:.4f}, "
            f"95% CI [{bootstrap['ci95_low']:.4f}, {bootstrap['ci95_high']:.4f}], "
            f"paired sign-test p={sign['p_value']:.4g}."
        )
    lines.extend(
        [
            "",
            "## Post-hoc mechanism diagnostics (not for model or sample-size selection)",
            "",
        ]
    )
    for item in report["posthoc_mechanism_diagnostics"]:
        status = "signal present" if item["mechanism_signal_present"] else "flagged"
        reasons = ", ".join(item["diagnostic_flags"]) or "none"
        lines.append(f"- {item['held_out_center']}: **{status}** ({reasons})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
