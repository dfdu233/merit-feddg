import inspect
import json

import numpy as np
import pytest

from merit_feddg.qualification import (
    ExpertIdentity,
    QualificationArtifact,
    QualificationGate,
    QualificationPolicy,
    fingerprint_payload,
    fit_feature_reference,
    fit_qualification_artifact,
)


def _identity(**changes):
    values = {
        "expert_id": "pathology-conch",
        "model_fingerprint": fingerprint_payload(b"frozen-model-weights"),
        "adapter_fingerprint": fingerprint_payload({"adapter": "conch", "version": 2}),
        "modality": "pathology",
        "capability": "classification",
        "task": "three-class-tissue-diagnosis",
    }
    values.update(changes)
    return ExpertIdentity(**values)


def _source_data():
    # Three genuine task classes across three explicitly named source domains.
    labels = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=int)
    domains = np.asarray(["tcga-site-a"] * 3 + ["tcga-site-b"] * 3 + ["panda-center-c"] * 3)
    probabilities = np.asarray(
        [
            [0.82, 0.10, 0.08],
            [0.08, 0.84, 0.08],
            [0.05, 0.10, 0.85],
            [0.74, 0.16, 0.10],
            [0.18, 0.70, 0.12],
            [0.08, 0.22, 0.70],
            [0.68, 0.18, 0.14],
            [0.15, 0.66, 0.19],
            [0.15, 0.20, 0.65],
        ]
    )
    cheap_features = np.asarray(
        [
            [-2.2, -2.0],
            [0.0, 0.1],
            [2.0, 2.1],
            [-1.8, -2.1],
            [0.2, -0.1],
            [2.2, 1.8],
            [-2.0, -1.8],
            [-0.2, 0.0],
            [1.8, 2.2],
        ]
    )
    native_features = np.column_stack(
        [cheap_features[:, 0], cheap_features[:, 1], cheap_features[:, 0] * 0.5]
    )
    return labels, domains, probabilities, cheap_features, native_features


def _artifact():
    labels, domains, probabilities, cheap_features, native_features = _source_data()
    return fit_qualification_artifact(
        _identity(),
        class_names=("benign", "in-situ", "invasive"),
        source_labels=labels,
        source_probabilities=probabilities,
        source_domains=domains,
        source_latency_ms=(22.0, 24.0, 30.0, 28.0),
        source_cheap_features=cheap_features,
        cheap_encoder_fingerprint="sha256:shared-router-v1",
        source_native_features=native_features,
        native_encoder_fingerprint="sha256:conch-visual-v1",
        cvar_alpha=1 / 3,
        shrinkage=0.25,
        metadata={"split": "frozen-source-validation", "seed": 7},
    )


def test_fits_real_multiclass_metrics_for_each_source_domain():
    artifact = _artifact()

    assert artifact.class_names == ("benign", "in-situ", "invasive")
    assert artifact.source_domain_count == 3
    assert artifact.source_sample_count == 9
    assert {item.domain for item in artifact.per_domain_metrics} == {
        "tcga-site-a",
        "tcga-site-b",
        "panda-center-c",
    }
    assert all(len(item.per_class_f1) == 3 for item in artifact.per_domain_metrics)
    assert artifact.aggregate_metrics.accuracy == pytest.approx(1.0)
    assert artifact.aggregate_metrics.multiclass_brier < 0.2
    assert 0.0 <= artifact.performance_lcb <= artifact.performance_cvar <= 1.0
    assert artifact.target_labels_used is False


def test_missing_source_classes_remain_in_frozen_task_macro_f1():
    artifact = fit_qualification_artifact(
        _identity(),
        class_names=("benign", "in-situ", "invasive"),
        source_labels=np.asarray([0, 0, 0, 0]),
        source_probabilities=np.asarray([[0.9, 0.05, 0.05]] * 4),
        source_domains=np.asarray(["site-a", "site-a", "site-b", "site-b"]),
        source_latency_ms=(10.0,),
        source_cheap_features=np.asarray([[0.0], [0.1], [0.2], [0.3]]),
        cheap_encoder_fingerprint="cheap",
        source_native_features=np.asarray([[0.0], [0.1], [0.2], [0.3]]),
        native_encoder_fingerprint="native",
        lcb_z=0.0,
    )
    assert artifact.aggregate_metrics.accuracy == 1.0
    assert artifact.aggregate_metrics.macro_f1 == pytest.approx(1.0 / 3.0)
    assert all(item.macro_f1 == pytest.approx(1.0 / 3.0) for item in artifact.per_domain_metrics)
    decision = QualificationGate().authorize(_identity(), artifact)
    assert not decision.allowed
    assert decision.reason == "insufficient-source-class-support"


def test_minimum_aggregate_support_is_enforced_for_every_frozen_class():
    decision = QualificationGate(QualificationPolicy(minimum_samples_per_class=4)).authorize(
        _identity(), _artifact()
    )

    assert not decision.allowed
    assert decision.reason == "insufficient-source-class-support"


def test_ood_class_outside_frozen_taxonomy_fails_closed():
    assessment = QualificationGate().post_call_native_ood(
        _identity(),
        _artifact(),
        [-2.0, -2.0, -1.0],
        "sha256:conch-visual-v1",
        class_index=99,
    )

    assert not assessment.allowed
    assert assessment.score == 1.0
    assert assessment.reason == "class-outside-frozen-taxonomy"


def test_artifact_api_has_no_target_label_input_or_stored_target_arrays():
    parameters = inspect.signature(fit_qualification_artifact).parameters
    assert all("target" not in name for name in parameters)

    payload = _artifact().to_dict()
    assert payload["target_labels_used"] is False
    assert "source_labels" not in payload
    assert "source_probabilities" not in payload
    assert all("target" not in key or key == "target_labels_used" for key in payload)


def test_class_conditional_shrinkage_mahalanobis_detects_shift():
    _, _, _, cheap_features, _ = _source_data()
    labels = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2])
    reference = fit_feature_reference(
        cheap_features,
        stage="pre_call",
        encoder_fingerprint="cheap-v1",
        source_labels=labels,
        shrinkage=0.4,
    )

    in_score, in_distance, selected_class, reason = reference.score([-2.0, -2.0], 0)
    shifted_score, shifted_distance, _, _ = reference.score([30.0, -25.0], 0)

    assert selected_class == 0
    assert reason == "class-conditional"
    assert shifted_distance > in_distance
    assert shifted_score > in_score
    assert all(value > 0 for value in reference.class_references[0].variance)


def test_high_dimensional_cross_fit_does_not_map_held_out_id_almost_all_to_one():
    rng = np.random.default_rng(123)
    dimension = 512
    source = np.vstack(
        [rng.normal(shift, 1.0, size=(12, dimension)) for shift in (-0.08, 0.0, 0.08)]
    )
    domains = np.repeat(["center-a", "center-b", "center-c"], 12)
    reference = fit_feature_reference(
        source,
        stage="pre_call",
        encoder_fingerprint="high-dimensional-frozen-encoder",
        source_domains=domains,
        shrinkage=0.1,
    )
    held_out_id = rng.normal(0.0, 1.0, size=(48, dimension))
    scores = np.asarray([reference.score(feature)[0] for feature in held_out_id])
    in_sample_distances = np.sort(
        [reference.global_reference.squared_mahalanobis(feature) for feature in source]
    )

    assert reference.calibration_method == "source-only-cross-fit"
    assert not np.allclose(reference.global_reference.calibration_distances, in_sample_distances)
    assert float(np.mean(scores >= 0.99)) < 0.25


def test_pre_call_and_post_call_ood_are_distinct_and_source_calibrated():
    artifact = _artifact()
    gate = QualificationGate(QualificationPolicy(minimum_source_domains=3))

    pre = gate.pre_call_ood(
        _identity(), artifact, [-2.0, -2.0], "sha256:shared-router-v1", class_index=0
    )
    post = gate.post_call_native_ood(
        _identity(),
        artifact,
        [-2.0, -2.0, -1.0],
        "sha256:conch-visual-v1",
        class_index=0,
    )

    assert pre.allowed and post.allowed
    assert pre.stage == "pre_call"
    assert post.stage == "post_call"
    assert pre.squared_distance is not None
    assert post.squared_distance is not None


@pytest.mark.parametrize(
    ("artifact", "runtime_identity", "reason"),
    [
        (None, _identity(), "missing-qualification"),
        (
            _artifact(),
            _identity(model_fingerprint="sha256:other-model"),
            "model-fingerprint-mismatch",
        ),
        (
            _artifact(),
            _identity(adapter_fingerprint="sha256:other-adapter"),
            "adapter-fingerprint-mismatch",
        ),
        (_artifact(), _identity(task="different-task"), "task-mismatch"),
    ],
)
def test_missing_or_mismatched_qualification_fails_closed(artifact, runtime_identity, reason):
    decision = QualificationGate().authorize(runtime_identity, artifact)

    assert not decision.allowed
    assert decision.reason == reason


def test_feature_encoder_fingerprint_mismatch_fails_closed():
    assessment = QualificationGate().pre_call_ood(
        _identity(), _artifact(), [-2.0, -2.0], "sha256:wrong-cheap-encoder", 0
    )

    assert not assessment.allowed
    assert assessment.score == 1.0
    assert assessment.reason == "feature-encoder-fingerprint-mismatch"


def test_policy_rejects_weak_or_insufficient_source_qualification():
    artifact = _artifact()
    too_strict = QualificationGate(
        QualificationPolicy(
            minimum_source_domains=4,
            minimum_performance_lcb=1.0,
        )
    ).authorize(_identity(), artifact)

    assert not too_strict.allowed
    assert too_strict.reason == "insufficient-source-domains"


def test_json_round_trip_preserves_all_metrics_features_and_fingerprints():
    artifact = _artifact()
    encoded = artifact.to_json()
    decoded = QualificationArtifact.from_json(encoded)

    assert decoded == artifact
    assert decoded.model_fingerprint == artifact.identity.model_fingerprint
    assert decoded.adapter_fingerprint == artifact.identity.adapter_fingerprint
    assert decoded.modality == "pathology"
    assert decoded.capability == "classification"
    assert decoded.task == "three-class-tissue-diagnosis"
    assert decoded.artifact_fingerprint == artifact.artifact_fingerprint
    parsed = json.loads(encoded)
    assert parsed["identity"]["model_fingerprint"] == artifact.identity.model_fingerprint
    assert parsed["identity"]["adapter_fingerprint"] == artifact.identity.adapter_fingerprint
    assert parsed["cheap_feature_reference"]["stage"] == "pre_call"
    assert parsed["native_feature_reference"]["stage"] == "post_call"
    assert len(parsed["native_feature_reference"]["class_references"]) == 3


def test_legacy_self_included_ood_artifact_fails_closed():
    payload = _artifact().to_dict()
    payload["cheap_feature_reference"].pop("calibration_method")
    payload["native_feature_reference"].pop("calibration_method")
    legacy = QualificationArtifact.from_dict(payload)

    decision = QualificationGate().authorize(_identity(), legacy)
    assert not decision.allowed
    assert decision.reason == "non-cross-fitted-ood-calibration"


def test_deserialization_rejects_any_target_label_provenance():
    payload = _artifact().to_dict()
    payload["target_labels_used"] = True

    with pytest.raises(ValueError, match="target-label"):
        QualificationArtifact.from_dict(payload)

    metadata_payload = _artifact().to_dict()
    metadata_payload["metadata"]["target_labels"] = [0, 1]
    with pytest.raises(ValueError, match="target-label"):
        QualificationArtifact.from_dict(metadata_payload)


def test_invalid_multiclass_probabilities_are_rejected():
    labels, domains, probabilities, cheap_features, native_features = _source_data()
    probabilities[0] = [0.9, 0.9, 0.1]

    with pytest.raises(ValueError, match="sum to one"):
        fit_qualification_artifact(
            _identity(),
            class_names=("benign", "in-situ", "invasive"),
            source_labels=labels,
            source_probabilities=probabilities,
            source_domains=domains,
            source_latency_ms=(20.0,),
            source_cheap_features=cheap_features,
            cheap_encoder_fingerprint="cheap-v1",
            source_native_features=native_features,
            native_encoder_fingerprint="native-v1",
        )
