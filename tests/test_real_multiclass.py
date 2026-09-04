import numpy as np

from merit_feddg.qualification import ExpertIdentity, fit_qualification_artifact
from merit_feddg.real_multiclass import (
    _method_decision,
    _shuffle_indices,
    _target_class_coverage,
    bounded_candidate_fusion,
    fit_multiclass_temperature,
    run_real_multiclass_loco,
)
from merit_feddg.types import EvidenceRecord


def _artifact():
    identity = ExpertIdentity(
        "pathology", "model", "adapter", "pathology", "classification", "task"
    )
    labels = np.asarray([0, 1, 2, 0, 1, 2])
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    features = np.asarray([[-2, -2], [0, 0], [2, 2], [-1.8, -2], [0.1, 0], [1.9, 2]])
    artifact = fit_qualification_artifact(
        identity,
        class_names=("a", "b", "c"),
        source_labels=labels,
        source_probabilities=probabilities,
        source_domains=np.asarray(["center-a"] * 3 + ["center-b"] * 3),
        source_latency_ms=[10, 11],
        source_cheap_features=features,
        cheap_encoder_fingerprint="cheap",
        source_native_features=features,
        native_encoder_fingerprint="native",
        lcb_z=0,
    )
    return identity, artifact


def _config():
    return {
        "med_defer": {
            "controller": {"uncertainty_threshold": 0.5},
            "domain_trust": {"ood_temperature": 2.0, "maximum_pre_call_ood": 1.0},
            "guidance": {"strength": 1.0, "max_bias_norm": 0.7},
        },
        "qualification": {"minimum_source_domains": 2},
    }


def test_bounded_candidate_fusion_supports_true_multiclass_and_caps_delta():
    base = np.asarray([0.9, 0.2, 0.1])
    fused, delta = bounded_candidate_fusion(
        base, np.asarray([0, 4, -1]), 1.0, strength=2, max_delta_norm=0.5
    )
    assert fused.shape == (3,)
    assert np.linalg.norm(delta) <= 0.5 + 1e-9
    assert fused[1] > base[1]


def test_temperature_is_fitted_from_real_multiclass_source_targets():
    logits = np.asarray([[8, 0, 0], [0, 8, 0], [0, 0, 8], [4, 0, 0], [0, 4, 0], [0, 0, 4]])
    labels = np.asarray([0, 1, 2, 0, 1, 2])
    temperature = fit_multiclass_temperature(logits, labels)
    assert 0.05 <= temperature <= 20


def test_full_policy_checks_high_confidence_first_claim_without_binary_risk_model():
    identity, artifact = _artifact()
    scores, decision = _method_decision(
        method="med_defer_full",
        sample_id="target-1",
        base_scores=np.asarray([12.0, 0.0, -2.0]),
        expert_scores=np.asarray([-2.0, 5.0, 0.0]),
        cheap_feature=np.asarray([-2.0, -2.0]),
        native_feature=np.asarray([-2.0, -2.0]),
        image_quality=1.0,
        identity=identity,
        cheap_fingerprint="cheap",
        native_fingerprint="native",
        artifact=artifact,
        study_config=_config(),
    )
    assert decision["expert_called"]
    assert decision["gate"] > 0
    assert scores[1] > 0


def test_full_policy_refuses_expert_when_a_frozen_source_class_is_unseen():
    identity = ExpertIdentity(
        "pathology", "model", "adapter", "pathology", "classification", "task"
    )
    artifact = fit_qualification_artifact(
        identity,
        class_names=("a", "b", "c"),
        source_labels=np.asarray([0, 1, 0, 1]),
        source_probabilities=np.asarray(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.2, 0.7, 0.1]]
        ),
        source_domains=np.asarray(["center-a", "center-a", "center-b", "center-b"]),
        source_latency_ms=[10],
        source_cheap_features=np.asarray([[-1, 0], [1, 0], [-1, 0.1], [1, 0.1]]),
        cheap_encoder_fingerprint="cheap",
        source_native_features=np.asarray([[-1, 0], [1, 0], [-1, 0.1], [1, 0.1]]),
        native_encoder_fingerprint="native",
        lcb_z=0,
    )
    base = np.asarray([2.0, 0.0, -1.0])
    scores, decision = _method_decision(
        method="med_defer_full",
        sample_id="target-1",
        base_scores=base,
        expert_scores=np.asarray([-1.0, 2.0, 0.0]),
        cheap_feature=np.asarray([-1.0, 0.0]),
        native_feature=np.asarray([-1.0, 0.0]),
        image_quality=1.0,
        identity=identity,
        cheap_fingerprint="cheap",
        native_fingerprint="native",
        artifact=artifact,
        study_config=_config(),
    )

    np.testing.assert_allclose(scores, base)
    assert not decision["expert_called"]
    assert decision["reason"] == "insufficient-source-class-support"


def test_pre_call_ood_does_not_use_confident_wrong_generalist_class_as_veto():
    identity, artifact = _artifact()
    config = _config()
    config["med_defer"]["domain_trust"]["maximum_pre_call_ood"] = 0.5
    _, decision = _method_decision(
        method="med_defer_full",
        sample_id="confident-wrong-class",
        # Generalist confidently predicts class 0, but the cheap feature lies
        # on the known source manifold near class 2.
        base_scores=np.asarray([12.0, 0.0, -2.0]),
        expert_scores=np.asarray([-2.0, 0.0, 5.0]),
        cheap_feature=np.asarray([2.0, 2.0]),
        native_feature=np.asarray([2.0, 2.0]),
        image_quality=1.0,
        identity=identity,
        cheap_fingerprint="cheap",
        native_fingerprint="native",
        artifact=artifact,
        study_config=config,
    )
    assert decision["expert_called"]
    assert decision["pre_call_ood"] <= 0.5
    assert decision["reason"] == "geomean-lcb-cvar-native-ood-guidance"


def test_uncertainty_ablation_still_misses_a_high_confidence_case():
    identity, artifact = _artifact()
    _, decision = _method_decision(
        method="uncertainty_only",
        sample_id="target-1",
        base_scores=np.asarray([12.0, 0.0, -2.0]),
        expert_scores=np.asarray([-2.0, 5.0, 0.0]),
        cheap_feature=np.asarray([-2.0, -2.0]),
        native_feature=np.asarray([-2.0, -2.0]),
        image_quality=1.0,
        identity=identity,
        cheap_fingerprint="cheap",
        native_fingerprint="native",
        artifact=artifact,
        study_config=_config(),
    )
    assert not decision["expert_called"]
    assert decision["reason"] == "confident-none"


def test_target_label_is_not_an_input_to_the_inference_decision():
    assert "label" not in _method_decision.__annotations__
    assert "target" not in _method_decision.__annotations__


def test_wrong_capability_fails_closed():
    identity, artifact = _artifact()
    base = np.asarray([1.0, 0.0, -1.0])
    scores, decision = _method_decision(
        method="wrong_capability",
        sample_id="target-1",
        base_scores=base,
        expert_scores=np.asarray([-1, 4, 0]),
        cheap_feature=np.asarray([-2.0, -2.0]),
        native_feature=np.asarray([-2.0, -2.0]),
        image_quality=1.0,
        identity=identity,
        cheap_fingerprint="cheap",
        native_fingerprint="native",
        artifact=artifact,
        study_config=_config(),
    )
    np.testing.assert_allclose(scores, base)
    assert not decision["expert_called"]


def test_shuffled_control_is_a_derangement():
    records = [
        EvidenceRecord(
            sample_id=f"sample-{index}",
            domain="target-center",
            modality="pathology",
            candidates=["a", "b", "c"],
            label=index % 3,
            general_null_logits=np.zeros(3),
            general_visual_layers=np.zeros((1, 3)),
            expert_scores={"pathology": np.zeros(3)},
            broad_specialist_scores=np.zeros(3),
            router_probs={"pathology": 1.0},
            metadata={"domain_kind": "real_medical_center"},
        )
        for index in range(7)
    ]
    mapping = _shuffle_indices(records, seed=9)
    assert sorted(mapping.values()) == list(range(len(records)))
    assert all(mapping[record.sample_id] != index for index, record in enumerate(records))


def test_target_coverage_separates_structural_from_label_blind_sample_absence():
    candidates = ("a", "b", "c", "d")
    records = [
        EvidenceRecord(
            sample_id=f"target-{label}",
            domain="center-c",
            modality="pathology",
            candidates=list(candidates),
            label=label,
            general_null_logits=np.zeros(4),
            general_visual_layers=np.zeros((1, 4)),
            expert_scores={"pathology": np.zeros(4)},
            broad_specialist_scores=np.zeros(4),
            router_probs={"pathology": 1.0},
            metadata={"domain_kind": "real_medical_center"},
        )
        for label in (0, 1)
    ]
    coverage = _target_class_coverage(
        records,
        candidates,
        "center-c",
        {"task": {"structurally_unavailable_classes_by_center": {"center-c": ["c"]}}},
    )

    assert coverage["target_class_support"] == {"a": 1, "b": 1, "c": 0, "d": 0}
    assert coverage["observed_target_classes"] == ["a", "b"]
    assert coverage["structurally_unavailable_target_classes"] == ["c"]
    assert coverage["unobserved_sampled_target_classes"] == ["d"]


def test_end_to_end_real_loco_uses_centers_and_never_target_labels(tmp_path):
    records = []
    candidates = ["adventitia", "muscularis propria", "tumor tissue"]
    for center_index, center in enumerate(("hospital-a", "hospital-b", "hospital-c")):
        for label in range(3):
            base = np.full(3, -1.0)
            base[(label + (1 if center == "hospital-c" else 0)) % 3] = 2.0
            expert = np.full(3, -2.0)
            expert[label] = 3.0
            feature = [float(label * 2), float(center_index) / 10]
            records.append(
                EvidenceRecord(
                    sample_id=f"{center}-{label}",
                    domain=center,
                    modality="pathology",
                    candidates=candidates,
                    label=label,
                    general_null_logits=np.zeros(3),
                    general_visual_layers=base[None, :],
                    expert_scores={"pathology": expert},
                    broad_specialist_scores=np.zeros(3),
                    router_probs={"pathology": 1.0},
                    metadata={
                        "domain_kind": "real_medical_center",
                        "slide_id": f"slide-{center}-{label}",
                        "cheap_domain_feature": feature,
                        "expert_native_features": {"pathology": feature},
                        "expert_latency_ms": {"pathology": 10.0},
                    },
                )
            )
    model_config = {
        "broad_specialist": {"id": "router", "adapter": "contrastive_biomedclip"},
        "experts": {"pathology": {"id": "conch", "adapter": "contrastive_conch"}},
    }
    study_config = {
        **_config(),
        "seed": 1,
        "task": {
            "id": "real-three-class",
            "expert_id": "pathology",
            "modality": "pathology",
            "capability": "classification",
        },
    }
    report = run_real_multiclass_loco(
        records,
        model_config,
        study_config,
        tmp_path / "result.json",
    )
    for fold in report["folds"]:
        assert fold["task_type"] == "real_multiclass"
        assert fold["binary_error_estimator"] is False
        assert fold["target_labels_used_during_qualification_or_inference"] is False
        assert fold["metrics"]["med_defer_full"]["target_label_used_by_decision"] is False
        assert fold["metrics"]["med_defer_full"]["paired_bootstrap"]["unit"] == ("slide_id cluster")
        assert (
            fold["metrics"]["med_defer_full"]["fixed_taxonomy_macro_f1"]
            == fold["metrics"]["med_defer_full"]["macro_f1"]
        )
        assert fold["target_class_support"] == dict.fromkeys(candidates, 1)
        assert fold["observed_target_classes"] == candidates
        assert fold["structurally_unavailable_target_classes"] == []
        assert fold["unobserved_sampled_target_classes"] == []
        assert fold["full_vs_shuffled"]["paired_bootstrap"]["unit"] == "slide_id cluster"
        assert fold["full_vs_shuffled"]["paired_sign_test"]["unit"] == (
            "slide_id mean patch-accuracy delta"
        )
        full = fold["predictions"]["med_defer_full"]
        shuffled = fold["predictions"]["shuffled_evidence"]
        assert [item["expert_called"] for item in shuffled] == [
            item["expert_called"] for item in full
        ]
        assert [item["gate"] for item in shuffled] == [item["gate"] for item in full]
        assert all(
            item["decision_reason"] == "shuffled-scores-with-matched-full-gate-and-call"
            for item in shuffled
        )
    assert report["aggregate"]["med_defer_full"]["worst_domain_accuracy"] == min(
        fold["metrics"]["med_defer_full"]["accuracy"] for fold in report["folds"]
    )
    assert "continue_scaling" not in report
    assert "posthoc_mechanism_diagnostics" in report
    assert (
        report["aggregate"]["med_defer_full"]["fixed_taxonomy_macro_f1"]
        == report["aggregate"]["med_defer_full"]["macro_f1"]
    )
    assert report["full_vs_shuffled"]["paired_bootstrap"]["clusters"] == 9
    assert report["full_vs_shuffled"]["paired_sign_test"]["unit"] == (
        "slide_id mean patch-accuracy delta"
    )
    assert (tmp_path / "result.json").is_file()
