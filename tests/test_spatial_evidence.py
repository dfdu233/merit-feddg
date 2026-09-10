"""Semantic limits, geometry, source balance and production hook tests (CPU)."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image

from merit_feddg.capabilities import CapabilityRequest, EvidenceItem
from merit_feddg.spatial_evidence import (
    SpatialEvidenceBridge,
    encode_soft_mask,
    mapped_soft_mask,
    spatial_packet,
)
from merit_feddg.tensor_evidence import preserves_tensor_records


def mask_item(label="lung", source="A", values=None):
    if values is None:
        values = [[1, 1], [0, 0]]
    return EvidenceItem(label, source, "segmentation", "anatomy", {"structures": [{
        "label": label, "soft_mask": encode_soft_mask(values),
        "mask_coordinate_system": "original_image"}]})


def test_native_geometry_returns_existing_features_without_parameters():
    bridge = SpatialEvidenceBridge(1, 2)
    packet = spatial_packet([mask_item()], (2, 2), grid_size=2)
    h = torch.tensor([[[0.], [4.], [8.], [12.]]])
    # Region mean=2, original and single expert each have unit mass.
    torch.testing.assert_close(bridge(h, packet), torch.tensor([[[1.], [3.], [8.], [12.]]]))
    assert list(bridge.parameters()) == [] and bridge.state_dict() == {}
    assert bridge.last_audit["max_abs_token_delta"] == 1
    assert not bridge.last_audit["class_scores_injected"]
    assert bridge(h, spatial_packet([], (2, 2), grid_size=2)) is h
    packet.importance[:] = 0
    assert bridge(h, packet) is h


def test_source_balance_duplicate_invariance_and_question_importance():
    a = mask_item()
    b = mask_item("heart", values=[[0, 0], [1, 1]])
    c = mask_item("kidney", source="B", values=[[1, 0], [1, 0]])
    p = spatial_packet([a, b, c], (2, 2), grid_size=2, question="Where is the lung?")
    assert sum(p.importance[i] for i, s in enumerate(p.sources) if s[0] == "A") == pytest.approx(.5)
    assert p.importance[p.labels.index("lung")] > p.importance[p.labels.index("heart")]
    duplicate = replace(a, evidence_id="duplicate")
    q = spatial_packet([c, b, duplicate, a], (2, 2), grid_size=2, question="Where is the lung?")
    h = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    bridge = SpatialEvidenceBridge(2, 2)
    torch.testing.assert_close(bridge(h, p), bridge(h, q))
    equal = spatial_packet([a, b], (2, 2), grid_size=2, weighting="equal", question="lung")
    np.testing.assert_allclose(equal.importance, [.5, .5])


def test_score_only_is_rejected_but_cam_is_explicitly_supported():
    item = EvidenceItem("cls", "A", "classification", "findings", {
        "findings": [{"finding": "opacity", "score": .8}],
        "score_semantics": "uncalibrated_independent_sigmoid"})
    packet = spatial_packet([item], (2, 2), grid_size=2)
    assert not len(packet)
    assert packet.rejected[0]["reason"] == "classification_without_spatial_support"
    item.payload["findings"][0]["spatial_support"] = {
        "soft_mask": encode_soft_mask([[1, .5], [0, 0]]),
        "mask_coordinate_system": "original_image"}
    packet = spatial_packet([item], (2, 2), grid_size=2)
    assert len(packet) == 1 and packet.numeric[0, 0] == pytest.approx(.8)
    np.testing.assert_allclose(packet.regions, [[1, .5, 0, 0]])


def test_new_labels_do_not_evict_prior_identity_and_targets_are_rejected():
    a, b = mask_item("zebra"), mask_item("aorta", source="B")
    before = spatial_packet([a], (2, 2), grid_size=2)
    after = spatial_packet([a, b], (2, 2), grid_size=2)
    assert preserves_tensor_records(before, after)
    forbidden = replace(a, provenance={"target_masks_used": True})
    assert not len(spatial_packet([forbidden], (2, 2), grid_size=2))
    assert not len(spatial_packet([mask_item(values=[[0, 0], [0, 0]])], (2, 2), grid_size=2))


def test_soft_masks_preserve_padding_crop_and_fractional_values():
    entry = {"soft_mask": encode_soft_mask([[.2, .6], [0, 0]]),
             "mask_coordinate_system": "model_grid_of_center_crop"}
    payload = {"image_transform": {"original_size_hw": [2, 4], "model_size_hw": [2, 2],
                                   "crop_box_xyxy_normalized": [.25, 0, .75, 1]}}
    mask = mapped_soft_mask(entry, payload, (4, 2))
    np.testing.assert_allclose(np.asarray(mask), [[0, .2, .6, 0], [0, 0, 0, 0]])
    item = EvidenceItem("s", "A", "segmentation", "anatomy", {**payload, "structures": [{"label": "lung", **entry}]})
    packet = spatial_packet([item], (4, 2), grid_size=2)
    np.testing.assert_allclose(packet.regions, [[.05, .15, 0, 0]], atol=1e-7)
    with pytest.raises(ValueError):
        mapped_soft_mask(entry, payload, (8, 2))


@pytest.mark.parametrize("value", [[[float("nan")]], [[-1]], [[2]], []])
def test_invalid_soft_values(value):
    with pytest.raises(ValueError):
        encode_soft_mask(value)


def test_corrupt_soft_payload_is_audited_not_injected():
    item = mask_item()
    item.payload["structures"][0]["soft_mask"]["data"] = "YWJj"
    packet = spatial_packet([item], (2, 2), grid_size=2)
    assert not len(packet)
    assert packet.rejected[0]["reason"] == "invalid_empty_or_unmapped_soft_mask"


def test_xrv_cam_captures_same_forward_and_does_not_change_weights():
    from merit_feddg.experts.native_xrv import XrvCapabilityAdapter

    class TinyDenseNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Identity()
            self.classifier = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.classifier(self.features(x).relu().mean((-2, -1))).sigmoid()

    model = TinyDenseNet().eval().requires_grad_(False)
    image = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    expected = model(image)[0].detach().numpy()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    adapter = XrvCapabilityAdapter.__new__(XrvCapabilityAdapter)
    adapter.capability, adapter.torch, adapter.model = "classification", torch, model
    adapter.targets = ("A", "B")
    adapter._inputs = lambda unused: (image, {})
    labels, scores, _, maps = adapter.classify_with_spatial(None)
    assert labels == ("A", "B") and maps.shape == (2, 2, 2)
    np.testing.assert_array_equal(scores, expected)
    assert not model.features._forward_hooks
    for key, value in model.state_dict().items():
        assert torch.equal(before[key], value)


def test_real_projector_hook_injection_and_cleanup():
    from merit_feddg.tensor_bridge import tensor_projector_context

    bridge, projector = SpatialEvidenceBridge(2, 2), torch.nn.Identity()
    probe = SimpleNamespace(tensor_bridge=bridge, deterministic_image_padding=True,
        image_processor=SimpleNamespace(crop_size={"height": 4, "width": 4},
            size={"shortest_edge": 4}, do_resize=True, do_center_crop=True),
        model=SimpleNamespace(config=SimpleNamespace(image_aspect_ratio="pad", hidden_size=2),
            get_vision_tower=lambda: SimpleNamespace(select_feature="patch", num_patches=4),
            get_model=lambda: SimpleNamespace(mm_projector=projector)))
    h = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    packet = spatial_packet([mask_item()], (2, 2), grid_size=2)
    with tensor_projector_context(probe, packet):
        assert not torch.equal(projector(h), h)
    assert projector(h) is h and not projector._forward_hooks
    with pytest.raises(RuntimeError), tensor_projector_context(probe, packet):
        raise RuntimeError("interrupted generation")
    assert not projector._forward_hooks and not probe._tensor_hook_active


def test_cam_uses_existing_class_weights_and_has_no_backward():
    from merit_feddg.experts.native_xrv import positive_class_maps

    features = torch.tensor([[[[1., 2.], [3., 4.]], [[4., 3.], [2., 1.]]]])
    weights = torch.tensor([[1., 0.], [0., 1.], [-1., -1.]])
    with torch.inference_mode():
        maps = positive_class_maps(features, weights, size=(2, 2))
    torch.testing.assert_close(maps[0, 0], features[0, 0] / 4)
    torch.testing.assert_close(maps[0, 1], features[0, 1] / 4)
    assert not maps[0, 2].any() and maps.grad_fn is None


def test_biomedparse_adapter_uses_soft_masks_not_target_statistics(monkeypatch):
    from merit_feddg.experts.native_biomedparse import BiomedParseCapabilityExpert

    expert = BiomedParseCapabilityExpert("unused", "bio", "unused", {"mri": ["MRI-FLAIR-Brain"]},
        max_prompts=1, group_requirements={"MRI-FLAIR-Brain": [["flair"]]})
    request = CapabilityRequest("id", Image.new("RGB", (2, 2)), "Where is edema?", "mri",
                                "open_vqa", "test", "image", "segmentation", scope="biomedical_objects")
    monkeypatch.setattr(expert, "_load", lambda: None)
    assert expert.infer(request).reason == "unknown_anatomy_or_sequence"
    expert.objects = {"MRI-FLAIR-Brain": ["tumor core", "edema"]}
    seen = []
    expert.predict = lambda model, image, prompts: seen.append(prompts) or np.array([[[.2, .4], [.6, .8]]])
    result = expert.infer(replace(request, question="Where is edema in this FLAIR image?"))
    assert seen == [["edema"]]
    assert result.items[0].payload["prompt_budget_omitted"] == ["tumor core"]
    packet = spatial_packet(result.items, (2, 2), grid_size=2)
    np.testing.assert_allclose(packet.regions, [[.2, .4, .6, .8]])


def test_strict_config_and_four_matched_arms():
    from merit_feddg.capability_runtime import ValueGenerationConfig
    from merit_feddg.generalist_factory import load_generalist
    from merit_feddg.io import load_experiment_yaml
    from merit_feddg.matched_evaluation import experiment_arms

    config = load_experiment_yaml("configs/matched_vector_gate.yaml")
    assert config["generalist"]["training_free_spatial"]
    assert "tensor_bridge_checkpoint" not in config["generalist"]
    decoder = ValueGenerationConfig(**config["capability_value"]["generation"])
    arms = experiment_arms(decoder, "spatial")
    assert set(arms) == {"generalist", "spatial_equal", "spatial_weighted", "spatial_gate"}
    assert all(a.max_new_tokens == a.vector_gate_probe_tokens == 64 for a in arms.values())
    assert replace(arms["spatial_gate"], vector_gate="off") == arms["spatial_weighted"]
    with pytest.raises(ValueError, match="retired"):
        load_generalist({"tensor_bridge_checkpoint": "unused"})
