"""Contract and numerical tests; no clinical checkpoints or labels required."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from merit_feddg.capabilities import EvidenceItem
from merit_feddg.tensor_evidence import TensorContract, compile_tensor_evidence


def contract(max_records=64):
    return TensorContract(("lung", "opacity"), ("chest",), tuple(
        {"expert": expert, "scope": "cxr", "label": label,
         "concept": concept, "scope_id": "chest"}
        for expert in ("old", "replacement")
        for label, concept in (("Left Lung", "lung"), ("Lung Opacity", "opacity"))
    ), grid_size=2, max_records=max_records)


def classification(score=0.8, expert="old"):
    return EvidenceItem("c", expert, "classification", "cxr", {
        "findings": [{"finding": "Lung Opacity", "score": score}],
        "score_semantics": "uncalibrated_independent_sigmoid",
    })


def segmentation():
    return EvidenceItem("s", "old", "segmentation", "cxr", {"structures": [{
        "anatomical_structure": "Left Lung", "mask_coordinate_system": "original_image",
        "mask": {"encoding": "rle-row-major-zero-first", "size": [2, 4], "counts": [0, 4, 4]},
    }]})


def detection():
    return EvidenceItem("d", "old", "detection", "cxr", {"detections": [{
        "label": "Lung Opacity", "bbox_xyxy_normalized_original_image": [0, 0, 0.5, 1],
    }]})


def test_mixed_outputs_preserve_geometry_and_semantics():
    packet = compile_tensor_evidence([classification(), segmentation(), detection()], (4, 2), contract())
    assert len(packet) == 3 and not packet.rejected
    assert packet.semantics.tolist() == [2, 0, 0]
    assert packet.numeric[0, :4].tolist() == pytest.approx([0.8, 1, 0, 0])
    np.testing.assert_array_equal(packet.regions[0], [0, 0, 0, 0])
    # Original 4x2 image is vertically padded by one pixel: no stretch to square.
    np.testing.assert_array_equal(packet.regions[1], [0.5, 0.5, 0, 0])
    np.testing.assert_array_equal(packet.regions[2], [0.5, 0, 0.5, 0])


def test_expert_order_replacement_and_budget_are_contract_invariant():
    a = compile_tensor_evidence([classification(), segmentation()], (4, 2), contract(1))
    b = compile_tensor_evidence([segmentation(), classification(expert="replacement")], (4, 2), contract(1))
    np.testing.assert_array_equal(a.numeric, b.numeric)
    np.testing.assert_array_equal(a.concept, b.concept)
    assert a.sources != b.sources and a.rejected[0]["reason"] == "token_budget"


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.2, 1.2, True])
def test_invalid_classification_scores_rejected(score):
    packet = compile_tensor_evidence([classification(score)], (4, 2), contract())
    assert not len(packet) and packet.rejected


def test_unknown_concept_and_score_semantics_do_not_get_guessed():
    item = classification()
    item.payload["findings"][0]["finding"] = "unknown new disease"
    packet = compile_tensor_evidence([item], (4, 2), contract())
    assert packet.rejected[0]["reason"] == "unregistered_concept_or_scope"
    item = classification()
    item.payload.pop("score_semantics")
    assert not len(compile_tensor_evidence([item], (4, 2), contract()))


def test_empty_mask_cannot_fall_back_to_prompt_box():
    item = segmentation()
    entry = item.payload["structures"][0]
    entry["mask"]["counts"] = [8]
    entry["bbox_xyxy_normalized_original_image"] = [0, 0, 1, 1]
    assert not len(compile_tensor_evidence([item], (4, 2), contract()))


def test_crop_mask_transform_checked_and_restored():
    item = segmentation()
    entry = item.payload["structures"][0]
    entry["mask_coordinate_system"] = "model_grid_of_center_crop"
    entry["mask"] = {"encoding": "rle-row-major-zero-first", "size": [2, 2], "counts": [0, 4]}
    item.payload["image_transform"] = {
        "original_size_hw": [2, 4], "model_size_hw": [2, 2],
        "crop_box_xyxy_normalized": [0.25, 0, 0.75, 1],
    }
    packet = compile_tensor_evidence([item], (4, 2), contract())
    np.testing.assert_array_equal(packet.regions, [[0.25] * 4])
    assert not len(compile_tensor_evidence([item], (8, 2), contract()))


def test_tiny_box_has_nonzero_coverage_without_inventing_large_roi():
    item = detection()
    item.payload["detections"][0]["bbox_xyxy_normalized_original_image"] = [0, 0, 0.001, 0.001]
    packet = compile_tensor_evidence([item], (4, 2), contract())
    assert len(packet) == 1 and 0 < packet.regions.sum() < 0.001


def test_targets_and_generation_are_not_native_tensor_evidence():
    item = replace(segmentation(), provenance={"target_masks_used": True})
    text = EvidenceItem("g", "old", "generation", "cxr", {"generated_text": "normal"})
    packet = compile_tensor_evidence([item, text], (4, 2), contract())
    assert not len(packet) and len(packet.rejected) == 2


def bridge_fixture():
    torch = pytest.importorskip("torch")
    from merit_feddg.tensor_bridge import NativeTensorBridge

    torch.manual_seed(7)
    bridge = NativeTensorBridge(contract(), visual_dim=8, width=8, queries=2, heads=2)
    packet = compile_tensor_evidence([classification(), segmentation()], (4, 2), contract())
    return torch, bridge, packet, torch.randn(1, 4, 8)


def test_zero_gate_empty_evidence_identity_and_trainable_gate():
    torch, bridge, packet, visual = bridge_fixture()
    assert bridge.eval()(visual, packet) is visual
    empty = compile_tensor_evidence([], (4, 2), contract())
    assert bridge(visual, empty) is visual
    result = bridge.train()(visual, packet)
    torch.testing.assert_close(result, visual, rtol=0, atol=0)
    result.square().sum().backward()
    assert bridge.gate.grad is not None and bridge.gate.grad.abs() > 0


def test_evidence_affects_visual_tokens_and_encoder_receives_gradient():
    torch, bridge, packet, visual = bridge_fixture()
    with torch.no_grad():
        bridge.gate.fill_(0.5)
    alternate = compile_tensor_evidence([classification(0.1), segmentation()], (4, 2), contract())
    a, b = bridge(visual, packet), bridge(visual, alternate)
    assert not torch.allclose(a, b, atol=1e-7)
    a.square().sum().backward()
    assert bridge.numeric_encoders[0][0].weight.grad.abs().sum() > 0
    assert bridge.shapes.weight.grad.abs().sum() > 0


def test_reader_invariance_to_record_order_and_expert_identity():
    torch, bridge, packet, visual = bridge_fixture()
    with torch.no_grad():
        bridge.gate.fill_(0.5)
    reordered = replace(packet, **{key: getattr(packet, key)[::-1].copy() for key in (
        "kind", "concept", "scope", "semantics", "numeric", "regions"
    )})
    torch.testing.assert_close(bridge(visual, packet), bridge(visual, reordered))


def test_checkpoint_roundtrip_and_reject_untrained(tmp_path):
    torch, bridge, packet, visual = bridge_fixture()
    path = tmp_path / "bridge.pt"
    with pytest.raises(ValueError, match="nonzero training"):
        bridge.save(path, training_steps=0, source_domains=["a"])
    bridge.save(path, training_steps=1, source_domains=["a"])
    restored = type(bridge).load(path)
    assert asdict(restored.contract) == asdict(bridge.contract)
    torch.testing.assert_close(restored(visual, packet), bridge.eval()(visual, packet))


def test_hook_cleanup_and_actual_projection_injection():
    torch, bridge, packet, visual = bridge_fixture()
    from merit_feddg.tensor_bridge import tensor_projector_context

    with torch.no_grad():
        bridge.gate.fill_(0.5)
    projector = torch.nn.Identity()
    probe = SimpleNamespace(
        tensor_bridge=bridge, deterministic_image_padding=True,
        image_processor=SimpleNamespace(crop_size={"height": 4, "width": 4},
                                        size={"shortest_edge": 4}, do_resize=True, do_center_crop=True),
        model=SimpleNamespace(config=SimpleNamespace(image_aspect_ratio="pad", hidden_size=8),
                              get_vision_tower=lambda: SimpleNamespace(select_feature="patch", num_patches=4),
                              get_model=lambda: SimpleNamespace(mm_projector=projector)),
    )
    with tensor_projector_context(probe, packet):
        assert not torch.equal(projector(visual), visual)
        with pytest.raises(RuntimeError, match="reentrant"), tensor_projector_context(probe, packet):
            pass
    assert projector(visual) is visual and not projector._forward_hooks
    with pytest.raises(ValueError, match="test error"), tensor_projector_context(probe, packet):
        raise ValueError("test error")
    assert not projector._forward_hooks and not probe._tensor_hook_active


def test_tensor_runtime_has_no_payload_text_and_keeps_prefix():
    from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
    from merit_feddg.evidence_need import evidence_memory

    calls = []
    probe = SimpleNamespace(tensor_bridge=object(), new_tensor_answer_session=lambda image, prompt, items:
                            SimpleNamespace(propose=lambda prefix, **kw: calls.append((prompt, items, prefix)) or ["ok"]))
    config = ValueGenerationConfig(evidence_style="tensor", visual_views=0)
    state = NativeState(prefix=(3, 4), items=(classification(),))
    session = NativeSession(probe, Image.new("RGB", (4, 2)), "Question", "Question", config)
    assert session.propose(state, 2) == "ok"
    assert calls == [("Question", state.items, (3, 4))]
    assert evidence_memory(state.items, "Question", config) == []
    with pytest.raises(ValueError, match="visual_views=0"):
        ValueGenerationConfig(evidence_style="tensor", visual_views=1)


def test_source_training_rejects_target_cases(tmp_path):
    pytest.importorskip("torch")
    import json

    from merit_feddg.tensor_train import source_rows

    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps({"id": "1", "split": "target", "domain": "a", "image": "a.png",
                               "prompt": "q", "answer": "a", "evidence": []}))
    with pytest.raises(ValueError, match="source-only"):
        source_rows(path)


def test_teacher_forcing_trains_bridge_through_frozen_model(tmp_path):
    torch = pytest.importorskip("torch")
    from merit_feddg.llava_generalist import LlavaMedGeneralist
    from merit_feddg.tensor_train import train

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mm_projector = torch.nn.Linear(8, 8)
            self.embedding = torch.nn.Embedding(16, 8)
            self.config = SimpleNamespace(hidden_size=8, image_aspect_ratio="pad")

        def get_input_embeddings(self):
            return self.embedding

        def get_model(self):
            return self

        def get_vision_tower(self):
            return SimpleNamespace(select_feature="patch", num_patches=4)

        def forward(self, input_ids, labels, **kwargs):
            assert labels.tolist() == [[-100, -100, 7, 2]]
            assert input_ids.tolist() == [[1, -200, 7, 2]]
            h = self.mm_projector(torch.ones(1, 4, 8))
            return SimpleNamespace(loss=(h - 1).square().mean())

    probe = LlavaMedGeneralist.__new__(LlavaMedGeneralist)
    probe.model = TinyModel()
    probe.deterministic_image_padding = True
    probe.image_processor = SimpleNamespace(crop_size={"height": 4, "width": 4},
                                            size={"shortest_edge": 4}, do_center_crop=True, do_resize=True)
    probe.tokenizer = SimpleNamespace(encode=lambda *a, **kw: [7], eos_token_id=2)
    probe._inputs = lambda *a: {"inputs": torch.tensor([[1, -200]]),
                               "attention_mask": torch.ones(1, 2, dtype=torch.long)}
    probe._validate_context = lambda *a: None
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 2)).save(image)
    rows = [{"id": "a", "split": "source", "domain": "hospital-a", "image": str(image),
             "prompt": "q", "answer": "a", "evidence": [asdict(classification())]}]
    before = {k: v.detach().clone() for k, v in probe.model.named_parameters()}
    bridge, losses = train(probe, rows, contract(), epochs=3, width=8)
    assert len(losses) == 3 and np.isfinite(losses).all() and bridge.gate.item() != 0
    for key, value in probe.model.named_parameters():
        assert torch.equal(value, before[key]) and value.grad is None


def test_new_expert_binding_changes_no_weights():
    torch, bridge, _packet, _visual = bridge_fixture()
    before = {k: v.clone() for k, v in bridge.state_dict().items()}
    bridge.bind_expert([{"expert": "unseen", "scope": "new_scope", "label": "new_alias",
                        "concept": "opacity", "scope_id": "chest"}])
    item = classification(expert="unseen")
    item = replace(item, scope="new_scope")
    item.payload["findings"][0]["finding"] = "new_alias"
    assert len(compile_tensor_evidence([item], (4, 2), bridge.contract)) == 1
    assert all(torch.equal(value, before[key]) for key, value in bridge.state_dict().items())
    with pytest.raises(ValueError, match="unregistered"):
        bridge.bind_expert([{"expert": "unseen2", "scope": "cxr", "label": "new disease",
                            "concept": "new disease", "scope_id": "chest"}])


def test_production_loader_rejects_same_dimension_wrong_base(tmp_path):
    from merit_feddg.llava_generalist import LlavaMedGeneralist

    _torch, bridge, _packet, _visual = bridge_fixture()
    path = tmp_path / "bridge.pt"
    bridge.save(path, training_steps=1, source_domains=["a"], base_identity="base-a")
    probe = LlavaMedGeneralist.__new__(LlavaMedGeneralist)
    with pytest.raises(ValueError, match="base-model provenance"):
        probe.load_tensor_bridge(path, expected_base_identity="base-b")
    with pytest.raises(ValueError, match="base-model provenance"):
        probe.load_tensor_bridge(path)


def test_tensor_does_not_silently_ignore_uncertainty_or_run_text_diagnostics():
    from merit_feddg.capability_diagnostics import collect_diagnostic_case
    from merit_feddg.capability_runtime import ValueGenerationConfig

    with pytest.raises(ValueError, match="uncertainty envelopes"):
        ValueGenerationConfig(evidence_style="tensor", visual_views=0,
                              uncertainty_from_probe=True, behavior_probe="audit")
    runtime = SimpleNamespace(config=ValueGenerationConfig(evidence_style="tensor", visual_views=0))
    with pytest.raises(ValueError, match="text channel diagnostics"):
        collect_diagnostic_case(runtime, None, None)
