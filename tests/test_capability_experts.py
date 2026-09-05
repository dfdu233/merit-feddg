"""Native tool semantics and leakage audits; these are not clinical accuracy tests."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

import merit_feddg.capability_experts as module
from merit_feddg.capabilities import CapabilityRequest, CapabilityResult, EvidenceItem
from merit_feddg.capability_experts import CapabilityPool, encode_binary_mask


def request(image, capability="classification", **kwargs):
    values = {
        "sample_id": "query",
        "image": str(image),
        "question": "What is visible?",
        "modality": "pathology",
        "task": "open_vqa",
        "domain": "held-out",
        "group_id": "query-group",
        "capability": capability,
    }
    values.update(kwargs)
    return CapabilityRequest(**values)


def picture(tmp_path, name="query", color=(230, 100, 40)):
    path = tmp_path / f"{name}.png"
    Image.new("RGB", (8, 5), color).save(path)
    return path


class Encoder:
    def __init__(self):
        self.image_calls = 0
        self.text_prompts = []

    def domain_embedding(self, image):
        self.image_calls += 1
        with Image.open(image) as opened:
            return np.asarray(opened.convert("RGB"), dtype=float).mean(axis=(0, 1))

    def _text_embeddings(self, prompts):
        self.text_prompts.append(prompts)
        return np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])[: len(prompts)]


@pytest.fixture
def fake_encoder(monkeypatch):
    model = Encoder()
    loads = []

    def loader(spec, artifacts):
        loads.append(spec)
        return model

    monkeypatch.setattr(module, "_expert_from_spec", loader)
    return model, loads


def classification_spec(**kwargs):
    return {
        "id": "microsoft/test",
        "adapter": "biomedclip",
        "capabilities": ["classification"],
        "catalog": [
            {"name": "tissue", "prompt": "A tissue image"},
            {"name": "cells", "prompt": "An image of cells"},
        ],
        **kwargs,
    }


def source(path, name, domain="source-a", group=None, **kwargs):
    return {
        "id": name,
        "image": str(path),
        "domain": domain,
        "group_id": group or name,
        "question": f"Question for {name}",
        "modality": "pathology",
        "role": "source",
        **kwargs,
    }


def test_classification_uses_catalog_not_question_or_continuation(tmp_path, fake_encoder):
    model, _ = fake_encoder
    image = picture(tmp_path)
    pool = CapabilityPool({"concept": classification_spec()}, None)
    result = pool.infer("concept", request(image, question="TARGET SECRET", generated_prefix="XYZ"))
    evidence = result.items[0]
    assert model.text_prompts == [["A tissue image", "An image of cells"]]
    assert evidence.confidence is None
    assert evidence.payload["score_semantics"] == "relative_similarity"
    assert evidence.payload["unlisted_concepts"] == "unknown"
    scores = [entry["similarity"] for entry in evidence.payload["catalog"]]
    assert sum(scores) != pytest.approx(1.0)
    assert "TARGET SECRET" not in evidence.summary
    assert evidence.provenance["target_candidates_used"] is False


def test_catalog_is_required_and_static(tmp_path, fake_encoder):
    pool = CapabilityPool({"concept": classification_spec(catalog=[])}, None)
    with pytest.raises(ValueError, match="fixed config catalog"):
        pool.infer("concept", request(picture(tmp_path)))


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"capability": "segmentation"}, "wrong_capability"),
        ({"modality": "cxr"}, "wrong_modality"),
        ({"task": "report_generation"}, "wrong_task"),
        ({"scope": "diagnosis"}, "wrong_scope"),
    ],
)
def test_incompatible_request_never_loads_model(tmp_path, fake_encoder, changes, reason):
    _, loads = fake_encoder
    pool = CapabilityPool(
        {
            "tool": classification_spec(
                modalities=["pathology"], tasks=["open_vqa"], scope="morphology"
            )
        },
        None,
    )
    result = pool.infer("tool", request(picture(tmp_path), **changes))
    assert result.reason == reason
    assert not result.items and not loads


def test_retrieval_excludes_domain_group_and_identical_pixels(tmp_path, fake_encoder):
    query = picture(tmp_path)
    copied = picture(tmp_path, "copy")
    rows = [
        source(picture(tmp_path, "same-domain", (1, 2, 3)), "same-domain", "source-a"),
        source(picture(tmp_path, "same-group", (4, 5, 6)), "same-group", "source-b", "query-group"),
        source(copied, "same-pixels", "source-b"),
        source(picture(tmp_path, "allowed", (220, 90, 30)), "allowed", "source-b"),
        source(
            picture(tmp_path, "modality", (230, 100, 41)),
            "wrong-modality",
            "source-b",
            modality="cxr",
        ),
    ]
    refs = {row["id"]: [f"Source record {row['id']}"] for row in rows}
    pool = CapabilityPool(
        {"retrieval": {"id": "microsoft/test", "adapter": "source_retrieval"}}, None, rows, refs
    )
    result = pool.infer("retrieval", request(query, "retrieval", domain="source-a"))
    evidence = result.items[0]
    assert [x["source_id"] for x in evidence.payload["references"]] == ["allowed"]
    assert evidence.payload["references"][0]["source_reference"] == ["Source record allowed"]
    assert evidence.payload["query_diagnosis"] == "not_inferred"
    assert evidence.provenance["excluded_domain"] == "source-a"
    assert evidence.confidence is None


def test_empty_retrieval_does_not_load_encoder(tmp_path, fake_encoder):
    _, loads = fake_encoder
    image = picture(tmp_path)
    rows = [source(image, "query", "source-a")]
    pool = CapabilityPool(
        {"r": {"id": "microsoft/test", "adapter": "source_retrieval"}}, None, rows
    )
    result = pool.infer("r", request(image, "retrieval", domain="source-a"))
    assert result.reason == "no_eligible_sources"
    assert not loads


def test_target_records_and_references_rejected(tmp_path):
    row = source(picture(tmp_path), "source")
    with pytest.raises(ValueError, match="target records"):
        CapabilityPool({}, None, [{**row, "role": "target"}])
    with pytest.raises(ValueError, match="source IDs only"):
        CapabilityPool({}, None, [row], {"target": ["SECRET"]})


def test_retrieval_and_classification_share_model_and_features(tmp_path, fake_encoder):
    model, loads = fake_encoder
    query = picture(tmp_path)
    rows = [source(picture(tmp_path, "source", (12, 13, 14)), "source")]
    pool = CapabilityPool(
        {
            "c": classification_spec(),
            "r": {"id": "microsoft/test", "adapter": "source_retrieval"},
        },
        None,
        rows,
        {"source": ["reference"]},
    )
    pool.infer("c", request(query))
    pool.infer("r", request(query, "retrieval"))
    pool.infer("r", request(query, "retrieval", generated_prefix="a new prefix"))
    assert len(loads) == 1
    assert model.image_calls == 2
    pool.reset_case()
    pool.infer("r", request(query, "retrieval"))
    assert model.image_calls == 3  # only query recomputed; source index persists


def test_index_identity_changes_with_reference_and_actual_pixels(tmp_path, fake_encoder):
    path = picture(tmp_path)
    rows = [source(path, "source")]
    first = CapabilityPool({}, None, rows, {"source": ["one"]}).source_index_identity()
    second = CapabilityPool({}, None, rows, {"source": ["two"]}).source_index_identity()
    Image.new("RGB", (8, 5), (0, 0, 0)).save(path)
    third = CapabilityPool({}, None, rows, {"source": ["one"]}).source_index_identity()
    assert len({first["fingerprint"], second["fingerprint"], third["fingerprint"]}) == 3


def test_native_factory_preserves_masks_boxes_text(tmp_path, monkeypatch):
    payload = {
        "mask": [[0, 1], [1, 0]],
        "boxes": [[0.0, 0.0, 1.0, 1.0]],
        "generated_text": "Native tool description",
    }

    class Native:
        def infer(self, req):
            return CapabilityResult(
                "native",
                req.capability,
                (
                    EvidenceItem(
                        "native-1",
                        "native",
                        "segmentation",
                        "region",
                        payload,
                        summary="Foreground",
                    ),
                ),
            )

    monkeypatch.setattr(module, "_expert_from_spec", lambda *_: Native())
    pool = CapabilityPool(
        {
            "native": {
                "id": "external/model",
                "factory": "external.module:factory",
                "capabilities": ["segmentation"],
                "scope": "region",
            }
        },
        None,
    )
    result = pool.infer("native", request(picture(tmp_path), "segmentation", scope="region"))
    assert result.items[0].payload == payload


def test_native_factory_cannot_spoof_expert_identity(tmp_path, monkeypatch):
    class Native:
        def infer(self, req):
            return CapabilityResult("someone-else", req.capability, ())

    monkeypatch.setattr(module, "_expert_from_spec", lambda *_: Native())
    pool = CapabilityPool(
        {"native": {"id": "external/model", "factory": "m:f", "capabilities": ["generation"]}}, None
    )
    with pytest.raises(ValueError, match="identity"):
        pool.infer("native", request(picture(tmp_path), "generation"))


def test_medsam_without_roi_does_not_load(tmp_path, fake_encoder):
    pool = CapabilityPool({"mask": {"id": "wanglab/medsam-vit-base", "adapter": "medsam"}}, None)
    result = pool.infer("mask", request(picture(tmp_path), "segmentation"))
    assert result.reason == "explicit_roi_required"
    assert not pool.models


def test_medsam_mask_is_not_a_diagnosis(tmp_path, monkeypatch):
    class Segmenter:
        def __init__(self, *args, **kwargs):
            pass

        def segment(self, image, region):
            assert region == (0.1, 0.1, 0.9, 0.9)
            return np.array([[0, 1], [0, 0]], dtype=np.uint8), 0.81

    monkeypatch.setattr(module, "MedSamCapabilityAdapter", Segmenter)
    pool = CapabilityPool({"mask": {"id": "wanglab/medsam-vit-base", "adapter": "medsam"}}, None)
    evidence = pool.infer(
        "mask", request(picture(tmp_path), "segmentation", region=(0.1, 0.1, 0.9, 0.9))
    ).items[0]
    assert evidence.payload["semantic_class"] == "unknown"
    assert evidence.payload["foreground_fraction_of_image"] == 0.25
    assert evidence.payload["foreground_bbox_xyxy_normalized"] == [0.5, 0.0, 1.0, 0.5]
    assert evidence.payload["foreground_centroid_xy_normalized"] == [0.75, 0.25]
    assert evidence.confidence is None
    assert evidence.payload["predicted_iou_is_calibrated"] is False


def test_medsam_different_regions_have_different_evidence_ids(tmp_path, monkeypatch):
    class Segmenter:
        def __init__(self, *args, **kwargs):
            pass

        def segment(self, image, region):
            return np.zeros((2, 2), dtype=np.uint8), 0.4

    monkeypatch.setattr(module, "MedSamCapabilityAdapter", Segmenter)
    pool = CapabilityPool({"mask": {"id": "test", "adapter": "medsam"}}, None)
    image = picture(tmp_path)
    first = pool.infer("mask", request(image, "segmentation", region=(0, 0, 0.5, 0.5))).items[0]
    second = pool.infer("mask", request(image, "segmentation", region=(0.5, 0.5, 1, 1))).items[0]
    assert first.evidence_id != second.evidence_id
    assert first.payload["foreground_bbox_xyxy_normalized"] is None
    assert first.payload["foreground_centroid_xy_normalized"] is None


@pytest.mark.parametrize(
    "adapter,capability", [("biomedclip", "classification"), ("source_retrieval", "retrieval")]
)
def test_global_image_tools_reject_roi(tmp_path, fake_encoder, adapter, capability):
    _, loads = fake_encoder
    pool = CapabilityPool({"tool": {"id": "test", "adapter": adapter}}, None)
    result = pool.infer("tool", request(picture(tmp_path), capability, region=(0, 0, 0.5, 0.5)))
    assert result.reason == "unsupported_region"
    assert not loads


@pytest.mark.parametrize(
    "mask", [np.zeros((2, 3)), np.ones((2, 3)), np.array([[0, 1, 0], [1, 0, 1]])]
)
def test_mask_rle_round_trip(mask):
    encoded = encode_binary_mask(mask)
    flat = []
    for i, count in enumerate(encoded["counts"]):
        flat.extend([i % 2] * count)
    np.testing.assert_array_equal(np.asarray(flat).reshape(encoded["size"]), mask)


def test_invalid_roi_rejected_before_model_load(tmp_path, fake_encoder):
    pool = CapabilityPool({"mask": {"id": "wanglab/medsam-vit-base", "adapter": "medsam"}}, None)
    req = request(picture(tmp_path), "segmentation")
    with pytest.raises(ValueError, match="region"):
        pool.infer("mask", replace(req, region=(1, 0, 0, 1)))
    assert not pool.models


def test_real_tiny_sam_processor_and_forward_without_download(tmp_path):
    """Checks spatial plumbing with random weights; NOT MedSAM clinical validation."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import SamConfig, SamImageProcessor, SamModel, SamProcessor

    config = SamConfig(
        vision_config={
            "hidden_size": 32,
            "output_channels": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 32,
            "patch_size": 4,
            "num_pos_feats": 16,
            "global_attn_indexes": [0],
            "window_size": 4,
            "mlp_dim": 64,
        },
        prompt_encoder_config={
            "hidden_size": 32,
            "image_size": 32,
            "image_embedding_size": 8,
            "patch_size": 4,
            "mask_input_channels": 8,
        },
        mask_decoder_config={
            "hidden_size": 32,
            "mlp_dim": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "iou_head_hidden_dim": 32,
        },
    )
    adapter = object.__new__(module.MedSamCapabilityAdapter)
    adapter.torch, adapter.device = torch, torch.device("cpu")
    adapter.model = SamModel(config).eval()
    adapter.processor = SamProcessor(
        SamImageProcessor(
            size={"longest_edge": 32},
            pad_size={"height": 32, "width": 32},
            mask_size={"longest_edge": 32},
            mask_pad_size={"height": 32, "width": 32},
        )
    )
    mask, predicted_iou = adapter.segment(picture(tmp_path), (0.1, 0.1, 0.9, 0.9))
    assert mask.shape == (5, 8)
    assert np.isin(mask, (0, 1)).all()
    assert np.isfinite(predicted_iou)
