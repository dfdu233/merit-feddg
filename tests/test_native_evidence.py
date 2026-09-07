import copy
import json

import numpy as np
import pytest
from PIL import Image

from merit_feddg.capabilities import EvidenceItem
from merit_feddg.capability_experts import encode_binary_mask
from merit_feddg.native_evidence import compile_evidence, make_visual_evidence


def evidence(capability, payload, **kwargs):
    return EvidenceItem("prediction-1", "expert-1", capability, "native", payload, **kwargs)


def test_compile_question_priority_preserves_scores_semantics_and_original():
    item = evidence("classification", {
        "findings": [{"finding": "Lung opacity", "score": 0.8},
                     {"finding": "Cardiomegaly", "score": 0.001}],
        "score_semantics": "uncalibrated_independent_sigmoid",
        "positive_threshold": "not_calibrated_for_query_domain",
        "unlisted_findings": "unknown", "usage": {"tokens": 900},
    })
    before = copy.deepcopy(item)
    result = compile_evidence([item], "Is cardiomegaly present?")
    assert result[0]["payload"]["findings"][0] == {"finding": "Cardiomegaly", "score": 0.001}
    assert result[0]["payload"]["unlisted_findings"] == "unknown"
    assert "usage" not in result[0]["payload"]
    assert "negative" not in json.dumps(result)
    assert item == before


def test_compile_all_capabilities_no_local_paths_or_empty_generation():
    items = [
        evidence("retrieval", {"references": [{"source_id": "case-1", "source_image": "/secret/image",
                 "source_question": "What organ?", "source_reference": "lung",
                 "reference_applies_to": "source_image_only"}], "query_diagnosis": "not_inferred"}),
        evidence("segmentation", {"mask": encode_binary_mask(np.zeros((2, 2))),
                 "empty_mask_means": "no_predicted_foreground_not_anatomical_absence"}),
        evidence("detection", {"detections": [{"label": "region", "bbox_xyxy_normalized": [0, 0, 1, 1]}]}),
        evidence("generation", {"generated_text": "Unverified specialist observation.",
                 "observation_status": "unverified_specialist_output", "usage": {"tokens": 3}}),
        evidence("generation", {"generated_text": "  ", "empty_generation": True}),
    ]
    result = compile_evidence(items, "organ", max_chars=4000)
    assert len(result) == 4
    serialized = json.dumps(result)
    assert "/secret" not in serialized and "counts" not in serialized
    assert "source_image_only" in serialized and "not_inferred" in serialized
    assert "not_anatomical_absence" in serialized
    assert "unverified_specialist_output" in serialized
    assert items[1].payload["mask"]["counts"] == [4]


def test_budget_keeps_complete_records_and_whole_relevant_entries():
    entries = [{"concept": "lung", "similarity": 0.1}]
    entries += [{"concept": f"unrelated-{i}", "similarity": 0.8} for i in range(30)]
    item = evidence("classification", {"catalog": entries, "catalog_exhaustive": False})
    for budget in (2, 140, 250, 500, 1800):
        result = compile_evidence([item], "lung", max_chars=budget)
        assert len(json.dumps(result)) <= budget
        assert json.loads(json.dumps(result)) == result
        if result:
            assert result[0]["payload"]["catalog"][0]["concept"] == "lung"
            assert all(key in result[0] for key in ("expert_id", "evidence_id", "capability", "scope", "payload"))
    assert len(item.payload["catalog"]) == 31
    assert compile_evidence([evidence("generation", {"generated_text": "x" * 2000})], "", 300) == []


@pytest.mark.parametrize("budget", [0, 1, -2, True, 1.3])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        compile_evidence([], "", budget)


def test_original_mask_overlay_and_image_are_immutable(monkeypatch):
    image = Image.new("RGB", (40, 30), "gray")
    original = image.tobytes()
    mask = np.zeros((30, 40), dtype=np.uint8)
    mask[10:20, 15:25] = 1
    item = evidence("segmentation", {
        "mask": encode_binary_mask(mask), "semantic_class": "unknown", "mask_path": "/do/not/read",
    }, provenance={"mask_resolution": "original_image", "target_mask_used": False})
    before = copy.deepcopy(item)
    monkeypatch.setattr(Image, "open", lambda *a, **kw: pytest.fail("Tool paths must not be opened"))
    views, meta = make_visual_evidence(image, [item], "foreground", max_views=5)
    assert len(views) == len(meta) == 1
    assert views[0].size == image.size and views[0].tobytes() != original
    assert image.tobytes() == original and item == before
    assert meta[0]["sources"][0]["coordinates"] == [15 / 40, 10 / 30, 25 / 40, 20 / 30]
    assert meta[0]["sources"][0]["representation"] == "predicted_mask"
    assert meta[0]["sources"][0]["label"] == "unknown"
    assert meta[0]["pixel_digest"] != meta[0]["original_pixel_digest"]
    assert make_visual_evidence(image, [item], "foreground")[1] == meta


def test_crop_grid_maps_to_original_coordinates_and_checks_transform():
    image = Image.new("RGB", (40, 20), "gray")
    payload = {
        "structures": [{"anatomical_structure": "Left Lung", "mask": encode_binary_mask([[1, 0], [0, 0]]),
                        "mask_coordinate_system": "model_grid_of_center_crop"}],
        "image_transform": {"original_size_hw": [20, 40], "model_size_hw": [2, 2],
                            "crop_box_xyxy_normalized": [0.25, 0, 0.75, 1]},
    }
    views, meta = make_visual_evidence(image, [evidence("segmentation", payload)], "lung")
    assert len(views) == 1
    assert meta[0]["sources"][0]["coordinates"] == [0.25, 0, 0.5, 0.5]
    payload["image_transform"]["original_size_hw"] = [40, 20]
    assert make_visual_evidence(image, [evidence("segmentation", payload)], "lung") == ([], [])


@pytest.mark.parametrize("mask", [
    {"encoding": "unknown", "size": [2, 2], "counts": [0, 4]},
    {"encoding": "rle-row-major-zero-first", "size": [2, 2], "counts": [-1, 5]},
    {"encoding": "rle-row-major-zero-first", "size": [2, 2], "counts": [0, 3]},
    {"encoding": "rle-row-major-zero-first", "size": [2, 2], "counts": [True, 3]},
    {"encoding": "rle-row-major-zero-first", "size": [2, 2], "counts": [0, 2, 0, 2]},
    {"encoding": "rle-row-major-zero-first", "size": [99999, 99999], "counts": [0, 9999800001]},
    {"encoding": "rle-row-major-zero-first", "size": [2, 2], "counts": [4]},
])
def test_malformed_or_empty_masks_do_not_fall_back_to_box(mask):
    item = evidence("segmentation", {"mask": mask, "prompt_box_xyxy_normalized": [0, 0, 1, 1],
                    "bbox_xyxy_normalized": [0, 0, 1, 1]},
                    provenance={"mask_resolution": "original_image"})
    assert make_visual_evidence(Image.new("RGB", (2, 2)), [item], "") == ([], [])


@pytest.mark.parametrize("box", [[0, 0, 1, 1], [0.25, 0.25, 0.5, 0.75]])
def test_explicit_detection_box_view(box):
    item = evidence("detection", {"detections": [{"label": "unverified region", "bbox_xyxy_normalized": box}]})
    views, meta = make_visual_evidence(Image.new("RGB", (40, 40)), [item], "")
    assert len(views) == 1
    assert meta[0]["sources"][0]["coordinates"] == box
    assert meta[0]["sources"][0]["representation"] == "predicted_bbox"


def test_crop_and_control_keep_equal_area_and_original_pixels():
    image = Image.fromarray(np.arange(40 * 40 * 3, dtype=np.uint8).reshape(40, 40, 3))
    item = evidence("detection", {"detections": [
        {"label": "anatomy", "bbox_xyxy_normalized": [.1, .1, .4, .4]}]})
    before = image.tobytes()
    crops, crop_meta = make_visual_evidence(image, [item], "", mode="crop")
    controls, control_meta = make_visual_evidence(image, [item], "", mode="control_crop")
    assert crops[0].size == controls[0].size == (12, 12)
    assert crop_meta[0]["sources"] == control_meta[0]["sources"]
    assert crop_meta[0]["crop_box_pixels"] != control_meta[0]["crop_box_pixels"]
    assert crops[0].tobytes() == image.crop((4, 4, 16, 16)).tobytes()
    assert image.tobytes() == before
    assert control_meta[0]["control_is_not_guaranteed_irrelevant"]


@pytest.mark.parametrize("box", [[1, 0, 0, 1], [0, 0, 40, 40], [0, 0, float("nan"), 1],
                                 [0, 0, float("inf"), 1], [True, 0, 1, 1], [], "0,0,1,1"])
def test_invalid_box_and_unknown_coordinates_skipped(box):
    item = evidence("detection", {"detections": [{"bbox_xyxy_normalized": box}]})
    assert make_visual_evidence(Image.new("RGB", (10, 10)), [item], "") == ([], [])
    unknown = evidence("detection", {"detections": [{"bbox": [0, 0, 1, 1]}]})
    assert make_visual_evidence(Image.new("RGB", (10, 10)), [unknown], "") == ([], [])


def test_no_visual_for_text_unknown_mask_coordinates_or_target_annotations():
    image = Image.new("RGB", (2, 2))
    mask = encode_binary_mask([[0, 1], [1, 0]])
    items = [evidence("generation", {"generated_text": "a finding"}),
             evidence("segmentation", {"mask": mask}),
             evidence("segmentation", {"mask": mask}, provenance={
                 "mask_resolution": "original_image", "target_mask_used": True})]
    assert make_visual_evidence(image, items, "") == ([], [])
    assert make_visual_evidence(image, items, "", max_views=0) == ([], [])


def test_question_relevant_annotation_first():
    item = evidence("detection", {"detections": [
        {"label": "lung", "bbox_xyxy_normalized": [0, 0, 0.3, 0.3]},
        {"label": "heart", "bbox_xyxy_normalized": [0.5, 0.5, 0.9, 0.9]},
    ]})
    _, metadata = make_visual_evidence(Image.new("RGB", (40, 40)), [item], "Where is the heart?")
    assert metadata[0]["sources"][0]["label"] == "heart"


def test_unknown_items_and_empty_text_are_skipped_without_crashing():
    items = [None, {}, {"capability": []}, evidence("generation", {"generated_text": ["not text"]})]
    assert compile_evidence(items, "") == []
    assert make_visual_evidence(Image.new("RGB", (5, 5)), items, "") == ([], [])
    record = evidence("classification", {
        "findings": [{"finding": "lung", "score": 0.1}], "catalog": None,
    })
    assert compile_evidence([record], "lung")[0]["payload"]["catalog"] is None


def test_prompt_mask_or_pixel_detection_box_is_never_rendered_as_prediction():
    items = [
        evidence("segmentation", {"prompt_box_xyxy_normalized": [0, 0, 1, 1]}),
        evidence("detection", {"detections": [{"bbox_xyxy": [0, 0, 1, 1]}]}),
    ]
    assert make_visual_evidence(Image.new("RGB", (5, 5)), items, "") == ([], [])


@pytest.mark.parametrize("capability,payload", [
    ("classification", {"catalog": [{"prompt": "discarded"}], "score_semantics": "relative_similarity"}),
    ("classification", {"findings": [{"finding": "lung", "score": float("nan")}]}),
    ("classification", {"catalog": [], "findings": [], "unlisted_findings": "unknown"}),
    ("retrieval", {"references": [{"source_image": "/private/path"}]}),
    ("retrieval", {"references": [{"source_id": "pointer-only", "similarity": 0.5}]}),
    ("segmentation", {"structures": [{"anatomical_structure": "heart", "mask": {}}]}),
    ("segmentation", {"mask": {"encoding": "rle-row-major-zero-first", "size": [2, 2], "counts": [5]}}),
    ("detection", {"detections": [{"label": "unverified", "bbox_xyxy_normalized": [1, 0, 0, 1]}]}),
    ("generation", {"generated_text": "", "observation_status": "unverified_specialist_output"}),
])
def test_metadata_without_usable_content_cannot_be_presented(capability, payload):
    assert compile_evidence([evidence(capability, payload)], "lung") == []


def test_budget_cannot_leave_metadata_only_record_after_oversized_finding():
    item = evidence("classification", {"catalog": [
        {"concept": "lung " + "large " * 1000, "similarity": 0.1},
        {"concept": "heart " + "large " * 1000, "similarity": 0.2},
    ], "score_semantics": "relative_similarity", "unlisted_concepts": "unknown"})
    assert compile_evidence([item], "lung", max_chars=350) == []
