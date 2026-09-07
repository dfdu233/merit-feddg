import numpy as np
import pytest
from PIL import Image

from merit_feddg.behavior_probe import probe_xrv, summarize_outputs, weak_views
from merit_feddg.capabilities import CapabilityRequest
from merit_feddg.capability_experts import CapabilityPool
from merit_feddg.capability_runtime import ValueGenerationConfig


def test_independent_scores_not_softmax_and_constant_wrong_is_not_certified():
    result = summarize_outputs([[.9, .9], [.9, .3]], "classification")
    assert result["sensitivity"] == pytest.approx(.6)
    assert result["per_label_change"] == pytest.approx([0, .6])
    result = summarize_outputs([[.99, .99], [.99, .99]], "classification")
    assert result["sensitivity"] == 0
    assert "correctness" not in result


def test_empty_masks_not_positive_evidence():
    a = np.zeros((2, 3, 3))
    result = summarize_outputs([a, a], "segmentation")
    assert result["sensitivity"] is None and not result["informative"]
    b = a.copy()
    b[0, 0, 0] = 1
    result = summarize_outputs([a, b], "segmentation")
    assert result["sensitivity"] == 1 and result["empty_pairs"] == 1


@pytest.mark.parametrize("outputs", [[[0.2], [float("nan")]], [[1.1], [.2]], [[.2], [.2, .3]]])
def test_bad_outputs_fail(outputs):
    with pytest.raises(ValueError):
        summarize_outputs(outputs, "classification")


def test_photometric_probe_preserves_original_and_reports_cost():
    image = Image.fromarray(np.arange(120, dtype=np.uint8).reshape(10, 12))
    before = image.tobytes()
    views = weak_views(image)
    assert len(views) == 3 and all(v.size == image.size for _, v in views)
    class Model:
        def classify(self, view):
            return ["a", "b"], [.8, .9], {"size": list(view.size)}
    result = probe_xrv(Model(), image, "classification")
    assert result["extra_model_calls"] == 3 and result["sensitivity"] == 0
    assert result["clinical_invariance_verified"] is False
    assert image.tobytes() == before


def test_reject_needs_explicit_threshold():
    with pytest.raises(ValueError):
        ValueGenerationConfig(behavior_probe="reject")
    with pytest.raises(ValueError):
        ValueGenerationConfig(behavior_max_sensitivity=float("nan"))
    assert ValueGenerationConfig().behavior_probe == "off"


def test_pool_does_not_load_unsupported_expert(monkeypatch):
    pool = CapabilityPool({"a": {"adapter": "conch"}}, "artifacts")
    monkeypatch.setattr(pool, "_model", lambda _: pytest.fail("must not load unsupported model"))
    request = CapabilityRequest("id", "unused", "q", "pathology", "open_vqa", "d", "g",
                                "classification", scope="tissue")
    assert pool.behavior_probe("a", request)["status"] == "unsupported"


def test_pool_segmentation_probe_uses_declared_structures(monkeypatch):
    pool = CapabilityPool({"a": {"adapter": "xrv_anatomy", "structures": ["Heart"]}}, "artifacts")
    class Model:
        def segment(self, image):
            return ["Lung", "Heart"], np.ones((2, 4, 4)), {"grid": [4, 4]}
    monkeypatch.setattr(pool, "_model", lambda _: Model())
    request = CapabilityRequest("id", Image.new("RGB", (4, 4)), "q", "cxr", "open_vqa",
                                "d", "g", "segmentation", scope="anatomy")
    result = pool.behavior_probe("a", request)
    assert result["labels"] == ["Heart"] and result["extra_model_calls"] == 3
    assert result["sensitivity"] == 0
