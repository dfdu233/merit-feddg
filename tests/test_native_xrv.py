"""Optional local XRV architecture/preprocessing integration; no clinical claims."""

import numpy as np
import pytest
from PIL import Image

from merit_feddg.experts.native_xrv import XrvCapabilityAdapter


def test_real_xrv_preprocessing_and_random_local_densenet(tmp_path):
    """Actual official architecture with random weights, never pretrained download."""
    torch = pytest.importorskip("torch")
    xrv = pytest.importorskip("torchxrayvision")
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = xrv.models.DenseNet(weights=None)
        checkpoint = tmp_path / "random-state.pt"
        torch.save(model.state_dict(), checkpoint)
        del model
        image = Image.fromarray(np.arange(80, dtype=np.uint8).reshape(8, 10))
        adapter = XrvCapabilityAdapter(checkpoint, "classification", device="cpu")
        labels, scores, transform = adapter.classify(image)
        assert len(labels) == 18 and scores.shape == (18,)
        assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
        assert adapter.model.op_threshs is None
        assert transform["crop_box_xyxy_normalized"] == [.1, 0., .9, 1.]
        assert transform["outside_crop"] == "unknown"
    finally:
        torch.set_num_threads(original_threads)


def test_xrv_preprocessing_matches_official_center_crop_for_odd_size():
    torch = pytest.importorskip("torch")
    xrv = pytest.importorskip("torchxrayvision")
    adapter = object.__new__(XrvCapabilityAdapter)
    adapter.xrv, adapter.torch, adapter.device = xrv, torch, torch.device("cpu")
    adapter.resolution = 224
    pixels = np.arange(40, dtype=np.uint8).reshape(5, 8)
    actual, transform = adapter._inputs(Image.fromarray(pixels))
    normalized = xrv.datasets.normalize(pixels, 255)[None]
    expected = xrv.datasets.XRayResizer(224)(xrv.datasets.XRayCenterCrop()(normalized))
    np.testing.assert_allclose(actual.numpy()[0], expected)
    assert transform["crop_box_xyxy_normalized"] == [.25, 0., .875, 1.]
