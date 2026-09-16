import hashlib

import pytest
from PIL import Image

from merit_feddg.open_study import fingerprint
from scripts.evaluate_control_evidence import distribution
from scripts.run_control_evidence import check_route, image_identity, sha


def test_file_and_legacy_rgb_identity(tmp_path):
    path = tmp_path / "image.png"
    image = Image.new("RGB", (4, 5), "red")
    image.save(path)
    row = {"image": str(path), "image_sha256": sha(path)}
    assert image_identity(row) == sha(path)
    row["image_sha256"] = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    assert image_identity(row) == sha(path)
    Image.new("RGB", (4, 5), "blue").save(path)
    with pytest.raises(ValueError, match="neither"):
        image_identity(row)


def test_legacy_route_requires_exact_identity():
    origin = {"identity": "donor", "config": {"routing": {"enabled": True}}}
    row = {"image_sha256": "pixels"}
    route = {"cache_key": fingerprint({"model_runtime": {"identity": "donor"},
                                      "image_sha256": "pixels", "routing": {"enabled": True}})}
    check_route(row, route, origin)
    with pytest.raises(ValueError, match="cache key"):
        check_route({"image_sha256": "different"}, route, origin)
    with pytest.raises(ValueError, match="mismatch"):
        check_route(row, {"group_id": "different", **route}, origin)


def test_unmeasured_distribution_is_not_zero():
    assert distribution([]) is None
    measured = distribution([0., .01, .02])
    assert measured["n"] == 3
    assert measured["median"] == .01
    assert measured["p95_nearest_rank"] == .02
