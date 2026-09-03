import numpy as np
from PIL import Image

from merit_feddg.io import load_yaml
from merit_feddg.prepare import image_bytes_are_valid, proxy_domain
from merit_feddg.runner import make_oracle_records
from merit_feddg.simulation import simulate_records


def test_proxy_domain_is_stable_and_image_level():
    key = "01234567" + "0" * 56
    assert proxy_domain("example", key) == proxy_domain("example", key)
    assert proxy_domain("example", key).endswith(("source_a", "source_b", "target"))


def test_image_validation_rejects_arbitrary_bytes(tmp_path):
    path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)
    assert image_bytes_are_valid(path.read_bytes())
    assert not image_bytes_are_valid(b"not-an-image")


def test_oracle_cache_routes_to_ground_truth_modality():
    records = simulate_records(load_yaml("configs/smoke.yaml"))[:3]
    converted = make_oracle_records(records)
    for record in converted:
        assert max(record.router_probs, key=record.router_probs.get) == record.modality
