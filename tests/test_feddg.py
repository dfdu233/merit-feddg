import numpy as np
from PIL import Image

from merit_feddg.feddg import FederatedReliabilityCalibrator, continuous_frequency_mix
from merit_feddg.io import load_yaml
from merit_feddg.simulation import simulate_records


def test_calibrator_never_uses_target_domains():
    config = load_yaml("configs/smoke.yaml")
    records = simulate_records(config)
    calibrator = FederatedReliabilityCalibrator()
    calibrator.fit(records, set(config["source_domains"]))
    observed_domains = {item.domain for item in calibrator.client_statistics}
    assert observed_domains == set(config["source_domains"])
    assert not observed_domains.intersection(config["target_domains"])


def test_frequency_mix_preserves_shape_and_range():
    source = Image.fromarray(np.full((16, 12, 3), 30, dtype=np.uint8))
    peer = Image.fromarray(np.full((16, 12, 3), 220, dtype=np.uint8))
    mixed = continuous_frequency_mix(source, peer, alpha=0.2)
    assert mixed.size == source.size
    values = np.asarray(mixed)
    assert values.min() >= 0 and values.max() <= 255
