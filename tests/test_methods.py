import numpy as np

from merit_feddg.io import load_yaml
from merit_feddg.methods import predict, specialist_lens
from merit_feddg.simulation import simulate_records


def test_specialist_magnitude_is_not_injected():
    config = load_yaml("configs/smoke.yaml")
    record = simulate_records(config)[0]
    route = record.modality
    base_scores, base_gate, _ = specialist_lens(record, route, 0.9, config["method"], 0.8)
    record.expert_scores[route] = record.expert_scores[route] * 1000.0
    scaled_scores, scaled_gate, _ = specialist_lens(record, route, 0.9, config["method"], 0.8)
    np.testing.assert_allclose(base_scores, scaled_scores, atol=1e-10)
    assert base_gate == scaled_gate


def test_correction_is_bounded():
    config = load_yaml("configs/smoke.yaml")
    record = simulate_records(config)[1]
    route = record.modality
    corrected, gate, _ = specialist_lens(record, route, 1.0, config["method"], 1.0)
    delta = corrected - record.general_final_logits
    assert np.linalg.norm(delta) <= config["method"]["max_correction_norm"] + 1e-8
    assert 0.0 <= gate <= 1.0


def test_uniform_router_abstains_from_intervention():
    config = load_yaml("configs/smoke.yaml")
    record = simulate_records(config)[2]
    record.router_probs = {name: 1.0 / len(record.router_probs) for name in record.router_probs}
    prediction = predict(record, "merit", config["method"])
    assert prediction.route_confidence == 0.0
    assert prediction.intervention_gate == 0.0
    np.testing.assert_allclose(prediction.scores, record.general_final_logits)
